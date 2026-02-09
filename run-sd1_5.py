import argparse
import gc
import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from diffusers.training_utils import compute_snr
from diffusers.utils import convert_state_dict_to_diffusers
from torch.optim.lr_scheduler import LambdaLR
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline
from peft import LoraConfig, get_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import set_seed, CLIPTokenizer, CLIPTextModel
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
    InterpolationMode,
)

from tqdm.auto import tqdm

from utils.eval_utils import SDText2ImageEvaluator
from utils.helper import naming_conversion, setup_logging
from utils.lora_utils import custom_init

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def save_lora_weights(unet, save_dir: str):
    unet_ = unet.to(torch.float32)  # stable serialization
    lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet_))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=save_dir,
        unet_lora_layers=lora_state,
        safe_serialization=True,
    )


def collate_fn(examples):
    pixel_values = torch.stack([example['pixel_values'] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example['input_ids'] for example in examples])
    return {'pixel_values': pixel_values, 'input_ids': input_ids}


def load_peft_unet(pretrained_model_name_or_path: str,
                   rank: int,
                   device: torch.device,
                   weight_dtype: torch.dtype,
                   lora_dropout: float = 0.0,
                   mode: str = None,):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", )

    if rank == 0:
        unet.requires_grad_(True)
        unet.to(device, dtype=weight_dtype)
        return unet

    unet.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)

    if mode is not None and 'alpha1' in mode:
        lora_alpha = 1
    elif mode is not None and 'constant1' in mode:
        lora_alpha = rank
    else:
        lora_alpha = 2 * rank

    lora_config = LoraConfig(r=rank,
                             lora_alpha=lora_alpha,
                             use_rslora=False,
                             target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                             lora_dropout=lora_dropout,
                             bias="none",
                             init_lora_weights="gaussian",)
    unet.add_adapter(lora_config)

    if mode is not None:
        if mode.startswith('initA'):
            custom_init(unet, 'initA')
        elif mode.startswith('initB'):
            custom_init(unet, 'initB')

    return unet


def get_optimizer_groups(unet_model,
                         base_n: int,
                         base_lr: float,
                         mode: str):
    module_fanin = {}
    for name, param in unet_model.named_parameters():
        if param.requires_grad and "lora_A" in name:
            prefix = name.split(".lora_A.")[0]
            module_fanin[prefix] = param.shape[1]

    lora_params_by_width = {}
    for name, param in unet_model.named_parameters():
        if not param.requires_grad:
            continue

        if "lora_A" in name:
            prefix = name.split(".lora_A.")[0]
        elif "lora_B" in name:
            prefix = name.split(".lora_B.")[0]
        else:
            continue

        width_n = module_fanin.get(prefix)
        if width_n not in lora_params_by_width:
            lora_params_by_width[width_n] = []
        lora_params_by_width[width_n].append(param)

    optimizer_groups = []
    for width, params in lora_params_by_width.items():
        if mode is not None and 'initB' in mode:
            scaling_factor = base_n / width
        elif mode is not None and 'initA' in mode:
            scaling_factor = math.sqrt(base_n / width)
        else:
            scaling_factor = 1.0

        optimizer_groups.append({
            "params": params,
            "lr": base_lr * scaling_factor
        })
    return optimizer_groups


def main(args):
    # scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    pred_type = noise_scheduler.config.prediction_type
    assert pred_type in ['epsilon', 'v_prediction'], f"Unsupported prediction_type: {pred_type}"

    # text tokenizer and encoder
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    text_encoder.to(args.device, dtype=args.weight_dtype)

    # image encoder
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae",)
    vae.requires_grad_(False)
    vae.to(args.device, dtype=args.weight_dtype)

    # unet
    unet = load_peft_unet(args.pretrained_model_name_or_path,
                          args.rank,
                          args.device,
                          args.weight_dtype,
                          lora_dropout=args.lora_dropout,
                          mode=args.init_method)

    # lora_params = [p for p in unet.parameters() if p.requires_grad]
    # logging.info(f"Trainable parameters: {sum(p.numel() for p in lora_params):,}")
    if args.rank == 0:
        trainable_params = [p for p in unet.parameters() if p.requires_grad]
        param_groups = [{"params": trainable_params, "lr": args.learning_rate}]
    else:
        param_groups = get_optimizer_groups(unet,
                                            base_n=320,
                                            base_lr=args.learning_rate,
                                            mode=args.init_method)
        trainable_params = [p for g in param_groups for p in g["params"]]

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if args.enable_compile:
        unet = torch.compile(unet, mode="reduce-overhead")

    optimizer = torch.optim.AdamW(param_groups,
                                  weight_decay=args.weight_decay,
                                  fused=True)

    # load dataset
    raw_data = load_dataset(f"lambdalabs/{args.task}", split='train')
    raw_data = raw_data.train_test_split(test_size=args.test_set_ratio,
                                         shuffle=True,
                                         seed=args.seed)
    train_set, test_set = raw_data['train'], raw_data['test']

    # setup for training
    train_transforms = Compose(
        [
            Resize(args.resolution, interpolation=InterpolationMode.LANCZOS),
            CenterCrop(args.resolution),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.5], [0.5])
        ]
    )

    def preprocess_train(examples,
                         image_column: str = 'image',
                         caption_column: str = 'text'):
        examples['pixel_values'] = [train_transforms(image.convert("RGB")) for image in examples[image_column]]

        inputs = tokenizer(examples[caption_column],
                           max_length=tokenizer.model_max_length,
                           padding='max_length',
                           truncation=True,
                           return_tensors='pt',)
        examples['input_ids'] = inputs.input_ids

        return examples

    train_set = train_set.with_transform(preprocess_train)
    train_dataloader = DataLoader(train_set,
                                  shuffle=True,
                                  drop_last=True,
                                  collate_fn=collate_fn,
                                  batch_size=args.batch_size,
                                  num_workers=16,
                                  prefetch_factor=4,
                                  pin_memory=True,
                                  persistent_workers=True,)

    # set up for evaluation
    eval_transforms = Compose(
        [
            Resize(args.resolution, interpolation=InterpolationMode.LANCZOS),
            CenterCrop(args.resolution),
        ]
    )

    eval_pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path,
                                                        unet=unet,
                                                        vae=vae,
                                                        text_encoder=text_encoder,
                                                        tokenizer=tokenizer,
                                                        safety_checker=None,
                                                        feature_extractor=None,)
    eval_pipe.to(args.device, dtype=args.weight_dtype)
    eval_pipe.set_progress_bar_config(disable=args.disable_eval_bar)

    evaluator = SDText2ImageEvaluator(test_set=test_set,
                                      image_transform=eval_transforms,
                                      device=args.device,
                                      sample_seed=args.seed,
                                      clip_model_name_or_path=args.clip_model_name_or_path,)

    # lr scheduler
    warmup_steps = int(args.max_train_steps * args.warmup_ratio)
    warmup_steps = max(0, warmup_steps)

    warmup_start_factor = 1e-4
    min_lr_mult = 0.1
    total_steps = args.max_train_steps

    def lr_mult(step: int) -> float:
        if warmup_steps > 0 and step <= warmup_steps:
            return warmup_start_factor + (1.0 - warmup_start_factor) * (step / warmup_steps)

        denom = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / denom
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_lr_mult + (1.0 - min_lr_mult) * cosine

    lr_scheduler = LambdaLR(optimizer,
                            lr_lambda=lr_mult)

    # training
    progress_bar = tqdm(range(0, args.max_train_steps),
                        initial=0,
                        desc='Training Steps',
                        disable=False)

    train_iterator = iter(train_dataloader)
    cur_iter_num = 0
    global_step = 0
    train_loss = 0.

    optimizer.zero_grad(set_to_none=True)
    while True:
        unet.train()

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        with torch.autocast(device_type="cuda", dtype=args.weight_dtype):
            # move data
            batch_pixel_values = batch["pixel_values"].to(args.device, dtype=args.weight_dtype, non_blocking=True)
            batch_input_ids = batch["input_ids"].to(args.device, non_blocking=True)

            # encode images into latent space
            latents = vae.encode(batch_pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # get noise and timestep
            noise = torch.randn_like(latents)

            bsz = latents.size(0)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=args.device).long()

            # forward diffusion process
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # get caption embedding
            encoder_hidden_states = text_encoder(batch_input_ids, return_dict=False)[0]

            # Predict the noise residual and compute loss
            if pred_type == "epsilon":
                target = noise
            else:
                target = noise_scheduler.get_velocity(latents, noise, timesteps)

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

            if args.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')
            else:
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                    dim=1
                )[0]
                if pred_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                else:
                    mse_loss_weights = mse_loss_weights / (snr + 1)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

            loss /= args.grad_accum
        loss.backward()

        train_loss += loss.item()
        cur_iter_num += 1

        if cur_iter_num % args.grad_accum == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress_bar.update(1)

            if global_step % args.logging_steps == 0:
                lrs = lr_scheduler.get_last_lr()
                wandb.log({'global_step': global_step,
                           "lr/min": min(lrs),
                           "lr/max": max(lrs),
                           "lr/first": lrs[0],
                           'grad_norm': grad_norm.item(),
                           'train_loss': train_loss / args.logging_steps})
                train_loss = 0.0

            if global_step % args.eval_steps == 0 or global_step == 1:
                metrics = evaluator.evaluate(pipe=eval_pipe,
                                             batch_size=args.eval_batch_size,
                                             num_inference_steps=args.eval_num_inference_steps,
                                             guidance_scale=args.eval_guidance_scale,
                                             num_images_per_prompt=args.eval_num_images_per_prompt,)

                wandb.log({"global_step": global_step,
                           "eval/fid": metrics["fid"],
                           "eval/clip_score": metrics["clip_score"],
                           "eval/num_generated": metrics["num_generated"],})

            if global_step >= args.max_train_steps:
                break

    save_lora_weights(unet, args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic arguments
    parser.add_argument('--base-model', type=str,
                        choices=['stable-diffusion-v1-5',],
                        default='stable-diffusion-v1-5')
    parser.add_argument('--task', type=str,
                        choices=['naruto-blip-captions'],
                        default='naruto-blip-captions')
    parser.add_argument('--resolution', type=int, default=512)

    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--lora-dropout', type=float, default=0.0)
    parser.add_argument('--init-method', type=str, default=None,
                        choices=[None, 'initA', 'initB', 'initA_alpha1', 'initB_alpha1', 'alpha1',
                                 'initA_constant1', 'initB_constant1', 'constant1'])

    parser.add_argument('--max-train-steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=48)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--learning-rate-exponent', type=float, required=True)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--snr-gamma', type=float, default=5.0)

    parser.add_argument('--logging-steps', type=int, default=50)
    parser.add_argument('--eval-steps', type=int, default=2000)
    parser.add_argument('--disable-eval-bar', action='store_true')

    parser.add_argument('--test-set-ratio', type=float, default=0.08)
    parser.add_argument("--eval-num-inference-steps", type=int, default=40)
    parser.add_argument("--eval-guidance-scale", type=float, default=7.5)
    parser.add_argument("--eval-num-images-per-prompt", type=int, default=2)
    parser.add_argument("--clip-model-name-or-path", type=str, default="openai/clip-vit-large-patch14")

    parser.add_argument('--enable-compile', action='store_true')
    parser.add_argument('--enable-log', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    args.learning_rate = 2 ** (-args.learning_rate_exponent)
    args.pretrained_model_name_or_path = f"stable-diffusion-v1-5/{args.base_model}"
    args.device = torch.device('cuda')
    args.weight_dtype = torch.bfloat16

    if args.init_method is not None and args.rank != 0:
        args.suffix = '-' + args.init_method
    else:
        args.suffix = ''

    save_dir = os.path.join('checkpoints', args.base_model, args.task, f'LoRA-{args.rank}{args.suffix}', f'LR-{naming_conversion(args.learning_rate_exponent)}')
    os.makedirs(save_dir, exist_ok=True)

    log_dir = os.path.join('logs', args.base_model, args.task, f'LoRA-{args.rank}{args.suffix}')
    setup_logging(log_dir)
    logging.info(args)

    wandb.init(project=f'sd',
               config=vars(args),
               mode='online' if args.enable_log else 'disabled', )

    args.log_dir = log_dir
    args.save_dir = save_dir
    set_seed(args.seed, deterministic=False)

    main(args)
