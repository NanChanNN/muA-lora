import argparse
import logging
import os
from functools import partial
from typing import Dict, Union, List

import evaluate
from peft import LoraConfig, get_peft_model
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, set_seed
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
    InterpolationMode,
)

import wandb
from utils.data_utils import vit_collate_fn, fetch_image_dataset
from utils.eval_utils import compute_image_cls_metrics
from utils.custom_trainer import FixedHeadLRTrainer
from utils.helper import naming_conversion, setup_logging, print_trainable_parameters
from utils.lora_utils import custom_init


def load_peft_vit(base_model_name: str,
                  label2id: Dict[str, int],
                  id2label: Dict[int, str],
                  rank: int,
                  target_modules : Union[List[str], str] = 'all-linear',
                  lora_dropout: float = 0.0,
                  init_lora_weights: bool = True,
                  mode: str = None):
    base_model = AutoModelForImageClassification.from_pretrained(f"google/{base_model_name}",
                                                                 label2id=label2id,
                                                                 id2label=id2label,
                                                                 ignore_mismatched_sizes=True,)

    if rank == 0:
        return base_model

    if mode is not None and 'alpha1' in mode:
        lora_alpha = 1
    elif mode is not None and 'constant1' in mode:
        lora_alpha = rank
    else:
        lora_alpha = 2 * rank

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        use_rslora=False,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        init_lora_weights=init_lora_weights,
        modules_to_save=["classifier"],
    )

    model = get_peft_model(base_model, lora_config)

    if mode is not None:
        if mode.startswith('initA'):
            custom_init(model, 'initA')
        elif mode.startswith('initB'):
            custom_init(model, 'initB')

    return model


def main(args):
    # Load image processor
    image_processor = AutoImageProcessor.from_pretrained(f"google/{args.base_model}",
                                                         use_fast=True)

    # Load dataset
    dataset = fetch_image_dataset(args.task)
    train_set = dataset['train']
    val_set = dataset['validation']

    labels = train_set.features["label"].names
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop(image_processor.size["height"], interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    val_transforms = Compose(
        [
            Resize(256, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_processor.size["height"]),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_train(example_batch):
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch.pop("image")]
        return example_batch

    def preprocess_val(example_batch):
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch.pop("image")]
        return example_batch

    train_set.set_transform(preprocess_train)
    val_set.set_transform(preprocess_val)

    # Load metric
    metric = evaluate.load("accuracy")

    # Construct model
    model = load_peft_vit(base_model_name=args.base_model,
                          label2id=label2id,
                          id2label=id2label,
                          rank=args.rank,
                          target_modules=args.target_modules,
                          lora_dropout=args.lora_dropout,
                          init_lora_weights=True,
                          mode=args.init_method,)
    print_trainable_parameters(model)

    training_args = TrainingArguments(output_dir=args.save_dir,
                                      per_device_train_batch_size=args.batch_size,
                                      per_device_eval_batch_size=args.eval_batch_size,
                                      gradient_accumulation_steps=args.grad_accum,
                                      learning_rate=args.learning_rate,
                                      weight_decay=args.weight_decay,
                                      optim='adamw_torch_fused',
                                      adam_beta1=args.adam_beta1,
                                      adam_beta2=args.adam_beta2,
                                      max_grad_norm=args.max_grad_norm,
                                      num_train_epochs=args.epochs,
                                      warmup_ratio=args.warmup_ratio,
                                      lr_scheduler_type=args.lr_scheduler_type,
                                      lr_scheduler_kwargs=args.lr_scheduler_kwargs,
                                      save_strategy="no",
                                      bf16=True,
                                      tf32=True,
                                      gradient_checkpointing=args.gradient_checkpointing,
                                      dataloader_pin_memory=True,
                                      dataloader_persistent_workers=True,
                                      dataloader_num_workers=12,
                                      dataloader_prefetch_factor=2,
                                      label_names=["labels"],
                                      logging_strategy='steps',
                                      logging_steps=args.logging_steps,
                                      eval_strategy="steps",
                                      eval_steps=args.eval_steps,
                                      report_to=['wandb'] if args.enable_log else [],
                                      disable_tqdm=False,
                                      remove_unused_columns=False,
                                      seed=args.seed,)

    if args.head_learning_rate is not None:
        training_args.head_learning_rate = args.head_learning_rate
        selected_trainer = FixedHeadLRTrainer
    else:
        selected_trainer = Trainer

    trainer = selected_trainer(model=model,
                               args=training_args,
                               train_dataset=train_set,
                               eval_dataset=val_set,
                               compute_metrics=partial(compute_image_cls_metrics, metric=metric),
                               data_collator=vit_collate_fn,)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic arguments
    parser.add_argument('--base-model', type=str,
                        choices=['vit-base-patch16-224-in21k',
                                 'vit-large-patch16-224-in21k',
                                 'vit-huge-patch14-224-in21k'],
                        default='vit-huge-patch14-224-in21k')
    parser.add_argument('--task', type=str,
                        choices=['food101', 'imagenet-1k'],
                        default='imagenet-1k')

    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--lora-dropout', type=float, default=0.0)
    parser.add_argument("--target-modules",
                        nargs="+",
                        metavar="MODULE",
                        default=["output.dense", "attention.output.dense", "intermediate.dense", "query", "key", "value"],)
    parser.add_argument('--init-method', type=str, default=None,
                        choices=[None, 'initA', 'initB', 'initA_alpha1', 'initB_alpha1', 'alpha1',
                                 'initA_constant1', 'initB_constant1', 'constant1'])

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=256)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--learning-rate-exponent', type=float, required=True)
    parser.add_argument('--head-learning-rate', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--lr-scheduler-type', type=str, default='cosine_with_min_lr')
    parser.add_argument('--save-steps', type=int, default=5000)
    parser.add_argument('--eval-steps', type=int, default=250)
    parser.add_argument('--logging-steps', type=int, default=20)

    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--enable-log', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    args.learning_rate = 2 ** (-args.learning_rate_exponent)

    if args.init_method is not None and args.rank != 0:
        args.suffix = '-' + args.init_method
    else:
        args.suffix = ''

    if args.lr_scheduler_type == 'cosine_with_min_lr':
        args.lr_scheduler_kwargs = {'min_lr_rate': 0.1}
    else:
        args.lr_scheduler_kwargs = None

    save_dir = os.path.join('checkpoints', args.base_model, args.task, f'LoRA-{args.rank}{args.suffix}', f'LR-{naming_conversion(args.learning_rate_exponent)}')
    os.makedirs(save_dir, exist_ok=True)

    log_dir = os.path.join('logs', args.base_model, args.task, f'LoRA-{args.rank}{args.suffix}')
    setup_logging(log_dir)
    logging.info(args)

    args.log_dir = log_dir
    args.save_dir = save_dir
    set_seed(args.seed, deterministic=False)

    main(args)
