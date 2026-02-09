import argparse
import logging
import os
from typing import Union, List

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from trl import SFTConfig, SFTTrainer

import wandb

from utils.helper import naming_conversion, print_trainable_parameters, setup_logging
from utils.lora_utils import custom_init
from utils.data_utils import preprocess_openthoughts, filter_by_length


ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    ddp_rank = int(os.environ['RANK'])
    master_process = ddp_rank == 0
else:
    master_process = True


def load_peft_qwen(base_model_name: str,
                   rank: int,
                   target_modules : Union[List[str], str] = 'all-linear',
                   lora_dropout: float = 0.0,
                   init_lora_weights: bool = True,
                   mode: str = None):
    base_model = AutoModelForCausalLM.from_pretrained(f"Qwen/{base_model_name}",
                                                      dtype=torch.bfloat16,
                                                      device_map="auto",
                                                      attn_implementation="kernels-community/vllm-flash-attn3")

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
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=init_lora_weights,
    )

    model = get_peft_model(base_model, lora_config)

    if mode is not None:
        if mode.startswith('initA'):
            custom_init(model, 'initA')
        elif mode.startswith('initB'):
            custom_init(model, 'initB')

    return model


def main(args):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{args.base_model}")
    with open('chat_templates/qwen_template.jinja', 'r') as f:
        tokenizer.chat_template = f.read()

    # Load dataset
    raw_data = load_dataset(f"open-thoughts/{args.task}",
                            "default",
                            split="train")

    raw_data = raw_data.map(preprocess_openthoughts, num_proc=16)
    raw_data = raw_data.remove_columns([c for c in raw_data.column_names if c != "messages"])

    if args.filter_length:
        raw_data = filter_by_length(raw_data, tokenizer, max_len=args.max_seq_length)

    raw_data = raw_data.train_test_split(test_size=args.test_set_ratio,
                                         shuffle=True,
                                         seed=args.seed)
    train_set, test_set = raw_data['train'], raw_data['test']

    # create model
    model = load_peft_qwen(base_model_name=args.base_model,
                           rank=args.rank,
                           target_modules=args.target_modules,
                           lora_dropout=args.lora_dropout,
                           init_lora_weights=True,
                           mode=args.init_method)
    if master_process:
        print_trainable_parameters(model)

    # training
    sft_config = SFTConfig(output_dir=args.save_dir,
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
                           save_strategy='no',
                           bf16=True,
                           tf32=True,
                           gradient_checkpointing=not args.disable_gradient_checkpointing,
                           gradient_checkpointing_kwargs={"use_reentrant": False},
                           ddp_find_unused_parameters=False,
                           dataloader_pin_memory=True,
                           dataloader_persistent_workers=True,
                           dataloader_num_workers=12,
                           dataloader_prefetch_factor=2,
                           logging_strategy='steps',
                           logging_steps=args.logging_steps,
                           eval_strategy='steps',
                           eval_steps=args.eval_steps,
                           report_to=['wandb'] if args.enable_log else [],
                           disable_tqdm=False,
                           seed=args.seed,
                           dataset_num_proc=16,
                           max_length=args.max_seq_length,
                           shuffle_dataset=True,
                           packing=True,
                           packing_strategy="bfd",
                           padding_free=True,
                           pad_to_multiple_of=8,
                           assistant_only_loss=True,)

    sft_trainer = SFTTrainer(model=model,
                             args=sft_config,
                             train_dataset=train_set,
                             eval_dataset=test_set,
                             processing_class=tokenizer, )

    sft_trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic arguments
    parser.add_argument('--base-model', type=str,
                        choices=['Qwen2.5-0.5B-Instruct', 'Qwen2.5-3B-Instruct'],
                        default='Qwen2.5-3B-Instruct')
    parser.add_argument('--task', type=str,
                        choices=['OpenThoughts-114k'],
                        default='OpenThoughts-114k')

    # lora relevant
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--lora-dropout', type=float, default=0.0)
    parser.add_argument("--target-modules",
                        nargs="+",
                        metavar="MODULE",
                        default=['gate_proj', 'up_proj', 'down_proj'],)
    parser.add_argument('--init-method', type=str, default=None,
                        choices=[None, 'initA', 'initB', 'initA_alpha1', 'initB_alpha1', 'alpha1',
                                 'initA_constant1', 'initB_constant1', 'constant1'])

    # training relevant
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--eval-batch-size', type=int, default=6)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--learning-rate-exponent', type=float, required=True)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--lr-scheduler-type', type=str, default='cosine_with_min_lr')
    parser.add_argument('--max-seq-length', type=int, default=8192)
    parser.add_argument('--save-steps', type=int, default=5000)
    parser.add_argument('--eval-steps', type=int, default=460)
    parser.add_argument('--logging-steps', type=int, default=20)

    # training set size
    parser.add_argument('--test-set-ratio', type=float, default=0.08)
    parser.add_argument('--disable-gradient-checkpointing', action='store_true')
    # logging
    parser.add_argument('--enable-log', action='store_true')
    parser.add_argument('--filter-length', action='store_true')
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

    if master_process:
        setup_logging()
        logging.info(args)

    args.save_dir = save_dir
    set_seed(args.seed, deterministic=False)

    main(args)
