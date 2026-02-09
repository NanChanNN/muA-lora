import argparse
import json
import logging
import os
import re
from typing import Union, List

import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed
from trl import SFTConfig, SFTTrainer

import wandb

from utils.helper import setup_logging, naming_conversion, print_trainable_parameters
from utils.lora_utils import custom_init


CANDIDATE_PATTERNS = [
    r"model\.visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc[12])",
    r"model\.visual\.(merger|deepstack_merger_list\.\d+)\.linear_fc[12]",
    r"model\.language_model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)",
    r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
]


def calculate_trainable_parameters(model_name: str,
                                   rank: int,
                                   target_patterns: List[int]):
    if model_name == 'Qwen3-VL-2B-Instruct':
        tparam_list = [
            ((3072 + 1024) + (4096 + 1024) + (1024 + 4096) + (1024 + 1024)) * rank * 24,
            ((4096 + 4096) + (2048 + 4096)) * rank * 4,
            ((6144 + 2048) + (6144 + 2048) + (2048 + 6144)) * rank * 28,
            ((2048 + 2048) + (1024 + 2048) + (1024 + 2048) + (2048 + 2048)) * rank * 28,
        ]
    else:
        raise KeyError(f"Model {model_name} is not supported.")

    sum = 0
    for p_idx in target_patterns:
        sum += tparam_list[p_idx]

    return sum


def load_peft_qwen3_vl(base_model_name: str,
                       rank: int,
                       compiled_patterns : List,
                       lora_dropout: float = 0.0,
                       init_lora_weights: bool = True,
                       mode: str = None):
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(f"Qwen/{base_model_name}",
                                                                 dtype=torch.bfloat16,)

    if rank == 0:
        return base_model

    target_modules = sorted([
        name for name, module in base_model.named_modules()
        if isinstance(module, nn.Linear) and any(p.fullmatch(name) for p in compiled_patterns)
    ])

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
    # Load processor
    processor = AutoProcessor.from_pretrained(f"Qwen/{args.base_model}")

    # Load dataset
    raw_data = load_dataset(f"trl-lib/{args.task}", split='train')

    if args.filter_length:
        filtered_id_path = os.path.join('auxiliary_data', f'{args.task}_filtered_ids.json')
        with open(filtered_id_path, 'r') as f:
            filtered_id = json.load(f)
        raw_data = raw_data.select(filtered_id)

    raw_data = raw_data.train_test_split(test_size=args.test_set_ratio,
                                         shuffle=True,
                                         seed=args.seed)
    train_set, test_set = raw_data['train'], raw_data['test']

    # compile patterns
    compiled = [re.compile(CANDIDATE_PATTERNS[p_idx] + r"$") for p_idx in args.target_patterns]

    # create model
    model = load_peft_qwen3_vl(base_model_name=args.base_model,
                               rank=args.rank,
                               compiled_patterns=compiled,
                               lora_dropout=args.lora_dropout,
                               init_lora_weights=True,
                               mode=args.init_method)
    trainable_params, _ = print_trainable_parameters(model, log=False)

    # check parameter count
    if args.rank > 0:
        theo_tparams = calculate_trainable_parameters(model_name=args.base_model,
                                                      rank=args.rank,
                                                      target_patterns=args.target_patterns,)
        assert theo_tparams == trainable_params, f"Trainable parameter mismatch: expected {theo_tparams}, got {trainable_params}"

    # training
    sft_config = SFTConfig(output_dir=args.save_dir,
                           per_device_train_batch_size=args.batch_size,
                           per_device_eval_batch_size=args.eval_batch_size,
                           gradient_accumulation_steps=args.grad_accum,
                           learning_rate=args.learning_rate,
                           optim='adamw_torch_fused',
                           weight_decay=args.weight_decay,
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
                           dataloader_pin_memory=True,
                           dataloader_persistent_workers=True,
                           dataloader_num_workers=8,
                           dataloader_prefetch_factor=2,
                           logging_strategy='steps',
                           logging_steps=args.logging_steps,
                           eval_strategy='steps',
                           eval_steps=args.eval_steps,
                           report_to=['wandb'] if args.enable_log else [],
                           disable_tqdm=False,
                           seed=args.seed,
                           dataset_num_proc=8,
                           max_length=args.max_seq_length,
                           completion_only_loss=True,)

    sft_trainer = SFTTrainer(model=model,
                             args=sft_config,
                             train_dataset=train_set,
                             eval_dataset=test_set,
                             processing_class=processor, )

    sft_trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic arguments
    parser.add_argument('--base-model', type=str,
                        choices=['Qwen3-VL-2B-Instruct',],
                        default='Qwen3-VL-2B-Instruct')
    parser.add_argument('--task', type=str,
                        choices=['llava-instruct-mix'],
                        default='llava-instruct-mix')

    # lora relevant
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--lora-dropout', type=float, default=0.0)
    parser.add_argument("--target-patterns",
                        nargs="+",
                        metavar="MODULE",
                        type=int,
                        default=[0, 1, 2],)
    parser.add_argument('--init-method', type=str, default=None,
                        choices=[None, 'initA', 'initB', 'initA_alpha1', 'initB_alpha1', 'alpha1',
                                 'initA_constant1', 'initB_constant1', 'constant1'])

    # training relevant
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-batch-size', type=int, default=48)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--learning-rate-exponent', type=float, required=True)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--lr-scheduler-type', type=str, default='cosine_with_min_lr')
    parser.add_argument('--max-seq-length', type=int, default=None)
    parser.add_argument('--save-steps', type=int, default=5000)
    parser.add_argument('--eval-steps', type=int, default=600)
    parser.add_argument('--logging-steps', type=int, default=20)

    # training set size
    parser.add_argument('--test-set-ratio', type=float, default=0.08)
    parser.add_argument('--filter-length', action='store_true')

    # logging
    parser.add_argument('--disable-gradient-checkpointing', action='store_true')
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
#
