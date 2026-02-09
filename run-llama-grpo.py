import argparse
import logging
import os
from functools import partial
from typing import Union, List

import torch
import wandb
from peft import LoraConfig, TaskType, get_peft_model
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset

from utils.data_utils import load_reasoning_dataset
from utils.grpo_utils import gsm8k_accuracy_reward_func, math_accuracy_reward_func, general_format_reward_func
from utils.helper import naming_conversion, setup_logging
from utils.lora_utils import custom_init


def load_peft_llam3_v2(base_model_name: str,
                       rank: int,
                       target_modules : Union[List[str], str] = 'all-linear',
                       lora_dropout: float = 0.0,
                       init_lora_weights: bool = True,
                       mode: str = None):
    base_model = AutoModelForCausalLM.from_pretrained(f"meta-llama/{base_model_name}",
                                                      dtype=torch.bfloat16,
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
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{args.base_model}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Dataset
    raw_data, preprocess_func = load_reasoning_dataset(args.task)
    if args.chat_model:
        preprocess_func = partial(preprocess_func,
                                  chat_model=True,
                                  tokenizer=tokenizer)
    if args.task == 'math':
        raw_data = raw_data.train_test_split(test_size=args.test_set_ratio,
                                             shuffle=True,
                                             seed=args.seed)
    train_set, val_set = raw_data['train'], raw_data['test']
    train_set = train_set.map(preprocess_func, num_proc=16, remove_columns=train_set.column_names)
    val_set = val_set.map(preprocess_func, num_proc=16, remove_columns=val_set.column_names)

    if args.task == 'gsm8k':
        train_set = train_set.shuffle(seed=args.seed).select(range(args.gsm8k_train_set_size))
        val_set = val_set.shuffle(seed=args.seed).select(range(args.gsm8k_test_set_size))

    # Model
    model = load_peft_llam3_v2(base_model_name=args.base_model,
                               rank=args.rank,
                               target_modules=args.target_modules,
                               lora_dropout=args.lora_dropout,
                               init_lora_weights=True,
                               mode=args.init_method)
    model.config.use_cache = False

    # Reward function
    if args.task == 'math':
        task_specific_reward = math_accuracy_reward_func
    elif args.task == 'gsm8k':
        task_specific_reward = gsm8k_accuracy_reward_func
    else:
        raise ValueError(f"Unknown task: {args.task}")

    def accuracy_reward_func(completions, solution, **kwargs):
        return task_specific_reward(
            completions=completions,
            solution=solution,
            reward_point=args.accuracy_reward_point,
            **kwargs,
        )

    def format_reward_func(completions, **kwargs):
        return general_format_reward_func(
            completions=completions,
            reward_point=args.format_reward_point,
            **kwargs,
        )

    # grpo config
    grpo_args = GRPOConfig(
        output_dir=args.save_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        optim="adamw_torch_fused",
        weight_decay=args.weight_decay,
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
        gradient_checkpointing=not args.disable_gradient_checkpointing,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=2,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        report_to=['wandb'] if args.enable_log else [],
        disable_tqdm=False,
        seed=args.seed,
        num_generations=args.num_generations,
        num_generations_eval=args.num_generations_eval,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        beta=args.kl_beta,
        scale_rewards=args.scale_rewards,
        loss_type=args.loss_type,
        mask_truncated_completions=True,
        log_completions=True,
        log_unique_prompts=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        reward_funcs=[accuracy_reward_func, format_reward_func],
        train_dataset=train_set,
        eval_dataset=val_set,
        processing_class=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic arguments
    parser.add_argument('--base-model', type=str,
                        choices=["Llama-3.1-8B", "Llama-3.1-8B-Instruct"],
                        default="Llama-3.1-8B")
    parser.add_argument('--task', type=str,
                        choices=["gsm8k", "math"],
                        default="gsm8k")

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
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=8)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--learning-rate-exponent', type=float, required=True)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--disable-gradient-checkpointing', action='store_true')
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--lr-scheduler-type', type=str, default='cosine_with_min_lr')
    parser.add_argument('--save-steps', type=int, default=5000)
    parser.add_argument('--eval-steps', type=int, default=50)
    parser.add_argument('--logging-steps', type=int, default=5)

    # GRPO-specific
    parser.add_argument('--accuracy-reward-point', type=float, default=1.0)
    parser.add_argument('--format-reward-point', type=float, default=0.10)

    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--num-generations-eval", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--kl_beta", type=float, default=0.0)
    parser.add_argument("--scale-rewards", type=str, default="group")
    parser.add_argument("--loss-type", type=str, default="dapo")

    # evaluation setting
    parser.add_argument('--test-set-ratio', type=float, default=0.08)
    parser.add_argument('--gsm8k-train-set-size', type=int, default=500)
    parser.add_argument('--gsm8k-test-set-size', type=int, default=50)

    # logging
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

    if 'Instruct' in args.base_model:
        args.chat_model = True
    else:
        args.chat_model = False

    save_dir = os.path.join('checkpoints', args.base_model, args.task, f'LoRA-{args.rank}{args.suffix}', f'LR-{naming_conversion(args.learning_rate_exponent)}')
    os.makedirs(save_dir, exist_ok=True)

    log_dir = os.path.join('logs', args.base_model, args.task, f'LoRA-{args.rank}{args.suffix}')
    setup_logging(log_dir)
    logging.info(args)

    args.log_dir = log_dir
    args.save_dir = save_dir
    set_seed(args.seed, deterministic=False)

    main(args)
