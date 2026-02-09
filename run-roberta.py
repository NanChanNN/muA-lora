import argparse
import logging
import os
from typing import Union, List

import wandb
from functools import partial

from peft import LoraConfig, TaskType, get_peft_model
from transformers import DataCollatorWithPadding, AutoConfig, TrainingArguments, Trainer, \
    AutoTokenizer, AutoModelForSequenceClassification, set_seed

from utils.custom_trainer import FixedHeadLRTrainer
from utils.data_utils import glue_task_to_keys, pre_process_GLUE, fetch_task_and_metric, glue_task_to_splits
from utils.helper import setup_logging, naming_conversion
from utils.lora_utils import custom_init
from utils.eval_utils import compute_glue_metrics


def load_peft_roberta(base_model_name: str,
                      config,
                      rank: int,
                      target_modules: Union[List[str], str] = 'all-linear',
                      exclude_modules: List[str] = None,
                      lora_dropout: float = 0.0,
                      init_lora_weights: bool = True,
                      mode: str = None):
    base_model = AutoModelForSequenceClassification.from_pretrained(f"FacebookAI/{base_model_name}",
                                                                    config=config,)

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
        exclude_modules=exclude_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        init_lora_weights=init_lora_weights,
        modules_to_save=["classifier"],)

    model = get_peft_model(base_model, lora_config)

    if mode is not None:
        if mode.startswith('initA'):
            custom_init(model, 'initA')
        elif mode.startswith('initB'):
            custom_init(model, 'initB')

    return model


def main(args):
    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"FacebookAI/{args.base_model}")

    # Load dataset
    raw_data, metric = fetch_task_and_metric(args.task)
    label_names = raw_data[args.split_keys[0]].features["label"].names
    num_labels = len(label_names)

    data_preprocess = partial(pre_process_GLUE,
                              tokenizer=tokenizer,
                              sentence1_key=args.key_pairs[0],
                              sentence2_key=args.key_pairs[1],
                              max_seq_length=args.max_seq_length)
    tokenized_dataset = raw_data.map(data_preprocess,
                                     batched=True,)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorWithPadding(tokenizer,
                                            pad_to_multiple_of=8)

    # Construct model
    config = AutoConfig.from_pretrained(f'FacebookAI/{args.base_model}',
                                        num_labels=num_labels,)
    config.label2id = {name: i for i, name in enumerate(label_names)}
    config.id2label = {i: name for i, name in enumerate(label_names)}

    model = load_peft_roberta(base_model_name=args.base_model,
                              config=config,
                              rank=args.rank,
                              target_modules=args.target_modules,
                              exclude_modules=args.exclude_modules,
                              lora_dropout=args.lora_dropout,
                              init_lora_weights=True,
                              mode=args.init_method,)

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
                                      gradient_checkpointing=False,
                                      dataloader_pin_memory=True,
                                      dataloader_persistent_workers=True,
                                      dataloader_num_workers=8,
                                      dataloader_prefetch_factor=4,
                                      label_names=['labels'],
                                      logging_strategy='steps',
                                      logging_steps=args.logging_steps,
                                      eval_strategy="steps",
                                      eval_steps=args.eval_steps,
                                      report_to=['wandb'] if args.enable_log else [],
                                      disable_tqdm=False,
                                      seed=args.seed,)

    if args.head_learning_rate is not None:
        training_args.head_learning_rate = args.head_learning_rate
        selected_trainer = FixedHeadLRTrainer
    else:
        selected_trainer = Trainer

    trainer = selected_trainer(model=model,
                               args=training_args,
                               data_collator=data_collator,
                               train_dataset=tokenized_dataset[args.split_keys[0]],
                               eval_dataset=tokenized_dataset[args.split_keys[1]],
                               processing_class=tokenizer,
                               compute_metrics=partial(compute_glue_metrics, metric=metric),)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic arguments
    parser.add_argument('--base-model', type=str,
                        choices=['roberta-base', 'roberta-large'],
                        default='roberta-large')
    parser.add_argument('--task', type=str,
                        choices=['mnli', 'anli', 'anli_r1', 'anli_r2', 'anli_r3'],
                        default='anli')

    # lora relevant
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--lora-dropout', type=float, default=0.0)
    parser.add_argument("--target-modules",
                        nargs="+",
                        metavar="MODULE",
                        default=["output.dense", "attention.output.dense", "intermediate.dense", "query", "key", "value"],)
    parser.add_argument("--exclude-modules",
                        nargs="+",
                        metavar="MODULE",
                        default=None,)
    parser.add_argument('--init-method', type=str, default=None,
                        choices=[None, 'initA', 'initB', 'initA_alpha1', 'initB_alpha1', 'alpha1',
                                 'initA_constant1', 'initB_constant1', 'constant1'])

    # training relevant
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-batch-size', type=int, default=384)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--learning-rate-exponent', type=float, required=True)
    parser.add_argument('--head-learning-rate', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--lr-scheduler-type', type=str, default='cosine_with_min_lr')
    parser.add_argument('--max-seq-length', type=int, default=128)
    parser.add_argument('--save-steps', type=int, default=5000)
    parser.add_argument('--eval-steps', type=int, default=250)
    parser.add_argument('--logging-steps', type=int, default=25)

    parser.add_argument('--enable-log', action='store_true')
    parser.add_argument('--wandb-suffix', type=str, default='')
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

    args.key_pairs = glue_task_to_keys[args.task]
    args.split_keys = glue_task_to_splits[args.task]
    
    save_dir = os.path.join('checkpoints', args.base_model, args.task, f'LoRA-{args.rank}{args.suffix}', f'LR-{naming_conversion(args.learning_rate_exponent)}')
    os.makedirs(save_dir, exist_ok=True)

    log_dir = os.path.join('logs', args.base_model, args.task, f'LoRA-{args.rank}{args.suffix}')
    setup_logging(log_dir)
    logging.info(args)

    args.log_dir = log_dir
    args.save_dir = save_dir
    set_seed(args.seed, deterministic=False)

    main(args)
