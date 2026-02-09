import re
import torch
from datasets import load_dataset, DatasetDict
import evaluate


glue_task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    'anli_r1': ('premise', 'hypothesis'),
    'anli_r2': ('premise', 'hypothesis'),
    'anli_r3': ('premise', 'hypothesis'),
    'anli': ('premise', 'hypothesis'),
}


glue_task_to_splits = {
    'mnli': ('train', 'validation_matched'),
    'anli_r1': ('train_r1', 'dev_r1', 'test_r1'),
    'anli_r2': ('train_r2', 'dev_r2', 'test_r2'),
    'anli_r3': ('train_r3', 'dev_r3', 'test_r3'),
    'anli': ('train', 'dev',)
}


def fetch_task_and_metric(task):
    if task in ['mnli']:
        ds = load_dataset("nyu-mll/glue", task)
        metric = evaluate.load("glue", task)
    elif task in ['anli_r1', 'anli_r2', 'anli_r3']:
        ds = load_dataset("facebook/anli")
        metric = evaluate.load("accuracy")
    elif task == 'anli':
        train_ds = load_dataset("facebook/anli", split='train_r1+train_r2+train_r3')
        dev_ds = load_dataset("facebook/anli", split='dev_r1+dev_r2+dev_r3')
        ds = DatasetDict({'train': train_ds, 'dev': dev_ds})
        metric = evaluate.load("accuracy")
    else:
        raise ValueError('Task {} not supported'.format(task))
    return ds, metric


def fetch_image_dataset(task):
    if task == 'food101':
        ds = load_dataset("ethz/food101")
    elif task == 'imagenet-1k':
        ds = load_dataset("ILSVRC/imagenet-1k", num_proc=12)
    else:
        raise ValueError('Task {} not supported'.format(task))
    return ds


def pre_process_GLUE(example,
                     tokenizer,
                     sentence1_key: str = 'premise',
                     sentence2_key: str = 'hypothesis',
                     max_seq_length: int = 128):
    args = (
        (example[sentence1_key],) if sentence2_key is None else (example[sentence1_key], example[sentence2_key])
    )
    result = tokenizer(*args,
                       truncation=True,
                       padding=False,
                       max_length=max_seq_length)
    result['labels'] = example['label']
    return result


def filter_by_length(dataset, tokenizer, min_len=256, max_len=1024, num_proc=32):

    def get_length(example):
        tok = tokenizer.apply_chat_template(example['messages'])
        return {'length': len(tok)}

    dataset = dataset.map(get_length, num_proc=num_proc)
    filtered = dataset.filter(lambda x: min_len <= x['length'] <= max_len, num_proc=num_proc)
    filtered = filtered.remove_columns('length')
    return filtered


def vit_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def pre_process_XSum(examples,
                     tokenizer,
                     input_key: str = 'document',
                     target_key: str = 'summary',
                     max_seq_length: int = 1024):
    model_inputs = tokenizer(examples[input_key],
                             max_length=max_seq_length,
                             padding=False,
                             truncation=True)
    labels = tokenizer(text_target=examples[target_key],
                       max_length=64,
                       padding=False,
                       truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def replace_thinking_token(content: str) -> str:
    content = content.replace('<|begin_of_thought|>', '<think>').replace('<|end_of_thought|>', '</think>')
    return content


def preprocess_openthoughts(example):
    messages = [
        {'role': 'system', 'content': replace_thinking_token(example['system'])},
        {'role': example['conversations'][0]['from'], 'content': replace_thinking_token(example['conversations'][0]['value'])},
        {'role': example['conversations'][1]['from'], 'content': replace_thinking_token(example['conversations'][1]['value'])},
    ]
    return {'messages': messages}


def _extract_gsm8k_ground_truth(answer_text: str) -> str:
    """Extract GSM8K ground-truth final answer.

    GSM8K stores the final answer as a line like: "#### 42".
    """
    m = re.search(r"####\s*([-+]?\d[\d,]*)", str(answer_text))
    return m.group(1).replace(",", "").strip()


def _build_gsm8k_prompt(question: str, ) -> str:
    question = str(question).strip()
    return (
        "Solve the following grade-school math problem. "
        "Show reasoning in <think>...</think>, "
        "then give the final numerical answer in <answer>...</answer>.\n\n"
        f"Question: {question}\n"
        "Solution: "
    )


def _build_gsm8k_prompt_chat(question: str, tokenizer) -> str:
    sys = (
        "You are a helpful assistant that solves grade-school math problems. "
        "You will be given a math problem. "
        "Provide the final numerical answer in <answer>...</answer>.\n\n"
        "Put your reasoning inside <think>...</think> before the final answer."
    )
    user = f"Question: {str(question).strip()}"

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]

    return tokenizer.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)


def preprocess_gsm8k(example,
                     chat_model:bool=False,
                     tokenizer=None):
    if chat_model:
        prompt = _build_gsm8k_prompt_chat(example["question"], tokenizer)
    else:
        prompt = _build_gsm8k_prompt(example["question"], )
    sol = _extract_gsm8k_ground_truth(example["answer"])

    return {
        "prompt": prompt,
        "solution": sol,
    }


def _build_math500_prompt(question: str, ) -> str:
    question = str(question).strip()
    return (
        "Solve the following competition math problem. "
        "Write your reasoning inside <think>...</think>. "
        "Then write the final simplified answer inside <answer>...</answer>, "
        "formatted in LaTeX (e.g., \\frac{1}{2}, \\sqrt{3}, x^2).\n\n"
        f"Problem: {question}\n"
        "Solution: "
    )


def _build_math500_prompt_chat(problem: str, tokenizer) -> str:
    sys = (
        "You are a helpful assistant that solves competition math problems. "
        "Write your reasoning inside <think>...</think>. "
        "Then write the final simplified answer inside <answer>...</answer>, "
        "formatted in LaTeX (e.g., \\frac{1}{2}, \\sqrt{3}, x^2)."
    )

    user = f"Problem: {str(problem).strip()}"
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)


def preprocess_math(example,
                    chat_model:bool=False,
                    tokenizer=None):
    if chat_model:
        prompt = _build_math500_prompt_chat(example["problem"], tokenizer)
    else:
        prompt = _build_math500_prompt(example["problem"],)
    sol = str(example["answer"]).strip()
    return {
        "prompt": prompt,
        "solution": sol,
    }


def load_reasoning_dataset(task: str):
    if task == "gsm8k":
        raw_data = load_dataset(f"openai/gsm8k", "main")
        data_preprocess_func = preprocess_gsm8k
    elif task == "math":
        raw_data = load_dataset("HuggingFaceH4/MATH-500", split='test')
        data_preprocess_func = preprocess_math
    else:
        raise ValueError(f"Unknown task: {task}")
    return raw_data, data_preprocess_func
