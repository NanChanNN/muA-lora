import logging
import os
from datetime import datetime
import numpy as np


def naming_conversion(lr_exponent: float):
    if lr_exponent == int(lr_exponent):
        return str(int(lr_exponent))
    else:
        return f"{lr_exponent:.1f}".replace('.', 'p')


def setup_logging(log_dir=None, log_level=logging.INFO):
    """
    Set up logging to console and optional file.
    """
    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=log_level, format=log_format, datefmt=date_format)

    log_file = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logging.getLogger().addHandler(file_handler)
        logging.info(f"Logging to file: {log_file}")
    else:
        logging.info("Logging to console only.")

    return log_file


def random_split(num_samples, seed: int=42):
    rng = np.random.default_rng(seed=seed)
    split_point = int(num_samples * 0.8)

    indices = np.arange(num_samples)
    shuffled_indices = rng.permutation(indices)

    return shuffled_indices[:split_point], shuffled_indices[split_point:]


def print_trainable_parameters(model, log: bool = True):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    if log:
        logging.info(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )
    else:
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    return trainable_params, all_param
