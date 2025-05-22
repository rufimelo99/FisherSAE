import json
import random

import numpy as np
import torch
import torch.nn.functional as F

from code_sae.logger import logger


def read_jsonl_file(jsonl_path):
    with open(jsonl_path, "r") as f:
        for line in f:
            yield json.loads(line)


def set_seed(seed: int = 42):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    return seed


def get_device() -> str:
    """
    Get the device to be used for training.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info("Getting device.", device=device)
    return device


def kl_div(original_logits: torch.Tensor, new_logits: torch.Tensor):
    log_probs_new = torch.nn.functional.log_softmax(new_logits, dim=-1)
    probs_orig = torch.nn.functional.softmax(original_logits, dim=-1)
    kl = torch.nn.functional.kl_div(log_probs_new, probs_orig, reduction="none")
    return kl.sum(dim=-1)


def js_div(original_logits: torch.Tensor, new_logits: torch.Tensor):
    original_probs = torch.nn.functional.softmax(original_logits, dim=-1)
    new_probs = torch.nn.functional.softmax(new_logits, dim=-1)
    m = (original_probs + new_probs) / 2
    kl_om = kl_div(original_logits, m)
    kl_nm = kl_div(new_logits, m)
    return (kl_om + kl_nm) / 2
