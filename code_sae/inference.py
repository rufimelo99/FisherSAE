import argparse
import json
import os
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import torch
from sae_lens import (
    SAE,
    HookedSAETransformer,
    LanguageModelSAERunnerConfig,
    SAETrainingRunner,
)

from code_sae.logger import logger
from code_sae.utils import get_device, set_seed

SEED = set_seed()
DEVICE = get_device()


def parse_args():
    parser = argparse.ArgumentParser(description="Inference on a model")
    parser.add_argument(
        "--path",
        type=str,
        default="checkpoints/some_folder/",
        help="Path to the checkpoint folder",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = args.path
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path {checkpoint_path} does not exist.")

    cfg = json.load(open(os.path.join(checkpoint_path, "cfg.json"), "r"))

    model = HookedSAETransformer.from_pretrained(
        cfg["model_name"],
        device=DEVICE,
    )

    sae = SAE.load_from_disk(path=checkpoint_path, device=DEVICE)
    _, cache = model.run_with_cache_with_saes("olá joca", saes=[sae])


if __name__ == "__main__":
    main()
