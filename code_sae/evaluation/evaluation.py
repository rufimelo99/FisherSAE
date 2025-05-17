import argparse
import json
import os

from sae_lens import PretokenizeRunner, PretokenizeRunnerConfig

from code_sae.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--config",
        type=str,
        default="tokenization_config.json",
        help="Path to the config file",
    )

    return parser.parse_args()
