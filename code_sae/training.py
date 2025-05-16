import argparse
import json
import os

from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner

from code_sae.logger import logger
from code_sae.utils import get_device, set_seed

SEED = set_seed()
DEVICE = get_device()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--config",
        type=str,
        default="training_config.json",
        help="Path to the config file",
    )

    return parser.parse_args()


def train(config):
    train_batch_size_tokens = config["train_batch_size_tokens"]

    total_training_steps = 10

    total_training_tokens = total_training_steps * train_batch_size_tokens

    lr_decay_steps = total_training_steps // 5  # 20% of training
    l1_warm_up_steps = total_training_steps // 20  # 5% of training

    cfg = LanguageModelSAERunnerConfig(
        **config,
        lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
        l1_warm_up_steps=l1_warm_up_steps,  # this can help avoid too many dead features initially.
        total_training_tokens=total_training_tokens,
        device=DEVICE,
        seed=SEED,
    )

    # look at the next cell to see some instruction for what to do while this is running.
    sparse_autoencoder = SAETrainingRunner(cfg).run()

    # save the model
    sparse_autoencoder.save_model(
        os.path.join(cfg.checkpoint_path, "sparse_autoencoder.pt")
    )


if __name__ == "__main__":
    args = parse_args()
    json_path = args.config
    if not os.path.exists(json_path):
        raise ValueError(f"Config path {json_path} does not exist.")
    with open(json_path, "r") as f:
        config = json.load(f)
    train(config)
