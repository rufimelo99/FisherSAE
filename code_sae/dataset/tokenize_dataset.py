import argparse
import json
import os

from sae_lens import PretokenizeRunner, PretokenizeRunnerConfig

from code_sae.logger import logger

TRAIN_SPLIT_RATIO = 0.8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize dataset with train/test splits"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="tokenization_config.json",
        help="Path to the config file",
    )
    parser.add_argument(
        "--training_ratio",
        type=float,
        default=TRAIN_SPLIT_RATIO,
        help="Ratio of the dataset to use for training (default: 0.8)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config
    if not os.path.exists(config_path):
        raise ValueError(f"Config path {config_path} does not exist.")

    with open(config_path, "r") as f:
        config = json.load(f)
    logger.info("Loading config", path=config_path)

    base_save_path = config.get("save_path", "pretokenized")
    train_split_ratio = args.training_ratio
    test_split_ratio = 1.0 - train_split_ratio

    # Create training split (80%)
    train_config = config.copy()
    train_config["save_path"] = f"{base_save_path}_train"
    train_config["split"] = f"train[:{int(train_split_ratio * 100)}%]"

    logger.info("Creating training split", split_ratio=train_split_ratio)
    train_cfg = PretokenizeRunnerConfig(**train_config)
    logger.info("Training config loaded", config=train_cfg)
    PretokenizeRunner(train_cfg).run()

    # Create testing split (20%)
    test_config = config.copy()
    test_config["save_path"] = f"{base_save_path}_test"
    test_config["split"] = f"train[{int(train_split_ratio * 100)}%:]"

    logger.info("Creating testing split", split_ratio=test_split_ratio)
    test_cfg = PretokenizeRunnerConfig(**test_config)
    logger.info("Testing config loaded", config=test_cfg)
    PretokenizeRunner(test_cfg).run()

    logger.info(
        "Tokenization complete",
        train_path=train_config["save_path"],
        test_path=test_config["save_path"],
    )


if __name__ == "__main__":
    main()
