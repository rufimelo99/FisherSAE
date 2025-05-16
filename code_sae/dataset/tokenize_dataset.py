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


def main():
    args = parse_args()
    config_path = args.config
    if not os.path.exists(config_path):
        raise ValueError(f"Config path {config_path} does not exist.")

    with open(config_path, "r") as f:
        config = json.load(f)
    logger.info("Loading config", path=config_path)
    cfg = PretokenizeRunnerConfig(
        **config,
    )
    logger.info("Config loaded", config=cfg)
    PretokenizeRunner(cfg).run()

if __name__ == "__main__":
    main()