import argparse
import json
import os
from datetime import datetime

from sae_lens import LanguageModelSAERunnerConfig, LoggingConfig, SAETrainingRunner
from sae_lens.saes.gated_sae import GatedSAEConfig
from sae_lens.saes.standard_sae import StandardSAEConfig

from code_sae.logger import logger as custom_logger
from code_sae.utils import get_device, set_seed

SAE_CONFIG_REGISTRY = {
    "gated": GatedSAEConfig,
    "standard": StandardSAEConfig,
    # "topk": TopKSAEConfig,
    # "relu": ReluSAEConfig,
}

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

    parser.add_argument(
        "--kwargs",
        type=str,
        default="{}",
        help="Additional keyword arguments to override config",
    )

    return parser.parse_args()


def train(config):
    sae_config = config.pop("sae", {})
    sae_class = SAE_CONFIG_REGISTRY.get(sae_config.pop("type"))
    if sae_class is None:
        raise ValueError(f"Unknown SAE type: {sae_config.get('type')}")

    cfg = LanguageModelSAERunnerConfig(
        **config,
        device=DEVICE,
        seed=SEED,
        # wandb_id=config.get("run_name", None),
        logger=LoggingConfig(
            log_to_wandb=True,
            wandb_project="Fisher_SAE",
            # run_name="experiment",
            wandb_log_frequency=30,
            eval_every_n_wandb_logs=20,
        ),
        sae=sae_class(
            **sae_config,
        ),
    )

    # look at the next cell to see some instruction for what to do while this is running.
    sparse_autoencoder = SAETrainingRunner(cfg).run()

    # save the model
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S_")

    sparse_autoencoder.save_model(
        os.path.join(
            f"{time_str}_{cfg.model_name}_{cfg.dataset_path}_{cfg.hook_name}",
            "sparse_autoencoder.pt",
        )
    )


if __name__ == "__main__":
    args = parse_args()
    json_path = args.config
    if not os.path.exists(json_path):
        raise ValueError(f"Config path {json_path} does not exist.")
    with open(json_path, "r") as f:
        config = json.load(f)

    kwargs = json.loads(args.kwargs)
    if kwargs:
        custom_logger.info(f"Overriding config with: {kwargs}")
        config.update(kwargs)
    train(config)
