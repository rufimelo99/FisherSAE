import argparse
import json
import os
from datetime import datetime

from sae_lens import LanguageModelSAERunnerConfig, LoggingConfig, SAETrainingRunner
from sae_lens.saes.gated_sae import GatedSAEConfig
from sae_lens.saes.standard_sae import StandardSAEConfig
from sae_lens.saes.topk_sae import TopKSAEConfig

from code_sae.logger import logger as custom_logger
from code_sae.topk_cl_sae import TopKCLTrainingSAEConfig
from code_sae.utils import get_device, set_seed

SAE_CONFIG_REGISTRY = {
    "gated": GatedSAEConfig,
    "standard": StandardSAEConfig,
    "topk": TopKSAEConfig,
    "topk_cl": TopKCLTrainingSAEConfig,
}

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

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. Overrides config seed if provided.",
    )

    return parser.parse_args()


def train(config, seed: int = 42):
    # Set seed for reproducibility
    seed = set_seed(seed)
    custom_logger.info(f"Using seed: {seed}")

    sae_config = config.pop("sae", {})
    sae_class = SAE_CONFIG_REGISTRY.get(sae_config.pop("type"))
    if sae_class is None:
        raise ValueError(f"Unknown SAE type: {sae_config.get('type')}")

    wandb_project = config.pop("wandb_project", "Vulnerable-Lens-SAE")

    cfg = LanguageModelSAERunnerConfig(
        **config,
        device=DEVICE,
        seed=seed,
        # wandb_id=config.get("run_name", None),
        logger=LoggingConfig(
            log_to_wandb=True,
            wandb_project=wandb_project,
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

    # Determine seed: CLI arg > config > default (42)
    seed = args.seed if args.seed is not None else config.pop("seed", 42)
    train(config, seed=seed)
