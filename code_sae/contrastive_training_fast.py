"""
Fast Contrastive SAE Training with Pre-computed Activations

This script trains a contrastive SAE using pre-computed activations,
eliminating the model forward pass bottleneck during training.

Workflow:
    1. First run: python code_sae/precompute_activations.py --config code_sae/precompute_config.json
    2. Then run: python code_sae/contrastive_training_fast.py --config code_sae/contrastive_training_fast_config.json

Expected speedup: 50-100x faster than on-the-fly activation extraction.
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from code_sae.logger import logger
from code_sae.topk_cl_sae import TopKCLTrainingSAE, TopKCLTrainingSAEConfig
from code_sae.utils import get_device, set_seed

SEED = set_seed()
DEVICE = get_device()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FastContrastiveTrainingConfig:
    """Configuration for fast contrastive SAE training with pre-computed activations."""

    # Pre-computed activations paths
    positive_activations_path: str = "data/activations/positive_acts.pt"
    negative_activations_path: str = "data/activations/negative_acts.pt"

    # Training
    batch_size: int = 256  # Can be much larger now!
    lr: float = 3e-4
    training_steps: int = 10000
    warmup_steps: int = 500
    log_every: int = 50
    save_every: int = 1000

    # DataLoader optimization
    num_workers: int = 4
    prefetch_factor: int = 4
    pin_memory: bool = True

    # SAE config
    d_in: int = 3584
    d_sae: int = 16384
    k: int = 64
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.07
    contrastive_mode: Literal["infonce", "triplet", "cosine"] = "infonce"
    use_feature_contrastive: bool = True
    dtype: str = "float32"

    # Logging
    wandb_project: str = "Contrastive-SAE"
    wandb_run_name: str | None = None
    checkpoint_dir: str = "checkpoints/contrastive_sae"


def load_config(config_path: str) -> FastContrastiveTrainingConfig:
    """Load config from JSON file."""
    with open(config_path) as f:
        config_dict = json.load(f)

    sae_config = config_dict.pop("sae", {})

    return FastContrastiveTrainingConfig(
        positive_activations_path=config_dict.get("positive_activations_path", ""),
        negative_activations_path=config_dict.get("negative_activations_path", ""),
        batch_size=config_dict.get("batch_size", 256),
        lr=config_dict.get("lr", 3e-4),
        training_steps=config_dict.get("training_steps", 10000),
        warmup_steps=config_dict.get("warmup_steps", 500),
        log_every=config_dict.get("log_every", 50),
        save_every=config_dict.get("save_every", 1000),
        num_workers=config_dict.get("num_workers", 4),
        prefetch_factor=config_dict.get("prefetch_factor", 4),
        pin_memory=config_dict.get("pin_memory", True),
        d_in=sae_config.get("d_in", 3584),
        d_sae=sae_config.get("d_sae", 16384),
        k=sae_config.get("k", 64),
        contrastive_weight=sae_config.get("contrastive_weight", 0.1),
        contrastive_temperature=sae_config.get("contrastive_temperature", 0.07),
        contrastive_mode=sae_config.get("contrastive_mode", "infonce"),
        use_feature_contrastive=sae_config.get("use_feature_contrastive", True),
        dtype=sae_config.get("dtype", "float32"),
        wandb_project=config_dict.get("wandb_project", "Contrastive-SAE"),
        wandb_run_name=config_dict.get("wandb_run_name"),
        checkpoint_dir=config_dict.get("checkpoint_dir", "checkpoints/contrastive_sae"),
    )


class PrecomputedActivationsDataset(Dataset):
    """
    Dataset for pre-computed paired activations.

    Efficiently loads activations from disk with minimal overhead.
    """

    def __init__(
        self,
        positive_path: str,
        negative_path: str,
        device: str = "cpu",  # Keep on CPU for DataLoader, move to GPU in training
    ):
        logger.info(f"Loading positive activations from: {positive_path}")
        self.positive_acts = torch.load(
            positive_path, map_location="cpu", weights_only=True
        )
        logger.info(f"Loading negative activations from: {negative_path}")
        self.negative_acts = torch.load(
            negative_path, map_location="cpu", weights_only=True
        )

        # Ensure same number of samples
        min_len = min(len(self.positive_acts), len(self.negative_acts))
        self.positive_acts = self.positive_acts[:min_len]
        self.negative_acts = self.negative_acts[:min_len]

        logger.info(f"Loaded {min_len} paired activation samples")
        logger.info(f"Activation shape: {self.positive_acts.shape}")

    def __len__(self) -> int:
        return len(self.positive_acts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "activations_a": self.positive_acts[idx],
            "activations_b": self.negative_acts[idx],
        }


class FastContrastiveSAETrainer:
    """Fast trainer for contrastive SAE with pre-computed activations."""

    def __init__(
        self,
        config: FastContrastiveTrainingConfig,
        sae: TopKCLTrainingSAE,
        dataloader: DataLoader,
    ):
        self.config = config
        self.sae = sae.to(DEVICE)
        self.dataloader = dataloader

        self.optimizer = torch.optim.AdamW(
            sae.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
        )

        def lr_lambda(step: int) -> float:
            if step < config.warmup_steps:
                return step / config.warmup_steps
            return 1.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.step = 0
        self.best_loss = float("inf")

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single training step."""
        self.sae.train()
        self.optimizer.zero_grad()

        act_a = batch["activations_a"].to(DEVICE, non_blocking=True)
        act_b = batch["activations_b"].to(DEVICE, non_blocking=True)

        from sae_lens.saes.sae import TrainStepInput

        step_input = TrainStepInput(
            sae_in={"activations_a": act_a, "activations_b": act_b},
            coefficients=self.sae.get_coefficients(),
            dead_neuron_mask=None,
            n_training_steps=self.step,
        )

        output = self.sae.training_forward_pass(step_input)

        output.loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.step += 1

        metrics = {
            "total_loss": output.loss.item(),
            "mse_loss": output.losses.get("mse_loss", torch.tensor(0.0)).item(),
            "contrastive_loss": output.losses.get(
                "contrastive_loss", torch.tensor(0.0)
            ).item(),
            "lr": self.scheduler.get_last_lr()[0],
            "step": self.step,
        }

        return metrics

    def train(self) -> None:
        """Full training loop with efficient data loading."""
        logger.info(f"Starting FAST contrastive SAE training on {DEVICE}")
        logger.info(
            f"SAE: d_in={self.config.d_in}, d_sae={self.config.d_sae}, k={self.config.k}"
        )
        logger.info(
            f"Contrastive: weight={self.config.contrastive_weight}, mode={self.config.contrastive_mode}"
        )
        logger.info(
            f"Batch size: {self.config.batch_size}, Workers: {self.config.num_workers}"
        )
        logger.info("-" * 50)

        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config=self.config.__dict__,
        )

        pbar = tqdm(total=self.config.training_steps, desc="Training")

        epoch = 0
        while self.step < self.config.training_steps:
            epoch += 1
            for batch in self.dataloader:
                if self.step >= self.config.training_steps:
                    break

                metrics = self.train_step(batch)

                if self.step % self.config.log_every == 0:
                    pbar.set_postfix(
                        loss=f"{metrics['total_loss']:.4f}",
                        mse=f"{metrics['mse_loss']:.4f}",
                        cl=f"{metrics['contrastive_loss']:.4f}",
                        epoch=epoch,
                    )
                    wandb.log(metrics, step=self.step)

                if self.step % self.config.save_every == 0:
                    self.save_checkpoint()

                pbar.update(1)

        pbar.close()

        self.save_checkpoint(final=True)
        wandb.finish()

        logger.info("Training complete!")

    def save_checkpoint(self, final: bool = False) -> None:
        """Save model checkpoint in sae_lens format (cfg.json + sae_weights.safetensors)."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if final:
            sae_dir = checkpoint_dir / "final"
        else:
            sae_dir = checkpoint_dir / f"step_{self.step}"

        # Save in sae_lens format using built-in save_model method
        self.sae.save_model(sae_dir)

        # Also save training state for resuming
        training_state_path = sae_dir / "training_state.pt"
        torch.save(
            {
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "step": self.step,
            },
            training_state_path,
        )
        logger.info(f"Saved checkpoint to {sae_dir}")


def create_sae(config: FastContrastiveTrainingConfig) -> TopKCLTrainingSAE:
    """Create a TopKCLTrainingSAE from config."""
    sae_config = TopKCLTrainingSAEConfig(
        d_in=config.d_in,
        d_sae=config.d_sae,
        k=config.k,
        dtype=config.dtype,
        device=DEVICE,
        contrastive_weight=config.contrastive_weight,
        contrastive_temperature=config.contrastive_temperature,
        contrastive_mode=config.contrastive_mode,
        use_feature_contrastive=config.use_feature_contrastive,
    )
    return TopKCLTrainingSAE(sae_config)


def main():
    parser = argparse.ArgumentParser(
        description="Fast contrastive SAE training with pre-computed activations"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Create dataset
    logger.info("Loading pre-computed activations...")
    dataset = PrecomputedActivationsDataset(
        positive_path=config.positive_activations_path,
        negative_path=config.negative_activations_path,
    )

    # Create efficient DataLoader with prefetching
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )

    logger.info(f"DataLoader created: {len(dataloader)} batches per epoch")

    # Create SAE
    logger.info("Creating contrastive SAE...")
    sae = create_sae(config)

    # Train
    trainer = FastContrastiveSAETrainer(
        config=config,
        sae=sae,
        dataloader=dataloader,
    )

    trainer.train()


if __name__ == "__main__":
    main()
