"""
Contrastive SAE Training with Paired HuggingFace Datasets

This script trains a contrastive SAE (TopKCLTrainingSAE) using a HuggingFace dataset
that has two columns: one for positive samples and one for negative samples.

Usage:
    python code_sae/contrastive_training.py --config code_sae/contrastive_training_config.json
"""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal

import torch
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer

from code_sae.logger import logger
from code_sae.topk_cl_sae import TopKCLTrainingSAE, TopKCLTrainingSAEConfig
from code_sae.utils import get_device, set_seed

SEED = set_seed()
DEVICE = get_device()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ContrastiveTrainingConfig:
    """Configuration for contrastive SAE training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    hook_name: str = "blocks.14.hook_resid_post"
    dtype: str = "float32"

    # Dataset - two separate HF repos for positive/negative samples
    positive_dataset_path: str = ""  # HF repo with positive (e.g., secure) samples
    negative_dataset_path: str = ""  # HF repo with negative (e.g., vulnerable) samples
    input_ids_column: str = "input_ids"  # Column name for tokenized input
    streaming: bool = True
    max_seq_len: int = 512

    # Training
    batch_size: int = 32
    lr: float = 3e-4
    training_steps: int = 10000
    warmup_steps: int = 500
    log_every: int = 50
    save_every: int = 1000
    eval_every: int = 500

    # SAE config
    d_in: int = 3584
    d_sae: int = 16384
    k: int = 64  # TopK sparsity
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.07
    contrastive_mode: Literal["infonce", "triplet", "cosine"] = "infonce"
    use_feature_contrastive: bool = True

    # Logging
    wandb_project: str = "Contrastive-SAE"
    wandb_run_name: str | None = None
    checkpoint_dir: str = "checkpoints/contrastive_sae"


def load_config(config_path: str) -> ContrastiveTrainingConfig:
    """Load config from JSON file."""
    with open(config_path) as f:
        config_dict = json.load(f)

    # Extract SAE-specific config
    sae_config = config_dict.pop("sae", {})

    # Map config fields
    return ContrastiveTrainingConfig(
        model_name=config_dict.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
        hook_name=config_dict.get("hook_name", "blocks.14.hook_resid_post"),
        dtype=config_dict.get("dtype", "float32"),
        positive_dataset_path=config_dict.get("positive_dataset_path", ""),
        negative_dataset_path=config_dict.get("negative_dataset_path", ""),
        input_ids_column=config_dict.get("input_ids_column", "input_ids"),
        streaming=config_dict.get("streaming", True),
        max_seq_len=config_dict.get("max_seq_len", 512),
        batch_size=config_dict.get("batch_size", 32),
        lr=config_dict.get("lr", 3e-4),
        training_steps=config_dict.get("training_steps", 10000),
        warmup_steps=config_dict.get("warmup_steps", 500),
        log_every=config_dict.get("log_every", 50),
        save_every=config_dict.get("save_every", 1000),
        eval_every=config_dict.get("eval_every", 500),
        d_in=sae_config.get("d_in", 3584),
        d_sae=sae_config.get("d_sae", 16384),
        k=sae_config.get("k", 64),
        contrastive_weight=sae_config.get("contrastive_weight", 0.1),
        contrastive_temperature=sae_config.get("contrastive_temperature", 0.07),
        contrastive_mode=sae_config.get("contrastive_mode", "infonce"),
        use_feature_contrastive=sae_config.get("use_feature_contrastive", True),
        wandb_project=config_dict.get("wandb_project", "Contrastive-SAE"),
        wandb_run_name=config_dict.get("wandb_run_name"),
        checkpoint_dir=config_dict.get("checkpoint_dir", "checkpoints/contrastive_sae"),
    )


class PairedActivationExtractor:
    """
    Extracts activations from paired samples in a HuggingFace dataset.

    Expects a dataset with two columns containing tokenized input_ids.
    """

    def __init__(
        self,
        model: HookedTransformer,
        positive_dataset_path: str,
        negative_dataset_path: str,
        input_ids_column: str,
        hook_name: str,
        batch_size: int = 32,
        max_seq_len: int = 512,
        streaming: bool = True,
        device: str = "cuda",
    ):
        self.model = model
        self.hook_name = hook_name
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.input_ids_column = input_ids_column
        self.positive_dataset_path = positive_dataset_path
        self.negative_dataset_path = negative_dataset_path

        # Load datasets
        logger.info(f"Loading positive dataset: {positive_dataset_path}")
        self.positive_dataset = load_dataset(positive_dataset_path, streaming=streaming, split="train")
        logger.info(f"Loading negative dataset: {negative_dataset_path}")
        self.negative_dataset = load_dataset(negative_dataset_path, streaming=streaming, split="train")

        # Validate column exists in both datasets
        pos_sample = next(iter(self.positive_dataset))
        neg_sample = next(iter(self.negative_dataset))
        if input_ids_column not in pos_sample:
            raise ValueError(
                f"Column '{input_ids_column}' not found in positive dataset. "
                f"Available columns: {list(pos_sample.keys())}"
            )
        if input_ids_column not in neg_sample:
            raise ValueError(
                f"Column '{input_ids_column}' not found in negative dataset. "
                f"Available columns: {list(neg_sample.keys())}"
            )

        logger.info(f"Datasets loaded. Using column: {input_ids_column}")

    def _pad_sequences(self, sequences: list[list[int]]) -> torch.Tensor:
        """Pad sequences to max_seq_len."""
        padded = []
        for seq in sequences:
            if len(seq) > self.max_seq_len:
                seq = seq[: self.max_seq_len]
            else:
                seq = seq + [0] * (self.max_seq_len - len(seq))
            padded.append(seq)
        return torch.tensor(padded, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def _get_activations(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract activations from the model at the specified hook."""
        _, cache = self.model.run_with_cache(
            input_ids,
            names_filter=[self.hook_name],
            stop_at_layer=int(self.hook_name.split(".")[1]) + 1,
        )
        # Shape: [batch, seq_len, d_model] -> [batch * seq_len, d_model]
        acts = cache[self.hook_name]
        return acts.reshape(-1, acts.shape[-1])

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield batches of paired activations."""
        batch_pos = []
        batch_neg = []

        for pos_sample, neg_sample in zip(self.positive_dataset, self.negative_dataset):
            batch_pos.append(pos_sample[self.input_ids_column])
            batch_neg.append(neg_sample[self.input_ids_column])

            if len(batch_pos) >= self.batch_size:
                # Pad and get activations
                pos_ids = self._pad_sequences(batch_pos)
                neg_ids = self._pad_sequences(batch_neg)

                pos_acts = self._get_activations(pos_ids)
                neg_acts = self._get_activations(neg_ids)

                yield {
                    "activations_a": pos_acts,
                    "activations_b": neg_acts,
                }

                batch_pos = []
                batch_neg = []


class ContrastiveSAETrainer:
    """Trainer for contrastive SAE with paired activations."""

    def __init__(
        self,
        config: ContrastiveTrainingConfig,
        model: HookedTransformer,
        sae: TopKCLTrainingSAE,
        activation_extractor: PairedActivationExtractor,
    ):
        self.config = config
        self.model = model
        self.sae = sae.to(DEVICE)
        self.activation_extractor = activation_extractor

        self.optimizer = torch.optim.AdamW(
            sae.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler with warmup
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

        # Get paired activations
        act_a = batch["activations_a"].to(DEVICE)
        act_b = batch["activations_b"].to(DEVICE)

        # Create step input for contrastive training
        from sae_lens.saes.sae import TrainStepInput

        step_input = TrainStepInput(
            sae_in={"activations_a": act_a, "activations_b": act_b},
            coefficients=self.sae.get_coefficients(),
            dead_neuron_mask=None,
            n_training_steps=self.step,
        )

        # Forward pass
        output = self.sae.training_forward_pass(step_input)

        # Backward pass
        output.loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.step += 1

        # Extract metrics
        metrics = {
            "total_loss": output.loss.item(),
            "mse_loss": output.losses.get("mse_loss", torch.tensor(0.0)).item(),
            "contrastive_loss": output.losses.get("contrastive_loss", torch.tensor(0.0)).item(),
            "lr": self.scheduler.get_last_lr()[0],
            "step": self.step,
        }

        return metrics

    def train(self) -> None:
        """Full training loop."""
        logger.info(f"Starting contrastive SAE training on {DEVICE}")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Hook: {self.config.hook_name}")
        logger.info(f"SAE: d_in={self.config.d_in}, d_sae={self.config.d_sae}, k={self.config.k}")
        logger.info(f"Contrastive: weight={self.config.contrastive_weight}, mode={self.config.contrastive_mode}")
        logger.info("-" * 50)

        # Initialize wandb
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config=self.config.__dict__,
        )

        pbar = tqdm(total=self.config.training_steps, desc="Training")

        for batch in self.activation_extractor:
            if self.step >= self.config.training_steps:
                break

            metrics = self.train_step(batch)

            # Logging
            if self.step % self.config.log_every == 0:
                pbar.set_postfix(
                    loss=f"{metrics['total_loss']:.4f}",
                    mse=f"{metrics['mse_loss']:.4f}",
                    cl=f"{metrics['contrastive_loss']:.4f}",
                )
                wandb.log(metrics, step=self.step)

            # Save checkpoint
            if self.step % self.config.save_every == 0:
                self.save_checkpoint()

            pbar.update(1)

        pbar.close()

        # Save final model
        self.save_checkpoint(final=True)
        wandb.finish()

        logger.info("Training complete!")

    def save_checkpoint(self, final: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if final:
            path = checkpoint_dir / "contrastive_sae_final.pt"
        else:
            path = checkpoint_dir / f"checkpoint_step_{self.step}.pt"

        torch.save(
            {
                "sae_state_dict": self.sae.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "step": self.step,
                "config": self.config.__dict__,
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path}")


def create_sae(config: ContrastiveTrainingConfig) -> TopKCLTrainingSAE:
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
    parser = argparse.ArgumentParser(description="Train a contrastive SAE")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Load model
    logger.info(f"Loading model: {config.model_name}")
    model = HookedTransformer.from_pretrained(
        config.model_name,
        device=DEVICE,
        dtype=getattr(torch, config.dtype),
    )
    model.eval()

    # Create SAE
    logger.info("Creating contrastive SAE...")
    sae = create_sae(config)

    # Create activation extractor
    activation_extractor = PairedActivationExtractor(
        model=model,
        positive_dataset_path=config.positive_dataset_path,
        negative_dataset_path=config.negative_dataset_path,
        input_ids_column=config.input_ids_column,
        hook_name=config.hook_name,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        streaming=config.streaming,
        device=DEVICE,
    )

    # Create trainer and train
    trainer = ContrastiveSAETrainer(
        config=config,
        model=model,
        sae=sae,
        activation_extractor=activation_extractor,
    )

    trainer.train()


if __name__ == "__main__":
    main()
