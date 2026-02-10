"""
Contrastive SAE Training with SAELens

This module extends SAELens to train Sparse Autoencoders with an additional
contrastive constraint. The contrastive loss encourages the SAE to learn
representations that can distinguish between paired samples (e.g., secure
vs vulnerable code).
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
from torch.utils.data import DataLoader, Dataset

from sae_lens.saes.sae import TrainingSAE, TrainingSAEConfig

@dataclass
class ContrastiveSAEConfig(TrainingSAEConfig):
    """Extended config for contrastive SAE training."""

    # Contrastive loss parameters
    contrastive_weight: float = 0.1  # Weight for contrastive loss term
    contrastive_temperature: float = 0.07  # Temperature for InfoNCE loss
    contrastive_mode: str = "infonce"  # "infonce", "triplet", or "cosine"
    triplet_margin: float = 1.0  # Margin for triplet loss
    use_feature_contrastive: bool = True  # Apply contrastive on SAE features (vs reconstructions)


class ContrastiveSAE(TrainingSAE):
    """
    Sparse Autoencoder with contrastive learning support.

    Extends SAELens TrainingSAE to add contrastive loss computation
    between paired samples (e.g., secure vs vulnerable code activations).
    """

    def __init__(self, cfg: ContrastiveSAEConfig):
        super().__init__(cfg)
        self.contrastive_cfg = cfg

    def compute_contrastive_loss(
        self,
        features_a: torch.Tensor,  # [batch, d_sae]
        features_b: torch.Tensor,  # [batch, d_sae]
        labels: Optional[torch.Tensor] = None,  # [batch] - 1 if same class, 0 if different
    ) -> torch.Tensor:
        """
        Compute contrastive loss between paired feature representations.

        Args:
            features_a: First set of SAE features (e.g., secure code)
            features_b: Second set of SAE features (e.g., vulnerable code)
            labels: Optional labels indicating if pairs should be similar (1) or different (0)
                   If None, assumes all pairs should be different (secure != vulnerable)

        Returns:
            Contrastive loss value
        """
        mode = self.contrastive_cfg.contrastive_mode

        if mode == "infonce":
            return self._infonce_loss(features_a, features_b)
        elif mode == "triplet":
            return self._triplet_loss(features_a, features_b)
        elif mode == "cosine":
            return self._cosine_contrastive_loss(features_a, features_b, labels)
        else:
            raise ValueError(f"Unknown contrastive mode: {mode}")

    def _infonce_loss(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss.

        Treats features_a[i] as anchor, features_a[j] (j!=i) as positives of same type,
        and features_b as negatives (different class).
        """
        temperature = self.contrastive_cfg.contrastive_temperature
        batch_size = features_a.shape[0]

        # Normalize features
        features_a = F.normalize(features_a, dim=-1)
        features_b = F.normalize(features_b, dim=-1)

        # Compute similarity matrices
        # Positive: similarity within same class (a with other a's)
        sim_aa = torch.mm(features_a, features_a.t()) / temperature
        # Negative: similarity between different classes (a with b's)
        sim_ab = torch.mm(features_a, features_b.t()) / temperature

        # For each anchor in a, treat other a's as positive and all b's as negative
        # Mask out self-similarity
        mask = torch.eye(batch_size, device=features_a.device).bool()
        sim_aa = sim_aa.masked_fill(mask, float('-inf'))

        # Concatenate positive and negative similarities
        # logits: [batch, batch-1 + batch] where first batch-1 are positives
        logits = torch.cat([sim_aa, sim_ab], dim=1)

        # Labels: positives are at indices 0 to batch-2 (excluding self)
        # We want to maximize similarity with positives (same class)
        # Simple approach: treat first non-self element as the positive
        labels = torch.zeros(batch_size, dtype=torch.long, device=features_a.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def _triplet_loss(
        self,
        anchor: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Triplet loss where anchor and positive are from same class,
        negative is from different class.

        Uses in-batch positives (other samples of same class as anchor).
        """
        margin = self.contrastive_cfg.triplet_margin

        # Normalize
        anchor = F.normalize(anchor, dim=-1)
        negative = F.normalize(negative, dim=-1)

        batch_size = anchor.shape[0]

        # For each anchor, use another random sample from same batch as positive
        # (assuming batch contains multiple samples of same class)
        perm = torch.randperm(batch_size, device=anchor.device)
        positive = anchor[perm]

        # Compute distances
        pos_dist = (anchor - positive).pow(2).sum(dim=-1)
        neg_dist = (anchor - negative).pow(2).sum(dim=-1)

        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + margin).mean()
        return loss

    def _cosine_contrastive_loss(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cosine embedding contrastive loss.

        If labels is None, assumes pairs should be different (label=-1).
        """
        if labels is None:
            # Default: pairs should be different (secure != vulnerable)
            labels = -torch.ones(features_a.shape[0], device=features_a.device)

        loss = F.cosine_embedding_loss(
            features_a,
            features_b,
            labels,
            margin=0.5
        )
        return loss

    def forward(
        self,
        x: torch.Tensor,
        dead_neuron_mask: Optional[torch.Tensor] = None,  # noqa: ARG002
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning reconstruction and features.

        Returns:
            sae_out: Reconstructed activations
            feature_acts: Sparse feature activations
            loss: Standard SAE loss (reconstruction + sparsity)
            mse_loss: Mean squared error loss
            l1_loss: L1 sparsity loss
        """
        # Standard SAE forward pass
        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)

        # Compute standard losses
        mse_loss = F.mse_loss(sae_out, x, reduction='mean')
        l1_loss = feature_acts.abs().sum(dim=-1).mean()

        loss = mse_loss + self.cfg.l1_coefficient * l1_loss

        return sae_out, feature_acts, loss, mse_loss, l1_loss


class ContrastivePairDataset(Dataset):
    """
    Dataset for paired activations (e.g., secure and vulnerable code).

    Expects pre-computed activations from a language model.
    """

    def __init__(
        self,
        activations_a: torch.Tensor,  # [n_samples, d_model]
        activations_b: torch.Tensor,  # [n_samples, d_model]
        labels: Optional[torch.Tensor] = None,  # [n_samples]
    ):
        assert activations_a.shape == activations_b.shape
        self.activations_a = activations_a
        self.activations_b = activations_b
        self.labels = labels

    def __len__(self) -> int:
        return self.activations_a.shape[0]

    def __getitem__(self, idx: int) -> dict:
        item = {
            "activations_a": self.activations_a[idx],
            "activations_b": self.activations_b[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class ContrastiveSAETrainer:
    """
    Trainer for Contrastive SAE that combines standard SAE loss with contrastive loss.
    """

    def __init__(
        self,
        sae: ContrastiveSAE,
        dataset: ContrastivePairDataset,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        device: str = "cuda",
    ):
        self.sae = sae.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.optimizer = torch.optim.Adam(
            sae.parameters(),
            lr=learning_rate,
        )

        self.step = 0
        self.losses_log: List[dict] = []

    def train_step(self, batch: dict) -> dict:
        """
        Single training step with combined SAE + contrastive loss.
        """
        self.sae.train()
        self.optimizer.zero_grad()

        act_a = batch["activations_a"].to(self.device)
        act_b = batch["activations_b"].to(self.device)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(self.device)

        # Forward pass for both activation sets
        sae_out_a, features_a, loss_a, mse_a, l1_a = self.sae(act_a)
        sae_out_b, features_b, loss_b, mse_b, l1_b = self.sae(act_b)

        # Standard SAE loss (average of both)
        sae_loss = (loss_a + loss_b) / 2
        mse_loss = (mse_a + mse_b) / 2
        l1_loss = (l1_a + l1_b) / 2

        # Contrastive loss on features or reconstructions
        if self.sae.contrastive_cfg.use_feature_contrastive:
            contrastive_loss = self.sae.compute_contrastive_loss(
                features_a, features_b, labels
            )
        else:
            contrastive_loss = self.sae.compute_contrastive_loss(
                sae_out_a, sae_out_b, labels
            )

        # Combined loss
        total_loss = sae_loss + self.sae.contrastive_cfg.contrastive_weight * contrastive_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        self.step += 1

        metrics = {
            "total_loss": total_loss.item(),
            "sae_loss": sae_loss.item(),
            "mse_loss": mse_loss.item(),
            "l1_loss": l1_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "step": self.step,
        }
        self.losses_log.append(metrics)

        return metrics

    def train(
        self,
        n_epochs: int = 10,
        log_every: int = 100,
        save_every: int = 1000,
        save_path: Optional[str] = None,
    ) -> List[dict]:
        """
        Full training loop.
        """
        for epoch in range(n_epochs):
            epoch_losses = []

            for batch in self.dataloader:
                metrics = self.train_step(batch)
                epoch_losses.append(metrics["total_loss"])

                if self.step % log_every == 0:
                    print(
                        f"Epoch {epoch+1}/{n_epochs} | "
                        f"Step {self.step} | "
                        f"Total Loss: {metrics['total_loss']:.4f} | "
                        f"SAE Loss: {metrics['sae_loss']:.4f} | "
                        f"Contrastive: {metrics['contrastive_loss']:.4f}"
                    )

                if save_path and self.step % save_every == 0:
                    self.save_checkpoint(f"{save_path}/checkpoint_{self.step}.pt")

            print(f"Epoch {epoch+1} complete. Average loss: {sum(epoch_losses)/len(epoch_losses):.4f}")

        return self.losses_log

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "sae_state_dict": self.sae.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.step,
            "config": self.sae.contrastive_cfg,
        }, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.sae.load_state_dict(checkpoint["sae_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]
        print(f"Loaded checkpoint from {path} at step {self.step}")


# ============================================================================
# Example usage
# ============================================================================

def create_contrastive_sae(
    d_model: int = 4096,
    d_sae: int = 16384,
    contrastive_weight: float = 0.1,
    contrastive_mode: str = "infonce",
    l1_coefficient: float = 1e-3,
    **kwargs,
) -> ContrastiveSAE:
    """
    Factory function to create a ContrastiveSAE with common defaults.
    """
    config = ContrastiveSAEConfig(
        d_in=d_model,
        d_sae=d_sae,
        l1_coefficient=l1_coefficient,
        contrastive_weight=contrastive_weight,
        contrastive_mode=contrastive_mode,
        **kwargs,
    )
    return ContrastiveSAE(config)


if __name__ == "__main__":
    # Example: Training a contrastive SAE on paired code activations

    # Configuration
    D_MODEL = 4096  # Hidden size of base LLM
    D_SAE = 16384   # SAE dictionary size
    BATCH_SIZE = 32
    N_EPOCHS = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dummy paired data (replace with your actual activations)
    # activations_secure: residual stream activations from secure code
    # activations_vuln: residual stream activations from vulnerable code
    n_samples = 1000
    activations_secure = torch.randn(n_samples, D_MODEL)
    activations_vuln = torch.randn(n_samples, D_MODEL)

    # Create dataset
    dataset = ContrastivePairDataset(
        activations_a=activations_secure,
        activations_b=activations_vuln,
    )

    # Create contrastive SAE
    sae = create_contrastive_sae(
        d_model=D_MODEL,
        d_sae=D_SAE,
        contrastive_weight=0.1,      # Weight for contrastive loss
        contrastive_mode="infonce",   # Options: "infonce", "triplet", "cosine"
        l1_coefficient=1e-3,          # Sparsity penalty
    )

    # Create trainer
    trainer = ContrastiveSAETrainer(
        sae=sae,
        dataset=dataset,
        batch_size=BATCH_SIZE,
        learning_rate=1e-4,
        device=DEVICE,
    )

    # Train
    print(f"Training Contrastive SAE on {DEVICE}")
    print(f"Model: d_model={D_MODEL}, d_sae={D_SAE}")
    print(f"Contrastive mode: {sae.contrastive_cfg.contrastive_mode}")
    print(f"Contrastive weight: {sae.contrastive_cfg.contrastive_weight}")
    print("-" * 50)

    losses = trainer.train(
        n_epochs=N_EPOCHS,
        log_every=50,
        save_every=500,
        save_path="./checkpoints",
    )

    # Save final model
    trainer.save_checkpoint("./checkpoints/contrastive_sae_final.pt")
    print("Training complete!")
