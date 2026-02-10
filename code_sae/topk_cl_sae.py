"""
TopK Contrastive Learning SAE

Defines TopKCLSAE (inference) and TopKCLTrainingSAE (training) that extend
SAELens TopK variants with contrastive learning during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from sae_lens.saes.sae import TrainStepInput, TrainStepOutput
from sae_lens.saes.topk_sae import TopKSAE, TopKSAEConfig, TopKTrainingSAE, TopKTrainingSAEConfig


@dataclass
class TopKCLSAEConfig(TopKSAEConfig):
    """Config for TopKCLSAE (inference-only)."""


class TopKCLSAE(TopKSAE):
    """Inference-only TopK SAE used with contrastive training."""

    cfg: TopKCLSAEConfig  # type: ignore[assignment]


@dataclass
class TopKCLTrainingSAEConfig(TopKTrainingSAEConfig):
    """
    Config for TopKCLTrainingSAE with contrastive learning parameters.
    """

    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.07
    contrastive_mode: Literal["infonce", "triplet", "cosine"] = "infonce"
    triplet_margin: float = 1.0
    use_feature_contrastive: bool = True


class TopKCLTrainingSAE(TopKTrainingSAE):
    """
    TopKTrainingSAE variant that adds contrastive learning.

    Expects paired inputs during training. `step_input.sae_in` can be:
    - tuple/list: (acts_a, acts_b)
    - dict: {"activations_a": ..., "activations_b": ..., "labels"?: ...}
            or {"sae_in_a": ..., "sae_in_b": ..., "labels"?: ...}
    If a single tensor is provided, this falls back to standard TopK training.
    """

    cfg: TopKCLTrainingSAEConfig  # type: ignore[assignment]

    def _split_contrastive_inputs(
        self,
        sae_in: torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | list[torch.Tensor]
        | dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if isinstance(sae_in, (tuple, list)):
            if len(sae_in) != 2:
                raise ValueError("Expected a pair of tensors for contrastive training.")
            return sae_in[0], sae_in[1], None
        if isinstance(sae_in, dict):
            if "activations_a" in sae_in and "activations_b" in sae_in:
                return sae_in["activations_a"], sae_in["activations_b"], sae_in.get(
                    "labels"
                )
            if "sae_in_a" in sae_in and "sae_in_b" in sae_in:
                return sae_in["sae_in_a"], sae_in["sae_in_b"], sae_in.get("labels")
            raise ValueError(
                "Contrastive input dict must contain activations_a/activations_b or sae_in_a/sae_in_b."
            )
        raise ValueError(
            "TopKCLTrainingSAE requires paired inputs for contrastive training."
        )

    def _infonce_loss(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
    ) -> torch.Tensor:
        temperature = self.cfg.contrastive_temperature
        features_a = F.normalize(features_a, dim=-1)
        features_b = F.normalize(features_b, dim=-1)

        logits = (features_a @ features_b.t()) / temperature
        labels = torch.arange(logits.shape[0], device=logits.device)

        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)
        return (loss_a + loss_b) / 2

    def _triplet_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
    ) -> torch.Tensor:
        margin = self.cfg.triplet_margin
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)

        if anchor.shape[0] < 2:
            return anchor.new_tensor(0.0)

        negative = torch.roll(positive, shifts=1, dims=0)
        pos_dist = (anchor - positive).pow(2).sum(dim=-1)
        neg_dist = (anchor - negative).pow(2).sum(dim=-1)
        return F.relu(pos_dist - neg_dist + margin).mean()

    def _cosine_contrastive_loss(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if labels is None:
            labels = -torch.ones(features_a.shape[0], device=features_a.device)
        if labels.min().item() >= 0:
            labels = labels * 2 - 1
        return F.cosine_embedding_loss(features_a, features_b, labels, margin=0.5)

    def compute_contrastive_loss(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mode = self.cfg.contrastive_mode
        if mode == "infonce":
            return self._infonce_loss(features_a, features_b)
        if mode == "triplet":
            return self._triplet_loss(features_a, features_b)
        if mode == "cosine":
            return self._cosine_contrastive_loss(features_a, features_b, labels)
        raise ValueError(f"Unknown contrastive mode: {mode}")

    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        if isinstance(step_input.sae_in, torch.Tensor):
            return super().training_forward_pass(step_input)

        act_a, act_b, labels = self._split_contrastive_inputs(step_input.sae_in)

        feature_acts_a, hidden_pre_a = self.encode_with_hidden_pre(act_a)
        sae_out_a = self.decode(feature_acts_a)

        feature_acts_b, hidden_pre_b = self.encode_with_hidden_pre(act_b)
        sae_out_b = self.decode(feature_acts_b)

        per_item_mse_a = self.mse_loss_fn(sae_out_a, act_a)
        per_item_mse_b = self.mse_loss_fn(sae_out_b, act_b)
        mse_loss = (per_item_mse_a.sum(dim=-1).mean() + per_item_mse_b.sum(dim=-1).mean()) / 2

        step_input_a = TrainStepInput(
            sae_in=act_a,
            coefficients=step_input.coefficients,
            dead_neuron_mask=step_input.dead_neuron_mask,
            n_training_steps=step_input.n_training_steps,
        )
        step_input_b = TrainStepInput(
            sae_in=act_b,
            coefficients=step_input.coefficients,
            dead_neuron_mask=step_input.dead_neuron_mask,
            n_training_steps=step_input.n_training_steps,
        )

        aux_a = self.calculate_aux_loss(
            step_input=step_input_a,
            feature_acts=feature_acts_a,
            hidden_pre=hidden_pre_a,
            sae_out=sae_out_a,
        )
        aux_b = self.calculate_aux_loss(
            step_input=step_input_b,
            feature_acts=feature_acts_b,
            hidden_pre=hidden_pre_b,
            sae_out=sae_out_b,
        )

        losses: dict[str, torch.Tensor] = {"mse_loss": mse_loss}

        total_loss = mse_loss
        if isinstance(aux_a, dict) and isinstance(aux_b, dict):
            for key in set(aux_a.keys()) | set(aux_b.keys()):
                val_a = aux_a.get(key)
                val_b = aux_b.get(key)
                if val_a is None:
                    avg_val = val_b
                elif val_b is None:
                    avg_val = val_a
                else:
                    avg_val = (val_a + val_b) / 2
                losses[key] = avg_val
                total_loss = total_loss + avg_val
        else:
            aux_tensor_a = aux_a if isinstance(aux_a, torch.Tensor) else torch.zeros_like(mse_loss)
            aux_tensor_b = aux_b if isinstance(aux_b, torch.Tensor) else torch.zeros_like(mse_loss)
            aux_avg = (aux_tensor_a + aux_tensor_b) / 2
            losses["aux_loss"] = aux_avg
            total_loss = total_loss + aux_avg

        if self.cfg.use_feature_contrastive:
            contrastive_loss = self.compute_contrastive_loss(
                feature_acts_a, feature_acts_b, labels
            )
        else:
            contrastive_loss = self.compute_contrastive_loss(
                sae_out_a, sae_out_b, labels
            )

        losses["contrastive_loss"] = contrastive_loss
        total_loss = total_loss + self.cfg.contrastive_weight * contrastive_loss

        return TrainStepOutput(
            sae_in=act_a,
            sae_out=sae_out_a,
            feature_acts=feature_acts_a,
            hidden_pre=hidden_pre_a,
            loss=total_loss,
            losses=losses,
        )
