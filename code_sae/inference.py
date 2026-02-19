import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import einops
import torch
import wandb
from sae_lens import SAE, HookedSAETransformer
from sae_lens.training.activations_store import ActivationsStore
from tqdm import tqdm

from code_sae.logger import logger
from code_sae.utils import get_device, set_seed

DEVICE = get_device()
set_seed()


@dataclass
class InferenceMetrics:
    mse: float
    l0: float
    l1: float
    cossim: float
    l2_norm_ratio: float
    n_samples: int

    def to_dict(self):
        return {
            "mse": self.mse,
            "l0": self.l0,
            "l1": self.l1,
            "cossim": self.cossim,
            "l2_norm_ratio": self.l2_norm_ratio,
            "n_samples": self.n_samples,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAE inference on test set")
    parser.add_argument(
        "--config",
        type=str,
        default="inference_config.json",
        help="Path to the config file",
    )
    parser.add_argument(
        "--kwargs",
        type=str,
        default="{}",
        help="Additional keyword arguments to override config",
    )
    return parser.parse_args()


@torch.no_grad()
def compute_reconstruction_mse(
    sae: SAE,
    model: HookedSAETransformer,
    batch_tokens: torch.Tensor,
    activation_store: ActivationsStore,
) -> dict:
    """
    Compute reconstruction MSE and related metrics for a batch of tokens.
    """
    hook_name = sae.cfg.hook_name
    hook_head_index = sae.cfg.hook_head_index

    # Get activations from the model
    _, cache = model.run_with_cache(
        batch_tokens,
        prepend_bos=False,
        names_filter=[hook_name],
        stop_at_layer=sae.cfg.hook_layer + 1,
    )

    # Extract activations based on hook type
    has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
    if hook_head_index is not None:
        original_act = cache[hook_name][:, :, hook_head_index]
    elif any(substring in hook_name for substring in has_head_dim_key_substrings):
        original_act = cache[hook_name].flatten(-2, -1)
    else:
        original_act = cache[hook_name]

    # Normalize if necessary
    if activation_store.normalize_activations == "expected_average_only_in":
        original_act = activation_store.apply_norm_scaling_factor(original_act)

    # Encode and decode through SAE
    sae_feature_activations = sae.encode(original_act.to(sae.device))
    sae_out = sae.decode(sae_feature_activations).to(original_act.device)

    # Unscale if needed
    if activation_store.normalize_activations == "expected_average_only_in":
        sae_out = activation_store.unscale(sae_out)

    del cache

    # Flatten for metric computation
    flattened_sae_input = einops.rearrange(original_act, "b ctx d -> (b ctx) d")
    flattened_sae_feature_acts = einops.rearrange(
        sae_feature_activations, "b ctx d -> (b ctx) d"
    )
    flattened_sae_out = einops.rearrange(sae_out, "b ctx d -> (b ctx) d")

    # Compute MSE
    mse = (flattened_sae_input - flattened_sae_out).pow(2).mean().item()

    # Compute L0 (average number of active features per token)
    l0 = (flattened_sae_feature_acts > 0).float().sum(dim=-1).mean().item()

    # Compute L1
    l1 = flattened_sae_feature_acts.abs().sum(dim=-1).mean().item()

    # Compute cosine similarity
    x_normed = flattened_sae_input / torch.norm(
        flattened_sae_input, dim=-1, keepdim=True
    ).clamp(min=1e-8)
    x_hat_normed = flattened_sae_out / torch.norm(
        flattened_sae_out, dim=-1, keepdim=True
    ).clamp(min=1e-8)
    cossim = (x_normed * x_hat_normed).sum(dim=-1).mean().item()

    # Compute L2 norm ratio
    l2_norm_in = torch.norm(flattened_sae_input, dim=-1)
    l2_norm_out = torch.norm(flattened_sae_out, dim=-1)
    l2_norm_in_safe = l2_norm_in.clone()
    l2_norm_in_safe[torch.abs(l2_norm_in_safe) < 1e-4] = 1.0
    l2_norm_ratio = (l2_norm_out / l2_norm_in_safe).mean().item()

    return {
        "mse": mse,
        "l0": l0,
        "l1": l1,
        "cossim": cossim,
        "l2_norm_ratio": l2_norm_ratio,
        "n_tokens": flattened_sae_input.shape[0],
    }


@torch.no_grad()
def run_inference(
    sae: SAE,
    model: HookedSAETransformer,
    activation_store: ActivationsStore,
    n_batches: int,
    batch_size: int,
    verbose: bool = True,
    log_to_wandb: bool = True,
) -> InferenceMetrics:
    """
    Run inference on multiple batches and aggregate metrics.
    """
    all_metrics = {
        "mse": [],
        "l0": [],
        "l1": [],
        "cossim": [],
        "l2_norm_ratio": [],
        "n_tokens": [],
    }

    batch_iter = range(n_batches)
    if verbose:
        batch_iter = tqdm(batch_iter, desc="Running inference")

    for batch_idx, _ in enumerate(batch_iter):
        batch_tokens = activation_store.get_batch_tokens(batch_size)
        metrics = compute_reconstruction_mse(sae, model, batch_tokens, activation_store)

        for key in all_metrics:
            all_metrics[key].append(metrics[key])

        # Log per-batch metrics to wandb
        if log_to_wandb:
            wandb.log({
                "batch/mse": metrics["mse"],
                "batch/l0": metrics["l0"],
                "batch/l1": metrics["l1"],
                "batch/cossim": metrics["cossim"],
                "batch/l2_norm_ratio": metrics["l2_norm_ratio"],
                "batch_idx": batch_idx,
            })

    # Compute weighted averages (weighted by number of tokens per batch)
    total_tokens = sum(all_metrics["n_tokens"])
    weights = [n / total_tokens for n in all_metrics["n_tokens"]]

    weighted_mse = sum(m * w for m, w in zip(all_metrics["mse"], weights))
    weighted_l0 = sum(m * w for m, w in zip(all_metrics["l0"], weights))
    weighted_l1 = sum(m * w for m, w in zip(all_metrics["l1"], weights))
    weighted_cossim = sum(m * w for m, w in zip(all_metrics["cossim"], weights))
    weighted_l2_ratio = sum(m * w for m, w in zip(all_metrics["l2_norm_ratio"], weights))

    return InferenceMetrics(
        mse=weighted_mse,
        l0=weighted_l0,
        l1=weighted_l1,
        cossim=weighted_cossim,
        l2_norm_ratio=weighted_l2_ratio,
        n_samples=total_tokens,
    )


def inference(config: dict):
    """
    Main inference function.

    Supports loading SAE from:
    - HuggingFace: provide `sae_release` and `sae_id`
    - Local disk: provide `sae_path`
    """
    # SAE loading configuration
    sae_path = config.get("sae_path", None)
    sae_release = config.get("sae_release", None)
    sae_id = config.get("sae_id", None)
    model_name = config.get("model_name", None)

    dataset = config["dataset"]
    dataset_split = config.get("dataset_split", "test")
    n_batches = config.get("n_batches", 100)
    batch_size = config.get("batch_size", 32)
    ctx_len = config.get("ctx_len", 128)
    output_dir = config.get("output_dir", "inference_results")

    # Wandb configuration
    wandb_project = config.get("wandb_project", "SAE-Inference")
    wandb_run_name = config.get("wandb_run_name", None)
    log_to_wandb = config.get("log_to_wandb", True)

    # Determine SAE source
    load_from_hf = sae_release is not None and sae_id is not None
    load_from_disk = sae_path is not None

    if not load_from_hf and not load_from_disk:
        raise ValueError(
            "Must provide either `sae_path` (local) or `sae_release` + `sae_id` (HuggingFace)"
        )

    # Initialize wandb
    if log_to_wandb:
        wandb_config = {
            "dataset": dataset,
            "dataset_split": dataset_split,
            "n_batches": n_batches,
            "batch_size": batch_size,
            "ctx_len": ctx_len,
        }
        if load_from_hf:
            wandb_config["sae_release"] = sae_release
            wandb_config["sae_id"] = sae_id
        else:
            wandb_config["sae_path"] = sae_path

        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=wandb_config,
        )
        logger.info("Wandb initialized", project=wandb_project, run_name=wandb_run_name)

    # Load SAE
    if load_from_hf:
        logger.info("Loading SAE from HuggingFace", release=sae_release, sae_id=sae_id)
        sae, _, _ = SAE.from_pretrained(
            release=sae_release, sae_id=sae_id, device=DEVICE
        )
        if model_name is None:
            model_name = sae.cfg.model_name
    else:
        logger.info("Loading SAE from disk", path=sae_path)
        sae, _, _ = SAE.load_from_disk(path=sae_path, device=DEVICE)
        if model_name is None:
            sae_cfg = json.load(open(os.path.join(sae_path, "cfg.json"), "r"))
            model_name = sae_cfg["model_name"]

    logger.info("Loading model", model=model_name)

    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name,
        device=DEVICE,
        **(sae.cfg.metadata.model_from_pretrained_kwargs or {}),
    )

    # Create activation store for the dataset
    # Format: "dataset_name" with split specified separately
    dataset_with_split = f"{dataset}"
    logger.info("Creating activation store", dataset=dataset, split=dataset_split)
    activation_store = ActivationsStore.from_sae(
        model,
        sae,
        context_size=ctx_len,
        dataset=dataset_with_split,
        streaming=True,
        split=dataset_split,
    )
    activation_store.shuffle_input_dataset(seed=42)
    activation_store.set_norm_scaling_factor_if_needed()

    # Run inference
    logger.info("Running inference", n_batches=n_batches, batch_size=batch_size)
    metrics = run_inference(
        sae=sae,
        model=model,
        activation_store=activation_store,
        n_batches=n_batches,
        batch_size=batch_size,
        verbose=True,
        log_to_wandb=log_to_wandb,
    )

    # Log results
    logger.info(
        "Inference complete",
        mse=f"{metrics.mse:.6f}",
        l0=f"{metrics.l0:.2f}",
        l1=f"{metrics.l1:.4f}",
        cossim=f"{metrics.cossim:.4f}",
        l2_norm_ratio=f"{metrics.l2_norm_ratio:.4f}",
        n_samples=metrics.n_samples,
    )

    # Log to wandb
    if log_to_wandb:
        wandb.log({
            "mse": metrics.mse,
            "l0": metrics.l0,
            "l1": metrics.l1,
            "cossim": metrics.cossim,
            "l2_norm_ratio": metrics.l2_norm_ratio,
            "n_samples": metrics.n_samples,
        })
        wandb.summary.update(metrics.to_dict())

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if load_from_hf:
        sae_name = f"{sae_release}_{sae_id}".replace("/", "_")
    else:
        sae_name = Path(sae_path).stem
    output_file = output_path / f"{sae_name}_inference_results.json"

    results = {
        "dataset": dataset,
        "dataset_split": dataset_split,
        "n_batches": n_batches,
        "batch_size": batch_size,
        "ctx_len": ctx_len,
        "metrics": metrics.to_dict(),
    }
    if load_from_hf:
        results["sae_release"] = sae_release
        results["sae_id"] = sae_id
    else:
        results["sae_path"] = sae_path

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved", path=str(output_file))

    # Finish wandb run
    if log_to_wandb:
        wandb.finish()

    return metrics


if __name__ == "__main__":
    args = parse_args()
    json_path = args.config
    if not os.path.exists(json_path):
        raise ValueError(f"Config path {json_path} does not exist.")

    with open(json_path, "r") as f:
        config = json.load(f)

    kwargs = json.loads(args.kwargs)
    if kwargs:
        logger.info("Overriding config", kwargs=kwargs)
        config.update(kwargs)

    inference(config)
