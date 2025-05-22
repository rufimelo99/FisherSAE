import argparse
import json
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, List, Tuple

import einops
import torch
from sae_lens import (
    SAE,
    HookedSAETransformer,
)
from sae_lens.training.activations_store import ActivationsStore
from tqdm import tqdm

from code_sae.logger import logger
from code_sae.utils import get_device, js_div, kl_div, set_seed

DEVICE = get_device()
SEED = set_seed()


@dataclass
class ReconstructionMetrics:
    recons_kl_div: torch.Tensor
    recons_js_div: torch.Tensor
    zero_abl_kl_div: torch.Tensor
    zero_abl_js_div: torch.Tensor
    recons_ce_loss: torch.Tensor
    zero_abl_ce_loss: torch.Tensor


@dataclass
class SparsityMetrics:
    l0: torch.Tensor
    l1: torch.Tensor
    l2_norm_ratio: torch.Tensor
    cossim: torch.Tensor
    mse: torch.Tensor
    cossim: torch.Tensor


@dataclass
class SingleEvaluationResult:
    unique_id: str
    dataset: str
    recons_kl_div: float
    recons_js_div: float
    zero_abl_kl_div: float
    zero_abl_js_div: float
    recons_ce_loss: float
    zero_abl_ce_loss: float
    l0: float
    l1: float
    l2_norm_ratio: float
    cossim: float
    mse: float

    def to_dict(self):
        return {
            "unique_id": self.unique_id,
            "dataset": self.dataset,
            "recons_kl_div": self.recons_kl_div,
            "recons_js_div": self.recons_js_div,
            "zero_abl_kl_div": self.zero_abl_kl_div,
            "zero_abl_js_div": self.zero_abl_js_div,
            "recons_ce_loss": self.recons_ce_loss,
            "zero_abl_ce_loss": self.zero_abl_ce_loss,
            "l0": self.l0,
            "l1": self.l1,
            "l2_norm_ratio": self.l2_norm_ratio,
            "cossim": self.cossim,
            "mse": self.mse,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a SAE")
    parser.add_argument(
        "--config",
        type=str,
        default="evaluation_config.json",
        help="Path to the config file",
    )
    return parser.parse_args()


def get_sae_and_model_paths(config) -> Tuple[List[str], HookedSAETransformer]:
    sae_paths = config["sae_paths"]
    assert len(sae_paths) > 0, "No SAEs provided in the config file."

    model_name = config["model_name"]
    assert model_name is not None, "No model name provided in the config file."

    # Read each config file and see if it has the same model name
    logger.info("Checking if all SAEs have the same model name...")
    for sae_path in tqdm(sae_paths):
        with open(os.path.join(sae_path, "cfg.json"), "r") as f:
            cfg = json.load(f)
            assert (
                cfg["model_name"] == model_name
            ), f"Model name mismatch: {cfg['model_name']} != {model_name}"

    logger.info("All SAEs have the same model name.")

    return sae_paths, model_name


@torch.no_grad()
def get_reconstruction_metrics(
    sae: SAE,
    model: HookedSAETransformer,
    batch_tokens: torch.Tensor,
    activation_store: ActivationsStore,
):
    original_logits = model.forward(batch_tokens, loss_per_token=True)

    hook_name = sae.cfg.hook_name
    mask = torch.ones_like(batch_tokens, dtype=torch.bool)

    # TODO(tomMcGrath): the rescaling below is a bit of a hack and could probably be tidied up
    def standard_replacement_hook(activations: torch.Tensor, hook: Any):  # noqa: ARG001
        original_device = activations.device
        activations = activations.to(sae.device)

        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.apply_norm_scaling_factor(activations)

        # SAE class agnost forward forward pass.
        new_activations = sae.decode(sae.encode(activations)).to(activations.dtype)

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            new_activations = activation_store.unscale(new_activations)

        new_activations = torch.where(mask[..., None], new_activations, activations)

        return new_activations.to(original_device)

    def standard_zero_ablate_hook(activations: torch.Tensor, hook: Any):  # noqa: ARG001
        original_device = activations.device
        activations = activations.to(sae.device)
        activations = torch.zeros_like(activations)
        return activations.to(original_device)

    has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
    if any(substring in hook_name for substring in has_head_dim_key_substrings):
        # Look into SAELens/sae_lens/evals.py !
        raise NotImplementedError(
            "We would need to modify the SAE to work with head dim keys."
        )
    replacement_hook = standard_replacement_hook
    zero_ablate_hook = standard_zero_ablate_hook

    recons_logits, recons_ce_loss = model.run_with_hooks(
        batch_tokens,
        return_type="both",
        fwd_hooks=[(hook_name, partial(replacement_hook))],
        loss_per_token=True,
    )
    zero_abl_logits, zero_abl_ce_loss = model.run_with_hooks(
        batch_tokens,
        return_type="both",
        fwd_hooks=[(hook_name, zero_ablate_hook)],
        loss_per_token=True,
    )

    recons_kl_div = kl_div(original_logits, recons_logits).mean(dim=-1).to("cpu")
    zero_abl_kl_div = kl_div(original_logits, zero_abl_logits).mean(dim=-1).to("cpu")
    recons_js_div = js_div(original_logits, recons_logits).mean(dim=-1).to("cpu")
    zero_abl_js_div = js_div(original_logits, zero_abl_logits).mean(dim=-1).to("cpu")
    recons_ce_loss = recons_ce_loss.mean(dim=1).to("cpu")
    zero_abl_ce_loss = zero_abl_ce_loss.mean(dim=1).to("cpu")

    return ReconstructionMetrics(
        recons_kl_div=recons_kl_div,
        recons_js_div=recons_js_div,
        zero_abl_kl_div=zero_abl_kl_div,
        zero_abl_js_div=zero_abl_js_div,
        recons_ce_loss=recons_ce_loss,
        zero_abl_ce_loss=zero_abl_ce_loss,
    )


def get_sparsity_metrics(
    sae: SAE,
    model: HookedSAETransformer,
    batch_tokens: torch.Tensor,
    activation_store: ActivationsStore,
):
    hook_name = sae.cfg.hook_name
    hook_head_index = sae.cfg.hook_head_index

    mask = torch.ones_like(batch_tokens, dtype=torch.bool)
    flattened_mask = mask.flatten()

    total_feature_acts = torch.zeros(sae.cfg.d_sae, device=sae.device)
    total_feature_prompts = torch.zeros(sae.cfg.d_sae, device=sae.device)
    logits, cache = model.run_with_cache(
        batch_tokens,
        prepend_bos=False,
        names_filter=[hook_name],
        stop_at_layer=sae.cfg.hook_layer + 1,
    )

    has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
    if hook_head_index is not None:
        original_act = cache[hook_name][:, :, hook_head_index]
    elif any(substring in hook_name for substring in has_head_dim_key_substrings):
        original_act = cache[hook_name].flatten(-2, -1)
    else:
        original_act = cache[hook_name]

    # normalise if necessary (necessary in training only, otherwise we should fold the scaling in)
    if activation_store.normalize_activations == "expected_average_only_in":
        original_act = activation_store.apply_norm_scaling_factor(original_act)

    # send the (maybe normalised) activations into the SAE
    sae_feature_activations = sae.encode(original_act.to(sae.device))
    sae_out = sae.decode(sae_feature_activations).to(original_act.device)
    del cache

    if activation_store.normalize_activations == "expected_average_only_in":
        sae_out = activation_store.unscale(sae_out)

    flattened_sae_input = einops.rearrange(original_act, "b ctx d -> (b ctx) d")
    flattened_sae_feature_acts = einops.rearrange(
        sae_feature_activations, "b ctx d -> (b ctx) d"
    )
    flattened_sae_out = einops.rearrange(sae_out, "b ctx d -> (b ctx) d")

    masked_sae_feature_activations = sae_feature_activations * mask.unsqueeze(-1)
    flattened_sae_input = flattened_sae_input[
        flattened_mask.to(flattened_sae_input.device)
    ]
    flattened_sae_feature_acts = flattened_sae_feature_acts[
        flattened_mask.to(flattened_sae_feature_acts.device)
    ]
    flattened_sae_out = flattened_sae_out[flattened_mask.to(flattened_sae_out.device)]

    l2_norm_in = torch.norm(flattened_sae_input, dim=-1)
    l2_norm_out = torch.norm(flattened_sae_out, dim=-1)
    l2_norm_in_for_div = l2_norm_in.clone()
    l2_norm_in_for_div[torch.abs(l2_norm_in_for_div) < 0.0001] = 1
    l2_norm_ratio = l2_norm_out / l2_norm_in_for_div
    l2_norm_ratio = l2_norm_ratio.to("cpu")

    l0 = (flattened_sae_feature_acts > 0).sum(dim=-1).float().to("cpu")
    l1 = flattened_sae_feature_acts.sum(dim=-1).to("cpu")

    # Variance Metrics
    resid_sum_of_squares = (flattened_sae_input - flattened_sae_out).pow(2).sum(dim=-1)

    mse = resid_sum_of_squares / flattened_mask.sum()
    mse = mse.to("cpu")

    x_normed = flattened_sae_input / torch.norm(
        flattened_sae_input, dim=-1, keepdim=True
    )
    x_hat_normed = flattened_sae_out / torch.norm(
        flattened_sae_out, dim=-1, keepdim=True
    )
    cossim = (x_normed * x_hat_normed).sum(dim=-1).to("cpu")

    # Feature-wise metrics
    sae_feature_activations_bool = (masked_sae_feature_activations > 0).float()
    total_feature_acts += sae_feature_activations_bool.sum(dim=1).sum(dim=0)
    total_feature_prompts += (sae_feature_activations_bool.sum(dim=1) > 0).sum(dim=0)
    sparsity_metrics = SparsityMetrics(
        l0=l0,
        l1=l1,
        cossim=cossim,
        l2_norm_ratio=l2_norm_ratio,
        mse=mse,
    )
    return sparsity_metrics


@torch.no_grad()
def run_evals(
    sae,
    activation_store: ActivationsStore,
    model: HookedSAETransformer,
    unique_id: str,
    dataset,
    n_batches,
    eval_batch_size_prompts,
    verbose=False,
) -> SingleEvaluationResult:
    batch_iter = range(n_batches)
    if verbose:
        batch_iter = tqdm(batch_iter, desc="Reconstruction Batches")

    run_reconst_metrics = ReconstructionMetrics(
        recons_kl_div=torch.tensor([]),
        recons_js_div=torch.tensor([]),
        zero_abl_kl_div=torch.tensor([]),
        zero_abl_js_div=torch.tensor([]),
        recons_ce_loss=torch.tensor([]),
        zero_abl_ce_loss=torch.tensor([]),
    )

    run_sparsity_metrics = SparsityMetrics(
        l0=torch.tensor([]),
        l1=torch.tensor([]),
        cossim=torch.tensor([]),
        l2_norm_ratio=torch.tensor([]),
        mse=torch.tensor([]),
    )

    for _ in batch_iter:
        batch_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)
        # current_model = HookedTransformer.from_pretrained_no_processing("gpt2", device=DEVICE, **sae.cfg.model_from_pretrained_kwargs)

        recons_metrics: ReconstructionMetrics = get_reconstruction_metrics(
            sae,
            model,
            batch_tokens,
            activation_store,
        )

        sparsity_metrics: SparsityMetrics = get_sparsity_metrics(
            sae,
            model,
            batch_tokens,
            activation_store,
        )

        run_reconst_metrics.recons_kl_div = torch.cat(
            [run_reconst_metrics.recons_kl_div, recons_metrics.recons_kl_div]
        )
        run_reconst_metrics.recons_js_div = torch.cat(
            [run_reconst_metrics.recons_js_div, recons_metrics.recons_js_div]
        )
        run_reconst_metrics.zero_abl_kl_div = torch.cat(
            [run_reconst_metrics.zero_abl_kl_div, recons_metrics.zero_abl_kl_div]
        )
        run_reconst_metrics.zero_abl_js_div = torch.cat(
            [run_reconst_metrics.zero_abl_js_div, recons_metrics.zero_abl_js_div]
        )
        run_reconst_metrics.recons_ce_loss = torch.cat(
            [run_reconst_metrics.recons_ce_loss, recons_metrics.recons_ce_loss]
        )
        run_reconst_metrics.zero_abl_ce_loss = torch.cat(
            [run_reconst_metrics.zero_abl_ce_loss, recons_metrics.zero_abl_ce_loss]
        )
        run_sparsity_metrics.l0 = torch.cat(
            [run_sparsity_metrics.l0, sparsity_metrics.l0]
        )
        run_sparsity_metrics.l1 = torch.cat(
            [run_sparsity_metrics.l1, sparsity_metrics.l1]
        )
        run_sparsity_metrics.cossim = torch.cat(
            [run_sparsity_metrics.cossim, sparsity_metrics.cossim]
        )
        run_sparsity_metrics.l2_norm_ratio = torch.cat(
            [run_sparsity_metrics.l2_norm_ratio, sparsity_metrics.l2_norm_ratio]
        )
        run_sparsity_metrics.mse = torch.cat(
            [run_sparsity_metrics.mse, sparsity_metrics.mse]
        )
        run_sparsity_metrics.cossim = torch.cat(
            [run_sparsity_metrics.cossim, sparsity_metrics.cossim]
        )

    # Average it out
    run_reconst_metrics.recons_kl_div = run_reconst_metrics.recons_kl_div.mean()
    run_reconst_metrics.recons_js_div = run_reconst_metrics.recons_js_div.mean()
    run_reconst_metrics.zero_abl_kl_div = run_reconst_metrics.zero_abl_kl_div.mean()
    run_reconst_metrics.zero_abl_js_div = run_reconst_metrics.zero_abl_js_div.mean()
    run_reconst_metrics.recons_ce_loss = run_reconst_metrics.recons_ce_loss.mean()
    run_reconst_metrics.zero_abl_ce_loss = run_reconst_metrics.zero_abl_ce_loss.mean()

    run_sparsity_metrics.l0 = run_sparsity_metrics.l0.mean()
    run_sparsity_metrics.l1 = run_sparsity_metrics.l1.mean()
    run_sparsity_metrics.cossim = run_sparsity_metrics.cossim.mean()
    run_sparsity_metrics.l2_norm_ratio = run_sparsity_metrics.l2_norm_ratio.mean()
    run_sparsity_metrics.mse = run_sparsity_metrics.mse.mean()
    run_sparsity_metrics.cossim = run_sparsity_metrics.cossim.mean()

    final_results = SingleEvaluationResult(
        unique_id=unique_id,
        dataset=dataset,
        recons_kl_div=run_reconst_metrics.recons_kl_div.item(),
        recons_js_div=run_reconst_metrics.recons_js_div.item(),
        zero_abl_kl_div=run_reconst_metrics.zero_abl_kl_div.item(),
        zero_abl_js_div=run_reconst_metrics.zero_abl_js_div.item(),
        recons_ce_loss=run_reconst_metrics.recons_ce_loss.item(),
        zero_abl_ce_loss=run_reconst_metrics.zero_abl_ce_loss.item(),
        l0=run_sparsity_metrics.l0.item(),
        l1=run_sparsity_metrics.l1.item(),
        cossim=run_sparsity_metrics.cossim.item(),
        l2_norm_ratio=run_sparsity_metrics.l2_norm_ratio.item(),
        mse=run_sparsity_metrics.mse.item(),
    )

    return final_results


def run_evaluations(args: argparse.Namespace):
    # Load the config file
    with open(args.config, "r") as f:
        config = json.load(f)

    sae_paths, model_path = get_sae_and_model_paths(config)

    eval_results = []
    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    for sae_path in tqdm(sae_paths):
        logger.info(f"Loading SAE from {sae_path}...")
        sae = SAE.load_from_disk(path=sae_path, device=DEVICE)
        logger.info("SAE loaded.")
        model = HookedSAETransformer.from_pretrained_no_processing(
            model_path, device=DEVICE, **sae.cfg.model_from_pretrained_kwargs
        )

        unique_id = f"{model.cfg.model_name}_{Path(sae_path).stem}"
        logger.info("Running evaluation", unique_id=unique_id)

        for dataset in tqdm(config["datasets"]):
            activation_store = ActivationsStore.from_sae(
                model, sae, context_size=config["ctx_len"], dataset=dataset
            )
            activation_store.shuffle_input_dataset(seed=42)
            activation_store.set_norm_scaling_factor_if_needed()
            final_results = run_evals(
                sae,
                activation_store,
                model,
                unique_id,
                dataset,
                n_batches=config["n_batches"],
                eval_batch_size_prompts=config["eval_batch_size_prompts"],
                verbose=True,
            )

            results = final_results.to_dict()
            # Save the results
            output_file = output_path / f"{Path(sae_path).stem}_results.json"
            with open(output_file, "w") as f:
                json.dump(results, f)

    eval_results.append(results)


if __name__ == "__main__":
    args = parse_args()
    run_evaluations(args)
