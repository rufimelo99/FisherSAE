import argparse
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Tuple

import torch
from sae_lens import (
    SAE,
    HookedSAETransformer,
    PretokenizeRunner,
    PretokenizeRunnerConfig,
)
from sae_lens.training.activations_store import ActivationsStore
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookedRootModule

from code_sae.logger import logger
from code_sae.utils import get_device, set_seed

DEVICE = get_device()
SEED = set_seed()


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


@dataclass
class SingleReconstructionResult:
    reconstruction_loss: float
    l0_sparsity: float
    l1_sparsity: float
    l2_sparsity: float


@dataclass
class SingleEvaluationResult:
    unique_id: str
    dataset: str
    reconstruction_results: List[SingleReconstructionResult]


@torch.no_grad()
def get_recons_loss(
    sae: SAE,
    model: HookedSAETransformer,
    batch_tokens: torch.Tensor,
    activation_store: ActivationsStore,
):
    print("oi")
    original_logits, original_ce_loss = model(
        batch_tokens,
        loss_per_token=True,
    )
    print("original_logits", original_logits.shape)
    print("original_ce_loss", original_ce_loss.shape)
    breakpoint()


@torch.no_grad()
def run_evals(
    sae,
    activation_store: ActivationsStore,
    model,
    dataset,
    n_batches=8,
    eval_batch_size_prompts=512,
    verbose=False,
):
    batch_iter = range(n_batches)
    if verbose:
        batch_iter = tqdm(batch_iter, desc="Reconstruction Batches")

    for _ in batch_iter:
        batch_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)
        current_model = HookedTransformer.from_pretrained_no_processing(
            "gpt2", device=DEVICE, **sae.cfg.model_from_pretrained_kwargs
        )

        breakpoint()
        original_logits, original_ce_loss = current_model(
            batch_tokens, loss_per_token=True
        )

        get_recons_loss(
            sae,
            model,
            batch_tokens,
            activation_store,
        ).items()
        break


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

            run_evals(
                sae,
                activation_store,
                model,
                dataset,
                n_batches=config["n_batches"],
                eval_batch_size_prompts=config["eval_batch_size_prompts"],
                verbose=True,
            )

            # Save the results
            output_file = output_path / f"{Path(sae_path).stem}_results.json"
            with open(output_file, "w") as f:
                json.dump(results, f)

    eval_results.append(results)


if __name__ == "__main__":
    args = parse_args()
    run_evaluations(args)
