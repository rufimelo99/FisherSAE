import argparse
import json
import os

from sae_lens import SAE, HookedSAETransformer

from code_sae.logger import logger
from code_sae.utils import get_device, set_seed

SEED = set_seed()
DEVICE = get_device()


def compute_l0(f):
    """Compute average number of active features per sample (L0 sparsity)."""
    return (f > 0).float().sum(dim=1).mean().item()


def parse_args():
    parser = argparse.ArgumentParser(description="Inference on a model")
    parser.add_argument(
        "--path",
        type=str,
        default="checkpoints/some_folder/",
        help="Path to the checkpoint folder",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = args.path
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path {checkpoint_path} does not exist.")

    cfg = json.load(open(os.path.join(checkpoint_path, "cfg.json"), "r"))

    model = HookedSAETransformer.from_pretrained(
        cfg["model_name"],
        device=DEVICE,
    )
    sae = SAE.load_from_disk(path=checkpoint_path, device=DEVICE)
    text = "olá joca"

    tokens = model.to_tokens(text).to(DEVICE)
    import torch

    # --- Baseline (no SAE) ---
    out_no_sae = model.forward(tokens)
    loss_no_sae = torch.nn.functional.cross_entropy(
        out_no_sae[:, :-1].reshape(-1, out_no_sae.size(-1)),
        tokens[:, 1:].reshape(-1),
        reduction="mean",
    ).item()

    # --- With SAE inserted ---
    out_sae, cache = model.run_with_cache_with_saes(text, saes=[sae])

    loss_sae = torch.nn.functional.cross_entropy(
        out_sae[:, :-1].reshape(-1, out_sae.size(-1)),
        tokens[:, 1:].reshape(-1),
        reduction="mean",
    ).item()

    delta_lm_loss = loss_sae - loss_no_sae
    breakpoint()
    # --- Compute L0 sparsity ---
    f = sae.encode(cache[sae.cfg.hook_name + ".hook_sae_input"])  # SAE activations
    l0_sparsity = compute_l0(f)
    # --- Report ---
    print(f"Prompt: {text}")
    print(f"Cross-entropy loss (no SAE): {loss_no_sae:.5f}")
    print(f"Cross-entropy loss (with SAE): {loss_sae:.5f}")
    print(f"Δ LM Loss: {delta_lm_loss:.5f}")
    print(f"Average L0 (active features per token): {l0_sparsity:.2f}")


if __name__ == "__main__":
    main()
