"""
Pre-compute activations for contrastive SAE training.

This script pre-computes and saves activations from paired datasets to disk,
dramatically speeding up training by avoiding repeated forward passes.

Usage:
    python code_sae/precompute_activations.py --config code_sae/precompute_config.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Literal

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer

from code_sae.logger import logger
from code_sae.utils import get_device, set_seed

SEED = set_seed()
DEVICE = get_device()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def precompute_activations(
    model_name: str,
    hook_name: str,
    dataset_path: str,
    output_path: str,
    input_ids_column: str = "input_ids",
    max_seq_len: int = 512,
    batch_size: int = 32,
    max_samples: int | None = None,
    dtype: str = "float32",
    save_format: Literal["memmap", "hdf5", "pt"] = "memmap",
    device: str = "cuda",
) -> None:
    """
    Pre-compute activations from a dataset and save to disk.

    Args:
        model_name: HuggingFace model name
        hook_name: Hook point to extract activations from
        dataset_path: Path to HuggingFace dataset
        output_path: Path to save activations
        input_ids_column: Column name for tokenized inputs
        max_seq_len: Maximum sequence length
        batch_size: Batch size for inference
        max_samples: Maximum number of samples to process (None = all)
        dtype: Data type for activations
        save_format: Format to save activations (memmap, hdf5, or pt)
        device: Device to run inference on
    """
    logger.info(f"Loading model: {model_name}")
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=getattr(torch, dtype),
    )
    model.eval()

    logger.info(f"Loading dataset: {dataset_path}")
    dataset = load_dataset(dataset_path, streaming=False, split="train")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    total_samples = len(dataset)
    logger.info(f"Processing {total_samples} samples")

    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    def pad_sequence(seq: list[int]) -> list[int]:
        """Pad or truncate sequence to max_seq_len."""
        if len(seq) > max_seq_len:
            return seq[:max_seq_len]
        return seq + [0] * (max_seq_len - len(seq))

    @torch.no_grad()
    def get_activations_batch(input_ids: torch.Tensor) -> torch.Tensor:
        """Extract activations for a batch."""
        _, cache = model.run_with_cache(
            input_ids,
            names_filter=[hook_name],
            stop_at_layer=int(hook_name.split(".")[1]) + 1,
        )
        acts = cache[hook_name]
        return acts.reshape(-1, acts.shape[-1])

    # Determine activation shape
    sample_ids = torch.tensor(
        [pad_sequence(dataset[0][input_ids_column])],
        dtype=torch.long,
        device=device,
    )
    sample_acts = get_activations_batch(sample_ids)
    d_model = sample_acts.shape[-1]
    acts_per_sample = sample_acts.shape[0]

    logger.info(f"Activation shape: {acts_per_sample} tokens x {d_model} features per sample")

    if save_format == "memmap":
        # Create memory-mapped array for efficient storage
        acts_shape = (total_samples * acts_per_sample, d_model)
        acts_file = Path(output_path)
        acts_memmap = torch.zeros(acts_shape, dtype=torch.float32)

        # Process in batches
        idx = 0
        for i in tqdm(range(0, total_samples, batch_size), desc="Extracting activations"):
            batch_end = min(i + batch_size, total_samples)
            batch = dataset[i:batch_end]

            # Pad sequences
            padded = [pad_sequence(sample[input_ids_column]) for sample in batch]
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)

            # Get activations
            acts = get_activations_batch(input_ids).cpu()

            # Store in memmap
            batch_size_actual = acts.shape[0]
            acts_memmap[idx : idx + batch_size_actual] = acts
            idx += batch_size_actual

        # Save memmap to disk
        torch.save(acts_memmap, acts_file)
        logger.info(f"Saved activations to {acts_file}")

        # Save metadata
        metadata = {
            "shape": list(acts_shape),
            "d_model": d_model,
            "acts_per_sample": acts_per_sample,
            "total_samples": total_samples,
            "max_seq_len": max_seq_len,
            "model_name": model_name,
            "hook_name": hook_name,
            "dtype": dtype,
        }
        metadata_file = acts_file.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")

    elif save_format == "pt":
        # Save as single PyTorch tensor
        all_acts = []

        for i in tqdm(range(0, total_samples, batch_size), desc="Extracting activations"):
            batch_end = min(i + batch_size, total_samples)
            batch = dataset[i:batch_end]

            padded = [pad_sequence(sample[input_ids_column]) for sample in batch]
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)

            acts = get_activations_batch(input_ids).cpu()
            all_acts.append(acts)

        all_acts = torch.cat(all_acts, dim=0)
        torch.save(all_acts, output_path)
        logger.info(f"Saved {all_acts.shape[0]} activations to {output_path}")

    else:
        raise ValueError(f"Unsupported save format: {save_format}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute activations for contrastive SAE")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # Pre-compute positive activations
    logger.info("=" * 60)
    logger.info("Pre-computing POSITIVE activations")
    logger.info("=" * 60)
    precompute_activations(
        model_name=config["model_name"],
        hook_name=config["hook_name"],
        dataset_path=config["positive_dataset_path"],
        output_path=config["positive_activations_path"],
        input_ids_column=config.get("input_ids_column", "input_ids"),
        max_seq_len=config.get("max_seq_len", 512),
        batch_size=config.get("precompute_batch_size", 32),
        max_samples=config.get("max_samples"),
        dtype=config.get("dtype", "float32"),
        save_format=config.get("save_format", "pt"),
        device=DEVICE,
    )

    # Pre-compute negative activations
    logger.info("=" * 60)
    logger.info("Pre-computing NEGATIVE activations")
    logger.info("=" * 60)
    precompute_activations(
        model_name=config["model_name"],
        hook_name=config["hook_name"],
        dataset_path=config["negative_dataset_path"],
        output_path=config["negative_activations_path"],
        input_ids_column=config.get("input_ids_column", "input_ids"),
        max_seq_len=config.get("max_seq_len", 512),
        batch_size=config.get("precompute_batch_size", 32),
        max_samples=config.get("max_samples"),
        dtype=config.get("dtype", "float32"),
        save_format=config.get("save_format", "pt"),
        device=DEVICE,
    )

    logger.info("=" * 60)
    logger.info("Pre-computation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
