"""
Script to push SAE to Hugging Face Hub for use with SAELens.

Usage:
    python code_sae/store_sae_hf.py --repo_id rufimelo/secure_code_qwen_coder_block_0_hook_resid_post_16384 --sae_dir artifacts/sae_DeltaSecommits_qwen_coder_blocks_0_hook_resid_post

After upload, load with SAELens:
    from sae_lens import SAE
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="YOUR_USERNAME/YOUR_REPO_NAME",
        sae_id="blocks.0.hook_resid_post"
    )
"""

import argparse
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def create_readme(sae_dir: Path, repo_id: str) -> str:
    """Generate a README.md for the SAE repository."""
    cfg_path = sae_dir / "cfg.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    metadata = cfg.get("metadata", {})
    model_name = metadata.get("model_name", "Unknown")
    hook_name = metadata.get("hook_name", "Unknown")
    d_in = cfg.get("d_in", "Unknown")
    d_sae = cfg.get("d_sae", "Unknown")
    architecture = cfg.get("architecture", "standard")
    dataset_path = metadata.get("dataset_path", "Unknown")

    readme = f"""---
library_name: sae_lens
tags:
  - sparse-autoencoder
  - mechanistic-interpretability
  - sae
---

# Sparse Autoencoder for {model_name}

This is a Sparse Autoencoder (SAE) trained using [SAELens](https://github.com/jbloomAus/SAELens).

## Model Details

| Property | Value |
|----------|-------|
| **Base Model** | `{model_name}` |
| **Hook Point** | `{hook_name}` |
| **Architecture** | `{architecture}` |
| **Input Dimension** | {d_in} |
| **SAE Dimension** | {d_sae} |
| **Training Dataset** | `{dataset_path}` |

## Usage

```python
from sae_lens import SAE

# Load the SAE
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="{repo_id}",
    sae_id="{hook_name}"
)

# Use with TransformerLens
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("{model_name}")

# Get activations and encode
_, cache = model.run_with_cache("your text here")
activations = cache["{hook_name}"]
features = sae.encode(activations)
```

## Files

- `{hook_name}/cfg.json` - SAE configuration
- `{hook_name}/sae_weights.safetensors` - Model weights
- `{hook_name}/sparsity.safetensors` - Feature sparsity statistics

## Training

This SAE was trained with SAELens version {metadata.get("sae_lens_training_version", "Unknown")}.
"""
    return readme


def push_sae_to_hub(
    sae_dir: str,
    repo_id: str,
    private: bool = False,
    token: str | None = None,
):
    """
    Push an SAE to the Hugging Face Hub.

    Args:
        sae_dir: Path to the local SAE directory containing cfg.json, sae_weights.safetensors, etc.
        repo_id: Hugging Face repo ID (e.g., "username/sae-model-name")
        private: Whether to make the repository private
        token: Hugging Face token (uses cached token if not provided)
    """
    sae_path = Path(sae_dir)

    if not sae_path.exists():
        raise FileNotFoundError(f"SAE directory not found: {sae_path}")

    required_files = ["cfg.json", "sae_weights.safetensors"]
    for fname in required_files:
        if not (sae_path / fname).exists():
            raise FileNotFoundError(f"Required file not found: {sae_path / fname}")

    # Load config to get hook name for subdirectory
    with open(sae_path / "cfg.json", "r") as f:
        cfg = json.load(f)

    hook_name = cfg.get("metadata", {}).get("hook_name", "sae")

    api = HfApi(token=token)

    # Create the repository
    print(f"Creating repository: {repo_id}")
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
        token=token,
    )

    # Create a temporary directory with the proper structure
    # SAELens expects: repo_id/hook_name/files
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sae_subdir = tmp_path / hook_name
        sae_subdir.mkdir(parents=True)

        # Copy SAE files to subdirectory
        files_to_upload = ["cfg.json", "sae_weights.safetensors", "sparsity.safetensors"]
        for fname in files_to_upload:
            src = sae_path / fname
            if src.exists():
                shutil.copy(src, sae_subdir / fname)
                print(f"  Added: {hook_name}/{fname}")

        # Create README
        readme_content = create_readme(sae_path, repo_id)
        readme_path = tmp_path / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)
        print("  Added: README.md")

        # Upload the entire folder
        print(f"\nUploading to {repo_id}...")
        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )

    print(f"\n✓ Successfully uploaded SAE to: https://huggingface.co/{repo_id}")
    print(f"\nTo load with SAELens:")
    print(f'  sae, cfg_dict, sparsity = SAE.from_pretrained("{repo_id}", sae_id="{hook_name}")')


def main():
    parser = argparse.ArgumentParser(
        description="Push SAE to Hugging Face Hub for use with SAELens"
    )
    parser.add_argument(
        "--sae_dir",
        type=str,
        default="sae_DeltaSecommits_qwen_coder_blocks.0.hook_resid_post",
        help="Path to the SAE directory",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face repo ID (e.g., 'username/sae-qwen-layer0')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (uses cached token if not provided)",
    )

    args = parser.parse_args()

    # Handle relative path (resolve from current working directory)
    sae_dir = Path(args.sae_dir)
    if not sae_dir.is_absolute():
        sae_dir = Path.cwd() / sae_dir

    push_sae_to_hub(
        sae_dir=str(sae_dir),
        repo_id=args.repo_id,
        private=args.private,
        token=args.token,
    )


if __name__ == "__main__":
    main()
