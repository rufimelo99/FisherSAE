"""
Script to push SAE(s) to Hugging Face Hub for use with SAELens.

Usage (single SAE):
    python code_sae/store_sae_hf.py --repo_id rufimelo/secure_code_qwen_coder_block_0_hook_resid_post_16384 --sae_dir artifacts/sae_DeltaSecommits_qwen_coder_blocks_0_hook_resid_post

Usage (multiple SAEs):
    python code_sae/store_sae_hf.py --repo_id rufimelo/secure_code_qwen_coder_strd_16384 --sae_dir ~/Downloads/sae_blocks.0.hook_resid_post/ ~/Downloads/sae_blocks.14.hook_resid_post/ ~/Downloads/sae_blocks.27.hook_resid_post/

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


def create_readme(sae_dirs: list[Path], repo_id: str) -> str:
    """Generate a README.md for the SAE repository with multiple SAEs."""
    # Load first config for common info
    first_cfg_path = sae_dirs[0] / "cfg.json"
    with open(first_cfg_path, "r") as f:
        first_cfg = json.load(f)

    first_metadata = first_cfg.get("metadata", {})
    model_name = first_metadata.get("model_name", "Unknown")
    architecture = first_cfg.get("architecture", "standard")
    dataset_path = first_metadata.get("dataset_path", "Unknown")
    d_in = first_cfg.get("d_in", "Unknown")
    d_sae = first_cfg.get("d_sae", "Unknown")
    sae_lens_version = first_metadata.get("sae_lens_training_version", "Unknown")

    # Collect all hook names
    hook_names = []
    for sae_dir in sae_dirs:
        cfg_path = sae_dir / "cfg.json"
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        hook_name = cfg.get("metadata", {}).get("hook_name", "sae")
        hook_names.append(hook_name)

    # Generate hook points table
    hook_points_list = "\n".join([f"| `{hook}` |" for hook in hook_names])

    # Generate files list
    files_list = "\n".join(
        [
            f"- `{hook}/cfg.json` - SAE configuration\n- `{hook}/sae_weights.safetensors` - Model weights\n- `{hook}/sparsity.safetensors` - Feature sparsity statistics"
            for hook in hook_names
        ]
    )

    # Generate usage example with first hook
    first_hook = hook_names[0]

    readme = f"""---
library_name: sae_lens
tags:
  - sparse-autoencoder
  - mechanistic-interpretability
  - sae
---

# Sparse Autoencoders for {model_name}

This repository contains {len(sae_dirs)} Sparse Autoencoder(s) (SAE) trained using [SAELens](https://github.com/jbloomAus/SAELens).

## Model Details

| Property | Value |
|----------|-------|
| **Base Model** | `{model_name}` |
| **Architecture** | `{architecture}` |
| **Input Dimension** | {d_in} |
| **SAE Dimension** | {d_sae} |
| **Training Dataset** | `{dataset_path}` |

## Available Hook Points

| Hook Point |
|------------|
{hook_points_list}

## Usage

```python
from sae_lens import SAE

# Load an SAE for a specific hook point
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="{repo_id}",
    sae_id="{first_hook}"  # Choose from available hook points above
)

# Use with TransformerLens
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("{model_name}")

# Get activations and encode
_, cache = model.run_with_cache("your text here")
activations = cache["{first_hook}"]
features = sae.encode(activations)
```

## Files

{files_list}

## Training

These SAEs were trained with SAELens version {sae_lens_version}.
"""
    return readme


def push_sae_to_hub(
    sae_dirs: list[str],
    repo_id: str,
    private: bool = False,
    token: str | None = None,
):
    """
    Push one or more SAEs to the Hugging Face Hub.

    Args:
        sae_dirs: List of paths to local SAE directories containing cfg.json, sae_weights.safetensors, etc.
        repo_id: Hugging Face repo ID (e.g., "username/sae-model-name")
        private: Whether to make the repository private
        token: Hugging Face token (uses cached token if not provided)
    """
    sae_paths = [Path(d) for d in sae_dirs]

    # Validate all directories
    for sae_path in sae_paths:
        if not sae_path.exists():
            raise FileNotFoundError(f"SAE directory not found: {sae_path}")

        required_files = ["cfg.json", "sae_weights.safetensors"]
        for fname in required_files:
            if not (sae_path / fname).exists():
                raise FileNotFoundError(f"Required file not found: {sae_path / fname}")

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
        hook_names = []

        for sae_path in sae_paths:
            # Load config to get hook name for subdirectory
            with open(sae_path / "cfg.json", "r") as f:
                cfg = json.load(f)

            hook_name = cfg.get("metadata", {}).get("hook_name", "sae")
            hook_names.append(hook_name)

            sae_subdir = tmp_path / hook_name
            sae_subdir.mkdir(parents=True, exist_ok=True)

            # Copy SAE files to subdirectory
            files_to_upload = [
                "cfg.json",
                "sae_weights.safetensors",
                "sparsity.safetensors",
            ]
            for fname in files_to_upload:
                src = sae_path / fname
                if src.exists():
                    shutil.copy(src, sae_subdir / fname)
                    print(f"  Added: {hook_name}/{fname}")

        # Create README
        readme_content = create_readme(sae_paths, repo_id)
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

    print(
        f"\n✓ Successfully uploaded {len(sae_dirs)} SAE(s) to: https://huggingface.co/{repo_id}"
    )
    print(f"\nAvailable sae_ids: {', '.join(hook_names)}")
    print(f"\nTo load with SAELens:")
    print(
        f'  sae, cfg_dict, sparsity = SAE.from_pretrained("{repo_id}", sae_id="{hook_names[0]}")'
    )


def main():
    parser = argparse.ArgumentParser(
        description="Push SAE(s) to Hugging Face Hub for use with SAELens"
    )
    parser.add_argument(
        "--sae_dir",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to the SAE directory(ies). Can specify multiple directories.",
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

    # Handle relative paths and expand ~ (resolve from current working directory)
    sae_dirs = []
    for sae_dir in args.sae_dir:
        path = Path(sae_dir).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        sae_dirs.append(str(path))

    push_sae_to_hub(
        sae_dirs=sae_dirs,
        repo_id=args.repo_id,
        private=args.private,
        token=args.token,
    )


if __name__ == "__main__":
    main()
