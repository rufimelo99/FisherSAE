"""Direct SAE upload to HF without temp directory."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def upload_sae_direct(sae_dir: str, repo_id: str, token: str | None = None):
    """Upload SAE directly from source without copying to temp."""
    sae_path = Path(sae_dir).expanduser()

    if not sae_path.exists():
        raise FileNotFoundError(f"SAE directory not found: {sae_path}")

    # Load config to get hook name
    cfg_path = sae_path / "cfg.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    hook_name = cfg.get("metadata", {}).get("hook_name", "sae")

    api = HfApi(token=token)

    # Create repo
    print(f"Creating repository: {repo_id}")
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        token=token,
    )

    # Upload files directly from source
    files_to_upload = [
        ("cfg.json", "SAE configuration"),
        ("sae_weights.safetensors", "Model weights"),
        ("sparsity.safetensors", "Feature sparsity"),
    ]

    print(f"\nUploading SAE from {sae_path}...")
    for fname, desc in files_to_upload:
        src = sae_path / fname
        if src.exists():
            file_path = f"{hook_name}/{fname}"
            print(f"  Uploading {file_path} ({desc})...")
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=file_path,
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )

    print(f"\n✓ Successfully uploaded SAE to: https://huggingface.co/{repo_id}")
    print(f"\nLoad with SAELens:")
    print(
        f'  sae, cfg_dict, sparsity = SAE.from_pretrained("{repo_id}", sae_id="{hook_name}")'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload SAE directly to HF")
    parser.add_argument(
        "--sae_dir", type=str, required=True, help="Path to SAE directory"
    )
    parser.add_argument("--repo_id", type=str, required=True, help="HF repo ID")
    parser.add_argument("--token", type=str, default=None, help="HF token")

    args = parser.parse_args()
    upload_sae_direct(args.sae_dir, args.repo_id, args.token)
