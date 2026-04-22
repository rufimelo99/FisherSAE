#!/usr/bin/env python
"""Generate training configs for multiple Llama-2-13B layers."""

import json
from pathlib import Path

BASE_CONFIG = {
    "wandb_project": "DeltaSecommits-Llama2-13B-vulnerable-SAE",
    "model_name": "meta-llama/Llama-2-13b-chat-hf",
    "model_class_name": "HookedTransformer",
    "hook_head_index": None,
    "dataset_path": "TQRG/DeltaSecommits_llama-2-13b-chat_tokenized_v3_vulnerable",
    "dataset_trust_remote_code": False,
    "streaming": True,
    "is_dataset_tokenized": True,
    "use_cached_activations": False,
    "cached_activations_path": None,
    "from_pretrained_path": None,
    "context_size": 100,
    "n_batches_in_buffer": 512,
    "store_batch_size_prompts": 8,
    "seqpos_slice": [None, None],
    "disable_concat_sequences": True,
    "sequence_separator_token": "bos",
    "act_store_device": "cpu",
    "dtype": "float32",
    "prepend_bos": True,
    "autocast": False,
    "autocast_lm": False,
    "compile_llm": False,
    "llm_compilation_mode": None,
    "compile_sae": False,
    "sae_compilation_mode": None,
    "train_batch_size_tokens": 4096,
    "training_tokens": 50000000,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "lr": 3e-5,
    "lr_scheduler_name": "cosineannealing",
    "lr_decay_steps": 20000000,
    "lr_end": 3e-6,
    "n_restart_cycles": 0,
    "dead_feature_window": 1000,
    "feature_sampling_window": 1000,
    "dead_feature_threshold": 1e-4,
    "n_eval_batches": 10,
    "eval_batch_size_prompts": 8,
    "test_dataset_split": "test",
    "test_eval_every_n_steps": 100,
    "n_checkpoints": 0,
    "checkpoint_path": "checkpoints",
    "verbose": True,
    "model_kwargs": {},
    "model_from_pretrained_kwargs": {},
    "sae_lens_version": "latest",
    "sae_lens_training_version": "latest",
    "exclude_special_tokens": False,
    "sae": {
        "d_in": 5120,
        "d_sae": 16384,
        "dtype": "float32",
        "device": "cpu",
        "apply_b_dec_to_input": True,
        "normalize_activations": "none",
        "reshape_activations": "none",
        "type": "standard",
    },
}

# Llama-2-13B has 40 layers. Select evenly distributed layers.
LAYERS = [0, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39]

config_dir = Path("artifacts/DeltaSecommits_Llama2-13B_vulnerable")
config_dir.mkdir(parents=True, exist_ok=True)

print(f"Generating configs for layers: {LAYERS}")

for layer in LAYERS:
    cfg = BASE_CONFIG.copy()
    cfg["hook_name"] = f"blocks.{layer}.hook_resid_post"

    config_path = (
        config_dir / f"_training_config_layer{layer}_standard_16384_lr_1e-4.json"
    )

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"✓ Created {config_path.name}")

print(f"\n✓ Generated {len(LAYERS)} configs")
