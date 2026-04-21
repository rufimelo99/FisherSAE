#!/usr/bin/env python
"""Run SAELens pretokenization with a config file."""

import json
import sys

from sae_lens.config import PretokenizeRunnerConfig
from sae_lens.pretokenize_runner import PretokenizeRunner

if len(sys.argv) != 2:
    print("Usage: python run_pretokenize.py <config_file>")
    sys.exit(1)

config_file = sys.argv[1]
print(f"Loading config from {config_file}")

with open(config_file, "r") as f:
    config_dict = json.load(f)

print(f"Config: {json.dumps(config_dict, indent=2)}")

cfg = PretokenizeRunnerConfig(**config_dict)
print(f"\nStarting pretokenization with context_size={cfg.context_size}...")

runner = PretokenizeRunner(cfg)
result = runner.run()

print(f"\n✓ Pretokenization complete!")
print(f"Dataset size: {len(result)}")
