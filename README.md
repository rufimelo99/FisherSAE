# CodeSAE

## Installation

```bash
conda create -n code_sae python=3.10
conda activate code_sae
cd CodeSAE
git checkout new_loss
pip install -e .
git submodule update --init --recursive
cd SAELens
git checkout new_loss
pip install -e .
```

python code_sae/training.py \
    --config scripts/devign/layer_0_residual_mid_gpt2_gated_fisher.json
