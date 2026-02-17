#!/bin/bash
python code_sae/training.py --config artifacts/DeltaSeccommits_QwenCoder_v2/_training_config_layer0_standard_16384_lr_1e-4.json 
python code_sae/training.py --config artifacts/DeltaSeccommits_QwenCoder_v2/_training_config_layer3_standard_16384_lr_1e-4.json 
python code_sae/training.py --config artifacts/DeltaSeccommits_QwenCoder_v2/_training_config_layer7_standard_16384_lr_1e-4.json 
python code_sae/training.py --config artifacts/DeltaSeccommits_QwenCoder_v2/_training_config_layer11_standard_16384_lr_1e-4.json 
python code_sae/training.py --config artifacts/DeltaSeccommits_QwenCoder_v2/_training_config_layer15_standard_16384_lr_1e-4.json 
python code_sae/training.py --config artifacts/DeltaSeccommits_QwenCoder_v2/_training_config_layer19_standard_16384_lr_1e-4.json 
python code_sae/training.py --config artifacts/DeltaSeccommits_QwenCoder_v2/_training_config_layer23_standard_16384_lr_1e-4.json 
python code_sae/training.py --config artifacts/DeltaSeccommits_QwenCoder_v2/_training_config_layer27_standard_16384_lr_1e-4.json