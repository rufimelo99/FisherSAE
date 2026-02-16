 python code_sae/precompute_activations.py --config artifacts/DeltaSeccommits_QwenCoder_CL/_precompute_layer0.json

 python code_sae/precompute_activations.py --config artifacts/DeltaSeccommits_QwenCoder_CL/_precompute_layer14.json

 python code_sae/precompute_activations.py --config artifacts/DeltaSeccommits_QwenCoder_CL/_precompute_layer27.json


python code_sae/contrastive_training_fast.py --config artifacts/DeltaSeccommits_QwenCoder_CL/_training_config_layer0_topk_cl_16384_lr_1e-4_fast.json 

python code_sae/contrastive_training_fast.py --config artifacts/DeltaSeccommits_QwenCoder_CL/_training_config_layer14_topk_cl_16384_lr_1e-4_fast.json 

python code_sae/contrastive_training_fast.py --config artifacts/DeltaSeccommits_QwenCoder_CL/_training_config_layer27_topk_cl_16384_lr_1e-4_fast.json 

