#!/bin/bash


# Use Zero 3 config setting as used during training (WIP)
# accelerate launch --config_file deep_speed_config.yaml src/inference_distributed.py \
#     --model_name_or_path output/checkpoint-1000/

# Use DDP setting
accelerate launch \
    --multi_gpu src/inference.py \
    --model_name_or_path output/checkpoint-1000/ \
    --is_model_id False \
    --is_train False \
    --quantization_algo 'bnb' \
    --inf_bnb_load_in_8bit True \
    --inf_bnb_load_in_4bit False \
    > output_8bit_inf.log 2>&1