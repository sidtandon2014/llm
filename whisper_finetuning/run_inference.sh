#!/bin/bash

# python src/inference.py --model_name_or_path output/checkpoint-1000/

# Use Zero 3 config setting as used during training
# accelerate launch --config_file deep_speed_config.yaml src/inference_distributed.py \
#     --model_name_or_path output/checkpoint-1000/

# Use DDP setting
accelerate launch --multi_gpu src/inference.py --model_name_or_path output/checkpoint-1000/ > output_inf.log 2>&1