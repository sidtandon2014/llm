#!/bin/bash


# Use Zero 3 config setting as used during training (WIP)
# accelerate launch --config_file deep_speed_config.yaml src/inference_distributed.py \
#     --model_name_or_path output/checkpoint-1000/

# Use DDP setting
# Use 4 bit 
# accelerate launch \
#     --multi_gpu src/inference.py \
#     --model_name_or_path output/checkpoint-1000/ \
#     --is_model_id False \
#     --is_train False \
#     --quantization_algo 'bnb' \
#     --inference_result_file_name 'results_4bit.csv' \
#     --inf_bnb_load_in_8bit False \
#     --inf_bnb_load_in_4bit True > output_4bit_inf.log 2>&1

# Use 8 bit
# accelerate launch \
#     --multi_gpu src/inference.py \
#     --model_name_or_path output/checkpoint-1000/ \
#     --is_model_id False \
#     --is_train False \
#     --quantization_algo 'bnb' \
#     --inference_result_file_name 'results_8bit.csv' \
#     --inf_bnb_load_in_8bit True \
#     --inf_bnb_load_in_4bit False > output_8bit_inf.log 2>&1

# No quantization
accelerate launch \
    --multi_gpu src/inference.py \
    --model_name_or_path output/checkpoint-1000/ \
    --is_model_id False \
    --is_train False \
    --quantization_algo None \
    --inference_result_file_name 'results.csv' \
    --inf_bnb_load_in_8bit False \
    --inf_bnb_load_in_4bit False > output_inf.log 2>&1