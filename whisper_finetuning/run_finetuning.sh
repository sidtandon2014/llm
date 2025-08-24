#!/bin/bash

# This script launches the fine-tuning process for the Whisper model.
# It uses accelerate for distributed training.

# Ensure you have logged in to Hugging Face CLI if you need to access gated models/datasets
# huggingface-cli login

export WANDB_DISABLED=true

accelerate launch --config_file accelerate_config.yaml src/main.py \
    --model_name_or_path="openai/whisper-large-v3" \
    --dataset_name="mozilla-foundation/common_voice_11_0" \
    --dataset_config_name="en" \
    --language="en" \
    --task="transcribe" \
    --output_dir="./whisper-large-v3-cv11-en-lora" \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --preprocessing_num_workers=16 \
    --learning_rate=1e-5 \
    --warmup_steps=500 \
    --max_steps=5000 \
    --gradient_checkpointing=True \
    --evaluation_strategy="steps" \
    --eval_steps=1000 \
    --save_strategy="steps" \
    --save_steps=1000 \
    --logging_steps=25 \
    --report_to="tensorboard" \
    --load_best_model_at_end=True \
    --metric_for_best_model="wer" \
    --greater_is_better=False \
    --push_to_hub=False \
    --load_in_4bit=True \
    --do_train \
    --do_eval
