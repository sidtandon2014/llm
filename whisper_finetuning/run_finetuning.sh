#!/bin/bash

# This script launches the fine-tuning process for the Whisper model.
# It uses accelerate for distributed training.

# Ensure you have logged in to Hugging Face CLI if you need to access gated models/datasets
# huggingface-cli login

export WANDB_DISABLED=true

accelerate launch --config_file accelerate_config.yaml src/main.py src/training_args.json
