#!/bin/bash

accelerate launch src/inference.py \
    --model_name_or_path ./checkpoints 
