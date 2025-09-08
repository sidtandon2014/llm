#!/bin/bash

accelerate launch src/inference.py \
    --model_name_or_path output/checkpoint-1000/
