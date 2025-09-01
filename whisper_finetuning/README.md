# Whisper Large Fine-Tuning 

This project fine-tune the `openai/whisper-large-v3` model on the Common Voice 11.0 dataset .

The codebase is built using the Hugging Face ecosystem (`transformers`, `datasets`, `accelerate`, `peft`) and is optimized for multi-GPU training on a single node.

## Project Structure

```
whisper_finetuning/
├── accelerate_config.yaml      # Configuration for multi-GPU training
├── requirements.txt            # Python dependencies
├── run_finetuning.sh           # Main script to launch the training
└── src/
    ├── data_preparation.py     # Handles dataset loading and preprocessing
    ├── main.py                 # Main training script orchestrating the process
    ├── model_utils.py          # Handles model loading, quantization, and LoRA setup
    └── training_args.py        # Defines custom arguments for model and data
```

## Setup

### 1. Install Dependencies

It is recommended to create a virtual environment first.

```bash
python -m venv .venv
source .venv/bin/activate
```

Then, install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Authenticate with Hugging Face

To download the Whisper model and the Common Voice dataset, you need to be logged into your Hugging Face account.

```bash
huggingface-cli login
```

You will also need to go to the dataset page for Common Voice 11.0 and agree to the terms to get access.

## How to Run

The fine-tuning process is launched via the `run_finetuning.sh` script. This script uses `accelerate` to handle the distributed training across your 8 GPUs.

To start training, simply run:

```bash
./run_finetuning.sh
```

### Customization

You can customize the training by modifying the arguments in the `run_finetuning.sh` script. Key parameters include:
- `--model_name_or_path`: The base Whisper model to use.
- `--dataset_name` & `--dataset_config_name`: The dataset and its configuration (e.g., language).
- `--output_dir`: Where to save the trained model checkpoints and logs.
- `--per_device_train_batch_size`: Batch size per GPU.
- `--learning_rate`: The learning rate for the optimizer.
- `--max_steps`: Total number of training steps.
- `--load_in_4bit`: Set to `True` to use 4-bit quantization, which is highly recommended for this model size.

The script is pre-configured with sensible defaults for fine-tuning on the English portion of Common Voice 11.0.
