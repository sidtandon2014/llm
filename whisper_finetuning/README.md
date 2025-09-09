# Whisper Large Fine-Tuning 

This project fine-tune the `openai/whisper-large-v3` model on the Common Voice 11.0 dataset .

The codebase is built using the Hugging Face ecosystem (`transformers`, `datasets`, `accelerate`, `peft`) and is optimized for multi-GPU training on a single node with 8 L4 GPUs.

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
## Dataset

Using ```mozilla-foundation/common_voice_11_0``` dataset with following stats:
1. audio: Resampled to 16k
2. sentence: Below is length stats on training dataset. Set the padding and truncation strategy accordingly durign training and evaluation 
```
count    948736.000000
mean         17.034940
std           4.168177
min           4.000000
10%          12.000000
25%          14.000000
50%          17.000000
75%          20.000000
90%          22.000000
99%          28.000000
max          54.000000

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

You can also create a .env file with token (That's what I am doing in codebase)

You will also need to go to the dataset page for Common Voice 11.0 and agree to the terms to get access.

## How to Run

The fine-tuning process is launched via the `run_finetuning.sh` script. This script uses `accelerate` to handle the distributed training across your NVIDIA 8xL4 GPUs.

### GPU configuration

```
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      PHB     PHB     PHB     SYS     SYS     SYS     SYS     0-23,48-71      0               N/A
GPU1    PHB      X      PHB     PHB     SYS     SYS     SYS     SYS     0-23,48-71      0               N/A
GPU2    PHB     PHB      X      PHB     SYS     SYS     SYS     SYS     0-23,48-71      0               N/A
GPU3    PHB     PHB     PHB      X      SYS     SYS     SYS     SYS     0-23,48-71      0               N/A
GPU4    SYS     SYS     SYS     SYS      X      PHB     PHB     PHB     24-47,72-95     1               N/A
GPU5    SYS     SYS     SYS     SYS     PHB      X      PHB     PHB     24-47,72-95     1               N/A
GPU6    SYS     SYS     SYS     SYS     PHB     PHB      X      PHB     24-47,72-95     1               N/A
GPU7    SYS     SYS     SYS     SYS     PHB     PHB     PHB      X      24-47,72-95     1               N/A

```
- There are two set of machines 
- For intra node connectivity there is PHB protocol whereas for inter node there is SYS
- As these are L4 machine NVLink is missing between machines 

### Bash commands
```bash
mkdir output
./run_finetuning.sh
```

## How to run Inference
The inference process is launched via the `run_inference.sh` script. 
- Make sure you have a fine-tuned model checkpoint saved in the `./checkpoints` directory.
- Place the audio file you want to transcribe in the root directory of the project and name it `test.wav`.
- You can change the path to the audio file and the model checkpoint in the `run_inference.sh` script.

```bash
./run_inference.sh
```

### Customization

You can customize the training by modifying the arguments in the `src/training_args.json` and `src/training_args.py`. Key parameters include:
- `--model_name_or_path`: The base Whisper model to use.
- `--dataset_name` & `--dataset_config_name`: The dataset and its configuration (e.g., language).
- `--output_dir`: Where to save the trained model checkpoints and logs.
- `--per_device_train_batch_size`: Batch size per GPU.
- `--learning_rate`: The learning rate for the optimizer.
- `--max_steps`: Total number of training steps.
- `--load_in_4bit`: Set to `True` to use 4-bit quantization, which is highly recommended for this model size.

The script is pre-configured with sensible defaults for fine-tuning on the English portion of Common Voice 11.0.


### Pending Features
- No shuffling on iterated dataset
- Profiling investigation
- Native Pytorch adaptability

## Metrics

### Training metrics


### Business Metrics