# WIP

import torch
from pathlib import Path
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, Accelerator
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

# Note: We do NOT initialize Accelerator here first.

def main():
    # 1. Initialize Accelerate with DeepSpeed
    accelerator = Accelerator()
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    checkpoint_path = os.path.join(Path(__file__).resolve().parent, "/output/checkpoint-1000/",)

    # 1. Load the model config first
    config = AutoConfig.from_pretrained(checkpoint_path)

    model_path = os.path.join(checkpoint_path,"pytorch_model.bin")
    if not os.path.exists(model_path):
        raise Exception(("pytorch_model.bin not found inside checkpoint directory),
                         ("Run ```./zero_to_fp32.py . pytorch_model.bin``` and then run this file")
                        ))

    model = WhisperForConditionalGeneration.from_pretrained(model_path
                                                            ,config)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    processed_datasets = load_from_disk(data_args.processed_dataset_dir)
    # test_dataset= processed_datasets["test"].to_iterable_dataset() #.take(100)

    # Preprocessing function
    def prepare_sample(batch):
        batch_audio = batch[data_args.audio_column_name]

        if isinstance(batch_audio, list):
            audio_array = [x["array"] for x in batch_audio]
        else:
            audio_array = batch_audio["array"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = processor.feature_extractor(
            audio_array, sampling_rate=data_args.sampling_rate
        ).input_features

        return batch["input_features"]


    test_dataset = test_dataset.map(
        prepare_sample,
        remove_columns=[data_args.audio_column_name]
    ).set_format("torch")





    # Now you can proceed with inference

    result = []
    for sample in test_dataset:
        print(example)
        sample = sample.to("cuda")

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=50)
            result.append(tokenizer.decode(output_ids, skip_special_tokens=True))




    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(generated_text)
    
if __name__ == "__main__":
    main()