# WIP
import os
import torch
from pathlib import Path
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, Accelerator
from transformers import (
    AutoConfig
    , WhisperForConditionalGeneration
    , AutoTokenizer
    , AutoProcessor
    , HfArgumentParser
)
from torch.utils.data import DataLoader
from datasets import load_from_disk
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from training_args import DataTrainingArguments, ModelArguments

@dataclass
class InfDataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    data_args: DataTrainingArguments

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        batch["input_features"] = torch.squeeze(batch["input_features"])
        
        if len(batch["input_features"].shape) == 2:
            batch["input_features"] = torch.unsqueeze(batch["input_features"], 0)
        
        batch["target"] = torch.stack([feature[self.data_args.text_column] for feature in features])
        return batch

def main():
    # 1. Initialize Accelerate with DeepSpeed
    accelerator = Accelerator()
    parser = HfArgumentParser((DataTrainingArguments, ModelArguments))
    data_args, model_args = parser.parse_args_into_dataclasses()
    # checkpoint_path = os.path.join(Path(__file__).resolve().parent, "/output/checkpoint-1000/",)
    checkpoint_path = os.path.join(Path(__file__).resolve().parent, model_args.model_name_or_path)
    if not os.path.isdir(checkpoint_path):
        raise Exception("model_name_or_path should be a checkpoint directory")

    # 1. Load the model config first
    config = AutoConfig.from_pretrained(checkpoint_path)

    model_path = os.path.join(checkpoint_path,"pytorch_model.bin")
    if not os.path.exists(model_path):
        raise Exception(("pytorch_model.bin not found inside checkpoint directory"),
                         ("Run ```./zero_to_fp32.py . pytorch_model.bin``` and then run this file")
                        )

    model = WhisperForConditionalGeneration.from_pretrained(model_path
                                                            ,config)

    processor = AutoProcessor.from_pretrained(checkpoint_path)
    data_collator = InfDataCollatorSpeechSeq2SeqWithPadding(processor, data_args)

    test_dataset = load_from_disk(data_args.processed_dataset_dir)["test"]
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

    data_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=data_collator)

    model,data_loader  = accelerator.prepare(model, data_loader)

    result = []
    for batch  in data_loader:
        input_features, target = batch
        print(input_features, target)
        print(input_features.shape, target.shape)
        print(input_features.device, target.device)
        # with torch.no_grad():
        #     output_ids = model.generate(**input_features, max_new_tokens=50)
        
        # if accelerator.is_main_process:
        #     result.extend((
        #         processor.tokenizer.decode(output_ids, skip_special_tokens=True)
        #         ,target
        #     ))
    # Now you can proceed with inference


if __name__ == "__main__":
    main()