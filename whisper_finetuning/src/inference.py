# WIP
import os
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, Accelerator
from torch.nn.parallel import DistributedDataParallel
import evaluate

from transformers import (
    AutoConfig
    , WhisperForConditionalGeneration
    , AutoTokenizer
    , AutoProcessor
    , HfArgumentParser
    , GenerationConfig
)
from torch.utils.data import DataLoader
from datasets import load_from_disk
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from data_classes import DataTrainingArguments, ModelArguments, InferenceArguments
from model_utils import load_model_and_processor

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
        
        # batch["input_features"] = batch["input_features"].to(torch.bfloat16)
        sentences = [feature[self.data_args.text_column] for feature in features]
        batch["target"] = sentences
        # batch["target"] = torch.stack([feature[self.data_args.text_column] for feature in features])
        return batch

def main():
    accelerator = Accelerator()
    parser = HfArgumentParser((DataTrainingArguments, ModelArguments, InferenceArguments))
    data_args, model_args, inference_args = parser.parse_args_into_dataclasses()
    # checkpoint_path = os.path.join(Path(__file__).resolve().parent, "/output/checkpoint-1000/",)


    model, processor = load_model_and_processor(model_args=model_args
                                                ,data_args=data_args
                                                ,inference_args)
    data_collator = InfDataCollatorSpeechSeq2SeqWithPadding(processor, data_args)

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

        return batch

    test_dataset = load_from_disk(data_args.processed_dataset_dir)["test"]
    with accelerator.main_process_first():
        test_dataset = test_dataset.map(
            prepare_sample,
            remove_columns=[data_args.audio_column_name],
            batched=True
        )
    test_dataset.set_format(type="torch",columns=["input_features","sentence"])

    data_loader = DataLoader(test_dataset
                             , batch_size=inference_args.batch_size
                             , shuffle=False
                             , collate_fn=data_collator)

    model,data_loader  = accelerator.prepare(model, data_loader)
    
    all_predictions = []
    all_references = []
    for batch in data_loader:
        input_features = batch["input_features"] #.to("cuda:0")
        # print(input_features.shape)
        with torch.no_grad():
            if isinstance(model, DistributedDataParallel):
                generated_ids  = model.module.generate(input_features
                                            ,max_new_tokens=50)
            else:
                generated_ids  = model.generate(input_features
                                            ,max_new_tokens=50)
            
        
        predictions = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # Gather predictions and references
        all_predictions.extend(predictions)
        all_references.extend(batch["target"])
       
    # 4. Gather results from all processes
    gathered_predictions = accelerator.gather_for_metrics(all_predictions)
    gathered_references = accelerator.gather_for_metrics(all_references)
    
    # Now you can calculate metrics on the main process
    if accelerator.is_main_process:
        print(gathered_predictions,gathered_references)
        wer_metric = evaluate.load("wer")
        wer = wer_metric.compute(predictions=gathered_predictions, references=gathered_references)
        print(f"WER: {wer}")
        
        results = pd.DataFrame(zip(gathered_predictions, gathered_references), columns=["predictions","references"])
        results.to_csv(os.path.join(checkpoint_path, "final_results.csv"), index=False)
        
#     print(all_predictions,all_references)
#     wer_metric = evaluate.load("wer")
#     wer = wer_metric.compute(predictions=all_predictions, references=all_references)
#     print(f"WER: {wer}")

#     results = pd.DataFrame(zip(all_predictions, all_references), columns=["predictions","references"])
#     results.to_csv(os.path.join(checkpoint_path, "final_results_single_gpu.csv", index=False))
    # Now you can proceed with inference


if __name__ == "__main__":
    main()