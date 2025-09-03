import logging
# from accelerate.logging import get_logger


import sys
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import evaluate
import torch
from datasets import Dataset
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperProcessor,
)
from transformers.trainer_utils import get_last_checkpoint
import torch.distributed as dist
from accelerate import Accelerator

from training_args import DataTrainingArguments, ModelArguments
from model_utils import load_model_and_processor
from data_preparation import prepare_dataset
from profiling import ProfilerCallback
from utils import display_ram_usage

logger = logging.getLogger(__name__)
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000
# accelerator = Accelerator()

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
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
        
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        
        labels = torch.stack([feature["labels"] for feature in features])
        if len(labels.shape) == 3:
            labels = torch.squeeze(labels)
        
        if len(labels.shape) == 1:
            labels = torch.unsqueeze(labels, 0)
        
        logger.debug(f"Labels Shape in collator: {labels.shape}")
        # att_mask = torch.stack([feature["attention_mask"] for feature in features])
        # att_mask = torch.squeeze(att_mask)
        
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        # batch["attention_mask"] = att_mask
        
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        
        # if rank ==0:
        logger.debug(f"Rank: {rank} Shape of input_features: {batch['input_features'].shape}")
        logger.debug(f"Rank: {rank} shape of labels: {batch['labels'].shape}")
        # logger.debug(f"Rank: {rank} shape of atention_mask: {batch['attention_mask'].shape}")
         
        return batch

def main():
    # 1. Parse arguments
    # The parser now accepts all three argument classes to parse from the JSON file.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    
    # 2. Setup logging
    log_file_path = f"{__name__}.log"
    # sys.stdout = open(log_file_path, 'a') # 'a' for append mode
    # sys.stderr = open(log_file_path, 'a')
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)
                 ,logging.FileHandler(log_file_path)
                 ],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    # 3. Load model and processor
    # with accelerator.main_process_first():
    processor, model = load_model_and_processor(model_args, data_args)

    # 4. Set language and task for generation
    # model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=data_args.language, task=data_args.task)
    # model.config.suppress_tokens = []

    # 5. Prepare dataset
    train_dataset,eval_dataset  = prepare_dataset(data_args, processor)
    # train_dataset = vectorized_datasets["train"]
    # eval_dataset = vectorized_datasets["test"]

    # 6. Define data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, data_args=data_args)

    # 7. Define evaluation metric
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
            
        pred_ids = pred.predictions
        label_ids = pred.label_ids.copy()
        
        logger.debug(f"Rank: {rank} pred_ids: {pred_ids.shape}")
        logger.debug(f"Rank: {rank} pred_ids_device: {pred_ids.device}")
        logger.debug(f"Rank: {rank} pred_ids_type: {type(pred_ids)}")
        logger.debug(f"Rank: {rank} label_ids: {label_ids.shape}")
        logger.debug(f"Rank: {rank} label_ids_device: {label_ids.device}")
        logger.debug(f"Rank: {rank} label_ids_type: {type(label_ids)}")
        logger.debug(f"Rank: {rank} pred_ids: {pred_ids}")
        logger.debug(f"Rank: {rank} label_ids: {label_ids}")
        
        if rank == 0:
            display_ram_usage()
        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        logger.debug("Label processed")
        # we do not want to group tokens when decoding
        pred_str = processor.batch_decode(pred_ids[:,:training_args.generation_max_length], skip_special_tokens=True, group_tokens=False)
        label_str = processor.batch_decode(label_ids[:,:training_args.generation_max_length], skip_special_tokens=True, group_tokens=False)

        logger.debug("Batch decoded")
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        logger.debug(f"Rank: {rank}, wer: {wer}")
        return {"wer": wer}

    # 8. Instantiate Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor,
        # callbacks=[ProfilerCallback(output_dir=training_args.output_dir)]
    )

    # 9. Start training
    if training_args.do_train:
        trainer.accelerator.print(f"{trainer.model}")
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        
        # For Memory profiling
        # torch.cuda.memory._record_memory_history(max_entries=100000)
        
        # Train model
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        
        # Dump memory profiling snapshot
        # torch.cuda.memory._dump_snapshot("profile_train.pkl")
        # Stop memroy profiling
        # torch.cuda.memory._record_memory_history(enabled=None)
        
        trainer.save_model()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 10. Evaluate
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
