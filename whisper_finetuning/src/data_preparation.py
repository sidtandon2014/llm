from datasets import load_dataset, DatasetDict, Audio, load_from_disk
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
load_dotenv()

def load_raw_dataset(data_args):
    """
    Loads the dataset for Whisper fine-tuning.
    """
    auth_token = os.getenv("token")
    processed_dataset_path = data_args.processed_dataset_dir
    if not os.path.exists(processed_dataset_path):
        raw_datasets = DatasetDict()
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split="train",
            cache_dir=data_args.dataset_cache_dir,
            token=auth_token,
            trust_remote_code=True
        )
        raw_datasets["test"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split="test",
            cache_dir=data_args.dataset_cache_dir,
            token=auth_token,
            trust_remote_code=True
        )

        def is_audio_in_length_range(sample):
            length = sample["array"].shape[0] / 16000
            return length < data_args.max_duration_in_seconds 

        raw_datasets = raw_datasets.remove_columns(
            ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]
        )

        # Resample audio to 16kHz
        raw_datasets = raw_datasets.cast_column(data_args.audio_column_name, Audio(sampling_rate=16000))

        # Calculate length 
        # There are only 3 samples that will be removed. Its be4tter to process it
        # But in case there is a dataset that has lot of such variation
        # Uncomment below lines
        # raw_datasets = raw_datasets.filter(
        #     is_audio_in_length_range,
        #     input_columns=[data_args.audio_column_name],
        #     num_proc=4
        #     )
        
        raw_datasets.save_to_disk(processed_dataset_path)
        return None

def prepare_dataset(data_args, processor):
    """
    Loads and preprocesses the dataset for Whisper fine-tuning.
    """
    # 1. Load Dataset
    processed_datasets = load_from_disk(data_args.processed_dataset_dir)
    train_dataset = processed_datasets["train"].to_iterable_dataset(num_shards=87) #.take(100)
    test_dataset= processed_datasets["test"].to_iterable_dataset(num_shards=8) #.take(100)

    # Preprocessing function
    def prepare_sample(batch, mode):
        batch_audio = batch[data_args.audio_column_name]

        if isinstance(batch_audio, list):
            audio_array = [x["array"] for x in batch_audio]
        else:
            audio_array = batch_audio["array"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = processor.feature_extractor(
            audio_array, sampling_rate=data_args.sampling_rate
        ).input_features

        # encode target text to label ids 
        if mode=="train":
            tokenized_text = processor.tokenizer(batch[data_args.text_column]
                                                ,padding='max_length'
                                                ,max_length=data_args.train_max_tokens_per_sentence
                                                ,truncation=True
                                                ,return_tensors="pt")
        elif mode=="eval":
            tokenized_text = processor.tokenizer(batch[data_args.text_column]
                                                ,padding='max_length'
                                                ,max_length=data_args.eval_max_tokens_per_sentence
                                                ,truncation=True
                                                ,return_tensors="pt")
        else:
            raise Exception("Not a valid mode")


        tokenized_text["input_ids"].masked_fill(tokenized_text.attention_mask.ne(1), -100)
        batch["labels"] = tokenized_text["input_ids"]
        # batch["attention_mask"] = tokenized_text["attention_mask"]

        return batch
    # Apply preprocessing
    train_dataset = train_dataset.map(
        prepare_sample,
        remove_columns=[data_args.audio_column_name,data_args.text_column] ,
        fn_kwargs={"mode": "train"}
    )

    test_dataset = test_dataset.map(
        prepare_sample,
        remove_columns=[data_args.audio_column_name,data_args.text_column] ,
        fn_kwargs={"mode": "eval"}
    )
    # vectorized_datasets.save_to_disk(data_args.vectorized_dataset_dir)
    return train_dataset, test_dataset

def prepare_inf_dataset(batch):
    pass

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
    
    