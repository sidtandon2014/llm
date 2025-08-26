from datasets import load_dataset, DatasetDict, Audio, load_from_disk, save_to_disk
import os

def load_raw_dataset(data_args):
    """
    Loads the dataset for Whisper fine-tuning.
    """
    processed_dataset_path = data_args.processed_dataset_dir
    if not os.path.exists(processed_dataset_path):
        raw_datasets = DatasetDict()
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split="train",
            cache_dir=data_args.dataset_cache_dir,
            token="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",
            trust_remote_code=True
        )
        raw_datasets["test"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split="test[:1000]",
            cache_dir=data_args.dataset_cache_dir,
            token="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",
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
        raw_datasets = raw_datasets.filter(
            is_audio_in_length_range,
            input_columns=[data_args.audio_column_name],
            num_proc=4
            )
        
        raw_datasets.save_to_disk(processed_dataset_path)
        return None

def prepare_dataset(data_args, processor):
    """
    Loads and preprocesses the dataset for Whisper fine-tuning.
    """
    # 1. Load Dataset
    processed_datasets = load_dataset(data_args.processed_dataset_path)

    # Select few samples for testing
    if data_args.max_train_samples is not None:
        processed_datasets["train"] = processed_datasets["train"].select(range(data_args.max_train_samples))

    if data_args.max_eval_samples is not None:
        processed_datasets["test"] = processed_datasets["test"].select(range(data_args.max_eval_samples))
    
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
        ).input_features[0]
        
        # encode target text to label ids 
        batch["labels"] = processor.tokenizer(batch[data_args.text_column]).input_ids
        return batch

    # Apply preprocessing
    vectorized_datasets = processed_datasets.map(
        prepare_sample,
        remove_columns=processed_datasets.column_names,
        num_proc=data_args.preprocessing_num_workers,
        desc="preprocess dataset",
    )


    return vectorized_datasets
