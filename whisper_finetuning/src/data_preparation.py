from datasets import load_dataset, DatasetDict, Audio

def prepare_dataset(data_args, processor):
    """
    Loads and preprocesses the dataset for Whisper fine-tuning.
    """
    # 1. Load Dataset
    raw_datasets = DatasetDict()
    raw_datasets["train"] = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        split="train+validation",
        cache_dir=data_args.dataset_cache_dir,
        use_auth_token=True,
    )
    raw_datasets["test"] = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        split="test",
        cache_dir=data_args.dataset_cache_dir,
        use_auth_token=True,
    )

    # 2. Remove unnecessary columns
    raw_datasets = raw_datasets.remove_columns(
        ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]
    )

    # 3. Resample audio to 16kHz
    raw_datasets = raw_datasets.cast_column(data_args.audio_column_name, Audio(sampling_rate=16000))

    # 4. Preprocessing function
    def prepare_sample(batch):
        audio = batch[data_args.audio_column_name]
        
        # compute log-Mel input features from input audio array 
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        
        # encode target text to label ids 
        batch["labels"] = processor.tokenizer(batch[data_args.text_column]).input_ids
        return batch

    # 5. Apply preprocessing
    vectorized_datasets = raw_datasets.map(
        prepare_sample,
        remove_columns=list(next(iter(raw_datasets.values())).features),
        num_proc=data_args.preprocessing_num_workers,
        desc="preprocess dataset",
    )

    # 6. Filter samples (optional)
    if data_args.max_train_samples is not None:
        vectorized_datasets["train"] = vectorized_datasets["train"].select(range(data_args.max_train_samples))

    if data_args.max_eval_samples is not None:
        vectorized_datasets["test"] = vectorized_datasets["test"].select(range(data_args.max_eval_samples))

    return vectorized_datasets["train"], vectorized_datasets["test"]
