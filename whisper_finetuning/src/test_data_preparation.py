from transformers import (
    HfArgumentParser,
    WhisperProcessor,
)
from data_preparation import prepare_dataset
from main import DataCollatorSpeechSeq2SeqWithPadding
from training_args import DataTrainingArguments, ModelArguments
from transformers import WhisperProcessor,WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
model_args, data_args = parser.parse_args_into_dataclasses([])

data_args.processed_dataset_dir = "../cache/processed_common_voice/"
train_ds, eval_ds = prepare_dataset(data_args, processor)
collator = DataCollatorSpeechSeq2SeqWithPadding(processor, data_args)


num_rows_to_get = 2
first_n_rows = []
for i, example in enumerate(train_ds):
    if i >= num_rows_to_get:
        break
    first_n_rows.append(example)
    
    
result = collator(first_n_rows)
assert result["input_features"].shape == (num_rows_to_get, 128, 3000), "Input features are not in right shape"
assert result["labels"].shape == (num_rows_to_get,data_args.train_max_tokens_per_sentence), "Labels are not in right shape"

print(f"input_features shapes: {result["input_features"].shape}")
print(f"labels shapes: {result["labels"].shape}")