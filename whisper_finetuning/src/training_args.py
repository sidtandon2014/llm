import dataclasses
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="openai/whisper-large-v3",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default="./cache/model/",
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_auth_token: bool = field(
        default=True,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    load_in_8bit: bool = field(
        default=False, metadata={"help": "Whether to load the model in 8bit."}
    )
    load_in_4bit: bool = field(
        default=False, metadata={"help": "Whether to load the model in 4bit."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: str = field(
        default="mozilla-foundation/common_voice_11_0", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default='en', metadata={"help": "The configuration name of the dataset to use (via the datasets library). For english this will be 'en'"}
    )
    text_column: str = field(
        default="sentence",
        metadata={"help": "The name of the column in the datasets containing the full text of the transcription."},
    )
    dataset_cache_dir: Optional[str] = field(
        default="./cache/", metadata={"help": "Path to cache directory for saving and loading datasets"}
    )
    processed_dataset_dir: Optional[str] = field(
        default="./cache/processed_common_voice/", metadata={"help": "Path to processed directory for saving and loading datasets"}
    )
    vectorized_dataset_dir: Optional[str] = field(
        default="./cache/vectorized_dataset/", metadata={"help": "Path to processed directory for saving and loading datasets"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=40,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=100,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=100,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    max_duration_in_seconds: float = field(
        default=30.0,
        metadata={"help": "Filter audio files that are longer than `max_duration_in_seconds` seconds"},
    )
    min_duration_in_seconds: float = field(
        default=0.0,
        metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"},
    )
    sampling_rate:int = field(
        default=16000,
        metadata={"help": "The sampling rate of the audio data. Defaults to 16000Hz (required by Whisper)."}
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": "Only run the preprocessing script to generate the cached dataset and do not run training. "
            "Useful for debugging the data preparation step."
        },
    )
    language: str = field(
        default="english",
        metadata={
            "help": "Language of the dataset. Important for models that support multiple languages. "
            "For English, set to `en`. See the model card for supported languages."
        },
    )
    task: str = field(
        default="transcribe",
        metadata={
            "help": "Task to perform: 'transcribe' for speech-to-text or 'translate' to translate to English."
        },
    )
    train_max_tokens_per_sentence: int = field(
        default=32,
        metadata={
            "help": "MAximum tokens in a sentence"
        },
    )
    eval_max_tokens_per_sentence: int = field(
        default=32,
        metadata={
            "help": "MAximum tokens in a sentence"
        },
    )
