import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from functools import partial
from dotenv import load_dotenv
import os
load_dotenv()

def load_model_and_processor(model_args, data_args):
    """
    Loads the Whisper model and processor, applying quantization and LoRA if specified.
    """
    auth_token = os.getenv("token")
    processor = WhisperProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=auth_token
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        load_in_4bit=model_args.load_in_4bit,
        load_in_8bit=model_args.load_in_8bit,
        cache_dir=model_args.cache_dir,
        token=auth_token
    )

    model.config.use_cache = False
    # set language and task for generation and re-enable cache
    model.generate = partial(
        model.generate
        , language=data_args.language
        , task=data_args.task
        , use_cache=True
        , forced_decoder_ids = None
    )

    if model_args.load_in_4bit or model_args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    # config = LoraConfig(
    #     r=8,
    #     lora_alpha=16,
    #     target_modules=["q_proj", "v_proj"],
    #     lora_dropout=0.05,
    #     bias="none"
    # )
    # model = get_peft_model(model, config)
    # model.print_trainable_parameters()

    # if model_args.freeze_feature_encoder:
    #     model.freeze_feature_encoder()

    return processor, model
