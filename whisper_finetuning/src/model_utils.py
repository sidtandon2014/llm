import torch
from transformers import (
    WhisperForConditionalGeneration
    , WhisperProcessor
    , AutoConfig
    , AutoProcessor
    , BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from functools import partial
from dotenv import load_dotenv
from pathlib import Path
import os
load_dotenv()

def load_model_and_processor(model_args, data_args, inference_args=None):
    """
    Loads the Whisper model and processor, applying quantization and LoRA if specified.
    """
    auth_token = os.getenv("token")

    model_path = model_args.model_name_or_path
    
    if not model_args.is_model_id:
        # load model from checkpoint
        # this code block should mostly be run during inference.
        # As during training, Trainer class will take care of loading the checkpoints

        assert inference_args is not None, "Please pass inference arguments"
        checkpoint_path = os.path.join(Path(__file__).resolve().parent.parent, model_args.model_name_or_path)
    
        print(checkpoint_path)
        if os.path.isdir(checkpoint_path):
            raise Exception("model_name_or_path should be a checkpoint directory")
        
        model_path = os.path.join(checkpoint_path,"pytorch_model.bin")
        if not os.path.exists(model_path):
            raise Exception(("pytorch_model.bin not found inside checkpoint directory"),
                            ("Run ```./zero_to_fp32.py . pytorch_model.bin``` and then run this file")
                            )
        
        quantization_config = None
        if inference_args.quantization_algo == "bnb":
            quantization_config = BitsAndBytesConfig(load_in_8bit=inference_args.inf_bnb_load_in_8bit
                                                    ,load_in_4bit=inference_args.inf_bnb_load_in_4bit)
        config = AutoConfig.from_pretrained(checkpoint_path)
        model = WhisperForConditionalGeneration.from_pretrained(model_path
                                                            ,config=config
                                                            ,quantization_config=quantization_config
                                                           ) #.to("cuda:0")

        processor = AutoProcessor.from_pretrained(checkpoint_path)
    else:
        # Load model from model_id and downlaod weights
        processor = WhisperProcessor.from_pretrained(
            pretrained_model_name_or_path=model_path,
            cache_dir=model_args.cache_dir,
            token=auth_token
        )

        model = WhisperForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            load_in_4bit=model_args.train_bnb_load_in_4bit,
            load_in_8bit=model_args.train_bnb_load_in_8bit,
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
        if model_args.train_bnb_load_in_8bit or model_args.train_bnb_load_in_4bit:
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
