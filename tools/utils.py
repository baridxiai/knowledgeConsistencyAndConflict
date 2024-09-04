import re
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from models.modelWrapper import initialize_model_and_tokenizer, DecoderLensWrapper, EncoderWrapper, initialize_encoder_model_and_tokenizer_per_task
from models.custom_mt0_bias import MT0ForConditionalGeneration
from models.custom_bert_bias import BertForMaskedLM

def initialize_wrapped_model_and_tokenizer(model_name:str, task_type:str, use_custom_bias_model: bool = False) -> [object, AutoTokenizer]:
    #initialize based on task
    task_type = 'cloze'
    if 'mt0' in model_name or 'mt5' in model_name: # encoder-decoder
        if not use_custom_bias_model:
            model, tokenizer = initialize_model_and_tokenizer(model_name)
        else:
            _, tokenizer = initialize_model_and_tokenizer(model_name)
            model = MT0ForConditionalGeneration.from_pretrained(model_name).to('cuda')
        wrapped_model = DecoderLensWrapper(model, tokenizer)
    else: # encoder
        if not use_custom_bias_model:
            model, tokenizer = initialize_encoder_model_and_tokenizer_per_task(model_name, task_type)
        else:
            _, tokenizer =  initialize_encoder_model_and_tokenizer_per_task(model_name, task_type)
            model = BertForMaskedLM.from_pretrained(model_name).to('cuda')
        wrapped_model = EncoderWrapper(model, tokenizer, task_type)
    return wrapped_model, tokenizer
