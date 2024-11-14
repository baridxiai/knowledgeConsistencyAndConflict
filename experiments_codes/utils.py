import re
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from models.model import initialize_model_and_tokenizer, DecoderLensWrapper, EncoderWrapper, initialize_encoder_model_and_tokenizer_per_task
from models.custom_mt0_bias import MT0ForConditionalGeneration
from models.custom_bert_bias import BertForMaskedLM
import random

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


def add_punctuations_whitespace(s: str) -> str:
    """
    To add whitespace in-between the token and punctuation to enable the these punctuations to be tokenized separately with words
    @param s(str): input string that we want to tokenize
    """

    s = re.sub('([.,!?():;])', r' \1 ', s)
    s = re.sub('\s{2,}', ' ', s)
    return s

def tokenize_sub(sub_label,tokenizer):
    """

    Tokenize object entity

    @param obj_labels: object entity tokens

    @return all_obj_tokens: tokenized object entity
    @return obj_token_lengths: the length of the object entity (excluding padding token)
    @return max_token_len: maximum object entity length (excluding padding token) within one batch
    """
    special_tok_id =tokenizer("▁")[0:-1]
    sub_tokens = tokenizer(sub_label)["input_ids"][1:-1]
    sub_length = len(sub_tokens)
    sub_mask = " ".join(["<mask>"]*sub_length)
    return sub_mask


def load_mlama(matrix_lang, target_lang,tokenizer):
    """
    Load paralllel matrix_lang-target_lang sentences from mLAMA dataset
    @param matrix_lang: matrix language
    @param embedded_lang: embeded language
    """
    m_lama = load_dataset("m_lama")["test"].shuffle(seed=42)
    m_lama_dict = dict()
    # if target_lang == "baseline":
    #     sub_dict = dict()
    #     for data in m_lama:
    #         if data['language'] != "en":
    #             if data['sub_uri'] not in sub_dict:
    #                 sub_dict[data['sub_uri']] = set()
    #             sub_dict[data['sub_uri']].add(data['sub_label'])

    for data in tqdm(m_lama):
        m_lama_id = f'{data["sub_uri"]}-{data["obj_uri"]}@{data["predicate_id"]}'
        if data['language'] == matrix_lang:
            if m_lama_id not in m_lama_dict:
                m_lama_dict[m_lama_id] = dict()
            m_lama_dict[m_lama_id]['template'] = add_punctuations_whitespace(data['template'])
            m_lama_dict[m_lama_id]['subj_label_same_lang'] = add_punctuations_whitespace(data['sub_label'])
            m_lama_dict[m_lama_id]['obj_label'] = add_punctuations_whitespace(data['obj_label'])
        elif data['language'] == target_lang:
            if m_lama_id not in m_lama_dict:
                m_lama_dict[m_lama_id] = dict()
            m_lama_dict[m_lama_id]['subj_label_cross_lang'] = add_punctuations_whitespace(data['sub_label'])
            m_lama_dict[m_lama_id]['obj_label_cross_lang'] = add_punctuations_whitespace(data['obj_label'])
        elif target_lang == "baseline":
            if m_lama_id not in m_lama_dict:
                m_lama_dict[m_lama_id] = dict()
            # random_uri = random.choice(list(set(sub_dict.keys())-set([data['sub_uri']])))
            sub_mask = tokenize_sub(data['sub_label'],tokenizer)
            #m_lama_dict[m_lama_id]['subj_label_cross_lang'] = add_punctuations_whitespace('<extra_id_0>')
            m_lama_dict[m_lama_id]['subj_label_cross_lang'] = add_punctuations_whitespace(sub_mask)
            m_lama_dict[m_lama_id]['obj_label_cross_lang'] = add_punctuations_whitespace(data['obj_label'])

    mlama_instances = [instance for instance in m_lama_dict.values() if 'subj_label_cross_lang' in instance and 'subj_label_same_lang' in instance] # filter out any subject that doesn't have its parallel subject

    return mlama_instances
