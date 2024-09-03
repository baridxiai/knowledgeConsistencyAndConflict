import re
from datasets import load_dataset
import tqdm

def add_punctuations_whitespace(s):
    """
    To add whitespace in-between the token and punctuation to enable the these punctuations to be tokenized separately with words
    @param s(str): input string that we want to tokenize
    """

    s = re.sub('([.,!?():;])', r' \1 ', s)
    s = re.sub('\s{2,}', ' ', s)
    return s

def _tokenize_obj(obj_labels,tokenizer):
    """

    Tokenize object entity

    @param obj_labels: object entity tokens

    @return all_obj_tokens: tokenized object entity
    @return obj_token_lengths: the length of the object entity (excluding padding token)
    @return max_token_len: maximum object entity length (excluding padding token) within one batch
    """
    all_obj_tokens = []
    obj_token_lengths = []
    special_tok_id =tokenizer("▁")[0:-1]
    for obj_label in obj_labels:
        obj_tokens = tokenizer(obj_label)["input_ids"]
        if special_tok_id in obj_tokens:
            obj_tokens = obj_tokens[2:-1]
        else:
            obj_tokens = obj_tokens[1:-1]
        obj_token_lengths.append(len(obj_tokens))
        all_obj_tokens.append(obj_tokens)

    max_token_len = max(obj_token_lengths)

    # add padding
    for i in range(len(all_obj_tokens)):
        num_pad_tokens = max_token_len-obj_token_lengths[i]
        all_obj_tokens[i] += [tokenizer.pad_token_id]*num_pad_tokens
    return all_obj_tokens, obj_token_lengths, max_token_len
def mask_sentences(prompts, obj_token_lengths,tokenizer):
    """
    Replace single mask into tokenizer's n-gram masks

    @param prompts: list of prompts/inputs]

    @return new_prompts: list of edited prompts/inputs
    """

    new_prompts = []
    for prompt, obj_token_length in zip(prompts, obj_token_lengths):
        new_mask = " ".join(["<mask>"]*obj_token_length)
        new_prompt = prompt.replace('[Y]', new_mask)
        new_prompt = new_prompt.replace('<mask>', tokenizer.mask_token)
        new_prompts.append(new_prompt)
    return new_prompts
def load_mlama(matrix_lang, target_lang):
    """
    Load paralllel matrix_lang-target_lang sentences from mLAMA dataset
    @param matrix_lang: matrix language
    @param embedded_lang: embeded language
    """

    m_lama = load_dataset("m_lama")["test"].shuffle(seed=42)
    m_lama_dict = dict()

    for data in tqdm(m_lama):
        m_lama_id = f'{data["sub_uri"]}-{data["obj_uri"]}@{data["predicate_id"]}'
        if data['language'] == matrix_lang:
            if m_lama_id not in m_lama_dict:
                m_lama_dict[m_lama_id] = dict()
            m_lama_dict[m_lama_id]['template'] = add_punctuations_whitespace(data['template'])
            m_lama_dict[m_lama_id]['subj_label_same_lang'] = add_punctuations_whitespace(data['sub_label'])
            m_lama_dict[m_lama_id]['obj_label'] = add_punctuations_whitespace(data['obj_label'])
        elif target_lang is not None and data['language'] == target_lang:
            if m_lama_id not in m_lama_dict:
                m_lama_dict[m_lama_id] = dict()
            m_lama_dict[m_lama_id]['subj_label_cross_lang'] = add_punctuations_whitespace(data['sub_label'])
    if target_lang is not None:
        mlama_instances = [instance for instance in m_lama_dict.values() if 'subj_label_cross_lang' in instance and 'subj_label_same_lang' in instance] # filter out any subject that doesn't have its parallel subject
    return mlama_instances