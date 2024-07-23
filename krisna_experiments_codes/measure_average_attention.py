from argparse import ArgumentParser
from model import initialize_model_and_tokenizer, DecoderLensWrapper, EncoderWrapper, initialize_encoder_model_and_tokenizer_per_task
from sentence_transformers import SentenceTransformer
from dataset import load_qa_dataset_for_inference, load_nli_dataset_for_inference
import gc
import torch
from eval import evaluate_sim, compute_query_matrix_and_norm, evaluate_f1_max, evaluate_f1_max, evaluate_exact_max, compute_acc_nli, compute_rankc, compute_mrr
from tqdm import tqdm
import pdb
from datasets import load_dataset
from tqdm import tqdm
import pickle
import re

def add_punctuations_whitespace(s):
    s = re.sub('([.,!?():;])', r' \1 ', s)
    s = re.sub('\s{2,}', ' ', s)
    return s 

def main(args):

    #initialize
    task_type = 'cloze'
    if 'mt0' in args.model_name or 'mt5' in args.model_name: # encoder-decoder
        model, tokenizer = initialize_model_and_tokenizer(args.model_name)
        wrapped_model = DecoderLensWrapper(model, tokenizer)
    else: # encoder
        model, tokenizer = initialize_encoder_model_and_tokenizer_per_task(args.model_name, task_type)
        wrapped_model = EncoderWrapper(model, tokenizer, task_type)
    
    m_lama = load_dataset("m_lama")["test"].shuffle(seed=42)
    m_lama_dict_source_lang = dict()
    m_lama_dict_target_lang = dict()

    # gather all data
    for data in tqdm(m_lama):
        m_lama_id = f'{data["sub_uri"]}-{data["obj_uri"]}@{data["predicate_id"]}'
        
        if data['language'] == args.source_lang:
            if m_lama_id not in m_lama_dict_source_lang:
                m_lama_dict_source_lang[m_lama_id] = dict()
            m_lama_dict_source_lang[m_lama_id]['template'] = add_punctuations_whitespace(data['template'])
            m_lama_dict_source_lang[m_lama_id]['subj_label_same_lang'] = add_punctuations_whitespace(data['sub_label'])
            m_lama_dict_source_lang[m_lama_id]['obj_label'] = add_punctuations_whitespace(data['obj_label'])
            
            if m_lama_id not in m_lama_dict_target_lang:
                m_lama_dict_target_lang[m_lama_id] = dict()
            m_lama_dict_target_lang[m_lama_id]['subj_label_cross_lang'] = add_punctuations_whitespace(data['sub_label'])
        
        elif data['language'] == args.target_lang:
            if m_lama_id not in m_lama_dict_target_lang:
                m_lama_dict_target_lang[m_lama_id] = dict()
            m_lama_dict_target_lang[m_lama_id]['template'] = add_punctuations_whitespace(data['template'])
            m_lama_dict_target_lang[m_lama_id]['subj_label_same_lang'] = add_punctuations_whitespace(data['sub_label'])
            m_lama_dict_target_lang[m_lama_id]['obj_label'] = add_punctuations_whitespace(data['obj_label'])
            
            if m_lama_id not in m_lama_dict_source_lang:
                m_lama_dict_source_lang[m_lama_id] = dict()
            m_lama_dict_source_lang[m_lama_id]['subj_label_cross_lang'] = add_punctuations_whitespace(data['sub_label'])
        
    mlama_instances_source_lang = [instance for instance in m_lama_dict_source_lang.values() if 'subj_label_cross_lang' in instance and 'subj_label_same_lang' in instance]  
    mlama_instances_target_lang = [instance for instance in m_lama_dict_target_lang.values() if 'subj_label_cross_lang' in instance and 'subj_label_same_lang' in instance]  

    source_mono_attentions, source_cs_attentions = wrapped_model.extract_attention_scores_subj_obj(mlama_instances_source_lang, args.batch_size, args.probed_layers)
    target_mono_attentions, target_cs_attentions = wrapped_model.extract_attention_scores_subj_obj(mlama_instances_target_lang, args.batch_size, args.probed_layers)

    source_mono_filepath = f"{args.output_prefix}_{args.source_lang}-attentions.pkl"
    source_cs_filepath = f"{args.output_prefix}_matrix-{args.source_lang}-embedded-{args.target_lang}-attentions.pkl"
    target_mono_filepath = f"{args.output_prefix}_{args.target_lang}-attentions.pkl"
    target_cs_filepath = f"{args.output_prefix}_matrix-{args.target_lang}-embedded-{args.source_lang}-attentions.pkl"
    
    with open(source_mono_filepath, 'wb') as f:
        pickle.dump(source_mono_attentions, f)
    
    with open(source_cs_filepath, 'wb') as f:
        pickle.dump(source_cs_attentions, f)
         
    
    with open(target_mono_filepath, 'wb') as f:
        pickle.dump(target_mono_attentions, f)
    
    with open(target_cs_filepath, 'wb') as f:
        pickle.dump(target_cs_attentions, f)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--probed_layers', type=int, nargs='+', default=[])
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--source_lang', type=str)
    parser.add_argument('--target_lang', type=str)
    parser.add_argument('--output_prefix', type=str, required=False)


    args = parser.parse_args()

    main(args)
