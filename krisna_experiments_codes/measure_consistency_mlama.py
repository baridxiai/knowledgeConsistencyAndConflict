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
            m_lama_dict_source_lang[m_lama_id]['template'] = data['template']
            m_lama_dict_source_lang[m_lama_id]['subj_label_same_lang'] = data['sub_label']
            m_lama_dict_source_lang[m_lama_id]['obj_label'] = data['obj_label']
            
            if m_lama_id not in m_lama_dict_target_lang:
                m_lama_dict_target_lang[m_lama_id] = dict()
            m_lama_dict_target_lang[m_lama_id]['subj_label_cross_lang'] = data['sub_label']
        
        elif data['language'] == args.target_lang:
            if m_lama_id not in m_lama_dict_target_lang:
                m_lama_dict_target_lang[m_lama_id] = dict()
            m_lama_dict_target_lang[m_lama_id]['template'] = data['template']
            m_lama_dict_target_lang[m_lama_id]['subj_label_same_lang'] = data['sub_label']
            m_lama_dict_target_lang[m_lama_id]['obj_label'] = data['obj_label']
            
            if m_lama_id not in m_lama_dict_source_lang:
                m_lama_dict_source_lang[m_lama_id] = dict()
            m_lama_dict_source_lang[m_lama_id]['subj_label_cross_lang'] = data['sub_label']
        
    mlama_instances_source_lang = [instance for instance in m_lama_dict_source_lang.values() if 'subj_label_cross_lang' in instance and 'subj_label_same_lang' in instance]  
    mlama_instances_target_lang = [instance for instance in m_lama_dict_target_lang.values() if 'subj_label_cross_lang' in instance and 'subj_label_same_lang' in instance]  

    # inference yay!
    source_mono_rank_preds, source_cs_rank_preds, source_gts = wrapped_model.inference_cloze_task(mlama_instances_source_lang, args.batch_size, args.probed_layers, args.beam_topk, args.ranking_topk)
    target_mono_rank_preds, target_cs_rank_preds,  target_gts = wrapped_model.inference_cloze_task(mlama_instances_target_lang, args.batch_size, args.probed_layers, args.beam_topk, args.ranking_topk)
    pdb.set_trace(source_mono_rank_preds)
    if len(args.probed_layers) == 0 or -1 in args.probed_layers:
        print(f"Matrix Language: {args.source_lang}, Embedded Language: {args.target_lang}")
        print(f"RankC score: {compute_rankc(source_cs_rank_preds, source_mono_rank_preds)}")
        print(f"Mono MRR score: {compute_mrr(source_mono_rank_preds, source_gts)}, CS MRR: {compute_mrr(compute_mrr(source_cs_rank_preds, source_gts))}\n")
    
        print(f"Matrix Language: {args.target_lang}, Embedded Language: {args.source_lang}")
        print(f"RankC score: {compute_rankc(target_cs_rank_preds, target_mono_rank_preds)}")
        print(f"Mono MRR score: {compute_mrr(target_mono_rank_preds, target_gts)}, CS MRR: {compute_mrr(target_cs_rank_preds, target_gts)}")
    else:
        source_out_dict, target_out_dict = dict(), dict()
        for layer in args.probed_layers:
            source_rankc_score = compute_rankc(source_cs_rank_preds[layer], source_mono_rank_preds[layer])
            source_mono_mrr = compute_mrr(source_mono_rank_preds[layer], source_gts)
            source_cs_mrr = compute_mrr(source_cs_rank_preds[layer], source_gts)
            source_out_dict[layer] = {
                'rankC': source_rankc_score,
                'mono_mrr': source_mono_mrr,
                'cs_mrr': source_cs_mrr,
                'mono_rank_preds': source_mono_rank_preds[layer],
                'cs_rank_preds': source_cs_rank_preds[layer]
            }

            target_rankc_score = compute_rankc(target_cs_rank_preds[layer], target_mono_rank_preds[layer])
            target_mono_mrr = compute_mrr(target_mono_rank_preds[layer], target_gts)
            target_cs_mrr = compute_mrr(target_cs_rank_preds[layer], target_gts)
            target_out_dict[layer] = {
                'rankC': target_rankc_score,
                'mono_mrr': target_mono_mrr,
                'cs_mrr': target_cs_mrr,
                'mono_rank_preds': target_mono_rank_preds[layer],
                'cs_rank_preds': target_cs_rank_preds[layer]
            }

            print(f"Layer: {layer}")
            print(f"Matrix Language: {args.source_lang}, Embedded Language: {args.target_lang}")
            print(f"RankC score: {source_rankc_score}")
            print(f"Mono MRR score: {source_mono_mrr}, CS MRR score: {source_cs_mrr}\n")
            print(f"Matrix Language: {args.target_lang}, Embedded Language: {args.source_lang}")
            print(f"RankC score: {target_rankc_score}")
            print(f"Mono MRR score: {target_mono_mrr}, CS MRR score: {target_cs_mrr}\n")

        
        source_filepath = f"{args.output_prefix}_matrix-{args.source_lang}-embedded-{args.target_lang}.pkl"
        target_filepath = f"{args.output_prefix}_matrix-{args.target_lang}-embedded-{args.source_lang}.pkl"
        
        with open(source_filepath, 'wb') as f:
            pickle.dump(source_out_dict, f)

        with open(target_filepath, 'wb') as f:
            pickle.dump(target_out_dict, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--probed_layers', type=int, nargs='+', default=[])
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--source_lang', type=str)
    parser.add_argument('--target_lang', type=str)
    parser.add_argument('--output_prefix', type=str, required=False)
    parser.add_argument('--beam_topk', type=int, default=1)
    parser.add_argument('--ranking_topk', type=int, default=3)


    args = parser.parse_args()

    main(args)
