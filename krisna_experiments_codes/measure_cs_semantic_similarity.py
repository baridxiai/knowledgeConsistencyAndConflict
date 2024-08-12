from argparse import ArgumentParser
from model import initialize_model_and_tokenizer, DecoderLensWrapper, EncoderWrapper, initialize_encoder_model_and_tokenizer_per_task
from sentence_transformers import SentenceTransformer
from dataset import load_qa_dataset_for_inference, load_nli_dataset_for_inference, gather_all_attribute_values_per_id
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
    if 'mt0' in args.model_name or 'mt5' in args.model_name: # encoder-decoder
        model, tokenizer = initialize_model_and_tokenizer(args.model_name)
        wrapped_model = DecoderLensWrapper(model, tokenizer)
    else: # encoder
        model, tokenizer = initialize_encoder_model_and_tokenizer_per_task(args.model_name, args.task_type)
        wrapped_model = EncoderWrapper(model, tokenizer, args.task_type)
    
    with open(args.mono_dataset_file, 'rb') as f:
        mono_dataset = pickle.load(f)
    
    with open(args.cs_dataset_file, 'rb') as f:
        cs_dataset = pickle.load(f)
    
    mono_attribute_values, cs_attribute_values = gather_all_attribute_values_per_id(cs_dataset, mono_dataset, args.attribute_field, args.second_attribute_field)
    avg_sim_score_per_layer = wrapped_model.measure_encoder_representations_cosine_similarity(mono_attribute_values, cs_attribute_values, args.probed_layers)
    for layer in avg_sim_score_per_layer.keys():
        print(f"Layer: {layer}")
        print(f"Cosine similarity: {avg_sim_score_per_layer[layer]}\n")
    
    with open(args.output_file, 'wb') as f:
        pickle.dump(avg_sim_score_per_layer, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--probed_layers', type=int, nargs='+', default=[])
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--cs_dataset_file', type=str)
    parser.add_argument('--mono_dataset_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--task_type', type=str)
    parser.add_argument('--attribute_field', type=str, default='query')
    parser.add_argument('--second_attribute_field', type=str, default=None)


    args = parser.parse_args()

    main(args)
