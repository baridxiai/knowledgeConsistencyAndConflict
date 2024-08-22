# Adapted implementation from https://github.com/theNamek/Bias-Neurons/blob/main/bias_neuron_src/1_analyze_mlm_bias.py\
from argparse import ArgumentParser
import torch
from tqdm import tqdm
from datasets import load_dataset
from tqdm import tqdm
import pickle
from transformers import AutoTokenizer
import numpy as np
from custom_mt0_bias import MT0ForConditionalGeneration
from custom_bert_bias import BertForMaskedLM
from utils import load_mlama
from typing import Union, List, Dict



def scaled_input(emb: torch.Tensor, batch_size: int, num_batch: int = 1) -> [torch.Tensor, np.array]:
    """"
    Create a batch of activations delta

    @param emb: activation tensor from the hidden states in FFN [1*intermediate_dim]
    @param batch_size: how man instances within one batch of deltas
    @param num_batch: total number of delta batches

    @return res: batches of activation delta [num_of_points*intermediate_dim]
    @return grad_step: batches of activation delta [1*intermediate_dim]
    """
    baseline = torch.zeros_like(emb) # 1*intermed_dim

    num_points = batch_size * num_batch
    grad_step = (emb - baseline) / num_points 

    res = torch.cat([torch.add(baseline, grad_step * i) for i in range(num_points)], dim=0) # batch
    return res, grad_step.detach().cpu().numpy()


def tokenize_obj(tokenizer: AutoTokenizer, obj_label: str, model_type: str) -> [Dict[str, List], Dict[str, torch.Tensor, List[int], List[int]]]:
    """
    Get the all necessary attributes with the tokenized object

    @param tokenizer: The model's tokenizer
    @param obj_label: Object entity
    @param model_typer: encoder-decoder or encoder

    @return obj_tokens: output from tokenizer given obj_label input
    @return obj_tokens_cuda: same as obj_tokens but in gpu
    @return obj_token_positions: span indices of the object entity
    @return labels: all object tokens
    @return obj_tokens_len: length of the object entity
    """
    if model_type == 'encoder-decoder':
        obj_tokens = tokenizer(f"<extra_id_0> {obj_label} <extra_id_1>")
        obj_tokens_cuda = tokenizer(f"<extra_id_0> {obj_label} <extra_id_1>", return_tensors='pt').to('cuda') 

        obj_token_input_ids = obj_tokens['input_ids']
        start_pos = obj_token_input_ids.index(250099)
        end_pos = obj_token_input_ids.index(250098)
        obj_token_positions = [idx for idx in range(start_pos+1, end_pos)]
        labels = [obj_token_input_ids[idx] for idx in obj_token_positions] 
        
        return obj_tokens, obj_tokens_cuda, obj_token_positions, labels
    
    else:
        obj_tokens = tokenizer(obj_label)
        obj_tokens_cuda = tokenizer(obj_label, return_tensors='pt').to('cuda') 
        obj_tokens_len = len(obj_tokens)
        return obj_tokens, obj_tokens_cuda, obj_tokens_len

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    torch.cuda.empty_cache()
    if args.model_type == 'encoder-decoder':
        model = MT0ForConditionalGeneration.from_pretrained(args.model_name)
    else:
        model = BertForMaskedLM.from_pretrained(args.model_name)

    model.to('cuda')
    mlama_instances = load_mlama(args.matrix_lang, args.embedded_lang)

    extra_id_token = 250099
    
    mono_ig2_avg_per_layer = dict()
    cs_ig2_avg_per_layer = dict()
    
    divider = len(mlama_instances)
    model.eval()

    for instance in tqdm(mlama_instances):
        if args.model_type == 'encoder-decoder':
            _, obj_tokens_cuda, obj_token_positions, labels = tokenize_obj(tokenizer, instance['obj_label'], args.model_type)
            instance_template = instance['template'].replace('[Y]', '<extra_id_0>')

            # Do code-mixing
            instance_mono_template = instance_template.replace('[X]', instance['subj_label_same_lang'])
            instance_cs_template = instance_template.replace('[X]', instance['subj_label_cross_lang'])

            tokenized_instance_mono_template = tokenizer(instance_mono_template)
            tokenized_instance_cs_template = tokenizer(instance_cs_template)

            # find the sentinel token in the encoder
            extra_token_position_mono = tokenized_instance_mono_template['input_ids'].index(extra_id_token)
            extra_token_position_cs = tokenized_instance_cs_template['input_ids'].index(extra_id_token)

            tokenized_instance_mono_template = tokenizer(instance_mono_template, return_tensors='pt', padding=False).to('cuda')
            tokenized_instance_cs_template = tokenizer(instance_cs_template, return_tensors='pt', padding=False).to('cuda')

            # monolingual ig2
            _, mono_ffn_states_per_layer = model(**tokenized_instance_mono_template, labels=obj_tokens_cuda['input_ids'], tgt_layers=args.probed_layers, encoder_tgt_pos = extra_token_position_mono, decoder_tgt_positions = obj_token_positions, tgt_labels=labels)
            for key, ffn_states in mono_ffn_states_per_layer.items():
                ffn_states = ffn_states.squeeze(1)
                scaled_weights, weights_step = scaled_input(ffn_states, args.integration_batch_size, args.integration_num_batch)  # (num_points, ffn_size), (ffn_size)
                scaled_weights.requires_grad_(True)
                ig2_mono = None
                for batch_idx in range(args.integration_num_batch):
                    batch_weights = scaled_weights[batch_idx * args.integration_batch_size:(batch_idx + 1) * args.integration_batch_size]
                    _, grad = model(**tokenized_instance_mono_template, labels=obj_tokens_cuda['input_ids'], tgt_layers=[key], modified_activation_values = batch_weights, encoder_tgt_pos = extra_token_position_mono, decoder_tgt_positions = obj_token_positions, tgt_labels=labels)  # (batch, n_vocab), (batch, ffn_size)
                    grad = grad.sum(axis=0)  # (ffn_size)
                    ig2_mono = grad if ig2_mono is None else np.add(ig2_mono, grad) # (ffn_size)
                ig2_mono = ig2_mono*weights_step
                if key not in mono_ig2_avg_per_layer:
                    mono_ig2_avg_per_layer[key] = ig2_mono.squeeze(0)
                else:
                    mono_ig2_avg_per_layer[key] = np.add(mono_ig2_avg_per_layer[key] , ig2_mono.squeeze(0))
            
            # codemixed ig2
            _, cs_ffn_states_per_layer = model(**tokenized_instance_cs_template, labels=obj_tokens_cuda['input_ids'], tgt_layers=args.probed_layers, encoder_tgt_pos = extra_token_position_cs, decoder_tgt_positions = obj_token_positions, tgt_labels=labels)
            for key, ffn_states in cs_ffn_states_per_layer.items():
                ffn_states = ffn_states.squeeze(1)
                scaled_weights, weights_step = scaled_input(ffn_states, args.integration_batch_size, args.integration_num_batch)  # (num_points, ffn_size), (ffn_size)
                scaled_weights.requires_grad_(True)
                ig2_cs = None
                for batch_idx in range(args.integration_num_batch):
                    batch_weights = scaled_weights[batch_idx * args.integration_batch_size:(batch_idx + 1) * args.integration_batch_size]
                    _, grad = model(**tokenized_instance_cs_template, labels=obj_tokens_cuda['input_ids'], tgt_layers=[key], modified_activation_values = batch_weights, encoder_tgt_pos = extra_token_position_cs, decoder_tgt_positions = obj_token_positions, tgt_labels=labels)  # (batch, n_vocab), (batch, ffn_size)
                    grad = grad.sum(axis=0)  # (ffn_size)
                    ig2_cs = grad if ig2_cs is None else np.add(ig2_cs, grad) # (ffn_size)
                ig2_cs = ig2_cs*weights_step
                if key not in cs_ig2_avg_per_layer:
                    cs_ig2_avg_per_layer[key] = ig2_cs.squeeze(0)
                else:
                    cs_ig2_avg_per_layer[key] = np.add(cs_ig2_avg_per_layer[key] , ig2_cs.squeeze(0))
        
        # encoder
        else:
            _, obj_tokens_cuda, obj_tokens_len  = tokenize_obj(tokenizer, instance['obj_label'], args.model_type)

            # replace the object
            obj_tokens_input_ids = obj_tokens_cuda['input_ids'] # for replacement
            instance_template = instance['template'].replace('[Y]', " ".join([tokenizer.mask_token]*obj_tokens_len))

            instance_mono_template = instance_template.replace('[X]', instance['subj_label_same_lang'])
            instance_cs_template = instance_template.replace('[X]', instance['subj_label_cross_lang'])

            tokenized_instance_mono_template = tokenizer(instance_mono_template, return_tensors='pt', padding=False).to('cuda')
            tokenized_instance_cs_template = tokenizer(instance_cs_template, return_tensors='pt', padding=False).to('cuda')

            start_tgt_pos_mono = tokenized_instance_mono_template['input_ids'][0].tolist().index(tokenizer.mask_token_id)
            start_tgt_pos_cs = tokenized_instance_cs_template['input_ids'][0].tolist().index(tokenizer.mask_token_id)

            
            for layer in args.probed_layers:
                # monolingual ig2
                mono_inputs_copy = {key: tensor.clone().to(torch.device('cuda')) for key, tensor in tokenized_instance_mono_template.items()}
                ig2_mono = None
                for curr_mask_pos in range(start_tgt_pos_mono, start_tgt_pos_mono+obj_tokens_len):
                    ffn_states, _ = model(**mono_inputs_copy, tgt_layer=layer, tgt_pos = curr_mask_pos)
                    scaled_weights, weights_step = scaled_input(ffn_states, args.integration_batch_size, args.integration_num_batch)  # (num_points, ffn_size), (ffn_size)
                    scaled_weights.requires_grad_(True)
                    total_grad = None
                    for batch_idx in range(args.integration_num_batch):
                        batch_weights = scaled_weights[batch_idx * args.integration_batch_size:(batch_idx + 1) * args.integration_batch_size]
                        _, grad = model(**mono_inputs_copy, tgt_layer=layer, tgt_pos = curr_mask_pos, tmp_score=batch_weights, tgt_label=obj_tokens_input_ids[0][curr_mask_pos-start_tgt_pos_mono])  # (batch, n_vocab), (batch, ffn_size)
                        grad = grad.sum(axis=0)  # (ffn_size)
                        total_grad = grad if total_grad is None else np.add(total_grad, grad) # (ffn_size)
                    mono_inputs_copy['input_ids'][0][curr_mask_pos] = obj_tokens_input_ids[0][curr_mask_pos-start_tgt_pos_mono]
                    total_grad = total_grad*weights_step
                    ig2_mono = total_grad if ig2_mono is None else np.add(ig2_mono, total_grad)
                ig2_mono = np.divide(ig2_mono, obj_tokens_len)
                if layer not in mono_ig2_avg_per_layer:
                    mono_ig2_avg_per_layer[layer] = ig2_mono.squeeze(0)
                else:
                    mono_ig2_avg_per_layer[layer] = np.add(mono_ig2_avg_per_layer[layer] , ig2_mono.squeeze(0))
                
                # cs ig2
                cs_inputs_copy = {key: tensor.clone().to(torch.device('cuda')) for key, tensor in tokenized_instance_cs_template.items()}
                ig2_cs = None
                for curr_mask_pos in range(start_tgt_pos_cs, start_tgt_pos_cs+obj_tokens_len):
                    ffn_states, _ = model(**cs_inputs_copy, tgt_layer=layer, tgt_pos = curr_mask_pos)
                    scaled_weights, weights_step = scaled_input(ffn_states, args.integration_batch_size, args.integration_num_batch)  # (num_points, ffn_size), (ffn_size)
                    scaled_weights.requires_grad_(True)
                    total_grad = None
                    for batch_idx in range(args.integration_num_batch):
                        batch_weights = scaled_weights[batch_idx * args.integration_batch_size:(batch_idx + 1) * args.integration_batch_size]
                        _, grad = model(**cs_inputs_copy, tgt_layer=layer, tgt_pos = curr_mask_pos, tmp_score=batch_weights, tgt_label=obj_tokens_input_ids[0][curr_mask_pos-start_tgt_pos_cs])  # (batch, n_vocab), (batch, ffn_size)
                        grad = grad.sum(axis=0)  # (ffn_size)
                        total_grad = grad if total_grad is None else np.add(total_grad, grad) # (ffn_size)
                    cs_inputs_copy['input_ids'][0][curr_mask_pos] = obj_tokens_input_ids[0][curr_mask_pos-start_tgt_pos_cs]
                    total_grad = total_grad*weights_step
                    ig2_cs = total_grad if ig2_cs is None else np.add(ig2_cs, total_grad)
                ig2_cs = np.divide(ig2_cs, obj_tokens_len)
                if layer not in cs_ig2_avg_per_layer:
                    cs_ig2_avg_per_layer[layer] = ig2_cs.squeeze(0)
                else:
                    cs_ig2_avg_per_layer[layer] = np.add(cs_ig2_avg_per_layer[layer] , ig2_cs.squeeze(0))

    for layer in args.probed_layers:
        mono_ig2_avg_per_layer[layer] = np.divide(mono_ig2_avg_per_layer[layer], divider)           
        cs_ig2_avg_per_layer[layer] = np.divide(cs_ig2_avg_per_layer[layer], divider)  
    
    mono_output_filepath = f"{args.output_prefix}-{args.source_lang}-{args.target_lang}-mono-ig2.pkl"
    cs_output_filepath = f"{args.output_prefix}-{args.source_lang}-{args.target_lang}-cm-ig2.pkl"

    with open(mono_output_filepath, 'wb') as f:
        pickle.dump(mono_ig2_avg_per_layer, f)


    with open(cs_output_filepath, 'wb') as f:
        pickle.dump(cs_ig2_avg_per_layer, f)

    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--integration_num_batch', type=int, default=1)
    parser.add_argument('--integration_batch_size', type=int, default=20)
    parser.add_argument('--probed_layers', type=int, nargs='+', default=[])
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--matrix_lang', type=str)
    parser.add_argument('--embedded_lang', type=str)
    parser.add_argument('--output_prefix', type=str, required=False)
    parser.add_argument('--model_type', type=str, choices=['encoder-decoder', 'encoder'])

    args = parser.parse_args()

    main(args)
