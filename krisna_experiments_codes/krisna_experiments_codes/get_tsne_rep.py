from sklearn.manifold import TSNE
from argparse import ArgumentParser
from model import initialize_model_and_tokenizer, DecoderLensWrapper
import pickle
import pdb
from tqdm import tqdm
from dataset import load_dataset_for_inference
import plotly.express as px
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def write_2d_manifold(args):
    with open(args.baseline1_file, 'rb') as f:
        baseline1_data = pickle.load(f)
    
    with open(args.baseline2_file, 'rb') as f:
        baseline2_data = pickle.load(f)
    
    with open(args.gcm_file, 'rb') as f:
        gcm_data = pickle.load(f)
    
    with open(args.random_file, 'rb') as f:
        random_data = pickle.load(f)
    
    num_generations = int(len(baseline1_data)/len(random_data))
    
    gcm_queries_dl, _, _, _, _, _, _ = load_dataset_for_inference(args.gcm_file, args.batch_size)
    random_queries_dl, _, _, _, _, _, _ = load_dataset_for_inference(args.random_file, args.batch_size)
    baseline1_queries_dl, _, _, _, _, _, _ = load_dataset_for_inference(args.baseline1_file, args.batch_size)
    baseline2_queries_dl, _, _, _, _, _, _ = load_dataset_for_inference(args.baseline2_file, args.batch_size)

    model, tokenizer = initialize_model_and_tokenizer(args.model_name)
    decoder_lens_model = DecoderLensWrapper(model, tokenizer)

    source_gcm_hidden_states, target_gcm_hidden_states = decoder_lens_model.get_encoder_representation(gcm_queries_dl, args.probed_layers)
    source_random_hidden_states, target_random_hidden_states = decoder_lens_model.get_encoder_representation(random_queries_dl, args.probed_layers)
    source_baseline1_hidden_states, target_baseline1_hidden_states = decoder_lens_model.get_encoder_representation(baseline1_queries_dl, args.probed_layers)
    target_baseline2_hidden_states, source_baseline2_hidden_states = decoder_lens_model.get_encoder_representation(baseline2_queries_dl, args.probed_layers)
    # pdb.set_trace()
    combined_hidden_states = dict()

    for layer_idx in tqdm(args.probed_layers):
        source_gcm_hidden_states_layer = source_gcm_hidden_states[layer_idx]
        source_random_hidden_states_layer = source_random_hidden_states[layer_idx]
        source_baseline1_hidden_states_layer = source_baseline1_hidden_states[layer_idx]
        source_baseline2_hidden_states_layer = source_baseline2_hidden_states[layer_idx]

        target_gcm_hidden_states_layer = target_gcm_hidden_states[layer_idx]
        target_random_hidden_states_layer = target_random_hidden_states[layer_idx]
        target_baseline1_hidden_states_layer = target_baseline1_hidden_states[layer_idx]
        target_baseline2_hidden_states_layer = target_baseline2_hidden_states[layer_idx]

        
        source_combined_layer = torch.cat([source_gcm_hidden_states_layer, source_random_hidden_states_layer, source_baseline1_hidden_states_layer, source_baseline2_hidden_states_layer], dim=0)
        target_combined_layer = torch.cat([target_gcm_hidden_states_layer, target_random_hidden_states_layer, target_baseline1_hidden_states_layer, target_baseline2_hidden_states_layer], dim=0)
        #source_combined_layer = torch.cat([gcm_hidden_states_layer, random_hidden_states_layer], dim=0)
        #source_combined_layer = torch.cat([baseline1_hidden_states_layer, baseline2_hidden_states_layer], dim=0)
        source_combined_layer = source_combined_layer.cpu().detach().numpy()
        target_combined_layer = target_combined_layer.cpu().detach().numpy()

        cats = [f'gcm_{args.source_lang}-{args.target_lang}']*len(source_gcm_hidden_states_layer) + [f'random_{args.source_lang}-{args.target_lang}']*len(source_random_hidden_states_layer) + [f'{args.source_lang}']*len(source_baseline1_hidden_states_layer) + [f'{args.target_lang}']*len(source_baseline2_hidden_states_layer)
        projected_combined_layer = TSNE(n_components=2).fit_transform(combined_layer)
        x_axis = [instance[0] for instance in projected_combined_layer]
        y_axis = [instance[1] for instance in projected_combined_layer]
        output_file_parts = args.output_file.rsplit('.', 1)
        output_filename, output_file_ext = output_file_parts[0], output_file_parts[1]
        layer_output_file = f"{output_filename}_layer_{layer_idx}-lang_{args.source_lang}.{output_file_ext}"
        data = {
            'x': x_axis,
            'y': y_axis,
            'label': cats
        }

        df = pd.DataFrame(data)
        fig = px.scatter(df, x='x', y='y', color='label', title=f'Layer {layer_idx} Encoder Embeddings Context/Answer Language:{args.source_lang}', opacity=0.7, size_max=20)
        fig.write_image(layer_output_file)

        cats = [f'gcm_{args.source_lang}-{args.target_lang}']*len(target_gcm_hidden_states_layer) + [f'random_{args.source_lang}-{args.target_lang}']*len(target_random_hidden_states_layer) + [f'{args.source_lang}']*len(target_baseline1_hidden_states_layer) + [f'{args.target_lang}']*len(target_baseline2_hidden_states_layer)
        projected_combined_layer = TSNE(n_components=2).fit_transform(combined_layer)
        x_axis = [instance[0] for instance in projected_combined_layer]
        y_axis = [instance[1] for instance in projected_combined_layer]
        output_file_parts = args.output_file.rsplit('.', 1)
        output_filename, output_file_ext = output_file_parts[0], output_file_parts[1]
        layer_output_file = f"{output_filename}_layer_{layer_idx}-lang_{args.target_lang}.{output_file_ext}"
        data = {
            'x': x_axis,
            'y': y_axis,
            'label': cats
        }

        df = pd.DataFrame(data)
        fig = px.scatter(df, x='x', y='y', color='label', title=f'Layer {layer_idx} Encoder Embeddings Context/Answer Language: {args.target_lang}', opacity=0.7, size_max=20)
        fig.write_image(layer_output_file)




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gcm_file', type=str)
    parser.add_argument('--random_file', type=str)
    parser.add_argument('--baseline1_file', type=str)
    parser.add_argument('--baseline2_file', type=str)
    parser.add_argument('--output_file', type=str)

    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--source_lang', type=str)
    parser.add_argument('--target_lang', type=str)
    parser.add_argument('--probed_layers', type=int, nargs='+')


    args = parser.parse_args()

    write_2d_manifold(args)
