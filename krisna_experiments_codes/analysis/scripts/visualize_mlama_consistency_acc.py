import pickle
from argparse import ArgumentParser
from eval import *
from tqdm.contrib import tzip

import plotly.graph_objects as go
import plotly.express as px



def visualize_consistency(args):
    fig_rankC = go.Figure()
    fig_accuracy = go.Figure()

    fig_rankC.update_layout(xaxis_title="Layer", yaxis_title="RankC(0-1)", title_text=f"RankC across Encoder Layers in {args.model_name}")
    fig_accuracy.update_layout(xaxis_title="Layer", yaxis_title="Accuracy(0-1)", title_text=f"Accuracy across Encoder Layers in {args.model_name}")

    for pred_file, label_name in tzip(args.prediction_files, args.label_names):
        with open(pred_file, 'rb') as f:
           data_obj = pickle.load(f)
        layers = [key for key in data_obj.keys()]
        rankC = [compute_rankc(layer_out['cs_rank_preds'], layer_out['mono_rank_preds']) for layer_out in data_obj.values()]
        acc = [compute_accuracy_top_n(layer_out['cs_rank_preds'], layer_out['mono_rank_preds']) for layer_out in data_obj.values()]
        fig_rankC.add_trace(go.Scatter(x=layers, y=rankC, mode='lines', name=label_name))
        fig_accuracy.add_trace(go.Scatter(x=layers, y=acc, mode='lines', name=label_name))

    fig_rankC.write_image(args.rankc_filepath)
    fig_accuracy.write_image(args.acc_filepath)

    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--prediction_files', type=str, nargs='+')
    parser.add_argument('--label_names', type=str, nargs='+')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--rankc_filepath', type=str)
    parser.add_argument('--acc_filepath', type=str)


    args = parser.parse_args()

    visualize_consistency(args)