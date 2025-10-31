import pickle
from argparse import ArgumentParser
from tqdm.contrib import tzip
import numpy as np

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from eval import compute_rankc, compute_accuracy_top_n

pio.kaleido.scope.mathjax = None

def visualize_layerwise_crosslingual_consistencies(args):
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
        if args.compressed:
            # group all layers into three sections
            #rankC
            early_rankC = rankC[0:len(rankC)//3]
            mid_rankC = rankC[len(rankC)//3:2*len(rankC)//3]
            last_rankC = rankC[2*len(rankC)//3:]
            rankC = [sum(early_rankC)/len(early_rankC), sum(mid_rankC)/len(mid_rankC), sum(last_rankC)/len(last_rankC)] 
            layers = [i for i in range(len(rankC))]

            #accuracy
            early_acc = acc[0:len(acc)//3]
            mid_acc = acc[len(acc)//3:2*len(acc)//3]
            last_acc = acc[2*len(acc)//3:]
            acc = [sum(early_acc)/len(early_acc), sum(mid_acc)/len(mid_acc), sum(last_acc)/len(last_acc)] 
        fig_rankC.add_trace(go.Scatter(x=layers, y=rankC, mode='lines', name=label_name))
        fig_accuracy.add_trace(go.Scatter(x=layers, y=acc, mode='lines', name=label_name))
    pio.full_figure_for_development(fig_rankC, warn=False)
    pio.full_figure_for_development(fig_accuracy, warn=False)
    fig_rankC.write_image(args.rankc_filepath, engine="kaleido")
    fig_accuracy.write_image(args.acc_filepath, engine="kaleido")

    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--prediction_files', type=str, nargs='+')
    parser.add_argument('--label_names', type=str, nargs='+')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--rankc_filepath', type=str)
    parser.add_argument('--acc_filepath', type=str)
    parser.add_argument('--compressed', action='store_true', default=False, help='Group all layers into three sections (early, mid, later)')


    args = parser.parse_args()

    visualize_layerwise_crosslingual_consistencies(args)