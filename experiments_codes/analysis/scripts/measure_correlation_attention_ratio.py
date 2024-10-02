import pickle
from argparse import ArgumentParser
from eval import *
from tqdm.contrib import tzip

import plotly.graph_objects as go
import plotly.express as px

from scipy.stats import spearmanr

def calculate_mean_attn_ratios(cm_head_attentions, mono_head_attentions):
    mono_attns = [mono_val for mono_val in mono_head_attentions.values()]
    mono_attns_avg = sum(mono_attns)/len(mono_attns)
    cm_attns = [cm_val for cm_val in cm_head_attentions.values()]
    cm_attns_avg = sum(cm_attns)/len(cm_attns)
    return mono_attns_avg/cm_attns_avg

def calculate_correlation(args):
    rankC_scores = []
    acc_scores = []
    attn_ratios = []

    for pred_file, cm_attn_file, mono_attn_file in tzip(args.prediction_files, args.cm_attn_files, args.mono_attn_files):
        with open(pred_file, 'rb') as f:
           pred_obj = pickle.load(f)

        with open(cm_attn_file, 'rb') as f:
           cm_obj = pickle.load(f)
        
        with open(mono_attn_file, 'rb') as f:
           mono_obj = pickle.load(f)
        
        rankC_batch = [compute_rankc(layer_out['cs_rank_preds'], layer_out['mono_rank_preds']) for layer_out in pred_obj.values()]
        acc_batch = [compute_accuracy_top_n(layer_out['cs_rank_preds'], layer_out['mono_rank_preds']) for layer_out in pred_obj.values()]

        attn_batch = [calculate_mean_attn_ratios(cm_obj[layer_idx], mono_obj[layer_idx]) for layer_idx in mono_obj.keys()]

        attn_ratios.extend(attn_batch)
        rankC_scores.extend(rankC_batch)
        acc_scores.extend(acc_batch)

    acc_rho, acc_p_value = spearmanr(attn_ratios, acc_scores)
    rankC_rho, rankC_p_value = spearmanr(attn_ratios, rankC_scores)
    print(f"Acc Spearman's rho: {acc_rho:.3f}")
    print(f"Acc P-value: {acc_p_value:.6f}")
    print(f"RankC Spearman's rho: {rankC_rho:.3f}")
    print(f"RankC P-value: {rankC_p_value:.6f}")
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--prediction_files', type=str, nargs='+')
    parser.add_argument('--cm_attn_files', type=str, nargs='+')
    parser.add_argument('--mono_attn_files', type=str, nargs='+')

    args = parser.parse_args()

    calculate_correlation(args)