import pickle
from argparse import ArgumentParser
from eval import *
from tqdm.contrib import tzip

import plotly.graph_objects as go
import plotly.express as px

from scipy.stats import spearmanr

def calculate_mean_absolute(cm_arr, mono_arr):
    mono_abs_values = [abs(val) for val in mono_arr]
    cm_abs_values = [abs(val) for val in cm_arr]
    mono_avg = sum(mono_abs_values)/len(mono_abs_values)
    cm_avg = sum(cm_abs_values)/len(cm_abs_values)
    return -abs(cm_avg-mono_avg)

def calculate_correlation(args):
    rankC_scores = []
    acc_scores = []
    ig2_diffs = []

    for pred_file, cm_ig2_file, mono_ig2_file in tzip(args.prediction_files, args.cm_ig2_files, args.mono_ig2_files):
        with open(pred_file, 'rb') as f:
           pred_obj = pickle.load(f)

        with open(cm_ig2_file, 'rb') as f:
           cm_obj = pickle.load(f)
        
        with open(mono_ig2_file, 'rb') as f:
           mono_obj = pickle.load(f)
        
        rankC_batch = [compute_rankc(layer_out['cs_rank_preds'], layer_out['mono_rank_preds']) for layer_out in pred_obj.values()]
        acc_batch = [compute_accuracy_top_n(layer_out['cs_rank_preds'], layer_out['mono_rank_preds']) for layer_out in pred_obj.values()]

        ig2_batch = [calculate_mean_absolute(cm_obj[layer_idx], mono_obj[layer_idx]) for layer_idx in mono_obj.keys()]

        ig2_diffs.extend(ig2_batch)
        rankC_scores.extend(rankC_batch)
        acc_scores.extend(acc_batch)

    acc_rho, acc_p_value = spearmanr(ig2_diffs, acc_scores)
    rankC_rho, rankC_p_value = spearmanr(ig2_diffs, rankC_scores)
    print(f"Acc Spearman's rho: {acc_rho:.3f}")
    print(f"Acc P-value: {acc_p_value:.6f}")
    print(f"RankC Spearman's rho: {rankC_rho:.3f}")
    print(f"RankC P-value: {rankC_p_value:.6f}")
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--prediction_files', type=str, nargs='+')
    parser.add_argument('--cm_ig2_files', type=str, nargs='+')
    parser.add_argument('--mono_ig2_files', type=str, nargs='+')

    args = parser.parse_args()

    calculate_correlation(args)