from glob import glob
import pickle
from argparse import ArgumentParser
import plotly.graph_objects as go
from eval import *


def visualize_overall_crosslingual_consistency(args):
    model_name = args.model_name
    modified_model_name = args.modified_model_name
    reg_pattern = f'analysis/evaluations/mlama-{model_name}-consistency*.pkl'
    modified_reg_pattern = f'analysis/evaluations/mlama-{modified_model_name}-consistency*.pkl'
    rankC_lang_scores = dict()
    accuracy_lang_scores = dict()
    modified_rankC_lang_scores = dict()
    modified_accuracy_lang_scores = dict()
    layer_to_pick = args.layer_to_pick

    # base model
    for filename in glob(reg_pattern):
        if 'intervention' in filename:
            continue
        prefix = filename.rsplit('.',  1)[0]
        lang_pair = prefix.split('_')[-1].replace('matrix-','').replace('embedded-','')
        with open(filename, 'rb') as f:
            obj_dict = pickle.load(f)
        rankC_score = float(compute_rankc(obj_dict[layer_to_pick]['cs_rank_preds'], obj_dict[layer_to_pick]['mono_rank_preds']))
        accuracy_score = float(compute_accuracy_top_n(obj_dict[layer_to_pick]['cs_rank_preds'], obj_dict[layer_to_pick]['mono_rank_preds']))
        rankC_lang_scores[lang_pair] = rankC_score
        accuracy_lang_scores[lang_pair] = accuracy_score


    # modified model
    if modified_model_name is not None:
        for filename in glob(modified_reg_pattern):
            prefix = filename.rsplit('.',  1)[0]
            lang_pair = prefix.split('_')[-1].replace('matrix-','').replace('embedded-','')
            with open(filename, 'rb') as f:
                obj_dict = pickle.load(f)
            rankC_score = float(compute_rankc(obj_dict[layer_to_pick]['cs_rank_preds'], obj_dict[layer_to_pick]['mono_rank_preds']))
            accuracy_score = float(compute_accuracy_top_n(obj_dict[layer_to_pick]['cs_rank_preds'], obj_dict[layer_to_pick]['mono_rank_preds']))
            modified_rankC_lang_scores[lang_pair] = rankC_score
            modified_accuracy_lang_scores[lang_pair] = accuracy_score


    #rankc sorting
    rankC_lang_scores_sorted = sorted(rankC_lang_scores, key=rankC_lang_scores.get, reverse=True)
    
    rankC_lang_scores_sorted_values = [rankC_lang_scores[key] for key in rankC_lang_scores_sorted]
    if modified_model_name is not None:
        modified_rankC_lang_scores_sorted_values = [modified_rankC_lang_scores[key] for key in rankC_lang_scores_sorted]
    
    rankC_lang_scores_average = sum(rankC_lang_scores_sorted_values)/len(rankC_lang_scores_sorted_values)

    if modified_model_name is not None:
        fig = go.Figure([go.Bar(x=rankC_lang_scores_sorted, y=rankC_lang_scores_sorted_values, name=model_name), go.Bar(x=rankC_lang_scores_sorted, y=modified_rankC_lang_scores_sorted_values, name=modified_model_name)])
    else:
        fig = go.Figure([go.Bar(x=rankC_lang_scores_sorted, y=rankC_lang_scores_sorted_values, name=model_name)])
    
    fig.update_layout(xaxis_title="Codemixed Language", yaxis_title="RankC (0-1)", title_text=f"Overall Crosslingual Consistency of {model_name}")
    fig.add_shape(
        type='line',
        x0=-0.5, x1=len(rankC_lang_scores_sorted_values) - 0.5,  # Adjust x0 and x1 to cover the full x-axis range
        y0=rankC_lang_scores_average, y1=rankC_lang_scores_average,
        line=dict(color='black', dash='dash')  # 'dash' for a dotted line; change to 'solid' for a solid line
    )
    fig.show()


    #accuracy sorting
    accuracy_lang_scores_sorted = sorted(accuracy_lang_scores, key=accuracy_lang_scores.get, reverse=True)
    accuracy_lang_scores_sorted_values = [accuracy_lang_scores[key] for key in accuracy_lang_scores_sorted]
    if modified_model_name is not None:
        modified_accuracy_lang_scores_sorted_values = [modified_accuracy_lang_scores[key] for key in accuracy_lang_scores_sorted]
    
    accuracy_lang_scores_average = sum(accuracy_lang_scores_sorted_values)/len(accuracy_lang_scores_sorted_values)

    if modified_model_name is not None:
        fig = go.Figure([go.Bar(x=accuracy_lang_scores_sorted, y=accuracy_lang_scores_sorted_values, name=model_name), go.Bar(x=accuracy_lang_scores_sorted, y=modified_accuracy_lang_scores_sorted_values, name=modified_model_name)])
    else:
        fig = go.Figure([go.Bar(x=accuracy_lang_scores_sorted, y=accuracy_lang_scores_sorted_values, name=model_name)])
    
    fig.update_layout(xaxis_title="Codemixed Language", yaxis_title="Accuracy (0-1)", title_text=f"Overall Crosslingual Consistency of {model_name}")
    fig.add_shape(
        type='line',
        x0=-0.5, x1=len(accuracy_lang_scores_sorted_values) - 0.5,  # Adjust x0 and x1 to cover the full x-axis range
        y0=accuracy_lang_scores_average, y1=accuracy_lang_scores_average,
        line=dict(color='black', dash='dash')  # 'dash' for a dotted line; change to 'solid' for a solid line
    )
    fig.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--modified_model_name', type=str, default=None, help='A model variant of model_name')
    parser.add_argument('--figures_folder', type=str)
    parser.add_argument('--layer_to_pick', type=int, help="layer that we want to evaluate the consistency")

    args = parser.parse_args()
    visualize_overall_crosslingual_consistency(args)