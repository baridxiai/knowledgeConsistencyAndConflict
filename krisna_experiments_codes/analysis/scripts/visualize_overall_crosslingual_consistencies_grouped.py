from glob import glob
import pickle
from argparse import ArgumentParser
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
import json
from eval import compute_rankc, compute_accuracy_top_n

pio.kaleido.scope.mathjax = None


def visualize_overall_crosslingual_consistency_categorized(args):
    model_name = args.model_name
    group_file_suffix = args.grouping_file.rsplit('/')[-1]
    
    category_name = group_file_suffix.replace('.json', '')
    modified_model_name = args.modified_model_name

    with open(args.grouping_file, 'r') as file:
        categorization_groups_dict = json.load(file)
    lang_to_category_val_dict = dict()
    category_scores = dict()
    
    if modified_model_name is not None:
        modified_category_scores = dict()
    
    for subcategory_name, langs in categorization_groups_dict.items():
        for lang in langs:
            lang_to_category_val_dict[lang] = subcategory_name
        category_scores[subcategory_name] = {
            'rankC': [],
            'accuracy': []
        }
        if modified_model_name is not None:
            modified_category_scores[subcategory_name] = {
                'rankC': [],
                'accuracy': []
            }
    

    
    reg_pattern = f'../evaluations/mlama-{model_name}-consistency*.pkl'
    modified_reg_pattern = f'../evaluations/mlama-{modified_model_name}-consistency*.pkl'
    layer_to_pick = args.layer_to_pick

    # base model
    for filename in glob(reg_pattern):
        if 'intervention' in filename:
            continue
        prefix = filename.rsplit('.',  1)[0]
        lang_pair = prefix.split('_')[-1].replace('matrix-','').replace('embedded-','')
        lang1, lang2 = lang_pair.split('-')

        with open(filename, 'rb') as f:
            obj_dict = pickle.load(f)
        rankC_score = float(compute_rankc(obj_dict[layer_to_pick]['cs_rank_preds'], obj_dict[layer_to_pick]['mono_rank_preds']))
        accuracy_score = float(compute_accuracy_top_n(obj_dict[layer_to_pick]['cs_rank_preds'], obj_dict[layer_to_pick]['mono_rank_preds']))

        corresponding_category = lang_to_category_val_dict[lang2]
        category_scores[corresponding_category]['rankC'].append(rankC_score)
        category_scores[corresponding_category]['accuracy'].append(accuracy_score)

    aggregated_category_scores = {
        "rankC": {
            'mean': [],
            'std': [],
            'category': []
        },
        "accuracy": {
            'mean': [],
            'std': [],
            'category': []
        }
    }

    # aggregate scores
    for category, metric_scores_dict in category_scores.items():
        for metric, scores in metric_scores_dict.items():
            metric_mean = np.mean(scores)
            metric_std = np.std(scores)
            aggregated_category_scores[metric]['mean'].append(metric_mean)
            aggregated_category_scores[metric]['std'].append(metric_std)
            aggregated_category_scores[metric]['category'].append(category)

    # modified model
    if modified_model_name is not None:
        modified_model_layer_to_pick = layer_to_pick
        if args.modified_model_layer_to_pick is not None:
            modified_model_layer_to_pick = args.modified_model_layer_to_pick
        for filename in glob(modified_reg_pattern):
            if 'intervention' in filename:
                continue
            prefix = filename.rsplit('.',  1)[0]
            lang_pair = prefix.split('_')[-1].replace('matrix-','').replace('embedded-','')
            lang1, lang2 = lang_pair.split('-')
        
            with open(filename, 'rb') as f:
                obj_dict = pickle.load(f)
            rankC_score = float(compute_rankc(obj_dict[modified_model_layer_to_pick]['cs_rank_preds'], obj_dict[modified_model_layer_to_pick]['mono_rank_preds']))
            accuracy_score = float(compute_accuracy_top_n(obj_dict[modified_model_layer_to_pick]['cs_rank_preds'], obj_dict[modified_model_layer_to_pick]['mono_rank_preds']))
            corresponding_category = lang_to_category_val_dict[lang2]
            modified_category_scores[corresponding_category]['rankC'].append(rankC_score)
            modified_category_scores[corresponding_category]['accuracy'].append(accuracy_score)

        modified_aggregated_category_scores = {
            "rankC": {
                'mean': [],
                'std': [],
                'category': []
            },
            "accuracy": {
                'mean': [],
                'std': [],
                'category': []
            }
        }

        # aggregate scores
        for category, metric_scores_dict in modified_category_scores.items():
            for metric, scores in metric_scores_dict.items():
                metric_mean = np.mean(scores)
                metric_std = np.std(scores)
                modified_aggregated_category_scores[metric]['mean'].append(metric_mean)
                modified_aggregated_category_scores[metric]['std'].append(metric_std)
                modified_aggregated_category_scores[metric]['category'].append(category)

    fig_model_name = args.model_name
    if args.modified_model_name is not None:
        fig_model_name = args.modified_model_name

    print("RankC Scores...")
    if modified_model_name is not None:
        fig = go.Figure([go.Bar(x=aggregated_category_scores['rankC']['category'], y=aggregated_category_scores['rankC']['mean'], name=model_name), go.Bar(x=modified_aggregated_category_scores['rankC']['category'], y=modified_aggregated_category_scores['rankC']['mean'], name=modified_model_name)])
    else:
        fig = go.Figure([go.Bar(x=aggregated_category_scores['rankC']['category'], y=aggregated_category_scores['rankC']['mean'], name=model_name)])
    fig.update_layout(xaxis_title=args.category_name, yaxis_title="RankC (0-1)", title_text=f"Overall Crosslingual Consistency of {args.model_name_title}",
    )
    fig.show()
    rankc_filepath = f"{args.figures_folder}/mlama-{args.model_name_title}-overall-crosslingual-consistency-rankC_{category_name}.pdf"
    pio.full_figure_for_development(fig, warn=False)
    fig.write_image(rankc_filepath,engine="kaleido")
    print(model_name)
    model_std_rankc_std_str = ""
    print("Standard Deviation For Each Category")
    for category, std in zip(aggregated_category_scores['rankC']['category'], aggregated_category_scores['rankC']['std']):
        model_std_rankc_std_str += f"{category}: {std}\n"
    print(model_std_rankc_std_str)

    if modified_model_name is not None:
        print(modified_model_name)
        model_std_rankc_std_str = ""
        print("Standard Deviation For Each Category")
        for category, std in zip(modified_aggregated_category_scores['rankC']['category'], modified_aggregated_category_scores['rankC']['std']):
            model_std_rankc_std_str += f"{category}: {std}\n"
        print(model_std_rankc_std_str)
    
    print("Accuracy Scores...")
    if modified_model_name is not None:
        fig = go.Figure([go.Bar(x=aggregated_category_scores['accuracy']['category'], y=aggregated_category_scores['accuracy']['mean'], name=model_name), go.Bar(x=modified_aggregated_category_scores['rankC']['category'], y=modified_aggregated_category_scores['rankC']['mean'], name=modified_model_name)])
    else:
        fig = go.Figure([go.Bar(x=aggregated_category_scores['accuracy']['category'], y=aggregated_category_scores['accuracy']['mean'], name=model_name)])
    fig.update_layout(xaxis_title=args.category_name, yaxis_title="Accuracy (0-1)", title_text=f"Overall Crosslingual Consistency of {args.model_name_title}",
    )
    fig.show()
    rankc_filepath = f"{args.figures_folder}/mlama-{args.model_name_title}-overall-crosslingual-consistency-accuracy_{category_name}.pdf"
    pio.full_figure_for_development(fig, warn=False)
    fig.write_image(rankc_filepath,engine="kaleido")
    print(model_name)
    model_std_acc_std_str = ""
    print("Standard Deviation For Each Category")
    for category, std in zip(aggregated_category_scores['accuracy']['category'], aggregated_category_scores['accuracy']['std']):
        model_std_acc_std_str += f"{category}: {std}\n"
    print(model_std_acc_std_str)
    
    if modified_model_name is not None:
        print(modified_model_name)
        model_std_acc_std_str = ""
        print("Standard Deviation For Each Category")
        for category, std in zip(modified_aggregated_category_scores['accuracy']['category'], modified_aggregated_category_scores['accuracy']['std']):
            model_std_acc_std_str += f"{category}: {std}\n"
        print(model_std_acc_std_str)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_name_title', type=str)
    parser.add_argument('--grouping_file', type=str, help='json file containing categorization of all languages')
    parser.add_argument('--category_name', type=str)
    parser.add_argument('--modified_model_name', type=str, default=None, help='A model variant of model_name')
    parser.add_argument('--figures_folder', type=str)
    parser.add_argument('--layer_to_pick', type=int, help="layer that we want to evaluate the consistency")
    parser.add_argument('--modified_model_layer_to_pick', type=int, default=None, help="layer that we want to evaluate the consistency") 

    args = parser.parse_args()
    visualize_overall_crosslingual_consistency_categorized(args)