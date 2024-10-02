

from argparse import ArgumentParser
import pickle
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import numpy as np
import einops
pio.kaleido.scope.mathjax = None

def visualize_attention_weights_heatmap(args):
    model, alt_model = args.model, args.model_2
    lang_pair = args.lang_pair
    lang1, lang2 = lang_pair.split('-')
    main_inp_file = f'{args.attention_folder}/mlama-{model}-attention-scores_matrix-{lang1}-embedded-{lang2}-{args.context_type}-attentions.pkl'
    contrasting_inp_file = f'{args.attention_folder}/mlama-{alt_model}-attention-scores_matrix-{lang1}-embedded-{lang2}-{args.context_type_2}-attentions.pkl' 
    lang_pair_title = args.lang_pair_title
    model_name = model if not args.use_second_model else alt_model
    plot_title = f'Average Attention Scores Difference in {model_name} for {lang_pair_title}'

    filepath = f'{args.figures_folder}/{alt_model}-mlama-{lang_pair}-difference-attention-aggregated.pdf'
    layer_indices = []
    head_indices = []
    attention_weights = []
    contrasting_attention_weights = []
    gathered_all_heads = False


    with open(main_inp_file, 'rb') as f:
        main_inp_dict = pickle.load(f)

    with open(contrasting_inp_file, 'rb') as f:
        contrasting_inp_dict = pickle.load(f) 

    all_layers = [key for key in main_inp_dict.keys()]
    group_size = len(all_layers)//3

    # main input
    for key, heads in  main_inp_dict.items():
        layer_indices.append(key)
        layer_attention_weights = []
        for head_idx, attn in heads.items():
            if not gathered_all_heads:
                head_indices.append(head_idx)
            layer_attention_weights.append(round(float(attn),2))
        gathered_all_heads = True
        attention_weights.append(layer_attention_weights) # layer*head
    
    # alternative input   
    for key, heads in  contrasting_inp_dict.items():
        layer_attention_weights = []
        for head_idx, attn in heads.items():
            layer_attention_weights.append(round(float(attn),2))
        contrasting_attention_weights.append(layer_attention_weights) # layer*headattention_weights = np.array(np.transpose(attention_weights))

    attention_weights = np.array(attention_weights)
    contrasting_attention_weights = np.array(contrasting_attention_weights)

    attention_weights_difference = contrasting_attention_weights-attention_weights

    if args.compress_layers:
        # aggregate
        if len(all_layers) % 2 == 0:
            aggregated_attention_weights = einops.reduce(attention_weights_difference, '(l b) h ->l h', 'mean', l=3, b=group_size)
        else:
            early_layers_group = attention_weights_difference[0:group_size,:]
            mid_layers_group = attention_weights_difference[group_size:2*group_size, :]
            later_layers_group = attention_weights_difference[2*group_size:, :]

            early_mean = early_layers_group.mean(axis=0)
            mid_mean = mid_layers_group.mean(axis=0)
            later_mean = later_layers_group.mean(axis=0)

            aggregated_attention_weights = np.stack([early_mean, mid_mean, later_mean], axis=0)
            
        aggregated_attention_weights = np.round(aggregated_attention_weights, decimals=3)
        grouped_layer_indices = [i for i in range(len(aggregated_attention_weights))]

        layer_indices = grouped_layer_indices
        attention_weights_difference = aggregated_attention_weights


    # row: head
    # col: layer
    attention_weights_difference = np.transpose(attention_weights_difference)
    fig = px.imshow(attention_weights_difference,
                    labels=dict(x="Encoder Layer", y="Head"),
                    x=layer_indices,
                    y=head_indices,
                    color_continuous_scale='rdbu',
                    title=plot_title,
                    range_color=[-0.6,0.6],
                    text_auto=True
                )

    fig.update_xaxes(
        tickmode='linear',  # Ensures that ticks are placed at every integer value
        tickvals=layer_indices,  # Set tick values to cover all columns
        ticktext=[str(i) for i in layer_indices]  # Customize tick labels if needed
    )

    fig.update_yaxes(
        tickmode='linear',  # Ensures that ticks are placed at every integer value
        tickvals=head_indices,  # Set tick values to cover all rows
        ticktext=[str(i) for i in head_indices]  # Customize tick labels if needed
    )

    fig.show()

    pio.write_image(fig, filepath)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--model_2', type=str)
    parser.add_argument('--attention_folder', type=str, help='Folder where we store attention weights for every head', default='../factors')
    parser.add_argument('--lang_pair', type=str, help='Language pair')
    parser.add_argument('--lang_pair_title', type=str, help='Language pair that is used for plot title')
    parser.add_argument('--figures_folder', type=str, default='../../figures', help="Folder where we want to store figures output")
    parser.add_argument('--context_type', type=str, choices=['cm', 'mono'])
    parser.add_argument('--context_type_2', type=str, choices=['cm', 'mono'])
    parser.add_argument('--use_second_model', action='store_true', default=False, help='use model_2 instead in the plot title')
    parser.add_argument('--compress_layers', action='store_true', default=False, help='Flag to group layers into three sections (early, mid, later)')
    


    args = parser.parse_args()
    visualize_attention_weights_heatmap(args)
    