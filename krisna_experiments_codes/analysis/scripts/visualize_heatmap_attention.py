

from argparse import ArgumentParser
import pickle
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import numpy as np

def visualize_attention_weights_heatmap(args):
    model = args.model
    inp_file = f'{args.attention_folder}/mlama-{model}-attention-scores_matrix-en-embedded-ar-cm-attentions.pkl'

    lang_pair = args.lang_pair
    lang_pair_title = args.lang_pair_title
    plot_title = f'Average Attention Scores in {model} {lang_pair_title}'
    filepath = f'{args.figures_folder}/{model}-mlama-{lang_pair}-cm-layerwise-attention.png'
    layer_indices = []
    head_indices = []
    attention_weights = []
    gathered_all_heads = False

    with open(inp_file, 'rb') as f:
        inp_dict = pickle.load(f)

    for key, heads in  inp_dict.items():
        layer_indices.append(key)
        layer_attention_weights = []
        for head_idx, attn in heads.items():
            if not gathered_all_heads:
                head_indices.append(head_idx)
            layer_attention_weights.append(round(float(attn),2))
        gathered_all_heads = True
        attention_weights.append(layer_attention_weights)
    attention_weights = np.array(np.transpose(attention_weights))

    # row: layer
    # col: head
    fig = px.imshow(attention_weights,
                    labels=dict(x="Encoder_Layer", y="Head", color="Attention Scores"),
                    x=layer_indices,
                    y=head_indices,
                    color_continuous_scale='peach',
                    title=plot_title,
                    range_color=[0,1],
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
    parser.add_argument('--attention_folder', type=str, help='Folder where we store attention weights for every head', default='analysis/factors')
    parser.add_argument('--lang_pair', type=str, help='Language pair')
    parser.add_argument('--lang_pair_title', type=str, help='Language pair that is used for plot title')
    parser.add_argument('--figures_folder', type=str, default='figures', help="Folder where we want to store figures output")

    args = parser.parse_args()
    visualize_attention_weights_heatmap(args)
    