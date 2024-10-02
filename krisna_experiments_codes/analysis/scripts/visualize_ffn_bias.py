import plotly.graph_objects as go
import pickle
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import numpy as np
from argparse import ArgumentParser
pio.kaleido.scope.mathjax = None

def visualize_mlp_bias(args):
    model = args.model
    lang_pair = args.lang_pair
    lang_pair_title = args.lang_pair_title
    cm_inp_file = f'{args.ig2_folder}/mlama-{model}-{lang_pair}-cm-ig2.pkl'
    mono_inp_file = f'{args.ig2_folder}/mlama-{model}-{lang_pair}-mono-ig2.pkl'

    mono_label = 'en'
    cm_label = lang_pair

    plot_title = f'Average Activation Values\' Gradient Sum on Probability in {model} {lang_pair_title}'
    filepath = f'../../figures/{model}-mlama-{lang_pair}-ffn-bias.png'

    cm_inp_dict = None
    mono_inp_dict = None

    with open(cm_inp_file, 'rb') as f:
        cm_inp_dict = pickle.load(f)

    with open(mono_inp_file, 'rb') as f:
        mono_inp_dict = pickle.load(f)

    x_cm = []
    y_cm = []
    for layer, act_vals in cm_inp_dict.items():
        for act_val in act_vals:
            x_cm.append(layer)
            y_cm.append(float(act_val))

    x_mono = []
    y_mono = []
    for layer, act_vals in mono_inp_dict.items():
        for act_val in act_vals:
            x_mono.append(layer)
            y_mono.append(float(act_val))


    fig = go.Figure()

    fig.add_trace(go.Box(
        y=y_mono,
        x=x_mono,
        name=mono_label
    ))

    fig.add_trace(go.Box(
        y=y_cm,
        x=x_cm,
        name=cm_label
    ))

    fig.update_xaxes(
        tickmode='linear',  # Ensures that ticks are placed at every integer value
        tickvals=x_cm,  # Set tick values to cover all columns
        ticktext=[str(i) for i in x_cm]  # Customize tick labels if needed
    )

    fig.update_layout(
        yaxis_title='average IG^2 sum',
        xaxis_title='Encoder Layer',
        title=plot_title,
        boxmode='group' # group together boxes of the different traces for each value of x
    )
    fig.show()

    pio.write_image(fig, filepath)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--lang_pair', type=str, help='Language pair')
    parser.add_argument('--lang_pair_title', type=str, help='Language pair that is used for plot title')
    parser.add_argument('--ig2_folder', type=str, help='Which folder containing the ig2 scores of all mlp neurons', default='../factors')
    parser.add_argument('--figures_folder', type=str, default='../../figures', help="Folder where we want to store figures output")

    args = parser.parse_args()
    visualize_mlp_bias(args)