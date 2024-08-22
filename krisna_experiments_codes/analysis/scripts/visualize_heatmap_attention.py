model = 'xlm-v-base'

import pdb
import pickle
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import numpy as np

inp_file = '../factors/mlama-xlm-v-attention-scores_matrix-en-embedded-ta-cm-attentions.pkl'

lang_pair = 'En-Ta'
plot_title = f'Average Attention Scores in {model} {lang_pair}'
filepath = f'../../figures/{model}-mlama-ta-en-cm-layerwise-attention.png'
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