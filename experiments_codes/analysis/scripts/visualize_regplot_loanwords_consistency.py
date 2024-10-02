import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from glob import glob
from sklearn.linear_model import LinearRegression
import pickle
import re
from datasets import load_dataset
from tqdm import tqdm
from utils import load_mlama

model_names = ['mt0-base', 'mt0-large', 'xlm-r', 'xlm-r-large', 'xlm-v']
probed_layers = [11, 23, 11, 23, 11]
lang_pairs = []

#hacky way to get language pairs
reg_pattern = f'../evaluations/mlama-{model_names[0]}-consistency*.pkl'
for filename in glob(reg_pattern):
    prefix = filename.rsplit('.',  1)[0]
    lang_pair = prefix.split('_')[-1].replace('matrix-','').replace('embedded-','')
    lang_pairs.append(lang_pair)

# get all object tokens statistics
mlama_crosslingual_borrowed_words_ratio = dict()
for lang_pair in lang_pairs:
    matrix_lang, embedded_lang = lang_pair.split('-')
    mlama_dataset = load_mlama(matrix_lang, embedded_lang)
    average_borrowed_words_ratio = []
    for val in mlama_dataset:
        ratio = 0
        for token in val['obj_label_cross_lang']:
            if token in val['obj_label']:
                ratio += 1
        average_borrowed_words_ratio.append(ratio/len(val['obj_label']))
    mlama_crosslingual_borrowed_words_ratio[lang_pair] = sum(average_borrowed_words_ratio)/len(average_borrowed_words_ratio)

borrowed_words_ratios = []
rankC_scores = []

for idx, model_name in enumerate(model_names):
    reg_pattern = f'../evaluations/mlama-{model_name}-consistency*.pkl'
    for filename in glob(reg_pattern):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
            prefix = filename.rsplit('.',  1)[0]
            lang_pair = prefix.split('_')[-1].replace('matrix-','').replace('embedded-','')
            rankC_score  = obj[probed_layers[idx]]['rankC']
            borrowed_words_ratio = mlama_crosslingual_borrowed_words_ratio[lang_pair]
            rankC_scores.append(rankC_score)
            borrowed_words_ratios.append(borrowed_words_ratio)

df = pd.DataFrame({'rankC': rankC_scores, 'ratio': borrowed_words_ratios})


# Create a scatter plot with Plotly Express
fig = px.scatter(df, x='ratio', y='rankC', title="Loarnwords Ratio vs Consistency")

# Fit a linear regression model
model = LinearRegression()
X = df[['ratio']]
model.fit(X, df['rankC'])
y_pred = model.predict(X)

# Add the regression line to the plot
fig.add_trace(go.Scatter(
    x=df['ratio'],
    y=y_pred,
    mode='lines',
    line=dict(color='red', width=2)
))

# Update layout
fig.update_layout(
    xaxis_title='Loanwords Ratio',
    yaxis_title='RankC(0-1)'
)

# Show the plot
fig.show()