import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from glob import glob
from sklearn.linear_model import LinearRegression
import pickle
import re
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from scipy.stats import spearmanr


def add_punctuations_whitespace(s: str) -> str:
    """
    To add whitespace in-between the token and punctuation to enable the these punctuations to be tokenized separately with words
    @param s(str): input string that we want to tokenize
    """

    s = re.sub('([.,!?():;])', r' \1 ', s)
    s = re.sub('\s{2,}', ' ', s)
    return s 

m_lama = load_dataset("m_lama", trust_remote_code=True)["test"].shuffle(seed=42)

def load_mlama(matrix_lang: str, target_lang: str):
    """
    Load paralllel matrix_lang-target_lang sentences from mLAMA dataset
    @param matrix_lang: matrix language
    @param embedded_lang: embeded language
    """
    
    
    m_lama_dict = dict()

    for data in tqdm(m_lama):
        m_lama_id = f'{data["sub_uri"]}-{data["obj_uri"]}@{data["predicate_id"]}'
        if data['language'] == matrix_lang:
            if m_lama_id not in m_lama_dict:
                m_lama_dict[m_lama_id] = dict()
            m_lama_dict[m_lama_id]['template'] = add_punctuations_whitespace(data['template'])
            m_lama_dict[m_lama_id]['subj_label_same_lang'] = add_punctuations_whitespace(data['sub_label'])
            m_lama_dict[m_lama_id]['obj_label'] = add_punctuations_whitespace(data['obj_label'])    
        elif data['language'] == target_lang:
            if m_lama_id not in m_lama_dict:
                m_lama_dict[m_lama_id] = dict()
            m_lama_dict[m_lama_id]['subj_label_cross_lang'] = add_punctuations_whitespace(data['sub_label'])
            m_lama_dict[m_lama_id]['obj_label_cross_lang'] = add_punctuations_whitespace(data['obj_label'])
        
    mlama_instances = [instance for instance in m_lama_dict.values() if 'subj_label_cross_lang' in instance and 'subj_label_same_lang' in instance] # filter out any subject that doesn't have its parallel subject 
    return mlama_instances

base_model_name = 'xlm-roberta-base' 
expanded_model_name = 'facebook/xlm-v-base' 

base_model_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
expanded_model_tokenizer = AutoTokenizer.from_pretrained(expanded_model_name)
lang_pairs = []

#hacky way to get language pairs
reg_pattern = f'../evaluations/mlama-xlm-r-consistency*.pkl'
for filename in glob(reg_pattern):
    prefix = filename.rsplit('.',  1)[0]
    lang_pair = prefix.split('_')[-1].replace('matrix-','').replace('embedded-','')
    lang_pairs.append(lang_pair)

# get all subject tokens statistics
mlama_crosslingual_token_ratio = dict()
for lang_pair in lang_pairs:
    matrix_lang, embedded_lang = lang_pair.split('-')
    mlama_dataset = load_mlama(matrix_lang, embedded_lang)
    average_token_ratio_per_lang = []
    for row in mlama_dataset:
        subj_entity=row['subj_label_cross_lang']
        tokenized_subj_base = base_model_tokenizer(subj_entity)['input_ids']
        tokenized_subj_expanded = expanded_model_tokenizer(subj_entity)['input_ids']
        tokenized_subj_expanded = [token for token in tokenized_subj_expanded if expanded_model_tokenizer.decode(token).strip() != '']
        average_token_ratio_per_lang.append(len(tokenized_subj_expanded)/len(tokenized_subj_base))
    mlama_crosslingual_token_ratio[lang_pair] = sum(average_token_ratio_per_lang)/len(average_token_ratio_per_lang)


probed_layer = 11

base_model_rankC_dict = dict()
expanded_model_rankC_dict = dict()

#base model
reg_pattern = f'../evaluations/mlama-xlm-r-consistency*.pkl'
for filename in glob(reg_pattern):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
        prefix = filename.rsplit('.',  1)[0]
        lang_pair = prefix.split('_')[-1].replace('matrix-','').replace('embedded-','')
        rankC_score  = obj[probed_layer]['rankC']
        base_model_rankC_dict[lang_pair] = rankC_score

#expanded model
reg_pattern = f'../evaluations/mlama-xlm-v-consistency*.pkl'
for filename in glob(reg_pattern):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
        prefix = filename.rsplit('.',  1)[0]
        lang_pair = prefix.split('_')[-1].replace('matrix-','').replace('embedded-','')
        rankC_score  = obj[probed_layer]['rankC']
        expanded_model_rankC_dict[lang_pair] = rankC_score

all_token_ratios = []
all_rankC_diffs = []

for key, val in mlama_crosslingual_token_ratio.items():
    all_token_ratios.append(val)
    all_rankC_diffs.append(expanded_model_rankC_dict[key]-base_model_rankC_dict[key])


df = pd.DataFrame({'rankC Delta': all_rankC_diffs, 'ratio': all_token_ratios})


# Create a scatter plot with Plotly Express
fig = px.scatter(df, x='ratio', y='rankC Delta', title="Token Parity Ratio vs Consistency Difference")

# Fit a linear regression model
model = LinearRegression()
X = df[['ratio']]
model.fit(X, df['rankC Delta'])
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
    xaxis_title='Token Parity Ratio',
    yaxis_title='Delta RankC(0-1)'
)

# Show the plot
fig.show()

rho, p_value = spearmanr(all_token_ratios, all_rankC_diffs)
print(f"Spearman's rho: {rho:.3f}")
print(f"P-value: {p_value:.6f}")
