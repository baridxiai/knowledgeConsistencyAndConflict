from glob import glob
import pickle
import plotly.graph_objects as go
import csv

model_name = 'xlm-align'
reg_pattern = f'../evaluations/mlama-{model_name}-consistency*.pkl'
csv_file = '../evaluations/mlama53_acc.csv'
lang_scores = dict()
layer_to_pick = 23
embedded_langs = []
for filename in glob(reg_pattern):
    prefix = filename.rsplit('.',  1)[0]
    lang_pair = prefix.split('_')[-1].replace('matrix-','').replace('embedded-','')
    embedded_lang = lang_pair.split('-')[-1]
    embedded_langs.append(embedded_lang)

missing_langs = []
with open(csv_file, newline='') as f:
    reader = csv.reader(f, delimiter=',')
    for idx, row in enumerate(reader):
        if idx == 0:
            col = row[3:]
            for lang in col:
                if lang not in embedded_langs:
                    missing_langs.append(lang)
print(missing_langs)
