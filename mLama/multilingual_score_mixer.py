# -*- coding: utf-8 -*-
# code warrior: Barid
import os

# import tensorflow as tf
from transformers import AutoTokenizer, AutoModel
import torch
import csv
import numpy as np
import datasets
import sys
from sklearn.decomposition import PCA
from sklearn import decomposition

model_names = [
    "facebook/xlm-roberta-xl",
    "facebook/xlm-v-base",
    "xlm-roberta-base",
    "xlm-roberta-large",
    "microsoft/xlm-align-base",
    "bert-base-multilingual-cased",
]
gpu_index = "cuda:1"
mode = "training"
LANG_CHECK = False
m_lama = datasets.load_dataset("m_lama")["test"].shuffle(seed=42)
XNLI_LANGS = ["en", "hi"]
MIN_COUNT = 2
# XNLI_LANGS = []
KEY_LIST = []
LANG_LIST = []
KEY_LIST_output = []
LANG_LIST_output = []


def percent(current, total):

    if current == total - 1:
        current = total
    bar_length = 20
    hashes = "#" * int(current / total * bar_length)
    spaces = " " * (bar_length - len(hashes))
    sys.stdout.write(
        "\rPercent: [%s] %d%%" % (hashes + spaces, int(100 * current / total))
    )


def bi_mixer(data):
    """
        "Paris is the capital of [MASK]"

        Change the sub or Paris to another language.

        data[0] and [1] are formated parallel data.
    """
    d1, d2 = data[0], data[1]
    org_1, sub_label_1, obj_label_1, langauge_1, key_1, predicate_id_1 = d1
    org_2, sub_label_2, obj_label_2, langauge_2, key_2, predicate_id_2 = d2
    org_1_mixed = org_1.replace(sub_label_1, sub_label_2)
    org_2_mixed = org_2.replace(sub_label_2, sub_label_1)
    if langauge_1 == "en":
        return [
            d1,
            [org_1_mixed, sub_label_1, obj_label_1, langauge_1, key_1, predicate_id_1],
        ], [
            d2,
            [org_2_mixed, sub_label_2, obj_label_2, langauge_2, key_2, predicate_id_2],
        ]
    else:
        return [
            d2,
            [org_2_mixed, sub_label_2, obj_label_2, langauge_2, key_2, predicate_id_2],
        ], [
            d1,
            [org_1_mixed, sub_label_1, obj_label_1, langauge_1, key_1, predicate_id_1],
        ]

def m_lama_cross_robust_parser(data):
    langauge = data["language"]
    obj_label = data["obj_label"]
    sub_label = data["sub_label"]
    template = data["template"]
    # masked = template.replace("[X]", sub_label).replace("[Y]", "<mask>")
    org = template.replace("[X]", sub_label).replace("[Y]", obj_label)
    predicate_id = str(data["predicate_id"])
    key = str(data["obj_uri"])+str(data["sub_uri"]) + "@" + str(data["predicate_id"])
    return predicate_id,key,langauge, org, sub_label,obj_label
def dictList_dictTFTensor(dict_input):
    tftensor = dict()
    for k, v in dict_input.items():
        tftensor[k] = torch.tensor(v).to(gpu_index)
    return tftensor
def query_answer_parser(c_s_o, tokenizer):
    context_batch = []
    obj_positions_batch = []
    lang_batch = []
    per_token = []
    for k, cso in enumerate(c_s_o):
        context_text, sub,obj, lang, k,_ = cso
        tok_obj_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if 6 in tok_obj_text:
            tok_obj_text.remove(6)
        doc_tokens = tokenizer.encode(context_text)
        if tok_obj_text[0] in doc_tokens:
            test = []
            start_p = doc_tokens.index(tok_obj_text[0])
            for new_start in range(start_p, len(doc_tokens)):
                for new_end in range(len(doc_tokens)-1, new_start - 1, -1):
                    text_span = doc_tokens[new_start : new_end + 1]
                    if text_span  == tok_obj_text and new_end >= new_start:
                        test.append((new_start, new_end))
            if len(test) > 0:
                obj_positions_batch.append(test[-1])
                context_batch.append(context_text)
                per_token.append(test[-1][1] + 1 - test[-1][0] )
                if LANG_CHECK:
                    language_id = tokenizer.lang2id[lang]
                    lang_batch.append(language_id)
                KEY_LIST.append(k)
                LANG_LIST.append(lang)
    masked_context_batch = tokenizer(
        [q for k, q in enumerate(context_batch)], padding=True
    )
    langs = []
    if LANG_CHECK:
        for k,v in enumerate(masked_context_batch["input_ids"]):
            langs.append([lang_batch[k]] * len(v))
        masked_context_batch["langs"] = langs
    if mode == "training":
        for k, v in enumerate(context_batch):
            for i in range(obj_positions_batch[k][0],obj_positions_batch[k][1]+1):
                masked_context_batch["input_ids"][k][i] = tokenizer.mask_token_id
    return masked_context_batch, obj_positions_batch,per_token
def get_rep(sample_id, model, tokenizer):
    inputs = dictList_dictTFTensor(sample_id)
    if "langs" in inputs:
        re = model(input_ids=inputs["input_ids"],langs=inputs["langs"], output_hidden_states=True)
    else:
        re = model(**inputs, output_hidden_states=True)
    return re.hidden_states[1:]
    # return re.hidden_states
def get_position_rep(rep_batch,posistion):
    layer_rep = []
    posistion_rep = []
    for j,rep_temp in enumerate(rep_batch):
        for k, rep in enumerate(rep_temp):
            rep = rep.cpu().detach().numpy()
            try:
                p_rep = np.mean(rep[posistion[k][0]:posistion[k][1]+1],0)
                # p_rep = np.mean(rep[k],0)
                posistion_rep.append(p_rep)
                if k==0:
                    KEY_LIST_output.append(KEY_LIST[j])
                    LANG_LIST_output.append(LANG_LIST[j])
            except Exception:
                print(rep_batch[k])
                print(posistion[k])
        layer_rep.append(np.array(posistion_rep).reshape((len(posistion_rep),-1)))
        posistion_rep = []
    return layer_rep
def maximum_explainable_variance(batch,model,tokenizer, row, bias =None):
    with torch.no_grad():
        if len(batch) > 60:
            count = int(len(batch)/60)
            for i in range(0,count):
                mini_batch,positions,per_token = query_answer_parser(batch[i*60:i*60+60],tokenizer)
                mini_rep = get_rep(mini_batch,model,tokenizer)
                if i == 0:
                    layer_rep = get_position_rep(mini_rep,positions)
                else:
                    mini_layer_rep = get_position_rep(mini_rep,positions)
                    for k,v in enumerate(layer_rep):
                        layer_rep[k] = np.concatenate((layer_rep[k],mini_layer_rep[k]),axis=0)
                if 'per_token' in row:
                    row["per_token"] = row["per_token"] + per_token
                else:
                    row["per_token"] = per_token
        else:
            batch,positions,per_token = query_answer_parser(batch,tokenizer)
            rep = get_rep(batch,model,tokenizer)
            layer_rep = get_position_rep(rep,positions)
            if 'per_token' in row:
                row["per_token"] = row["per_token"] + per_token
            else:
                row["per_token"] = per_token
        # return IsoScore(obj_rep)
    for k, v in enumerate(layer_rep):
        pca = decomposition.PCA(1)
        pca.fit(v)
        # k = str(k)
        k = str(k+1)
        mev = min(pca.explained_variance_ratio_[0],1.0)
        if bias is not None:
            mev -=  bias[k][0]
            mev = min(mev,1.0)
        if k in row:
            row[k].append(mev)
        else:
            row[k] = [mev]
        # break ###
    return row
m_lama_organized =dict()
n = 0
m_lama_bias = dict()
temp = []
for data in m_lama:
    n +=1
    percent(n,830000)
    predicate_id,key,langauge, org, sub_label,obj_label = m_lama_cross_robust_parser(data)
    if langauge in XNLI_LANGS or len(XNLI_LANGS) == 0:
        v = [org,sub_label,obj_label,langauge,key,predicate_id]
        temp.append(v)
        if predicate_id in m_lama_bias:
            m_lama_bias[predicate_id].append(v)
        else:
           m_lama_bias[predicate_id] = [v]
        if key in m_lama_organized:
            m_lama_organized[key].append(v)
        else:
            m_lama_organized[key] = [v]
re_score = []
re_bias = []
fieldnames = ['model','mode','per_token'] + list(map(lambda w: w, map(str, range(1, 36 +1))))
# fieldnames = ['model','mode','per_token'] + list(map(lambda w: w, map(str, range(0,1))))
writer = csv.DictWriter(open("./multilingual_score", 'w'), fieldnames=fieldnames)
writer.writeheader()

for model_name in model_names:
    variance_explained_scoreEN = dict()
    variance_explained_scoreX = dict()
    variance_explained_bias = dict()
    xlmR_tokenizer = AutoTokenizer.from_pretrained(model_name)
    xlmR = AutoModel.from_pretrained(model_name).to(gpu_index)
    xlmR_masked_cross_score = []
    n = 0
    m_lama_bias_length = len(m_lama_bias)
    print("###computing variance_explained_bias ####")
    # for k, v in m_lama_bias.items():
    #     n += 1
    #     temp_dict = {}
    #     percent(n, m_lama_bias_length)
    #     variance_explained_bias[k] = maximum_explainable_variance(
    #         v, xlmR, xlmR_tokenizer, temp_dict
    #     )

    n = 0
    m_lama_organized_length = len(m_lama_organized)
    print("###computing variance_explained_score ####")
    for k, v in m_lama_organized.items():
        try:
            # predicate_id,key,langauge, org, sub_label,obj_label
            key, predicate_id = k.split("@")
            if (
                len(v) == 2
                and v[0][-3] != v[1][-3]
                and variance_explained_bias[predicate_id] is not None
            ):
                n += 1
                percent(n, m_lama_organized_length)
                v0, v1 = bi_mixer(v)
                variance_explained_scoreEN = maximum_explainable_variance(
                    v0,
                    xlmR,
                    xlmR_tokenizer,
                    variance_explained_scoreEN,
                )
                variance_explained_scoreX = maximum_explainable_variance(
                    v1,
                    xlmR,
                    xlmR_tokenizer,
                    variance_explained_scoreX,
                )
        except Exception:
            pass
    # xlmR_masked_cross_score = np.mean(xlmR_masked_cross_score)
    # stat_lang = {s:0 for s in XNLI_LANGS}
    temp = []
    xlmR_raw_bias = []
    xlmR_masked_bias = []
    n = 0
    xlmR_masked_bias = np.mean(xlmR_masked_bias)
    re_bias.append([model_name, str(xlmR_masked_bias)])
    re_score.append([model_name, str(xlmR_masked_cross_score)])
    for k, v in variance_explained_scoreEN.items():
        variance_explained_scoreEN[k] = str(np.nanmean(variance_explained_scoreEN[k]))
        variance_explained_scoreX[k] = str(np.nanmean(variance_explained_scoreX[k]))
    variance_explained_scoreEN["mode"] = "scoreEN"
    variance_explained_scoreEN["model"] = model_name
    variance_explained_scoreX["mode"] = "scoreX"
    variance_explained_scoreX["model"] = model_name
    print("############score################")
    print(variance_explained_scoreEN)
    writer.writerow(variance_explained_scoreEN)
    print(variance_explained_scoreX)
    writer.writerow(variance_explained_scoreX)
    del xlmR_tokenizer, xlmR
del writer
print("###################################################")
print("Num of examination:" + str(n))
