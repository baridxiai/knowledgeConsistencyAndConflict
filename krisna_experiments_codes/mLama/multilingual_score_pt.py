# -*- coding: utf-8 -*-
# code warrior: Barid
import os

# import tensorflow as tf
from transformers import AutoTokenizer, AutoModel, XLMRobertaModel
import torch
import csv
import numpy as np
import datasets
import sys
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn import decomposition
from numpy import save

model_names = [
    "facebook/xlm-roberta-xll",
    "facebook/xlm-roberta-xl",
    "xlm-mlm-100-1280",
    "xlm-roberta-base",
    "facebook/xlm-v-base",
    "xlm-roberta-large",
    "microsoft/xlm-align-base",
    "bert-base-multilingual-cased",
]
# model_names  = ["xlm-roberta-large","microsoft/xlm-align-base","bert-base-multilingual-cased"]
# model_names  = ["facebook/xlm-v-base"]
# model_names  = ["microsoft/xlm-align-base","xlm-mlm-100-1280","bert-base-multilingual-cased","facebook/xlm-v-base"]
# model_names  = ["xlm-mlm-xnli15-1024","xlm-mlm-tlm-xnli15-1024"]
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import datetime
import torch

# torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600000))
gpu_index = "cuda:0"
mode = "training"
LANG_CHECK = False
m_lama = datasets.load_dataset("m_lama")["test"].shuffle(seed=42)
# XNLI_LANGS = ['de','en', 'es', 'fr' ,'zh']
# XNLI_LANGS = ["af","ar","az","be","bg","bn","ca","ceb","cs","cy","da","de","el","en","es","et","eu","fa","fi","fr","ga","gl","he","hi","hr","hu","hy","id","it","ja","ka","ko","la","lt","lv","ms","nl","pl"]
# MIN_COUNT = 4
MIN_COUNT = 20
XNLI_LANGS = []
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


def m_lama_cross_robust_parser(data):
    """
    parse mLAMA
    {
    'language': 'af',
    'lineid': 0,
    'obj_label': 'Frankryk',
    'obj_uri': 'Q142',
    'predicate_id': 'P1001',
    'sub_label': 'President van Frankryk',
    'sub_uri': 'Q191954',
    'template': "[X] is 'n wettige term in [Y].",
    'uuid': '3fe3d4da-9df9-45ba-8109-784ce5fba38a'
    }
    """
    langauge = data["language"]
    obj_label = data["obj_label"]
    sub_label = data["sub_label"]
    template = data["template"]
    # masked = template.replace("[X]", sub_label).replace("[Y]", "<mask>")
    org = template.replace("[X]", sub_label).replace("[Y]", obj_label)
    predicate_id = str(data["predicate_id"])
    key = str(data["obj_uri"]) + str(data["sub_uri"]) + "@" + str(data["predicate_id"])
    return predicate_id, key, langauge, org, sub_label, obj_label


def dictList_dictTFTensor(dict_input):
    tftensor = dict()
    for k, v in dict_input.items():
        tftensor[k] = torch.tensor(v).to(gpu_index)
    return tftensor


def query_answer_parser(c_s_o, tokenizer):
    """
    This data can be used for creating fill-in-the-blank queries like "Paris is the capital of [MASK]" across 53 languages.
    The obj is only dependent on the query and independent on any context.
    To predict the obj, the model has to utilize the internal knowledge.


    The script below is modified from SQUAD.
    """
    context_batch = []
    obj_positions_batch = []
    lang_batch = []
    per_token = []
    for k, cso in enumerate(c_s_o):
        context_text, sub, obj, lang, k, _ = cso
        tok_obj_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if 6 in tok_obj_text:
            tok_obj_text.remove(6) # need to remove the special token in matching.
        doc_tokens = tokenizer.encode(context_text)
        if tok_obj_text[0] in doc_tokens:
            test = []
            start_p = doc_tokens.index(tok_obj_text[0])
            for new_start in range(start_p, len(doc_tokens)):
                for new_end in range(len(doc_tokens) - 1, new_start - 1, -1):
                    text_span = doc_tokens[new_start : new_end + 1]
                    if text_span == tok_obj_text and new_end >= new_start:
                        test.append((new_start, new_end))
            if len(test) > 0:
                obj_positions_batch.append(test[-1]) # get the obj postion.
                context_batch.append(context_text)
                per_token.append(test[-1][1] + 1 - test[-1][0])
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
        for k, v in enumerate(masked_context_batch["input_ids"]):
            langs.append([lang_batch[k]] * len(v))
        masked_context_batch["langs"] = langs
    if mode == "training":
        for k, v in enumerate(context_batch):
            for i in range(obj_positions_batch[k][0], obj_positions_batch[k][1] + 1):
                masked_context_batch["input_ids"][k][i] = tokenizer.mask_token_id #  Mask the obj
    return masked_context_batch, obj_positions_batch, per_token


def get_rep(sample_id, model, tokenizer):
    inputs = dictList_dictTFTensor(sample_id)
    if "langs" in inputs:
        re = model(
            input_ids=inputs["input_ids"],
            langs=inputs["langs"],
            output_hidden_states=True,
        )
    else:
        re = model(**inputs, output_hidden_states=True)
    return re.hidden_states[1:]
    # return re.hidden_states


def get_position_rep(rep_batch, posistion):
    layer_rep = []
    posistion_rep = []
    for j, rep_temp in enumerate(rep_batch):
        for k, rep in enumerate(rep_temp):
            rep = rep.cpu().detach().numpy()
            try:
                p_rep = np.mean(rep[posistion[k][0] : posistion[k][1] + 1], 0)
                # p_rep = np.mean(rep[k],0)
                posistion_rep.append(p_rep)
                if k == 0:
                    KEY_LIST_output.append(KEY_LIST[j])
                    LANG_LIST_output.append(LANG_LIST[j])
            except Exception:
                print(rep_batch[k])
                print(posistion[k])
        layer_rep.append(np.array(posistion_rep).reshape((len(posistion_rep), -1)))
        posistion_rep = []
    return layer_rep


def maximum_explainable_variance(batch, model, tokenizer, row, bias=None):
    with torch.no_grad():
        if len(batch) > 300:
            count = int(len(batch) / 300)
            for i in range(0, count):
                mini_batch, positions, per_token = query_answer_parser(
                    batch[i * 300 : i * 300 + 300], tokenizer
                )
                mini_rep = get_rep(mini_batch, model, tokenizer)
                if i == 0:
                    layer_rep = get_position_rep(mini_rep, positions)
                else:
                    mini_layer_rep = get_position_rep(mini_rep, positions)
                    for k, v in enumerate(layer_rep):
                        layer_rep[k] = np.concatenate(
                            (layer_rep[k], mini_layer_rep[k]), axis=0
                        )
                if "per_token" in row:
                    row["per_token"] = row["per_token"] + per_token
                else:
                    row["per_token"] = per_token
        else:
            batch, positions, per_token = query_answer_parser(batch, tokenizer)
            rep = get_rep(batch, model, tokenizer)
            layer_rep = get_position_rep(rep, positions)
            if "per_token" in row:
                row["per_token"] = row["per_token"] + per_token
            else:
                row["per_token"] = per_token
        # return IsoScore(obj_rep)
    for k, v in enumerate(layer_rep):
        pca = decomposition.PCA(1)
        pca.fit(v)
        # k = str(k)
        k = str(k + 1)
        mev = min(pca.explained_variance_ratio_[0], 1.0)
        if bias is not None:
            mev -= bias[k][0]
        if k in row:
            row[k].append(mev)
        else:
            row[k] = [mev]
        # break ###
    return row


def increamental_maximum_explainable_variance(batch, model, tokenizer, row, pca=None):
    with torch.no_grad():
        batch, positions, per_token = query_answer_parser(batch, tokenizer)
        rep = get_rep(batch, model, tokenizer)
        layer_rep = get_position_rep(rep, positions)
        # return IsoScore(obj_rep)
    for k, v in enumerate(layer_rep):
        k = str(k + 1)
        if k in pca:
            pca_now = pca[k]
        else:
            pca[k] = IncrementalPCA(n_components=1)
            pca_now = pca[k]
        pca_now.partial_fit(v)
        # if k in row:
        #     row[k].append(min(pca_now.explained_variance_ratio_[0],1.0))
        # else:
        #     row[k] = [min(pca_now.explained_variance_ratio_[0],1.0)]
        if "per_token" in row:
            row["per_token"] = (row["per_token"] + per_token) / 2.0
        else:
            row["per_token"] = per_token
        # break ###
    return row


def layer_wise_rep(batch, model, tokenizer, row, pca=None):
    with torch.no_grad():
        batch, positions, per_token = query_answer_parser(batch, tokenizer)
        rep = get_rep(batch, model, tokenizer)
        layer_rep = get_position_rep(rep, positions)
        # return IsoScore(obj_rep)
    for k, v in enumerate(layer_rep):
        # k = str(k)
        k = str(k + 1)
        if k in row:
            row[k] = np.concatenate((row[k], v), axis=0)
        else:
            row[k] = v
        if "per_token" in row:
            row["per_token"] = (row["per_token"] + per_token) / 2.0
        else:
            row["per_token"] = per_token
        # break ###
    return row


m_lama_organized = dict()
n = 0
m_lama_bias = dict()
temp = []
for data in m_lama:
    n += 1
    percent(n, 830000)
    predicate_id, key, langauge, org, sub_label, obj_label = m_lama_cross_robust_parser(
        data
    )
    if langauge in XNLI_LANGS or len(XNLI_LANGS) == 0:
        v = [org, sub_label, obj_label, langauge, key, predicate_id]
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
fieldnames = ["model", "mode", "per_token"] + list(
    map(lambda w: w, map(str, range(1, 36 + 1)))
)
# fieldnames = ['model','mode','per_token'] + list(map(lambda w: w, map(str, range(0,1))))
writer = csv.DictWriter(open("./multilingual_score", "w"), fieldnames=fieldnames)
writer.writeheader()
for model_name in model_names:
    variance_explained_score = dict()
    variance_explained_bias = dict()
    incrementalPCA_dict = dict()
    xlmR_tokenizer = AutoTokenizer.from_pretrained(model_name)
    xlmR = AutoModel.from_pretrained(model_name).to(gpu_index)
    base_pca = decomposition.PCA(1)
    xlmR_masked_cross_score = []
    n = 0
    m_lama_bias_length = len(m_lama_bias)
    # print("###computing variance_explained_bias ####")
    # for k,v in m_lama_bias.items():
    #     n +=1
    #     temp_dict = {}
    #     percent(n,m_lama_bias_length)
    #     variance_explained_bias[k]=maximum_explainable_variance(v,xlmR,xlmR_tokenizer,temp_dict)

    n = 0
    m_lama_organized_length = len(m_lama_organized)
    print("###computing variance_explained_score ####")
    for k, v in m_lama_organized.items():
        lang_set = set()
        try:
            for d in v:
                lang_set.add(d[3])
            if len(lang_set) > MIN_COUNT:
                n += 1
                percent(n, m_lama_organized_length)
                key, predicate_id = k.split("@")
                variance_explained_score = maximum_explainable_variance(
                    v, xlmR, xlmR_tokenizer, variance_explained_score
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
    for k, v in variance_explained_score.items():
        variance_explained_score[k] = str(np.nanmean(variance_explained_score[k]))
    # variance_explained_score["per_token"] = np.mean(variance_explained_score["per_token"])
    variance_explained_score["mode"] = "score"
    variance_explained_score["model"] = model_name
    # print("############org################")
    print("############score################")
    print(variance_explained_score)
    writer.writerow(variance_explained_score)
    del xlmR_tokenizer, xlmR, base_pca
del writer
print("###################################################")
print("Num of examination:" + str(n))
