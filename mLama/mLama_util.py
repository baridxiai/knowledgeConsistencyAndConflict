# -*- coding: utf-8 -*-
# code warrior: Barid

import torch
import numpy as np
import datasets
import sys


import torch

# torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600000))
gpu_index = "cuda"
mode = "training"
LANG_CHECK = False
m_lama = datasets.load_dataset("m_lama")["test"].shuffle(seed=42)
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
