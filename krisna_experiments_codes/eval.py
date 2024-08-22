import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import collections, re, string

# preds: [num_layers, num_rows]
# all_gts: [num_rows, ?]] 
MIXED_SEGMENTATION_LANGS = ["zh_cn", "zh_hk", "zh_tw", "ja", "th", "km"]

ARTICLE_REGEX_BY_LANG = {
    "en": r"\b(a|an|the)\b",
    "es": r"\b(un|una|unos|unas|el|la|los|las)\b",
    "vi": r"\b(của|là|cái|chiếc|những)\b",
    "de": r"\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b",
    "ar": "\sال^|ال",
    "nl": r"\b(de|het|een|des|der|den)\b",
    "sv": r"\b(en|ett)\b",
    "da": r"\b(en|et)\b",
    "no": r"\b(en|et|ei)\b",
    "fr": r"\b(le|la|l'|les|du|de|d'|des|un|une|des)",
    "pt": r"\b(o|a|os|as|um|uma|uns|umas)\b",
    "it": r"\b(il|lo|la|l'|i|gli|le|del|dello|della|dell'|dei|degli|degl'|delle|un'|uno|una|un)",
    "fi": r"\b(se|yks|yksi)\b",
    "hu": r"\b(a|az|egy)\b",
}


def compute_query_matrix_and_norm(model, preds):
    queries_embed = model.encode(preds)
    queries_norm = np.linalg.norm(queries_embed, axis=1)
    return queries_embed, queries_norm

def evaluate_sim(model, query_embed, query_norm, all_gts):
    avg_sims = []
    for i in tqdm(range(0, len(query_embed))):
        answer_embeds = model.encode(all_gts[i])
        sim_matrix = np.dot(query_embed[i], answer_embeds.T)
        answer_norm = np.linalg.norm(answer_embeds, axis=1)
        cos_sim = sim_matrix/(np.outer(query_norm[i], answer_norm) + 1e-9)
        weighted_cos_sim = sum(cos_sim[0])/len(all_gts[i])
        avg_sims.append(weighted_cos_sim)
    return sum(avg_sims)/len(avg_sims)

def compute_query_matrix_and_norm_multi_preds(model, preds_cands):
    all_queries_embed = []
    all_queries_norm = []
    for pred_cands in preds_cands:
        queries_embed = model.encode(pred_cands)
        all_queries_embed.append(queries_embed)
        all_queries_norm.append(np.linalg.norm(queries_embed, axis=1))
    return all_queries_embed, all_queries_norm

def evaluate_sim_multi_preds(model, query_embed, query_norm, all_gts):
    avg_sims = []
    for i in tqdm(range(0, len(query_embed))):
        answer_embeds = model.encode(all_gts[i])
        sim_matrix = np.dot(query_embed[i], answer_embeds.T)
        answer_norm = np.linalg.norm(answer_embeds, axis=1)
        cos_sim = sim_matrix/(np.outer(query_norm[i], answer_norm) + 1e-9)
        weighted_cos_sim = cos_sim.mean(axis=-1, keepdims=False).mean(axis=0, keepdims=False)
        avg_sims.append(weighted_cos_sim)
    return sum(avg_sims)/len(avg_sims)

def whitespace_tokenize(text):
    return text.split()

def mixed_segmentation(text):
    segs_out = []
    temp_str = ""
    for char in text:
        if temp_str != "":
            ss = whitespace_tokenize(temp_str)
            segs_out.extend(ss)
            temp_str = ""
        segs_out.append(char)

    if temp_str != "":
        ss = whitespace_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out

def normalize_answer(s, lang):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text, lang):
        article_regex = ARTICLE_REGEX_BY_LANG.get(lang)
        if article_regex:
            return re.sub(article_regex, " ", text)
        else:
            return text

    def white_space_fix(text, lang):
        if lang in MIXED_SEGMENTATION_LANGS:
            tokens = mixed_segmentation(text)
        else:
            tokens = whitespace_tokenize(text)
        return " ".join([t for t in tokens if t.strip() != ""])

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s)), lang), lang)

def normalize_cloze_string(input_text):
    # Remove substrings matching <[0-9][0-9]+>
    cleaned_text = re.sub(r'<\d+>', '', input_text)
    
    # Remove leading and trailing whitespace and reduce excessive whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    cleaned_text = cleaned_text.lstrip().strip()
    
    return cleaned_text

def get_tokens(s, lang):
    if not s:
        return []
    return normalize_answer(s, lang).split()


def compute_exact(a_gold, a_pred, lang):
    return int(normalize_answer(a_gold, lang) == normalize_answer(a_pred, lang))


def compute_f1(a_gold, a_pred, lang):
    gold_toks = get_tokens(a_gold, lang)
    pred_toks = get_tokens(a_pred, lang)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max(metric, pred, labels, lang):
    scores = []
    for label in labels:
        score = metric(pred, label, lang)
        scores.append(score)
    return max(scores)

def metric_mean(metric, pred, labels, lang):
    scores = []
    for label in labels:
        score = metric(pred, label, lang)
        scores.append(score)
    return sum(scores)/len(scores)

def f1_max(pred, labels, lang):
    return metric_max(compute_f1, pred, labels, lang)

def f1_mean(pred, labels, lang):
    return metric_mean(compute_f1, pred, labels, lang)

def exact_max(pred, labels, lang):
    return metric_max(compute_exact, pred, labels, lang)

def evaluate_f1_mean(preds, all_gts, lang):
    all_scores = []
    for pred, gt in zip(preds, all_gts):
        all_scores.append(f1_mean(pred, gt, lang))
    return sum(all_scores)/len(all_scores)*100

def evaluate_accuracy(preds, all_gts):
    accs = []
    for pred, gt in zip(preds, all_gts):
        correct_cnt = [int(pred)==int(gt_cand) for gt_cand in gt]
        correct_acc = sum(correct_cnt)/len(correct_cnt)*100
        accs.append(correct_acc)
    return sum(accs)/len(accs)

def evaluate_f1_max(preds, all_gts, lang):
    all_scores = []
    for pred, gt in zip(preds, all_gts):
        all_scores.append(f1_max(pred, gt, lang))
    return sum(all_scores)/len(all_scores)*100

def evaluate_f1_means_max(preds_cands, all_gts, lang):
    all_scores = []
    for pred_cands, gt_cands in zip(preds_cands, all_gts):
        score_mean = []
        for pred_cand in pred_cands:
            score_mean.append(f1_max(pred_cand, gt_cands, lang))
        all_scores.append(sum(score_mean)/len(score_mean)*100)
    return sum(all_scores)/len(all_scores)



def evaluate_exact_max(preds, all_gts, lang):
    all_scores = []
    for pred, gt in zip(preds, all_gts):
        all_scores.append(exact_max(pred, gt, lang))
    return sum(all_scores)/len(all_scores)*100


def compute_acc_nli(preds, gts):
    correct_label = [int(pred)==int(gt) for pred, gt in zip(preds, gts)]
    return sum(correct_label)/len(correct_label)*100


def compute_rankc(preds1, preds2):
    def softmax(x):
        return np.exp(x)/np.sum(np.exp(x))
    consistent_sum = 0

    for rank_pred1, rank_pred2 in zip(preds1, preds2):
        N = len(rank_pred1)
        weight = np.array([N-i for i in range(N)])
        weight = softmax(weight)

        for i in range(N):
            candidates1 = set(rank_pred1[:i+1])
            candidates2 = set(rank_pred2[:i+1])

            common_cands = candidates1.intersection(candidates2)
            consistent_sum += weight[i]*len(common_cands)/len(candidates1)
    
    return consistent_sum/len(preds1)

def compute_mrr(preds, gts):
    all_ranks = []
    for pred_cands, gt in zip(preds, gts):
        rank = 0
        for idx, cand in enumerate(pred_cands):
            normalized_cand = normalize_cloze_string(cand)
            if normalized_cand == gt:
                rank = 1/(idx+1)
                break
        all_ranks.append(rank)
    return sum(all_ranks)/len(all_ranks)


def compute_accuracy_top_n(preds1, preds2, n=1):
    acc_avg = 0
    for rank_pred1, rank_pred2 in zip(preds1, preds2):
        rank_pred1 = rank_pred1[:n]
        rank_pred2 = rank_pred2[:n]
        correct_cnt = [pred_word1==pred_word2 for pred_word1, pred_word2 in zip(rank_pred1, rank_pred2)]
        instance_acc = sum(correct_cnt)/len(correct_cnt)
        acc_avg += instance_acc
    return acc_avg/len(preds1)
