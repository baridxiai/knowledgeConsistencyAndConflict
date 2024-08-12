import torch
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
from tqdm import tqdm
import pdb

# class MKQADataset(Dataset):
#     def __init__(self, dataset_file, source_lang, target_lang, tokenizer, max_length=512):
#         self.source_lang = source_lang
#         self.target_lang = target_lang
#         with open(dataset_file, 'rb') as f:
#             data_obj_dict = pkl.load(f)
#         self.data_obj = data_obj_dict
#         self.tokenizer = tokenizer
#         self.max_length = max_length
    
#     def __len__(self):
#         return len(self.data_obj)
    
#     def __getitem__(self, idx):
#         instance = self.data_obj[idx]

#         query = instance["query"]
#         same_lang_answers = instance['answers_same_lang']
#         cross_lang_answers = instance['answers_cross_lang']

#         tokenized_query
def load_nli_dataset_for_inference(dataset_file):
    with open(dataset_file, 'rb') as f:
        data_obj_dict = pkl.load(f)
    
    premise_hyphotesis_pairs = []
    pair_ids = []
    gts = []
    for instance in tqdm(data_obj_dict):
        premise_hyphotesis_pairs.append((instance['premise'], instance['hypothesis'], instance['label']))
        pair_ids.append(instance['id'])
        gts.append(instance['label'])

    return premise_hyphotesis_pairs, gts, pair_ids

def load_nli_dataset_dl_for_inference(dataset_file, batch_size=8):
    with open(dataset_file, 'rb') as f:
        data_obj_dict = pkl.load(f)
    gts = []
    premise_hyphotesis_pairs = []
    premise_hyphotesis_pairs_instance = []
    pair_ids = []
    for instance in tqdm(data_obj_dict):
        #pdb.set_trace()
        if len(premise_hyphotesis_pairs_instance) == batch_size:
            premise_hyphotesis_pairs.append(premise_hyphotesis_pairs_instance)
            
            premise_hyphotesis_pairs_instance = []
        gts.append(instance['label'])
        pair_ids.append(instance['id'])

        premise_hyphotesis_pairs_instance.append((instance['premise'], instance['hypothesis']))
        
    if len(premise_hyphotesis_pairs_instance) > 0:
        premise_hyphotesis_pairs.append(premise_hyphotesis_pairs_instance)
    return premise_hyphotesis_pairs, gts, pair_ids

def load_qa_dataset_for_encoder_inference(dataset_file, batch_size=8):
    with open(dataset_file, 'rb') as f:
        data_obj_dict = pkl.load(f)
    queries = []
    pos_same_lang_answers = []
    pos_cross_lang_answers = []
    neg_same_lang_answers = []
    neg_cross_lang_answers = []
    hard_neg_same_lang_answers = []
    hard_neg_cross_lang_answers = []
    query_ids = []

    query_instance = []
    for instance in tqdm(data_obj_dict):
        if len(query_instance) == batch_size:
            queries.append(query_instance)
            query_instance = []
        query = instance['query']
        source_context = instance['context_same_lang']
        target_context = instance['context_cross_lang']
        query_instance.append((query, (source_context, target_context)))
        query_ids.append(instance['id'])

        # positive answers
        pos_same_lang_answers.append(instance["answers_same_lang"])
        pos_cross_lang_answers.append(instance["answers_cross_lang"])

        # negative answers
        neg_same_lang_answers.append(instance["negative_answers_same_lang"])
        neg_cross_lang_answers.append(instance["negative_answers_cross_lang"])

        # hard negative answers
        hard_neg_same_lang_answers.append(instance["hard_negative_answers_same_lang"])
        hard_neg_cross_lang_answers.append(instance["hard_negative_answers_cross_lang"])
    if len(query_instance)  > 0:
        queries.append(query_instance)
    return queries, pos_same_lang_answers, pos_cross_lang_answers, neg_same_lang_answers, neg_cross_lang_answers, hard_neg_same_lang_answers, hard_neg_cross_lang_answers, query_ids
    

def load_qa_dataset_for_inference(dataset_file, batch_size=8):
    with open(dataset_file, 'rb') as f:
        data_obj_dict = pkl.load(f)
    queries = []
    query_ids = []
    pos_same_lang_answers = []
    pos_cross_lang_answers = []
    neg_same_lang_answers = []
    neg_cross_lang_answers = []
    hard_neg_same_lang_answers = []
    hard_neg_cross_lang_answers = []
    query_instance = []
    for instance in tqdm(data_obj_dict):
        if len(query_instance) == batch_size:
            queries.append(query_instance)
            query_instance = []
        query = instance['query']
        query_id = instance['id']
        source_query = instance['query']
        if instance['context_same_lang'] != '':
            source_query = f"{instance['context_same_lang']}. {source_query}"
        target_query = instance['query']
        if instance['context_cross_lang'] != '':
            target_query = f"{instance['context_cross_lang']}. {target_query}"
        query_instance.append((source_query, target_query))
        query_ids.append(query_id)

        # positive answers
        pos_same_lang_answers.append(instance["answers_same_lang"])
        pos_cross_lang_answers.append(instance["answers_cross_lang"])

        # negative answers
        neg_same_lang_answers.append(instance["negative_answers_same_lang"])
        neg_cross_lang_answers.append(instance["negative_answers_cross_lang"])

        # hard negative answers
        hard_neg_same_lang_answers.append(instance["hard_negative_answers_same_lang"])
        hard_neg_cross_lang_answers.append(instance["hard_negative_answers_cross_lang"])
    if len(query_instance)  > 0:
        queries.append(query_instance)

    return queries, pos_same_lang_answers, pos_cross_lang_answers, neg_same_lang_answers, neg_cross_lang_answers, hard_neg_same_lang_answers, hard_neg_cross_lang_answers, query_ids
