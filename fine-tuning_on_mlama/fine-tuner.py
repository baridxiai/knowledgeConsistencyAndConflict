import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["WANDB_PROJECT"]="codemixed_knowledge_consistency"
from argparse import ArgumentParser
import json
from transformers import (
    AutoTokenizer,
    Trainer,
    AutoModelForMaskedLM,TrainingArguments,DataCollatorForLanguageModeling,EarlyStoppingCallback
)
from torch.utils.data import  SequentialSampler
from datasets import Dataset,load_dataset
import pandas as pd
import torch
import datetime
from datasets import concatenate_datasets
#torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(days=1))

# Training

def tokenize_mlama_examples(examples, tokenizer):
    obj_label = examples["obj_label"]
    sub_label = examples["sub_label"]
    template = examples["template"]
    predicate_id = examples["predicate_id"][1:] ## omit prefix "P"
    examples = template.replace("[X]", sub_label).replace("[Y]", obj_label)
    tokens = tokenizer(examples, padding=True,  truncation=True)
    tokens["predicate_id"] = [predicate_id]
    return tokens
def tokenize_wiki_examples(examples, tokenizer):
    return tokenizer(examples["text"], padding=True,  truncation=True)
def load_training_validation_dataset(tokenizer):
    #  mlama 53 is sorted in order of statements.
    m_lama = load_dataset("parquet", data_files="./mlama53.parquet")["train"]
    val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenized_train = m_lama.map(lambda examples: tokenize_mlama_examples(examples, tokenizer), batched=False,remove_columns=m_lama.column_names)
    tokenized_val = val_dataset.map(lambda examples: tokenize_wiki_examples(examples, tokenizer), batched=True,remove_columns=val_dataset.column_names)

    return tokenized_train, tokenized_val

def load_training_arguments(data_file):
    with open(data_file, 'r') as f:
        train_args = json.load(f)
    train_args = TrainingArguments(**train_args)
    return train_args
def group_by(d, col, join):
    """from: https://github.com/huggingface/datasets/issues/3644"""
    # Get the indices of each group
    groups = {key: [] for key in d.unique(col)}
    def create_groups_indices(key, i):
        groups[key].append(i)
    d.map(create_groups_indices, with_indices=True, input_columns=col)
    # Get one dataset object per group
    groups = {key: d.select(indices) for key, indices in groups.items()}
    # Apply join function
    groups = {
        key: d
        for key, dataset_group in groups.items()
    }
    # Return concatenation of all the joined groups
    return concatenate_datasets(groups.values())
def non_shuffle(self):
    self.train_dataset.shuffle()
    self.train_dataset = group_by(self.train_dataset)
    return SequentialSampler(self.train_dataset)
def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
    train_dataset, val_dataset = load_training_validation_dataset(tokenizer)
    training_args = load_training_arguments(args.training_config_json)
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=5
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer._get_train_sampler =lambda: non_shuffle(trainer)

    trainer.train()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--training_config_json', type=str)
    args = parser.parse_args()
    train(args)