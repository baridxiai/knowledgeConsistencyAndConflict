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
from datasets import Dataset,load_dataset
import pandas as pd
import torch
import datetime
#torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(days=1))

# Training

def tokenize_mlama_examples(examples, tokenizer):
    obj_label = examples["obj_label"]
    sub_label = examples["sub_label"]
    template = examples["template"]
    examples = template.replace("[X]", sub_label).replace("[Y]", obj_label)

    return tokenizer(examples, padding=True,  truncation=True)
def tokenize_wiki_examples(examples, tokenizer):
    return tokenizer(examples["text"], padding=True,  truncation=True)
def load_training_validation_dataset(tokenizer):
    #  mlama 53 is sorted in order of statements.
    df = pd.read_parquet("./mlama53.parquet", engine="fastparquet")
    m_lama = Dataset.from_pandas(df)
    val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenized_train = m_lama.map(lambda examples: tokenize_mlama_examples(examples, tokenizer), batched=False,remove_columns=df.columns.values).batch(batch_size=53)
    tokenized_val = val_dataset.map(lambda examples: tokenize_wiki_examples(examples, tokenizer), batched=True,remove_columns=["text"])

    return tokenized_train, tokenized_val

def load_training_arguments(data_file):
    with open(data_file, 'r') as f:
        train_args = json.load(f)
    train_args = TrainingArguments(**train_args)
    return train_args

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
        callbacks=[early_stopping]
    )
    trainer.train()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--training_config_json', type=str)
    args = parser.parse_args()
    train(args)