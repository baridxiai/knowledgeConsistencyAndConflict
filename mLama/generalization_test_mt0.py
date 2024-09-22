import os
from transformers import AutoModelForTokenClassification

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["WANDB_PROJECT"] = "codemixed_knowledge_consistency"
import pandas as pd
from argparse import ArgumentParser
import json
from transformers import (
    AutoTokenizer,
    Trainer,
    AutoModelForMaskedLM,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from datasets import Dataset
import datetime
import torch
import evaluate
import numpy as np
precision_metric = evaluate.load("precision")

torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(days=1))

import torch
import torch.utils.data
from tools import mLama_util

def tokenize_mlama_examples(examples, tokenizer):
    obj_label = examples["obj_label"]
    _, obj_token_lengths, _ = mLama_util.tokenize_obj([obj_label], tokenizer)
    sub_label = examples["sub_label"]
    template = examples["template"]
    mono_prompts = template.replace("[X]", sub_label).replace("[y]","<extra_id_0>")
    mono_inputs = tokenizer(mono_prompts)
    labels = tokenizer(f"<extra_id_0> {obj_label} <extra_id_1>")
    labels[1] = -100
    labels[-2] = -100
    labels[-1] = -100
    mono_inputs["labels"] = labels
    return mono_inputs


def load_training_validation_dataset(tokenizer):
    #  mlama 53 is sorted in order of statements.
    # m_lama = load_dataset("m_lama")["test"].shuffle(seed=42)
    # m_lama = m_lama.filter(lambda x: x["language"] not in ["en","ar","id","ta"])
    df = pd.read_parquet("./mlama53.parquet", engine="fastparquet")
    m_lama = Dataset.from_pandas(df)
    tokenized_train = dict()
    for lang in ["en","ar","id","ta","de"]:
        temp_lama = m_lama.filter(lambda x: x["language"] not in [lang])
        temp_lama = temp_lama.map(lambda examples: tokenize_mlama_examples(examples, tokenizer),
        batched=False,
        remove_columns=m_lama.column_names)
        tokenized_train[lang] = temp_lama
    return tokenized_train, tokenized_train


def load_training_arguments(data_file):
    with open(data_file, "r") as f:
        train_args = json.load(f)
    train_args = TrainingArguments(**train_args)
    return train_args

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

# def precision_at_k(eval_pred):
#     logits, y_true = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     df = pd.DataFrame({'true': y_true, 'score': y_score}).sort('score')
#     threshold = df.iloc[int(k*len(df)),1]
#     y_pred = pd.Series([1 if i >= threshold else 0 for i in df['score']])
#     return metrics.precision_score(y_true, y_pred)

def train(args):
    training_args = TrainingArguments(
   output_dir="./mlama_generatization",
    overwrite_output_dir=True,
    num_train_epochs=5,
    save_steps=10000,
    save_total_limit=5,
    remove_unused_columns=False,
    per_gpu_eval_batch_size=32
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    train_dataset, val_dataset = load_training_validation_dataset(tokenizer)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.evaluate()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()
    train(args)
