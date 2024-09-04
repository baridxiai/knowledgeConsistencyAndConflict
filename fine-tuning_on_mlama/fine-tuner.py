import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["WANDB_PROJECT"] = "codemixed_knowledge_consistency"
from argparse import ArgumentParser
import json
from transformers import (
    AutoTokenizer,
    Trainer,
    AutoModelForMaskedLM,
    TrainingArguments,
    DataCollatorForTokenClassification,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import SequentialSampler
from datasets import Dataset, load_dataset
import datetime
import torch

torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(days=1))
import random

import torch
import torch.utils.data
from tools import mLama_util

# Training

from peft import LoraConfig

peft_config = LoraConfig(
    r=64,  # the rank of the LoRA matrices
    lora_alpha=16,  # the weight
    lora_dropout=0.1,  # dropout to add to the LoRA layers
    bias="none",  # add bias to the nn.Linear layers?
    task_type="MASKED_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],  # the name of the layers to add LoRA
    modules_to_save=None,  # layers to unfreeze and train from the original pre-trained model
)


class batchSeq(SequentialSampler):
    def __init__(self, data_source, span):
        self.data_source = data_source
        self.span = span

    def __iter__(self):
        groups = dict()
        for key in range(0, int(self.data_source.num_rows / self.span)):
            groups[key] = [i for i in range(key * self.span, (key + 1) * self.span)]
        keys = [k for k, values in groups.items()]
        random.shuffle(keys)
        group_index = []
        for k, index in enumerate(keys):
            group_index += groups[index]
        return iter(group_index)


# def tokenize_mlama_examples(examples, tokenizer):
#     obj_label = examples["obj_label"]
#     sub_label = examples["sub_label"]
#     template = examples["template"]
#     examples = template.replace("[X]", sub_label).replace("[Y]", obj_label)
#     tokens = tokenizer(examples, padding=True,  truncation=True)
#     return tokens
def tokenize_mlama_examples(examples, tokenizer):
    obj_label = examples["obj_label"]
    _, obj_token_lengths, _ = mLama_util.tokenize_obj([obj_label], tokenizer)
    sub_label = examples["sub_label"]
    template = examples["template"]
    mono_prompts = template.replace("[X]", sub_label)
    mono_prompts = mLama_util.mask_sentences(
        [mono_prompts], obj_token_lengths, tokenizer
    )[0]
    mono_inputs = tokenizer(mono_prompts)
    labels = template.replace("[X]", sub_label).replace("[Y]", obj_label)
    labels = tokenizer(
        labels,
        padding="max_length",
        max_length=len(mono_inputs["input_ids"]),
        truncation=True,
    )["input_ids"]
    for k, v in enumerate(mono_inputs["input_ids"]):
        if v != tokenizer.mask_token_id:
            labels[k] = -100
    mono_inputs["labels"] = labels
    return mono_inputs


def tokenize_wiki_examples(examples, tokenizer):
    batch = tokenizer(
        examples["text"], padding=True, truncation=True, max_length=256, return_tensors="pt"
    )
    batch["input_ids"], batch["labels"] = DataCollatorForLanguageModeling(
        tokenizer
    ).torch_mask_tokens(batch["input_ids"], None)
    return batch


def load_training_validation_dataset(tokenizer):
    #  mlama 53 is sorted in order of statements.
    m_lama = load_dataset("m_lama")["test"].shuffle(seed=42)
    m_lama = m_lama.filter(lambda x: x["language"] not in ["en","ar","id","ta"])
    val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenized_train = m_lama.map(
        lambda examples: tokenize_mlama_examples(examples, tokenizer),
        batched=False,
        remove_columns=m_lama.column_names,
    )
    tokenized_val = val_dataset.map(
        lambda examples: tokenize_wiki_examples(examples, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    return tokenized_train, tokenized_val


def load_training_arguments(data_file):
    with open(data_file, "r") as f:
        train_args = json.load(f)
    train_args = TrainingArguments(**train_args)
    return train_args


def non_shuffle(self, span):
    return batchSeq(self.train_dataset, span)


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    train_dataset, val_dataset = load_training_validation_dataset(tokenizer)
    training_args = load_training_arguments(args.training_config_json)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--training_config_json", type=str)
    parser.add_argument("--batch_consistency", type=int)
    args = parser.parse_args()
    train(args)
