import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["WANDB_PROJECT"]="codemixed_knowledge_consistency"
from argparse import ArgumentParser
import json
from transformers import (
    AutoTokenizer,
    Trainer,
    AutoModelForMaskedLM,TrainingArguments,DataCollatorForLanguageModeling,XLMRobertaForMaskedLM
)
from torch.utils.data import  SequentialSampler
from datasets import Dataset,load_dataset
import pandas as pd
import datetime
from datasets import concatenate_datasets
import torch
torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(days=1))
import random
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable as variable
import torch.utils.data
from tools import utils
from models.model import EncoderWrapper
# Training

class EWC(object):
    def __init__(self):

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self.knowledge_base = utils.load_mlama("en",None)
        self.modelWrapper = EncoderWrapper("FacebookAI/xlm-roberta-base", "FacebookAI/xlm-roberta-base")
        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)
        self._precision_matrices = self._diag_fisher()

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)
        self.modelWrapper.inference_cloze_grads(self.knowledge_base)
        for n, p in self.modelWrapper.model.named_parameters():
            precision_matrices[n].data += p.grad.data ** 2
        # output = self.model(input).view(1, -1)
        # label = output.max(1)[1].view(-1)
        # loss = F.nll_loss(F.log_softmax(output, dim=1), label,reduction=None)
        # loss.backward()
        # for input in self.dataset:
        #     # input = variable(input)
        #     self.model.zero_grad()
        #     output = self.model(input).logits.view(1, -1)
        #     label = output.max(1)[1].view(-1)
        #     loss = F.nll_loss(F.log_softmax(output, dim=1), label)
        #     loss.backward()


        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
EWC_model =EWC()
class EWC_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        re = super(EWC_Trainer, self).compute_loss(model, inputs, return_outputs=return_outputs)
        if return_outputs:
            loss, outputs = re
        else:
            loss = re
        ewc_penality = EWC_model.penalty(model)
        loss += ewc_penality
        return (loss, outputs) if return_outputs else loss
class batchSeq(SequentialSampler):
    def __init__(self, data_source,span):
        self.data_source = data_source
        self.span = span

    def __iter__(self):
        groups = dict()
        for key in range(0, int(self.data_source.num_rows/self.span)):
            groups[key] =[i for i in range(key * self.span, (key+1)*self.span)]
        keys = [k for k, values in groups.items()]
        random.shuffle(keys)
        group_index = []
        for k, index in enumerate(keys):
            group_index += groups[index]
        return iter(group_index)


def tokenize_mlama_examples(examples, tokenizer):
    obj_label = examples["obj_label"]
    sub_label = examples["sub_label"]
    template = examples["template"]
    examples = template.replace("[X]", sub_label).replace("[Y]", obj_label)
    tokens = tokenizer(examples, padding=True,  truncation=True)
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

def non_shuffle(self,span):
    return batchSeq(self.train_dataset,span)
def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
    train_dataset, val_dataset = load_training_validation_dataset(tokenizer)
    training_args = load_training_arguments(args.training_config_json)
    trainer = EWC_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # trainer._get_train_sampler =lambda: non_shuffle(trainer,span=53)
    trainer.training_step
    trainer.train()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--training_config_json', type=str)
    parser.add_argument('--batch_consistency', type=int)
    args = parser.parse_args()
    train(args)