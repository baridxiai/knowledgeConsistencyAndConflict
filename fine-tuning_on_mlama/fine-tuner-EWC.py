import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["WANDB_PROJECT"] = "codemixed_knowledge_consistency"
from argparse import ArgumentParser
import json
from transformers import (
    AutoTokenizer,
    Trainer,
    AutoModelForMaskedLM,
    TrainingArguments,
    DataCollatorWithPadding,
)
from torch.utils.data import SequentialSampler
from datasets import Dataset, load_dataset

# torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(days=1))
import random
from copy import deepcopy

from torch.autograd import Variable as variable
from tools import mLama_util
from models.modelWrapper import EncoderWrapper

# Training
KB =  load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

class EWC(object):
    def __init__(self, model, tokenizer):

        self.modelWrapper = EncoderWrapper(model, tokenizer, "cloze")
        self.model = self.modelWrapper.model
        self.params = {
            n: p for n, p in self.model.named_parameters() if p.requires_grad
        }
        self._means = {}
        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)
        self._precision_matrices = self._diag_fisher()

    def _diag_fisher(self):
        batch_size = 128
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)
        batch_cnt = len(KB)//batch_size

        for i in range(0, batch_cnt):
            batch = KB[i*64:min((i+1)*64, len(KB))]
            self.modelWrapper.inference_cloze_grads(batch, batch_size)
            for n, p in self.modelWrapper.model.named_parameters():
                precision_matrices[n].data += p.grad.data**2 / batch_cnt

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


class EWC_Trainer(Trainer):
    def __init__(
        self,
        model,
        args,
        train_dataset,
        eval_dataset,
        tokenizer,
        data_collator,
        EWC_base,
    ):
        super(EWC_Trainer, self).__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        self.EWC_base = EWC_base

    def compute_loss(self, model, inputs, return_outputs=False):
        re = super(EWC_Trainer, self).compute_loss(
            model, inputs, return_outputs=return_outputs
        )
        if return_outputs:
            loss, outputs = re
        else:
            loss = re
        ewc_penality = self.EWC_base.penalty(self.model)
        loss += ewc_penality*100
        return (loss, outputs) if return_outputs else loss


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


def tokenize_mlama_examples(examples, tokenizer):
    obj_label = examples["obj_label"]
    _, obj_token_lengths, _= mLama_util.tokenize_obj(obj_label,tokenizer)
    sub_label = examples["sub_label"]
    template = examples["template"]
    mono_prompts = template.replace("[X]", sub_label)
    mono_prompts = mLama_util.mask_sentences(mono_prompts, obj_token_lengths,tokenizer)
    mono_inputs = tokenizer(mono_prompts, padding=True, truncation=True, return_tensors='pt')
    labels = template.replace("[X]", sub_label).replace("[Y]", obj_label)
    labels = tokenizer(labels, padding=True, truncation=True)
    masked_indices = mono_inputs['input_ids'] == tokenizer.mask_token_id
    labels[~masked_indices] = -100
    mono_inputs['labels'] = labels

    return mono_inputs


def tokenize_wiki_examples(examples, tokenizer):
    return tokenizer(examples["text"], padding=True, truncation=True)


def load_training_validation_dataset(tokenizer):
    #  mlama 53 is sorted in order of statements.
    m_lama = m_lama = load_dataset("m_lama")["test"].shuffle(seed=42)
    #m_lama = m_lama.filter(lambda x: x["language"] !="en")
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
    model = AutoModelForMaskedLM.from_pretrained(args.model_name).to("cuda")


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataset, val_dataset = load_training_validation_dataset(tokenizer)
    training_args = load_training_arguments(args.training_config_json)
    EWC_base = EWC(model, tokenizer)
    trainer = EWC_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        EWC_base=EWC_base,
    )
    # trainer._get_train_sampler =lambda: non_shuffle(trainer,span=53)
    trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--training_config_json", type=str)
    parser.add_argument("--batch_consistency", type=int)
    args = parser.parse_args()
    train(args)
