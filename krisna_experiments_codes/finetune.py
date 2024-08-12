from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, EarlyStoppingCallback
import json
from datasets import load_dataset, concatenate_datasets
import evaluate
import os
import wandb

os.environ["WANDB_PROJECT"]="codemixed_knowledge_consistency"

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    global metric
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = predictions.argmax(axis=1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_examples(examples, tokenizer):
    inputs = tokenizer(
        examples['premise'],
        examples['hypothesis'],
        max_length=512,
        padding="max_length",
    )
    inputs['label'] = examples['label']

    return inputs
def load_training_validation_dataset(tokenizer):
    en_nli_dataset = load_dataset("facebook/xnli", 'en')
    de_nli_dataset = load_dataset("facebook/xnli", 'de')
    ar_nli_dataset = load_dataset("facebook/xnli", 'ar')

    train_nli_dataset = concatenate_datasets([en_nli_dataset['train'], de_nli_dataset['train'], ar_nli_dataset['train']])
    train_nli_dataset = train_nli_dataset.shuffle(seed=42)
    val_nli_dataset = concatenate_datasets([en_nli_dataset['validation'], de_nli_dataset['validation'], ar_nli_dataset['validation']])

    tokenized_train_nli = train_nli_dataset.map(lambda examples: tokenize_examples(examples, tokenizer), batched=True, remove_columns=train_nli_dataset.column_names)
    tokenized_val_nli = val_nli_dataset.map(lambda examples: tokenize_examples(examples, tokenizer), batched=True, remove_columns=val_nli_dataset.column_names)

    return tokenized_train_nli, tokenized_val_nli

def load_training_arguments(data_file):
    with open(data_file, 'r') as f:
        train_args = json.load(f)
    train_args = TrainingArguments(**train_args)
    return train_args

def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)
    
    data_collator = DataCollatorWithPadding(tokenizer, padding='max_length', max_length=512)
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
        compute_metrics=compute_metrics,
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
