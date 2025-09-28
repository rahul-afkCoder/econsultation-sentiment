# app/train.py
# NOTE: This is a simple illustrative finetuning script for a classification head using HuggingFace.
# For large-scale training, use proper distributed training and hyperparameter tuning.

from datasets import load_dataset, Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
import pandas as pd
import os

def prepare_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    # expects columns: comment_text, label (label values: 0=negative, 1=neutral,2=positive)
    return Dataset.from_pandas(df)

def train(csv_path: str, out_dir: str = './models/sentiment-distilbert', epochs: int = 2):
    ds = prepare_dataset(csv_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    def tokenize(batch):
        return tokenizer(batch['comment_text'], truncation=True, padding='max_length', max_length=128)
    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column('label', 'labels')
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    model = DistilBertForSequenceClassification.from_pretrained('distilbertbase-uncased', num_labels=3)
    
    args = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy='epoch',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        save_total_limit=1,
        logging_steps=50,
    )
    
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()
    trainer.save_model(out_dir)
    print(f"Saved model to {out_dir}")

if __name__ == '__main__':
    import sys
    csv = sys.argv[1]
    train(csv)
