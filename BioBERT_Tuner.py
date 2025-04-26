# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 11:26:14 2025

This file takes the BioBERT code and fine tunes the model on the given data.
It saves the new model as a local directory called "biobert-sentence-model".

"""
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from IPython.display import display

# Load CSV
csv_path = "phase1_sentences.csv"  # Replace with your filename
df = pd.read_csv(csv_path, header=0)
df = df.rename(columns={"text": "text", "has_data": "label"})
df["label"] = df["label"].astype(int)  # column 2 is label
df = df.rename(columns={1: "text", 2: "label"})
df = df[["text", "label"]]  # Drop other columns

# Convert to Hugging Face Dataset and split
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)

# Tokenizer and model
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns("text")  # Drop extra columns
dataset.set_format("torch")

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Training args
training_args = TrainingArguments(
    output_dir="./biobert-sentence-model",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Evaluate
eval_results = trainer.evaluate()
print(eval_results)
display(eval_results)

#Save the fine-tuned model
trainer.save_model("./biobert-sentence-model")
tokenizer.save_pretrained("./biobert-sentence-model")
