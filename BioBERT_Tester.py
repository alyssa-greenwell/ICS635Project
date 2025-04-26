# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 18:38:49 2025

This file takes a locally saved BioBERT fine-tuned model and compares it to the
base BioBERT model that has not been fine-tuned. This program returns a bar 
chart comparing the performance of the two models as well as confucion matrices
for each model.

If an error occurs where the program cannot find the fine-tuned model, make 
sure that the BioBERT_Tuner file has been run first.

"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Config
MODEL_PATH_FINE_TUNED = "./biobert-sentence-model"
MODEL_NAME_BASE = "dmis-lab/biobert-base-cased-v1.1"
TEST_CSV_PATH = "phase1_sentences.csv"  # <-- UPDATE THIS

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_BASE)

# Load test data
df = pd.read_csv(TEST_CSV_PATH)
df = df.rename(columns={"text": "text", "has_data": "label"})
df = df[["text", "label"]]
df["label"] = df["label"].astype(int)  # Ensure correct format

# Tokenize test data
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

from datasets import Dataset
test_dataset = Dataset.from_pandas(df)
test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.remove_columns(["text", "__index_level_0__"]) if "__index_level_0__" in test_dataset.column_names else test_dataset.remove_columns(["text"])
test_dataset.set_format("torch")

# Metric computation
def evaluate_model(model, dataset):
    trainer = Trainer(model=model, tokenizer=tokenizer)
    preds_output = trainer.predict(dataset)
    preds = np.argmax(preds_output.predictions, axis=1)
    labels = preds_output.label_ids
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Load models
model_fine_tuned = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH_FINE_TUNED)
model_base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_BASE, num_labels=2)

# Evaluate both
results_fine_tuned = evaluate_model(model_fine_tuned, test_dataset)
results_base = evaluate_model(model_base, test_dataset)

print("Fine-Tuned BioBERT:")
print(results_fine_tuned)
print("\nBase BioBERT (no fine-tuning):")
print(results_base)

# Predict again for confusion matrix
def get_predictions(model, dataset):
    trainer = Trainer(model=model, tokenizer=tokenizer)
    preds_output = trainer.predict(dataset)
    preds = np.argmax(preds_output.predictions, axis=1)
    labels = preds_output.label_ids
    return preds, labels

# Get predictions and labels
preds_fine, labels_fine = get_predictions(model_fine_tuned, test_dataset)
preds_base, labels_base = get_predictions(model_base, test_dataset)

# Confusion Matrix Plotter
def plot_confusion_matrix(preds, labels, title):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

# Plot Confusion Matrices
plot_confusion_matrix(preds_fine, labels_fine, "Confusion Matrix: Fine-Tuned BioBERT")
plot_confusion_matrix(preds_base, labels_base, "Confusion Matrix: Base BioBERT")

# Bar Chart Comparison
metrics = ["accuracy", "precision", "recall", "f1"]
fine_vals = [results_fine_tuned[m] for m in metrics]
base_vals = [results_base[m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, fine_vals, width, label="Fine-Tuned")
plt.bar(x + width/2, base_vals, width, label="Base")
plt.xticks(x, metrics)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Model Performance Comparison")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()
