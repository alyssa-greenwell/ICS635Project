# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 11:35:10 2025

This file runs a fine-tuned BioBERT NER model against a base BioBERT model and
compares their performance using precision, recall, and f1-score as metrics,
and generates a bar chart displaying the f1-score by category as well as 
confusion matrices for both models.

Run the BioBERT_NER_Tuner file first to generate the model, then run this to
evaluate it.
"""

import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Setup
fine_tuned_model_path = "./biobert-ner-model"
base_model_name = "dmis-lab/biobert-base-cased-v1.1"
#List of all classes, currently there are the ones that I could find
label_list = ["O", "B-CANCER_TYPE", "I-CANCER_TYPE", "B-GENOMIC_DATA_TYPE", 
              "I-GENOMIC_DATA_TYPE", "B-SAMPLE_COUNT", "I-SAMPLE_COUNT", 
              "B-DATA_ACCESSION", "I-DATA_ACCESSION", "B-DATA_SOURCE", 
              "I-DATA_SOURCE"] 
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Function to load data
def load_conll(file_path):
    tokens, labels = [], []
    temp_tokens, temp_labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                if temp_tokens:
                    tokens.append(temp_tokens)
                    labels.append(temp_labels)
                    temp_tokens, temp_labels = [], []
            else:
                parts = line.strip().split("\t")
                temp_tokens.append(parts[0])
                temp_labels.append(parts[4])
    if temp_tokens:
        tokens.append(temp_tokens)
        labels.append(temp_labels)
    return {"tokens": tokens, "ner_tags": labels}

dataset = load_conll("ner_testing_data.conll")  # ADD TEST FILE HERE
test_dataset = Dataset.from_dict(dataset)

# Tokenize
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=128
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id.get(label[word_idx], label2id["O"]))
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_test_dataset.set_format("torch")

# Define metrics
def compute_metrics(pred):
    logits = pred.predictions
    labels = pred.label_ids
    predictions = np.argmax(logits, axis=2)

    true_labels = [
        [id2label[label.item()] for (pred, label) in zip(predictions[i], labels[i]) if label.item() != -100]
        for i in range(len(labels))
    ]
    pred_labels = [
        [id2label[pred] for (pred, label) in zip(predictions[i], labels[i]) if label.item() != -100]
        for i in range(len(labels))
    ]

    return {
        "f1": f1_score(true_labels, pred_labels),
        "report": classification_report(true_labels, pred_labels, output_dict=True),
    }

# Load models
fine_tuned_model = AutoModelForTokenClassification.from_pretrained(fine_tuned_model_path, num_labels=len(label_list))
base_model = AutoModelForTokenClassification.from_pretrained(base_model_name, num_labels=len(label_list))

# Evaluate
trainer_fine = Trainer(model=fine_tuned_model, tokenizer=tokenizer)
trainer_base = Trainer(model=base_model, tokenizer=tokenizer)

pred_fine = trainer_fine.predict(tokenized_test_dataset)
pred_base = trainer_base.predict(tokenized_test_dataset)

results_fine = compute_metrics(pred_fine)
results_base = compute_metrics(pred_base)

# Print reports
print("\n--- Fine-tuned Model Report ---")
print(classification_report(
    [[id2label[label.item()] for label in labels if label.item() != -100] for labels in pred_fine.label_ids],
    [[id2label[np.argmax(logits)] for logits, label in zip(batch, labels) if label.item() != -100] for batch, labels in zip(pred_fine.predictions, pred_fine.label_ids)],
))
print("\n--- Base Model Report ---")
print(classification_report(
    [[id2label[label.item()] for label in labels if label.item() != -100] for labels in pred_base.label_ids],
    [[id2label[np.argmax(logits)] for logits, label in zip(batch, labels) if label.item() != -100] for batch, labels in zip(pred_base.predictions, pred_base.label_ids)],
))

# Bar chart comparison
labels_to_plot = [label for label in label_list if label != "O"]

fine_scores = []
base_scores = []

for label in labels_to_plot:
    fine_scores.append(results_fine["report"].get(label, {}).get("f1", 0.0))
    base_scores.append(results_base["report"].get(label, {}).get("f1", 0.0))

x = np.arange(len(labels_to_plot))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, fine_scores, width, label="Fine-tuned")
rects2 = ax.bar(x + width/2, base_scores, width, label="Base")

ax.set_ylabel('F1 Score')
ax.set_title('NER Entity F1 Scores: Fine-tuned vs Base BioBERT')
ax.set_xticks(x)
ax.set_xticklabels(labels_to_plot, rotation=45, ha="right")
ax.legend()

fig.tight_layout()
plt.show()

# Confusion Matrices
# Data Processing
true_labels_fine = [
    id2label[label.item()]
    for labels in pred_fine.label_ids
    for label in labels
    if label.item() != -100
]
pred_labels_fine = [
    id2label[np.argmax(logit)]
    for batch, labels in zip(pred_fine.predictions, pred_fine.label_ids)
    for logit, label in zip(batch, labels)
    if label.item() != -100
]

true_labels_base = [
    id2label[label.item()]
    for labels in pred_base.label_ids
    for label in labels
    if label.item() != -100
]
pred_labels_base = [
    id2label[np.argmax(logit)]
    for batch, labels in zip(pred_base.predictions, pred_base.label_ids)
    for logit, label in zip(batch, labels)
    if label.item() != -100
]

# Create confusion matrices
cm_fine = confusion_matrix(true_labels_fine, pred_labels_fine, labels=labels_to_plot)
cm_base = confusion_matrix(true_labels_base, pred_labels_base, labels=labels_to_plot)

# Plot Fine-tuned model confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_fine, display_labels=labels_to_plot)
disp.plot(include_values=True, cmap='Blues', ax=ax, xticks_rotation='vertical')
ax.set_title('Confusion Matrix: Fine-tuned BioBERT')
plt.show()

# Plot Base model confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_base, display_labels=labels_to_plot)
disp.plot(include_values=True, cmap='Oranges', ax=ax, xticks_rotation='vertical')
ax.set_title('Confusion Matrix: Base BioBERT')
plt.show()
