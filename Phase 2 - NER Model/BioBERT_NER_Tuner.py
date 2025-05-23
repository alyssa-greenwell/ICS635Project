# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 11:19:26 2025

This file creates a fine-tuned NER classifier based on the BioBERT model.
It creates a local directory that has the model.
"""

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer

# Parameters for ablation studies
max_tokens = 512; # Recommended range 128 to 512.
num_epochs = 3; # Recommended range 3 to 5.
learning_rate = 2e-5; # Recommended range 2e-5 to 5e-5.

# Function to Load CoNLL file
def load_conll(file_path):
    sentences = []
    labels = []
    words = []
    tags = []

    #Parse CoNLL Data into arrays for training
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                if words:
                    sentences.append(words)
                    labels.append(tags)
                    words = []
                    tags = []
            else:
                parts = line.strip().split('\t')
                if len(parts) >= 5:  # Check for a label
                    word = parts[0]
                    ner_tag = parts[4]
                    words.append(word)
                    tags.append(ner_tag)
        if words:
            sentences.append(words)
            labels.append(tags)

    return sentences, labels

train_sentences, train_labels = load_conll("ner_training_data.conll") #ADD TRAINING DATA HERE

# Map labels to integers for training with functions to translate between them
unique_labels = ["O", "B-CANCER_TYPE", "I-CANCER_TYPE", "B-GENOMIC_DATA_TYPE", 
              "I-GENOMIC_DATA_TYPE", "B-SAMPLE_COUNT", "I-SAMPLE_COUNT", 
              "B-DATA_ACCESSION", "I-DATA_ACCESSION", "B-DATA_SOURCE", 
              "I-DATA_SOURCE"] 
unique_labels.sort()
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

print(label2id)

#tokenize sentences and align labels
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

def tokenize_and_align_labels(sentences, labels):
    tokenized_inputs = tokenizer(
        sentences,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_tokens, #Can be edited for ablations
        return_tensors="pt"
    )

    all_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(label2id[label[word_idx]])  # or -100 to ignore subword tokens
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

# Hugging Face Dataset
train_dataset = Dataset.from_dict(tokenize_and_align_labels(train_sentences, train_labels))

#Load BioBERT model
model = AutoModelForTokenClassification.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1",
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id,
)

#Build the trainer
training_args = TrainingArguments(
    output_dir="./biobert-ner",
    save_strategy="epoch",
    learning_rate=learning_rate, #can be edited for ablations
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=num_epochs, #can be edited for ablations
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

#Fine-Tune
trainer.train()

#Save the fine-tuned model
trainer.save_model("./biobert-ner-model")
tokenizer.save_pretrained("./biobert-ner-model")