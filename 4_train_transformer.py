"""
STEP 4: Fine-tune DistilBERT
------------------------------
This is your "deep learning" model that you'll compare against the TF-IDF baseline.

TIME-SAVING NOTES (since you have 1 week):
  - Uses DistilBERT (40% smaller/faster than BERT, ~97% of its performance)
  - Trains on a SUBSAMPLE of the data by default (SAMPLE_SIZE below) so it finishes
    in a reasonable time on Colab's free GPU. Increase it if you have more time/compute.
  - Only 2 epochs by default — enough to show a meaningful result without hours of training.

HOW TO RUN:
  - Strongly recommended: run this in Google Colab with a free GPU
    (Runtime > Change runtime type > GPU)
  - Upload data/processed_news.csv to Colab, adjust DATA_PATH if needed
  - Run: python 4_train_transformer.py

Output: models/distilbert_fake_news/ (saved model + tokenizer)
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import os

DATA_PATH = "data/processed_news.csv"
MODEL_OUT = "models/distilbert_fake_news"
SAMPLE_SIZE = 8000       # reduce this if training is too slow (e.g. 4000); increase if you have time/GPU
MAX_LEN = 256
EPOCHS = 2
BATCH_SIZE = 16

os.makedirs("models", exist_ok=True)

# ---- Load & sample data ----
df = pd.read_csv(DATA_PATH)
if len(df) > SAMPLE_SIZE:
    df = df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(min(len(x), SAMPLE_SIZE // 2), random_state=42)
    ).reset_index(drop=True)

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

train_ds = Dataset.from_pandas(train_df[["clean_content", "label"]].rename(columns={"clean_content": "text"}))
test_ds = Dataset.from_pandas(test_df[["clean_content", "label"]].rename(columns={"clean_content": "text"}))

# ---- Tokenizer ----
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

train_ds = train_ds.map(tokenize_fn, batched=True)
test_ds = test_ds.map(tokenize_fn, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ---- Model ----
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

training_args = TrainingArguments(
    output_dir="models/checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    fp16=torch.cuda.is_available(),   # speeds up training on GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    print(f"Training on {len(train_df)} samples, evaluating on {len(test_df)} samples...")
    print(f"GPU available: {torch.cuda.is_available()}")

    trainer.train()
    results = trainer.evaluate()
    print("\n=== Final Evaluation ===")
    print(results)

    trainer.save_model(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")
    print("Compare these accuracy/F1 numbers to your TF-IDF baseline for your report!")
