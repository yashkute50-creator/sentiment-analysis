# src/train_bert.py
import os
import json
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

DATA_PATH = "data/raw/training.1600000.processed.noemoticon.csv"
OUT_DIR = "outputs/bert"
MODEL_NAME = "distilbert-base-uncased"

SAMPLE_PER_CLASS = 10000   # 20k total (fast)
MAX_LEN = 64
EPOCHS = 1
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", header=None)
    df.columns = ["target", "ids", "date", "flag", "user", "text"]
    df = df[["target", "text"]].dropna()
    df["label"] = df["target"].map({0: 0, 4: 1})
    return df[["text", "label"]]


def split_stratified(df: pd.DataFrame):
    from sklearn.model_selection import train_test_split

    X = df["text"]
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )

    train_df = pd.DataFrame({"text": X_train.values, "label": y_train.values})
    val_df   = pd.DataFrame({"text": X_val.values,   "label": y_val.values})
    test_df  = pd.DataFrame({"text": X_test.values,  "label": y_test.values})
    return train_df, val_df, test_df


def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_data(DATA_PATH)

    # balanced sample
    df_small = (
        df.groupby("label", group_keys=False)
          .sample(n=SAMPLE_PER_CLASS, random_state=SEED)
          .reset_index(drop=True)
    )

    train_df, val_df, test_df = split_stratified(df_small)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )

    train_ds = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True, remove_columns=["text"])
    val_ds   = Dataset.from_pandas(val_df).map(tokenize_fn, batched=True, remove_columns=["text"])
    test_ds  = Dataset.from_pandas(test_df).map(tokenize_fn, batched=True, remove_columns=["text"])

    train_ds.set_format("torch")
    val_ds.set_format("torch")
    test_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"],
        }

    args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=25,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_metrics = trainer.evaluate(test_ds)

    # save model + tokenizer
    model_dir = os.path.join(OUT_DIR, "model")
    os.makedirs(model_dir, exist_ok=True)
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    # save metrics + test predictions
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    pred_out = trainer.predict(test_ds)
    logits = pred_out.predictions
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    preds = logits.argmax(axis=1)

    results = test_df.reset_index(drop=True).copy()
    results["pred"] = preds
    results["correct"] = results["pred"] == results["label"]
    results["prob_pos"] = probs[:, 1]
    results["confidence"] = probs.max(axis=1)

    results.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    print(json.dumps(test_metrics, indent=2))
    print("Saved:", os.path.join(OUT_DIR, "test_predictions.csv"))


if __name__ == "__main__":
    main()