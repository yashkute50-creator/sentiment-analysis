# src/evaluate.py
import os
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

DATA_PATH = "data/raw/training.1600000.processed.noemoticon.csv"
MODEL_DIR = "outputs/baseline"
OUT_CSV = os.path.join(MODEL_DIR, "sample_predictions.csv")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", header=None)
    df.columns = ["target", "ids", "date", "flag", "user", "text"]
    df = df[["target", "text"]].dropna()
    df["label"] = df["target"].map({0: 0, 4: 1})
    return df[["text", "label"]]


def main(sample_size: int = 2000):
    tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf.joblib"))
    clf = joblib.load(os.path.join(MODEL_DIR, "logreg.joblib"))

    df = load_data(DATA_PATH).sample(sample_size, random_state=42).reset_index(drop=True)

    X = tfidf.transform(df["text"])
    df["pred"] = clf.predict(X)
    df["correct"] = df["pred"] == df["label"]

    print("Sample accuracy:", accuracy_score(df["label"], df["pred"]))
    print(df[["text", "label", "pred", "correct"]].head(20))

    os.makedirs(MODEL_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # Optional: write a small metrics summary too
    summary = {
        "sample_size": sample_size,
        "sample_accuracy": float(accuracy_score(df["label"], df["pred"])),
        "misclassified": int((~df["correct"]).sum()),
        "output_csv": OUT_CSV,
    }
    with open(os.path.join(MODEL_DIR, "sample_eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
