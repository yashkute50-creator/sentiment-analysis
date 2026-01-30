# src/train.py
import os
import json
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from .preprocessing import clean_text_basic, split_data, set_seed

DATA_PATH = "data/raw/training.1600000.processed.noemoticon.csv"
OUT_DIR = "outputs/baseline"
SAMPLE_PER_CLASS = 50000  # 100k total


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", header=None)
    df.columns = ["target", "ids", "date", "flag", "user", "text"]
    df = df[["target", "text"]].dropna()
    df["sentiment"] = df["target"].map({0: 0, 4: 1})
    df = df.drop(columns=["target"])
    return df


def main():
    set_seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load full dataset
    df = load_data(DATA_PATH)

    # Balanced sampling
    df_small = (
        df.groupby("sentiment", group_keys=False)
          .sample(n=SAMPLE_PER_CLASS, random_state=42)
          .reset_index(drop=True)
    )

    # Clean for classical baseline
    df_small_clean = df_small.copy()
    df_small_clean["text"] = df_small_clean["text"].apply(clean_text_basic)

    # Split (expects columns: text, sentiment)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_small_clean[["text", "sentiment"]])

    # Vectorize
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 1), min_df=5)
    X_train_t = tfidf.fit_transform(X_train)
    X_val_t = tfidf.transform(X_val)
    X_test_t = tfidf.transform(X_test)

    # Train
    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X_train_t, y_train)

    # Evaluate on test
    test_preds = clf.predict(X_test_t)

    metrics = {
        "model": "tfidf_logreg",
        "sample_per_class": SAMPLE_PER_CLASS,
        "test_accuracy": float(accuracy_score(y_test, test_preds)),
        "test_f1": float(f1_score(y_test, test_preds)),
        "confusion_matrix": confusion_matrix(y_test, test_preds).tolist(),
        "classification_report": classification_report(y_test, test_preds, output_dict=True),
    }

    # Save artifacts
    joblib.dump(tfidf, os.path.join(OUT_DIR, "tfidf.joblib"))
    joblib.dump(clf, os.path.join(OUT_DIR, "logreg.joblib"))
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
