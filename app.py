import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Sentiment Demo", layout="wide")

BASELINE_DIR = "outputs/baseline"
BERT_DIR = "outputs/bert"
BERT_MODEL_DIR = os.path.join(BERT_DIR, "model")


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


@st.cache_resource
def load_baseline():
    tfidf_path = os.path.join(BASELINE_DIR, "tfidf.joblib")
    clf_path = os.path.join(BASELINE_DIR, "logreg.joblib")
    tfidf = joblib.load(tfidf_path)
    clf = joblib.load(clf_path)
    return tfidf, clf


@st.cache_resource
def load_bert():
    tok = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    return tok, model


def predict_baseline(text, tfidf, clf):
    X = tfidf.transform([text])
    pred = int(clf.predict(X)[0])
    return pred


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict_bert(text, tok, model, max_len=64):
    inputs = tok(text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_len)
    with st.spinner("Running BERT..."):
        out = model(**inputs)
    logits = out.logits.detach().numpy()[0]
    probs = softmax(logits)
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))
    prob_pos = float(probs[1])
    return pred, conf, prob_pos


def label_name(y):
    return "Positive" if y == 1 else "Negative"


st.title("Sentiment Analysis Demo (UI only)")

# ---- Metrics row ----
col1, col2 = st.columns(2)

baseline_metrics = load_json(os.path.join(BASELINE_DIR, "metrics.json"))
bert_metrics = load_json(os.path.join(BERT_DIR, "metrics.json"))

with col1:
    st.subheader("Baseline (TF-IDF + Logistic Regression)")
    if baseline_metrics:
        st.metric("Test accuracy", f"{baseline_metrics['test_accuracy']:.4f}")
        st.metric("Test F1", f"{baseline_metrics['test_f1']:.4f}")
    else:
        st.info("Run: python -m src.train")

with col2:
    st.subheader("BERT (DistilBERT fine-tuned)")
    if bert_metrics:
        # Trainer keys are usually eval_accuracy/eval_f1
        acc = bert_metrics.get("eval_accuracy")
        f1 = bert_metrics.get("eval_f1")
        if acc is not None: st.metric("Test accuracy", f"{acc:.4f}")
        if f1 is not None: st.metric("Test F1", f"{f1:.4f}")
    else:
        st.info("Run: python -m src.train_bert")


st.divider()

# ---- Try it ----
st.subheader("Try a text")
text = st.text_area("Enter a tweet/text", value="I love this!", height=80)

run = st.button("Predict")

if run:
    if not os.path.exists(os.path.join(BASELINE_DIR, "tfidf.joblib")):
        st.error("Baseline model not found. Run: python -m src.train")
    elif not os.path.exists(BERT_MODEL_DIR):
        st.error("BERT model not found. Run: python -m src.train_bert")
    else:
        tfidf, clf = load_baseline()
        tok, model = load_bert()

        b_pred = predict_baseline(text, tfidf, clf)
        bert_pred, bert_conf, bert_prob_pos = predict_bert(text, tok, model)

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Baseline prediction**")
            st.success(label_name(b_pred))
        with c2:
            st.write("**BERT prediction**")
            st.success(f"{label_name(bert_pred)} (confidence={bert_conf:.2f}, prob_pos={bert_prob_pos:.2f})")

st.divider()

# ---- Misclassified table (BERT) ----
st.subheader("Misclassified examples (BERT test set)")
pred_csv = os.path.join(BERT_DIR, "test_predictions.csv")
if os.path.exists(pred_csv):
    dfp = pd.read_csv(pred_csv)
    only_wrong = st.checkbox("Show only wrong predictions", value=True)
    show_n = st.slider("Rows", 10, 200, 50)

    view = dfp[dfp["correct"] == False] if only_wrong else dfp
    st.dataframe(view[["text", "label", "pred", "confidence", "prob_pos", "correct"]].head(show_n), use_container_width=True)
else:
    st.info("No predictions CSV found. Run: python -m src.train_bert")
