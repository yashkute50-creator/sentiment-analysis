# Sentiment Analysis (Sentiment140) — Baseline vs DistilBERT

This project compares:
1) TF-IDF + Logistic Regression baseline
2) DistilBERT fine-tuning (Transformers)

Dataset: Sentiment140 (1.6M tweets)

## Project structure
- `notebooks/01_eda.ipynb` : load/EDA + baseline split work
- `notebooks/02_bert_training.ipynb` : DistilBERT fine-tuning
- `src/preprocessing.py` : reusable split + basic cleaning
- `data/raw/` : raw dataset (do not modify)
- `outputs/` : saved models/metrics (generated)

## Setup (macOS)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset
Download from Kaggle (Sentiment140):

Put file here:
data/raw/training.1600000.processed.noemoticon.csv

## Run (recommended)
Open notebooks in VS Code:

Run notebooks/01_eda.ipynb to reproduce TF-IDF+LR baseline results

Run notebooks/02_bert_training.ipynb to reproduce DistilBERT results

## Results (your numbers)
TF-IDF + Logistic Regression (200k): Accuracy 0.7925, F1 0.79

DistilBERT (20k, 1 epoch, max_len=64): Accuracy 0.8035, F1 0.8055

## Notes
If ModuleNotFoundError: src, add this cell at top of notebook:

```python
import os, sys
sys.path.append(os.path.abspath(".."))
```

Also add:
```bash
mkdir -p outputs
touch src/__init__.py
```
