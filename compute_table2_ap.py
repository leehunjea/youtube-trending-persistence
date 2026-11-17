from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "output"

files = [
    ("CatBoost", "preds_CatBoost_v2.csv"),
    ("XGBoost", "preds_XGB_v2.csv"),
    ("LightGBM", "preds_LGBM_v2.csv"),
    ("Random Forest", "preds_RF_v2.csv"),
    ("Transformer", "preds_Transformer_K14.csv"),
    ("LSTM", "preds_LSTM_K14.csv"),
    ("TFT", "preds_TFT_K14.csv"),
    ("GRU", "preds_GRU_K14.csv"),
]


def average_precision(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y)
    ranks = np.arange(1, len(y) + 1)
    precision = tp / ranks
    positives = y.sum()
    if positives == 0:
        return float("nan")
    return float((precision * y).sum() / positives)


rows = []
for model, filename in files:
    df = pd.read_csv(OUT / filename)
    ap = average_precision(df["y_true"], df["y_pred"])
    rows.append((model, ap))

for model, ap in rows:
    print(f"{model},{ap:.4f}")
