from pathlib import Path
import pickle

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[2]
out = root / "output"

print("== label metadata ==")
print(pd.read_csv(out / "label_metadata.csv").to_string(index=False))

print("\n== K sequence files ==")
for k in [5, 7, 10, 14]:
    data = np.load(out / f"kr_sequences_K{k}.npz", allow_pickle=True)
    y = data["y_cls"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    print(
        f"K={k}: n={len(y)}, pos={int(y.sum())}, pos_rate={y.mean():.6f}, "
        f"train={len(train_idx)}({y[train_idx].mean():.6f}), "
        f"val={len(val_idx)}({y[val_idx].mean():.6f}), "
        f"test={len(test_idx)}({y[test_idx].mean():.6f})"
    )

print("\n== K14 feature dimensions ==")
data = np.load(out / "kr_sequences_K14.npz", allow_pickle=True)
with open(out / "scalers.pkl", "rb") as f:
    scalers = pickle.load(f)
seq_features = list(scalers["seq_features"])
static_features = list(scalers["static_features"])
print("seq_features", len(seq_features), seq_features)
print("static_features", len(static_features), static_features)
print("X_seq shape", data["X_seq"].shape)
print("X_static shape", data["X_static"].shape)
print("basic summary 8 * seq", 8 * len(seq_features))

print("\n== threshold sensitivity ==")
print(pd.read_csv(out / "threshold_sensitivity.csv").to_string(index=False))

print("\n== category handling check ==")
print(pd.read_csv(out / "catboost_category_handling_check.csv").to_string(index=False))
