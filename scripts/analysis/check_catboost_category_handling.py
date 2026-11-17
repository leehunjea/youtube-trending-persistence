from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"


def make_ml_features(X_seq, X_static, seq_features, static_features):
    rows = []
    for i in range(len(X_seq)):
        feat = {}
        st = X_static[i]
        for j, name in enumerate(static_features):
            feat[name] = st[j]

        seq = X_seq[i]
        for j, name in enumerate(seq_features):
            vals = seq[:, j]
            feat[f"{name}_first"] = vals[0]
            feat[f"{name}_last"] = vals[-1]
            feat[f"{name}_mean"] = vals.mean()
            feat[f"{name}_std"] = vals.std()
            feat[f"{name}_max"] = vals.max()
            feat[f"{name}_min"] = vals.min()
            feat[f"{name}_range"] = vals.max() - vals.min()
            feat[f"{name}_slope"] = np.polyfit(np.arange(len(vals)), vals, 1)[0] if len(vals) > 1 else 0
        rows.append(feat)
    return pd.DataFrame(rows)


def main():
    data = np.load(OUTPUT / "kr_sequences_K14.npz", allow_pickle=True)
    with open(OUTPUT / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    seq_features = scalers["seq_features"]
    static_features = scalers["static_features"]
    static_scaler = scalers["static_scaler"]

    X = make_ml_features(data["X_seq"], data["X_static"], seq_features, static_features)
    y = data["y_cls"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    # Recover original static features from the standardized first block.
    n_static = len(static_features)
    raw_static = static_scaler.inverse_transform(data["X_static"][:, :n_static])
    raw_static_df = pd.DataFrame(raw_static, columns=static_features)
    X_cat = X.copy()
    X_cat["category_id"] = raw_static_df["category_id"].round().astype(int).astype(str)

    scale_pos = float((y[train_idx] == 0).sum() / (y[train_idx] == 1).sum())

    common_params = dict(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        scale_pos_weight=scale_pos,
        early_stopping_rounds=50,
        random_seed=42,
        verbose=0,
    )

    results = []
    for label, frame, cat_features in [
        ("numeric_standardized_category", X, None),
        ("catboost_categorical_category", X_cat, ["category_id"]),
    ]:
        model = CatBoostClassifier(**common_params)
        if cat_features is None:
            model.fit(frame.iloc[train_idx], y[train_idx], eval_set=(frame.iloc[val_idx], y[val_idx]))
        else:
            model.fit(
                frame.iloc[train_idx],
                y[train_idx],
                eval_set=(frame.iloc[val_idx], y[val_idx]),
                cat_features=cat_features,
            )
        prob = model.predict_proba(frame.iloc[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        results.append(
            {
                "setting": label,
                "roc_auc": roc_auc_score(y[test_idx], prob),
                "ap": average_precision_score(y[test_idx], prob),
                "f1": f1_score(y[test_idx], pred),
                "best_iteration": model.get_best_iteration(),
            }
        )

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT / "catboost_category_handling_check.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
