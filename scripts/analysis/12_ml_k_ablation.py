"""
K-wise ablation for classical ML models.

This complements the DL-only K ablation table by training the same feature
construction and the same ML model family across K=5, 7, 10, 14.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


OUTPUT = Path("./output")
RANDOM_STATE = 42
K_VALUES = [5, 7, 10, 14]


def make_ml_features(X_seq, X_static, seq_features, static_features):
    rows = []
    k = X_seq.shape[1]
    available_seq = seq_features[: X_seq.shape[2]]

    for i in range(len(X_seq)):
        seq = X_seq[i]
        st = X_static[i]
        feat = {}

        for j, name in enumerate(static_features):
            feat[name] = st[j]

        for j, name in enumerate(available_seq):
            col = seq[:, j]
            feat[f"{name}_first"] = col[0]
            feat[f"{name}_last"] = col[-1]
            feat[f"{name}_mean"] = col.mean()
            feat[f"{name}_std"] = col.std()
            feat[f"{name}_max"] = col.max()
            feat[f"{name}_min"] = col.min()
            feat[f"{name}_range"] = col.max() - col.min()
            feat[f"{name}_slope"] = np.polyfit(range(k), col, 1)[0] if k > 1 else 0

        if {"rank", "views_per_hour", "momentum", "rank_delta", "views_acceleration"}.issubset(
            set(available_seq)
        ):
            rank_col = seq[:, available_seq.index("rank")]
            vph_col = seq[:, available_seq.index("views_per_hour")]
            mom_col = seq[:, available_seq.index("momentum")]
            rd_col = seq[:, available_seq.index("rank_delta")]
            va_col = seq[:, available_seq.index("views_acceleration")]
            feat["rank_improvement"] = rank_col[0] - rank_col[-1]
            feat["peak_momentum"] = mom_col.max()
            feat["momentum_trend"] = mom_col[-1] - mom_col[0]
            feat["positive_rank_delta"] = (rd_col < 0).sum()
            feat["max_acceleration"] = va_col.max()
            feat["sustained_growth"] = (va_col > 0).sum() / k
            feat["vph_growth_rate"] = vph_col[-1] / (vph_col[0] + 1e-6)

        rows.append(feat)

    return pd.DataFrame(rows)


def build_models(scale_pos_weight):
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    }

    try:
        import xgboost as xgb

        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            verbosity=0,
            random_state=RANDOM_STATE,
        )
    except Exception as e:
        print(f"skip XGBoost: {e}")

    try:
        import lightgbm as lgb

        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            verbosity=-1,
            random_state=RANDOM_STATE,
        )
    except Exception as e:
        print(f"skip LightGBM: {e}")

    try:
        from catboost import CatBoostClassifier

        models["CatBoost"] = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            verbose=0,
            random_seed=RANDOM_STATE,
        )
    except Exception as e:
        print(f"skip CatBoost: {e}")

    return models


def main():
    with open(OUTPUT / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    seq_features = scalers["seq_features"]
    static_features = scalers["static_features"]
    rows = []

    for k in K_VALUES:
        print(f"\nK={k}")
        data = np.load(OUTPUT / f"kr_sequences_K{k}.npz", allow_pickle=True)
        X = make_ml_features(data["X_seq"], data["X_static"], seq_features, static_features)
        y = data["y_cls"].astype(int)
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]
        test_idx = data["test_idx"]
        trainval_idx = np.concatenate([train_idx, val_idx])

        X_train = X.iloc[trainval_idx].values
        X_test = X.iloc[test_idx].values
        y_train = y[trainval_idx]
        y_test = y[test_idx]
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        for model_name, model in build_models(scale_pos).items():
            model.fit(X_train, y_train)
            pred = model.predict_proba(X_test)[:, 1]
            y_hat = (pred >= 0.5).astype(int)
            row = {
                "model": model_name,
                "K_EARLY": k,
                "n_features": X.shape[1],
                "roc_auc": roc_auc_score(y_test, pred),
                "f1": f1_score(y_test, y_hat),
                "precision": precision_score(y_test, y_hat, zero_division=0),
                "recall": recall_score(y_test, y_hat, zero_division=0),
            }
            rows.append(row)
            print(f"  {model_name:<14} AUC={row['roc_auc']:.4f} F1={row['f1']:.4f}")

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "ml_k_ablation_results.csv", index=False, encoding="utf-8-sig")

    pivot = result.pivot_table(index="model", columns="K_EARLY", values="roc_auc").reset_index()
    pivot.to_csv(OUTPUT / "ml_k_ablation_auc_table.csv", index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print(f"  {OUTPUT / 'ml_k_ablation_results.csv'}")
    print(f"  {OUTPUT / 'ml_k_ablation_auc_table.csv'}")
    print("\nAUC table:")
    print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()
