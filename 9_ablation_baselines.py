"""
Additional KCI robustness experiment: simple baselines and feature-set ablations.

The main paper result uses full K=14 sequence summaries with engineered trend
features. This script quantifies how much each feature family contributes, using
the same train/val/test split and the same K=14 sequence file as the main run.
"""
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUTPUT_PATH = "./output"
FIGURE_PATH = "./figures"
RANDOM_STATE = 42

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(FIGURE_PATH, exist_ok=True)


def make_ml_features(X_seq, X_static, seq_features, static_features):
    rows = []
    k = X_seq.shape[1]
    for i in range(len(X_seq)):
        seq = X_seq[i]
        st = X_static[i]
        feat = {}

        for j, name in enumerate(static_features):
            feat[name] = st[j]

        for j, name in enumerate(seq_features):
            col = seq[:, j]
            feat[f"{name}_first"] = col[0]
            feat[f"{name}_last"] = col[-1]
            feat[f"{name}_mean"] = col.mean()
            feat[f"{name}_std"] = col.std()
            feat[f"{name}_max"] = col.max()
            feat[f"{name}_min"] = col.min()
            feat[f"{name}_range"] = col.max() - col.min()
            feat[f"{name}_slope"] = np.polyfit(range(k), col, 1)[0] if k > 1 else 0

        rank_col = seq[:, seq_features.index("rank")]
        vph_col = seq[:, seq_features.index("views_per_hour")]
        mom_col = seq[:, seq_features.index("momentum")]
        rd_col = seq[:, seq_features.index("rank_delta")]
        va_col = seq[:, seq_features.index("views_acceleration")]

        feat["rank_improvement"] = rank_col[0] - rank_col[-1]
        feat["peak_momentum"] = mom_col.max()
        feat["momentum_trend"] = mom_col[-1] - mom_col[0]
        feat["positive_rank_delta"] = (rd_col < 0).sum()
        feat["max_acceleration"] = va_col.max()
        feat["sustained_growth"] = (va_col > 0).sum() / k
        feat["vph_growth_rate"] = vph_col[-1] / (vph_col[0] + 1e-6)
        rows.append(feat)

    return pd.DataFrame(rows)


def best_f1_threshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    if len(thresholds) == 0:
        return 0.5
    return float(thresholds[int(np.nanargmax(f1[:-1]))])


def evaluate_scores(y_true, y_score, threshold=None):
    if threshold is None:
        threshold = 0.5
    y_pred = (y_score >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }


def fit_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    return model.decision_function(X_test)


def catboost_model(scale_pos_weight):
    return CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        verbose=0,
        random_seed=RANDOM_STATE,
    )


def main():
    print("=" * 72)
    print("  KCI Additional Experiment: Baselines and Feature Ablations")
    print("=" * 72)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] load K=14 sequence features")

    with open(f"{OUTPUT_PATH}/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    seq_features = scalers["seq_features"]
    static_features = scalers["static_features"]

    data = np.load(f"{OUTPUT_PATH}/kr_sequences_K14.npz", allow_pickle=True)
    X_seq = data["X_seq"]
    X_static = data["X_static"]
    y = data["y_cls"].astype(int)
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    video_ids = data["video_ids"].astype(str)
    trainval_idx = np.concatenate([train_idx, val_idx])

    X = make_ml_features(X_seq, X_static, seq_features, static_features)
    scale_pos = (y[trainval_idx] == 0).sum() / (y[trainval_idx] == 1).sum()

    basic_seq = ["rank", "views_per_hour", "comment_view_ratio", "hours_since_publish"]
    delta_seq = ["rank_delta", "views_acceleration", "momentum"]
    engineered_delta = [
        "rank_improvement",
        "peak_momentum",
        "momentum_trend",
        "positive_rank_delta",
        "max_acceleration",
        "sustained_growth",
        "vph_growth_rate",
    ]

    def summary_cols(names):
        suffixes = ["first", "last", "mean", "std", "max", "min", "range", "slope"]
        return [f"{name}_{suffix}" for name in names for suffix in suffixes]

    feature_sets = {
        "Static only": list(static_features),
        "First snapshot only": list(static_features) + [f"{name}_first" for name in basic_seq],
        "Basic sequence summaries": list(static_features) + summary_cols(basic_seq),
        "No delta/momentum": list(static_features) + summary_cols(basic_seq),
        "Delta/growth only": summary_cols(delta_seq) + engineered_delta,
        "Full ML v2 features": list(X.columns),
    }

    rows = []
    prediction_rows = []

    print(f"[{datetime.now().strftime('%H:%M:%S')}] simple score baselines")
    score_baselines = {
        "Score: first rank": -X["rank_first"].values,
        "Score: first views/hour": X["views_per_hour_first"].values,
        "Score: last rank": -X["rank_last"].values,
        "Score: last views/hour": X["views_per_hour_last"].values,
    }
    for name, score in score_baselines.items():
        threshold = best_f1_threshold(y[trainval_idx], score[trainval_idx])
        metrics = evaluate_scores(y[test_idx], score[test_idx], threshold)
        rows.append(
            {
                "experiment": name,
                "model": "Univariate score",
                "n_features": 1,
                **metrics,
            }
        )

    print(f"[{datetime.now().strftime('%H:%M:%S')}] logistic and CatBoost ablations")
    for set_name, cols in feature_sets.items():
        cols = [c for c in cols if c in X.columns]
        X_train = X.iloc[trainval_idx][cols].values
        X_test = X.iloc[test_idx][cols].values
        y_train = y[trainval_idx]
        y_test = y[test_idx]

        logit = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
        )
        logit_prob_train = fit_predict(logit, X_train, y_train, X_train)
        logit_prob = logit.predict_proba(X_test)[:, 1]
        logit_threshold = best_f1_threshold(y_train, logit_prob_train)
        metrics = evaluate_scores(y_test, logit_prob, logit_threshold)
        rows.append(
            {
                "experiment": set_name,
                "model": "Logistic regression",
                "n_features": len(cols),
                **metrics,
            }
        )

        cb = catboost_model(scale_pos)
        cb.fit(X_train, y_train)
        cb_train = cb.predict_proba(X_train)[:, 1]
        cb_prob = cb.predict_proba(X_test)[:, 1]
        cb_threshold = best_f1_threshold(y_train, cb_train)
        metrics = evaluate_scores(y_test, cb_prob, cb_threshold)
        rows.append(
            {
                "experiment": set_name,
                "model": "CatBoost",
                "n_features": len(cols),
                **metrics,
            }
        )

        prediction_rows.append(
            pd.DataFrame(
                {
                    "experiment": set_name,
                    "model": "CatBoost",
                    "video_id": video_ids[test_idx],
                    "y_true": y_test,
                    "y_pred": cb_prob,
                }
            )
        )
        print(f"  {set_name:<26} CatBoost AUC={metrics['auc']:.4f}")

    result = pd.DataFrame(rows).sort_values(["auc", "average_precision"], ascending=False)
    result.to_csv(f"{OUTPUT_PATH}/ablation_baselines.csv", index=False, encoding="utf-8-sig")
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        f"{OUTPUT_PATH}/ablation_predictions.csv", index=False, encoding="utf-8-sig"
    )

    plot_df = result[result["model"].isin(["CatBoost", "Univariate score"])].copy()
    plot_df["label"] = plot_df["experiment"] + "\n(" + plot_df["model"] + ")"
    plot_df = plot_df.sort_values("auc", ascending=True)

    plt.figure(figsize=(10, 6))
    colors = ["#7c8597" if m == "Univariate score" else "#2f80ed" for m in plot_df["model"]]
    plt.barh(plot_df["label"], plot_df["auc"], color=colors)
    plt.xlim(0.5, max(0.9, plot_df["auc"].max() + 0.02))
    plt.xlabel("ROC-AUC")
    plt.title("Baseline and Feature-Set Ablation (K=14 Test Set)")
    for i, value in enumerate(plot_df["auc"]):
        plt.text(value + 0.003, i, f"{value:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_PATH}/fig7_ablation_baselines.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] saved")
    print(f"  {OUTPUT_PATH}/ablation_baselines.csv")
    print(f"  {OUTPUT_PATH}/ablation_predictions.csv")
    print(f"  {FIGURE_PATH}/fig7_ablation_baselines.png")
    print("\nTop rows:")
    print(result.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
