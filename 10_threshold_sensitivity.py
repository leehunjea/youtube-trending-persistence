"""
Additional KCI robustness experiment: viral-threshold sensitivity.

The main label defines viral videos as the top quartile by trending_snapshots.
Reviewers may see that cutoff as arbitrary, so this script repeats the full
K=14 CatBoost experiment across nearby percentile cutoffs.
"""
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


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


def evaluate(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def main():
    print("=" * 72)
    print("  KCI Additional Experiment: Viral Threshold Sensitivity")
    print("=" * 72)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] load features and labels")

    with open(f"{OUTPUT_PATH}/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    seq_features = scalers["seq_features"]
    static_features = scalers["static_features"]

    data = np.load(f"{OUTPUT_PATH}/kr_sequences_K14.npz", allow_pickle=True)
    X = make_ml_features(data["X_seq"], data["X_static"], seq_features, static_features)
    video_ids = data["video_ids"].astype(str)
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    trainval_idx = np.concatenate([train_idx, val_idx])

    video_df = pd.read_csv(f"{OUTPUT_PATH}/kr_video_labeled.csv", encoding="utf-8-sig")
    video_df["video_id"] = video_df["video_id"].astype(str)
    snapshots = (
        video_df.set_index("video_id")
        .loc[video_ids, "trending_snapshots"]
        .astype(float)
        .values
    )

    X_train = X.iloc[trainval_idx].values
    X_test = X.iloc[test_idx].values

    rows = []
    prediction_rows = []
    percentiles = [65, 70, 75, 80, 85]

    for percentile in percentiles:
        cutoff = float(np.percentile(snapshots, percentile))
        y = (snapshots >= cutoff).astype(int)
        y_train = y[trainval_idx]
        y_test = y[test_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"  skip p{percentile}: single-class split")
            continue

        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            verbose=0,
            random_seed=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        train_prob = model.predict_proba(X_train)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
        threshold = best_f1_threshold(y_train, train_prob)
        metrics = evaluate(y_test, test_prob, threshold)

        row = {
            "percentile": percentile,
            "cutoff_snapshots": cutoff,
            "train_positive_rate": y_train.mean(),
            "test_positive_rate": y_test.mean(),
            "threshold": threshold,
            **metrics,
        }
        rows.append(row)
        prediction_rows.append(
            pd.DataFrame(
                {
                    "percentile": percentile,
                    "cutoff_snapshots": cutoff,
                    "video_id": video_ids[test_idx],
                    "y_true": y_test,
                    "y_pred": test_prob,
                }
            )
        )
        print(
            f"  p{percentile:<2} cutoff={cutoff:5.1f} "
            f"pos_test={y_test.mean():.3f} AUC={metrics['auc']:.4f} "
            f"AP={metrics['average_precision']:.4f}"
        )

    result = pd.DataFrame(rows)
    result.to_csv(f"{OUTPUT_PATH}/threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        f"{OUTPUT_PATH}/threshold_predictions.csv", index=False, encoding="utf-8-sig"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(result["percentile"], result["auc"], marker="o", label="ROC-AUC")
    plt.plot(
        result["percentile"],
        result["average_precision"],
        marker="s",
        label="Average precision",
    )
    plt.plot(result["percentile"], result["f1"], marker="^", label="F1")
    plt.xlabel("Viral label percentile cutoff")
    plt.ylabel("Test metric")
    plt.ylim(0.5, 1.0)
    plt.title("Threshold Sensitivity of CatBoost K=14")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURE_PATH}/fig8_threshold_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] saved")
    print(f"  {OUTPUT_PATH}/threshold_sensitivity.csv")
    print(f"  {OUTPUT_PATH}/threshold_predictions.csv")
    print(f"  {FIGURE_PATH}/fig8_threshold_sensitivity.png")
    print("\nResults:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
