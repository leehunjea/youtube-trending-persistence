"""
Strict temporal split robustness check for KCI review.

This script rebuilds K=14 features from raw snapshot tables and fits all
scalers on the temporal-train period only. It avoids reusing scalers from the
main random split experiment.
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

OUTPUT_PATH = "./output"
K_EARLY = 14
RANDOM_STATE = 42

SEQ_FEATURES = [
    "rank",
    "views_per_hour",
    "comment_view_ratio",
    "hours_since_publish",
    "rank_delta",
    "views_acceleration",
    "momentum",
]

STATIC_FEATURES = [
    "category_id",
    "tag_count",
    "publish_hour_sin",
    "publish_hour_cos",
    "publish_weekday_sin",
    "publish_weekday_cos",
    "hours_to_trend",
    "relative_first_views",
]


def extract_expert_features(df_snap_vid, k):
    feat = {}
    for col_name in SEQ_FEATURES:
        col = df_snap_vid[col_name].values[:k]
        feat[f"{col_name}_slope"] = np.polyfit(range(len(col)), col, 1)[0] if len(col) > 1 else 0
        feat[f"{col_name}_mean"] = col.mean()
        feat[f"{col_name}_std"] = col.std()

    rank_col = df_snap_vid["rank"].values[:k]
    vph_col = df_snap_vid["views_per_hour"].values[:k]
    mom_col = df_snap_vid["momentum"].values[:k]
    rd_col = df_snap_vid["rank_delta"].values[:k]
    va_col = df_snap_vid["views_acceleration"].values[:k]

    feat["rank_improvement"] = rank_col[0] - rank_col[-1]
    feat["peak_momentum"] = mom_col.max()
    feat["momentum_trend"] = mom_col[-1] - mom_col[0]
    feat["positive_rank_delta"] = (rd_col < 0).sum()
    feat["max_acceleration"] = va_col.max()
    feat["sustained_growth"] = (va_col > 0).sum() / k
    feat["vph_growth_rate"] = vph_col[-1] / (vph_col[0] + 1e-6)
    return feat


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


def load_optional_models(scale_pos_weight):
    models = {
        "RF_temporal": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "MLP_temporal": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=300,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }

    try:
        import xgboost as xgb

        models["XGB_temporal"] = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            verbosity=0,
            random_state=RANDOM_STATE,
        )
    except Exception as e:
        print(f"  skip XGBoost: {e}")

    try:
        import lightgbm as lgb

        models["LGBM_temporal"] = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            verbosity=-1,
            random_state=RANDOM_STATE,
        )
    except Exception as e:
        print(f"  skip LightGBM: {e}")

    try:
        from catboost import CatBoostClassifier

        models["CatBoost_temporal"] = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            verbose=0,
            random_seed=RANDOM_STATE,
        )
    except Exception as e:
        print(f"  skip CatBoost: {e}")

    return models


def build_temporal_arrays(df_snap, df_video, split):
    grouped = df_snap.sort_values(["video_id", "collection_date"]).groupby("video_id")

    train_ids = df_video.loc[split == "train", "video_id"].values
    df_snap_train = df_snap[df_snap["video_id"].isin(train_ids)]
    df_snap_train_early = df_snap_train.groupby("video_id").head(K_EARLY)

    seq_scaler = StandardScaler()
    seq_scaler.fit(df_snap_train_early[SEQ_FEATURES].fillna(0))

    static_scaler = StandardScaler()
    static_scaler.fit(df_video.loc[split == "train", STATIC_FEATURES].fillna(0))

    expert_train = []
    grouped_train = df_snap_train.sort_values(["video_id", "collection_date"]).groupby("video_id")
    for vid in train_ids:
        if vid in grouped_train.groups:
            expert_train.append(extract_expert_features(grouped_train.get_group(vid), K_EARLY))

    expert_scaler = StandardScaler()
    expert_scaler.fit(pd.DataFrame(expert_train).fillna(0))

    df_snap_scaled = df_snap.copy()
    df_snap_scaled[SEQ_FEATURES] = seq_scaler.transform(df_snap_scaled[SEQ_FEATURES].fillna(0))
    grouped_scaled = df_snap_scaled.sort_values(["video_id", "collection_date"]).groupby("video_id")

    static_scaled = static_scaler.transform(df_video[STATIC_FEATURES].fillna(0))
    video_ids = df_video["video_id"].values
    y = df_video["is_viral"].values

    X_seq = np.zeros((len(video_ids), K_EARLY, len(SEQ_FEATURES)), dtype=np.float32)
    mask = np.zeros((len(video_ids), K_EARLY), dtype=np.float32)
    expert_rows = []

    for i, vid in enumerate(video_ids):
        vals = grouped_scaled.get_group(vid)[SEQ_FEATURES].values[:K_EARLY]
        X_seq[i, : len(vals), :] = vals
        mask[i, : len(vals)] = 1
        expert_rows.append(extract_expert_features(grouped.get_group(vid), K_EARLY))

    expert_scaled = expert_scaler.transform(pd.DataFrame(expert_rows).fillna(0))
    X_static = np.concatenate([static_scaled, expert_scaled], axis=1).astype(np.float32)
    return X_seq, X_static, mask, y, video_ids


def main():
    print("=" * 72)
    print("  Strict Temporal Split Validation (K=14, train-only scalers)")
    print("=" * 72)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] load data")
    df_snap = pd.read_csv(f"{OUTPUT_PATH}/kr_snapshot.csv", encoding="utf-8-sig")
    df_video = pd.read_csv(f"{OUTPUT_PATH}/kr_video_labeled.csv", encoding="utf-8-sig")
    df_snap["video_id"] = df_snap["video_id"].astype(str)
    df_video["video_id"] = df_video["video_id"].astype(str)
    df_snap["collection_date"] = pd.to_datetime(df_snap["collection_date"], errors="coerce")
    df_video["first_trending_date"] = df_video["video_id"].map(
        df_snap.groupby("video_id")["collection_date"].min()
    )

    df_snap = df_snap.sort_values(["video_id", "collection_date"]).copy()
    df_snap["rank_delta"] = df_snap.groupby("video_id")["rank"].diff().fillna(0)
    df_snap["views_acceleration"] = df_snap.groupby("video_id")["views_per_hour"].diff().fillna(0)
    first_vph = df_snap.groupby("video_id")["views_per_hour"].transform("first")
    df_snap["momentum"] = df_snap["views_per_hour"] / (first_vph + 1)

    snap_count = df_snap.groupby("video_id")["collection_date"].nunique()
    valid_ids = snap_count[snap_count >= K_EARLY].index
    df_video = df_video[df_video["video_id"].isin(valid_ids)].copy()
    df_video = df_video.sort_values(["first_trending_date", "video_id"]).reset_index(drop=True)

    n = len(df_video)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    split = np.array(["train"] * n, dtype=object)
    split[train_end:val_end] = "val"
    split[val_end:] = "test"
    df_video["temporal_split"] = split

    print(f"[{datetime.now().strftime('%H:%M:%S')}] build train-only scaled arrays")
    X_seq, X_static, _mask, y, video_ids = build_temporal_arrays(df_snap, df_video, split)
    X = make_ml_features(X_seq, X_static, SEQ_FEATURES, STATIC_FEATURES)

    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    test_idx = np.where(split == "test")[0]

    X_train = X.iloc[train_idx].values
    y_train = y[train_idx]
    X_val = X.iloc[val_idx].values
    y_val = y[val_idx]
    X_test = X.iloc[test_idx].values
    y_test = y[test_idx]

    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models = load_optional_models(scale_pos)

    rows = []
    for name, model in models.items():
        print(f"\n>>> {name}")
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
        test_bin = (test_prob >= 0.5).astype(int)

        row = {
            "model": name,
            "train_n": len(train_idx),
            "val_n": len(val_idx),
            "test_n": len(test_idx),
            "train_positive_rate": round(float(y_train.mean()), 4),
            "val_positive_rate": round(float(y_val.mean()), 4),
            "test_positive_rate": round(float(y_test.mean()), 4),
            "val_auc": round(roc_auc_score(y_val, val_prob), 4),
            "test_auc": round(roc_auc_score(y_test, test_prob), 4),
            "test_ap": round(average_precision_score(y_test, test_prob), 4),
            "test_f1": round(f1_score(y_test, test_bin), 4),
            "test_precision": round(precision_score(y_test, test_bin), 4),
            "test_recall": round(recall_score(y_test, test_bin), 4),
        }
        rows.append(row)
        print(
            f"  val_auc={row['val_auc']:.4f} "
            f"test_auc={row['test_auc']:.4f} "
            f"test_f1={row['test_f1']:.4f}"
        )

        pd.DataFrame(
            {
                "video_id": video_ids[test_idx],
                "y_true": y_test,
                "y_pred": test_prob,
            }
        ).to_csv(f"{OUTPUT_PATH}/preds_{name}.csv", index=False)

    result = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    result.to_csv(f"{OUTPUT_PATH}/temporal_split_results.csv", index=False, encoding="utf-8-sig")

    split_meta = df_video.groupby("temporal_split").agg(
        n=("video_id", "count"),
        positive_rate=("is_viral", "mean"),
        date_min=("first_trending_date", "min"),
        date_max=("first_trending_date", "max"),
    )
    split_meta.to_csv(f"{OUTPUT_PATH}/temporal_split_metadata.csv", encoding="utf-8-sig")

    print("\nSaved:")
    print(f"  {OUTPUT_PATH}/temporal_split_results.csv")
    print(f"  {OUTPUT_PATH}/temporal_split_metadata.csv")
    print(f"  {OUTPUT_PATH}/preds_*_temporal.csv")


if __name__ == "__main__":
    main()
