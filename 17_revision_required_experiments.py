"""
Revision experiments for KCI review response.

This script adds two defense-oriented experiments using the feature construction
that matches the 71-feature CatBoost ablation setup used in the manuscript:

1. Fixed-sample K experiment:
   Re-evaluate K=5,7,10,14 on the identical video set and split defined by K=14.
2. CatBoost category handling check:
   Compare standardized integer category_id with CatBoost native categorical
   handling under the same 71-feature pipeline.
"""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
FIGURES = ROOT / "figures" / "revision"
RANDOM_STATE = 42
K_VALUES = [5, 7, 10, 14]

FIGURES.mkdir(parents=True, exist_ok=True)


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


def evaluate(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, prob),
        "ap": average_precision_score(y_true, prob),
        "f1": f1_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
    }


def catboost_model(scale_pos_weight):
    return CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        verbose=0,
        random_seed=RANDOM_STATE,
    )


def build_models(scale_pos_weight):
    models = {
        "CatBoost": catboost_model(scale_pos_weight),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
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
    except Exception as exc:
        print(f"skip XGBoost: {exc}")

    try:
        import lightgbm as lgb

        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            verbosity=-1,
            random_state=RANDOM_STATE,
        )
    except Exception as exc:
        print(f"skip LightGBM: {exc}")

    return models


def load_scalers():
    with open(OUTPUT / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    return scalers


def fixed_sample_k_experiment(seq_features, static_features):
    base = np.load(OUTPUT / "kr_sequences_K14.npz", allow_pickle=True)
    base_video_ids = base["video_ids"].astype(str)
    base_trainval_ids = set(base_video_ids[np.concatenate([base["train_idx"], base["val_idx"]])])
    base_test_ids = set(base_video_ids[base["test_idx"]])
    base_all_ids = base_trainval_ids | base_test_ids

    rows = []
    for k in K_VALUES:
        data = np.load(OUTPUT / f"kr_sequences_K{k}.npz", allow_pickle=True)
        video_ids = data["video_ids"].astype(str)
        id_to_idx = {vid: i for i, vid in enumerate(video_ids)}
        selected_ids = [vid for vid in base_video_ids if vid in id_to_idx and vid in base_all_ids]
        selected_idx = np.array([id_to_idx[vid] for vid in selected_ids], dtype=int)

        selected_id_set = set(selected_ids)
        trainval_idx = np.array([id_to_idx[vid] for vid in base_trainval_ids if vid in selected_id_set], dtype=int)
        test_idx = np.array([id_to_idx[vid] for vid in base_test_ids if vid in selected_id_set], dtype=int)

        X = make_ml_features(
            data["X_seq"][selected_idx],
            data["X_static"][selected_idx],
            seq_features,
            static_features,
        )
        y_selected = data["y_cls"].astype(int)[selected_idx]
        selected_index_by_original = {orig_idx: pos for pos, orig_idx in enumerate(selected_idx)}
        trainval_pos = np.array([selected_index_by_original[i] for i in trainval_idx], dtype=int)
        test_pos = np.array([selected_index_by_original[i] for i in test_idx], dtype=int)

        X_train = X.iloc[trainval_pos]
        X_test = X.iloc[test_pos]
        y_train = y_selected[trainval_pos]
        y_test = y_selected[test_pos]
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        for model_name, model in build_models(scale_pos).items():
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, prob)
            rows.append(
                {
                    "experiment": "fixed_K14_sample",
                    "model": model_name,
                    "K": k,
                    "n_total": len(selected_idx),
                    "n_trainval": len(trainval_pos),
                    "n_test": len(test_pos),
                    "trainval_positive_rate": y_train.mean(),
                    "test_positive_rate": y_test.mean(),
                    "n_features": X.shape[1],
                    **metrics,
                }
            )
            print(f"[fixed sample] K={k:<2} {model_name:<13} AUC={metrics['roc_auc']:.4f} AP={metrics['ap']:.4f}")

    result = pd.DataFrame(rows).sort_values(["model", "K"])
    result.to_csv(OUTPUT / "revision_fixed_sample_k_results.csv", index=False, encoding="utf-8-sig")

    pivot = result.pivot_table(index="model", columns="K", values="roc_auc").reset_index()
    pivot.to_csv(OUTPUT / "revision_fixed_sample_k_auc_table.csv", index=False, encoding="utf-8-sig")
    return result


def category_handling_experiment(seq_features, static_features, static_scaler):
    data = np.load(OUTPUT / "kr_sequences_K14.npz", allow_pickle=True)
    X = make_ml_features(data["X_seq"], data["X_static"], seq_features, static_features)
    y = data["y_cls"].astype(int)
    trainval_idx = np.concatenate([data["train_idx"], data["val_idx"]])
    test_idx = data["test_idx"]

    n_static = len(static_features)
    raw_static = static_scaler.inverse_transform(data["X_static"][:, :n_static])
    raw_static_df = pd.DataFrame(raw_static, columns=static_features)
    X_cat = X.copy()
    X_cat["category_id"] = raw_static_df["category_id"].round().astype(int).astype(str)

    y_train = y[trainval_idx]
    y_test = y[test_idx]
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    rows = []
    importance_frames = []
    for setting, frame, cat_features in [
        ("정수형 코드값 표준화", X, None),
        ("CatBoost 범주형 처리", X_cat, ["category_id"]),
    ]:
        model = catboost_model(scale_pos)
        if cat_features is None:
            model.fit(frame.iloc[trainval_idx], y_train)
        else:
            model.fit(frame.iloc[trainval_idx], y_train, cat_features=cat_features)
        prob = model.predict_proba(frame.iloc[test_idx])[:, 1]
        metrics = evaluate(y_test, prob)
        rows.append(
            {
                "setting": setting,
                "n_features": frame.shape[1],
                **metrics,
            }
        )
        imp = pd.DataFrame(
            {
                "setting": setting,
                "feature": frame.columns,
                "importance": model.get_feature_importance(),
            }
        ).sort_values("importance", ascending=False)
        importance_frames.append(imp)
        print(f"[category] {setting:<18} AUC={metrics['roc_auc']:.4f} AP={metrics['ap']:.4f}")

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "revision_catboost_category_71feature_results.csv", index=False, encoding="utf-8-sig")
    pd.concat(importance_frames, ignore_index=True).to_csv(
        OUTPUT / "revision_catboost_category_71feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result


def main():
    scalers = load_scalers()
    seq_features = scalers["seq_features"]
    static_features = scalers["static_features"]
    static_scaler = scalers["static_scaler"]

    print("Running fixed-sample K experiment...")
    fixed = fixed_sample_k_experiment(seq_features, static_features)

    print("\nRunning category handling experiment...")
    category = category_handling_experiment(seq_features, static_features, static_scaler)

    print("\nSaved:")
    print(f"  {OUTPUT / 'revision_fixed_sample_k_results.csv'}")
    print(f"  {OUTPUT / 'revision_fixed_sample_k_auc_table.csv'}")
    print(f"  {OUTPUT / 'revision_catboost_category_71feature_results.csv'}")
    print(f"  {OUTPUT / 'revision_catboost_category_71feature_importance.csv'}")
    print("\nFixed-sample K AUC:")
    print(fixed.pivot_table(index="model", columns="K", values="roc_auc").round(4).to_string())
    print("\nCategory handling:")
    print(category.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

