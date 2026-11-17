"""
Best-model search for revision.

The search is intentionally small and reproducible:
- Use the same K=14 71-feature representation as the manuscript.
- Select configurations only with the validation set.
- Report final metrics on the untouched test set.
- Try tuned CatBoost variants and validation-selected soft ensembles.
"""
from itertools import product
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"
RANDOM_STATE = 42


def make_ml_features(X_seq, X_static, seq_features, static_features):
    rows = []
    k = X_seq.shape[1]
    available_seq = seq_features[: X_seq.shape[2]]
    suffixes = ["first", "last", "mean", "std", "max", "min", "range", "slope"]

    for i in range(len(X_seq)):
        seq = X_seq[i]
        st = X_static[i]
        feat = {}

        for j, name in enumerate(static_features):
            feat[name] = st[j]

        for j, name in enumerate(available_seq):
            col = seq[:, j]
            values = {
                "first": col[0],
                "last": col[-1],
                "mean": col.mean(),
                "std": col.std(),
                "max": col.max(),
                "min": col.min(),
                "range": col.max() - col.min(),
                "slope": np.polyfit(range(k), col, 1)[0] if k > 1 else 0,
            }
            for suffix in suffixes:
                feat[f"{name}_{suffix}"] = values[suffix]

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


def metrics(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, prob),
        "ap": average_precision_score(y_true, prob),
        "f1": f1_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
    }


def catboost_params(scale_pos, depth=6, lr=0.05, l2=3.0, random_strength=1.0, bagging_temperature=1.0):
    return dict(
        iterations=1200,
        learning_rate=lr,
        depth=depth,
        l2_leaf_reg=l2,
        random_strength=random_strength,
        bagging_temperature=bagging_temperature,
        scale_pos_weight=scale_pos,
        eval_metric="AUC",
        early_stopping_rounds=80,
        verbose=0,
        random_seed=RANDOM_STATE,
    )


def main():
    data = np.load(OUTPUT / "kr_sequences_K14.npz", allow_pickle=True)
    with open(OUTPUT / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    X = make_ml_features(data["X_seq"], data["X_static"], scalers["seq_features"], scalers["static_features"])
    y = data["y_cls"].astype(int)
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    trainval_idx = np.concatenate([train_idx, val_idx])

    n_static = len(scalers["static_features"])
    raw_static = scalers["static_scaler"].inverse_transform(data["X_static"][:, :n_static])
    raw_static_df = pd.DataFrame(raw_static, columns=scalers["static_features"])
    X_cat = X.copy()
    X_cat["category_id"] = raw_static_df["category_id"].round().astype(int).astype(str)

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    scale_pos_trainval = (y[trainval_idx] == 0).sum() / max((y[trainval_idx] == 1).sum(), 1)

    rows = []
    val_predictions = {}
    test_predictions = {}

    # Baseline-like CatBoost variants selected by validation AUC.
    search_space = list(product([4, 6, 8], [0.03, 0.05], [1.0, 3.0, 6.0], [0.5, 1.0]))
    for use_cat in [False, True]:
        frame = X_cat if use_cat else X
        cat_features = ["category_id"] if use_cat else None
        for depth, lr, l2, bag_temp in search_space:
            label = f"CatBoost_{'cat' if use_cat else 'num'}_d{depth}_lr{lr}_l2{l2}_bt{bag_temp}"
            model = CatBoostClassifier(**catboost_params(scale_pos, depth=depth, lr=lr, l2=l2, bagging_temperature=bag_temp))
            if cat_features:
                model.fit(frame.iloc[train_idx], y_train, eval_set=(frame.iloc[val_idx], y_val), cat_features=cat_features)
            else:
                model.fit(frame.iloc[train_idx], y_train, eval_set=(frame.iloc[val_idx], y_val))
            val_prob = model.predict_proba(frame.iloc[val_idx])[:, 1]
            test_prob = model.predict_proba(frame.iloc[test_idx])[:, 1]
            val_m = metrics(y_val, val_prob)
            test_m = metrics(y_test, test_prob)
            rows.append(
                {
                    "model": label,
                    "family": "CatBoost tuned",
                    "selected_by": "validation AUC",
                    "best_iteration": model.get_best_iteration(),
                    "val_roc_auc": val_m["roc_auc"],
                    "val_ap": val_m["ap"],
                    **{f"test_{k}": v for k, v in test_m.items()},
                }
            )
            val_predictions[label] = val_prob
            test_predictions[label] = test_prob
            print(f"{label:<42} val={val_m['roc_auc']:.4f} test={test_m['roc_auc']:.4f} AP={test_m['ap']:.4f}")

    # Add a few non-CatBoost base learners for validation-selected ensembles.
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(
            n_estimators=900,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos,
            eval_metric="auc",
            early_stopping_rounds=80,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        xgb_model.fit(X.iloc[train_idx], y_train, eval_set=[(X.iloc[val_idx], y_val)], verbose=False)
        val_prob = xgb_model.predict_proba(X.iloc[val_idx])[:, 1]
        test_prob = xgb_model.predict_proba(X.iloc[test_idx])[:, 1]
        val_m = metrics(y_val, val_prob)
        test_m = metrics(y_test, test_prob)
        label = "XGBoost_tuned"
        rows.append({"model": label, "family": "XGBoost tuned", "selected_by": "validation AUC", "best_iteration": getattr(xgb_model, "best_iteration", None), "val_roc_auc": val_m["roc_auc"], "val_ap": val_m["ap"], **{f"test_{k}": v for k, v in test_m.items()}})
        val_predictions[label] = val_prob
        test_predictions[label] = test_prob
        print(f"{label:<42} val={val_m['roc_auc']:.4f} test={test_m['roc_auc']:.4f} AP={test_m['ap']:.4f}")
    except Exception as exc:
        print(f"skip xgboost tuned: {exc}")

    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMClassifier(
            n_estimators=900,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
        lgb_model.fit(X.iloc[train_idx], y_train)
        val_prob = lgb_model.predict_proba(X.iloc[val_idx])[:, 1]
        test_prob = lgb_model.predict_proba(X.iloc[test_idx])[:, 1]
        val_m = metrics(y_val, val_prob)
        test_m = metrics(y_test, test_prob)
        label = "LightGBM_tuned"
        rows.append({"model": label, "family": "LightGBM tuned", "selected_by": "validation AUC", "best_iteration": None, "val_roc_auc": val_m["roc_auc"], "val_ap": val_m["ap"], **{f"test_{k}": v for k, v in test_m.items()}})
        val_predictions[label] = val_prob
        test_predictions[label] = test_prob
        print(f"{label:<42} val={val_m['roc_auc']:.4f} test={test_m['roc_auc']:.4f} AP={test_m['ap']:.4f}")
    except Exception as exc:
        print(f"skip lightgbm tuned: {exc}")

    rf = RandomForestClassifier(n_estimators=700, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X.iloc[train_idx], y_train)
    val_prob = rf.predict_proba(X.iloc[val_idx])[:, 1]
    test_prob = rf.predict_proba(X.iloc[test_idx])[:, 1]
    val_m = metrics(y_val, val_prob)
    test_m = metrics(y_test, test_prob)
    label = "RandomForest_tuned"
    rows.append({"model": label, "family": "RandomForest tuned", "selected_by": "validation AUC", "best_iteration": None, "val_roc_auc": val_m["roc_auc"], "val_ap": val_m["ap"], **{f"test_{k}": v for k, v in test_m.items()}})
    val_predictions[label] = val_prob
    test_predictions[label] = test_prob
    print(f"{label:<42} val={val_m['roc_auc']:.4f} test={test_m['roc_auc']:.4f} AP={test_m['ap']:.4f}")

    result = pd.DataFrame(rows).sort_values(["val_roc_auc", "test_roc_auc"], ascending=False)
    result.to_csv(OUTPUT / "revision_best_model_search_results.csv", index=False, encoding="utf-8-sig")

    # Validation-selected soft ensembles from the top candidates.
    candidates = result.head(8)["model"].tolist()
    ensemble_rows = []
    for n in [2, 3, 4, 5]:
        names = candidates[:n]
        val_stack = np.column_stack([val_predictions[name] for name in names])
        test_stack = np.column_stack([test_predictions[name] for name in names])
        best = None
        # Coarse grid for weights. Validation only.
        grid = np.linspace(0, 1, 11)
        if n == 2:
            weight_sets = [(w, 1 - w) for w in grid]
        elif n == 3:
            weight_sets = [(a, b, 1 - a - b) for a in grid for b in grid if a + b <= 1]
        else:
            # Equal weight is safer and avoids excessive validation overfitting for n>=4.
            weight_sets = [tuple([1 / n] * n)]
        for weights in weight_sets:
            weights = np.array(weights)
            if (weights < -1e-9).any():
                continue
            val_prob_ens = val_stack @ weights
            val_auc = roc_auc_score(y_val, val_prob_ens)
            if best is None or val_auc > best[0]:
                best = (val_auc, weights)
        test_prob_ens = test_stack @ best[1]
        test_m = metrics(y_test, test_prob_ens)
        ensemble_rows.append(
            {
                "model": f"Soft ensemble top{n}",
                "members": " + ".join(names),
                "weights": ",".join(f"{w:.2f}" for w in best[1]),
                "val_roc_auc": best[0],
                **{f"test_{k}": v for k, v in test_m.items()},
            }
        )
        print(f"Soft ensemble top{n:<2} val={best[0]:.4f} test={test_m['roc_auc']:.4f} AP={test_m['ap']:.4f}")

    ensemble = pd.DataFrame(ensemble_rows).sort_values(["val_roc_auc", "test_roc_auc"], ascending=False)
    ensemble.to_csv(OUTPUT / "revision_best_model_ensembles.csv", index=False, encoding="utf-8-sig")

    print("\nTop single models by validation AUC:")
    print(result.head(10).round(4).to_string(index=False))
    print("\nEnsembles:")
    print(ensemble.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
