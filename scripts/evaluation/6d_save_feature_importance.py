import os
import pickle

import numpy as np
import pandas as pd

OUTPUT_PATH = "./output"
MODEL_PATH = "./models"


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


def main():
    data = np.load(f"{OUTPUT_PATH}/kr_sequences_K14.npz", allow_pickle=True)
    with open(f"{OUTPUT_PATH}/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    with open(f"{MODEL_PATH}/CatBoost_v2_model.pkl", "rb") as f:
        payload = pickle.load(f)

    model = payload["model"]
    X = make_ml_features(
        data["X_seq"],
        data["X_static"],
        scalers["seq_features"],
        scalers["static_features"],
    )

    if hasattr(model, "get_feature_importance"):
        importance = model.get_feature_importance()
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        raise RuntimeError("Model does not expose feature importance.")

    fi = (
        pd.DataFrame({"feature": X.columns, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    fi.to_csv(f"{OUTPUT_PATH}/CatBoost_v2_feature_importance.csv", index=False, encoding="utf-8-sig")
    print(fi.head(20).to_string(index=False))
    print(f"\nSaved: {OUTPUT_PATH}/CatBoost_v2_feature_importance.csv")


if __name__ == "__main__":
    main()
