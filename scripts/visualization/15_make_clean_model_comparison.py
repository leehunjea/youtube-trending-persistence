"""
Create a cleaner Korean version of the main model ROC-AUC comparison figure.

The figure intentionally removes K=14 from labels and uses simple horizontal
bars without confidence-interval error bars for cleaner manuscript/slide use.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).resolve().parents[2]
OUTPUT = BASE / "output"
WHITE = BASE / "figures" / "white"
KCI = BASE / "figures" / "kci"
WHITE.mkdir(parents=True, exist_ok=True)
KCI.mkdir(parents=True, exist_ok=True)


DISPLAY = {
    "CatBoost_v2": "CatBoost-Full",
    "XGB_v2": "XGBoost-Full",
    "LGBM_v2": "LightGBM-Full",
    "RF_v2": "Random Forest-Full",
    "Transformer_K14": "Transformer",
    "LSTM_K14": "LSTM",
    "TFT_K14": "TFT",
    "GRU_K14": "GRU",
}

MODEL_TYPE = {
    "CatBoost_v2": "트리 기반",
    "XGB_v2": "트리 기반",
    "LGBM_v2": "트리 기반",
    "RF_v2": "트리 기반",
    "Transformer_K14": "딥러닝",
    "LSTM_K14": "딥러닝",
    "TFT_K14": "딥러닝",
    "GRU_K14": "딥러닝",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": ["Malgun Gothic", "DejaVu Sans", "Arial"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#111111",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.8,
            "axes.titlesize": 17,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def main():
    setup_style()
    df = pd.read_csv(OUTPUT / "bootstrap_ci.csv")
    keep = list(DISPLAY.keys())
    df = df[df["model"].isin(keep)].copy()
    df["표시명"] = df["model"].map(DISPLAY)
    df["계열"] = df["model"].map(MODEL_TYPE)
    df = df.sort_values("auc", ascending=True)

    colors = df["계열"].map({"트리 기반": "#2F6BBA", "딥러닝": "#9AA4B2"}).tolist()

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    bars = ax.barh(df["표시명"], df["auc"], color=colors, height=0.62)

    ax.set_xlim(0.80, 0.855)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("주요 모델별 ROC-AUC 비교", pad=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.9)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    for bar, value in zip(bars, df["auc"]):
        ax.text(
            value + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#111111",
        )

    from matplotlib.patches import Patch

    legend_items = [
        Patch(facecolor="#2F6BBA", label="트리 기반 모델"),
        Patch(facecolor="#9AA4B2", label="딥러닝 모델"),
    ]
    ax.legend(handles=legend_items, loc="lower right", frameon=False)

    plt.tight_layout()

    outputs = [
        WHITE / "fig2_model_comparison.png",
        KCI / "kci_fig1_model_auc_ci_clean_kr.png",
    ]
    for out in outputs:
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"saved {out}")

    pdf_outputs = [
        WHITE / "fig2_model_comparison.pdf",
        KCI / "kci_fig1_model_auc_ci_clean_kr.pdf",
    ]
    for out in pdf_outputs:
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print(f"saved {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
