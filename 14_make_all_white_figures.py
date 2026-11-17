"""
Create white-background versions of all manuscript figures.

Original dark figures remain in ./figures. White variants are saved in
./figures/white using the same base filenames where possible.
"""
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve


BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "output"
FIG = BASE / "figures"
KCI = FIG / "kci"
WHITE = FIG / "white"
WHITE.mkdir(parents=True, exist_ok=True)


def setup():
    plt.rcParams.update(
        {
            "font.family": ["Malgun Gothic", "DejaVu Sans", "Arial"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#111111",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "grid.color": "#D9DEE7",
            "grid.linewidth": 0.8,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
        }
    )


def save(name):
    png = WHITE / name
    pdf = WHITE / name.replace(".png", ".pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved {png}")
    print(f"saved {pdf}")


def copy_kci_figures():
    mapping = {
        "fig1_k_ablation.png": "kci_fig2_k_ablation.png",
        "fig2_model_comparison.png": "kci_fig1_model_auc_ci.png",
        "fig3_significance.png": "kci_fig3_bootstrap_diff.png",
        "fig4_feature_importance.png": "kci_fig4_feature_importance.png",
        "fig7_ablation_baselines.png": "kci_fig5_ablation_baselines.png",
        "fig8_threshold_sensitivity.png": "kci_fig6_threshold_sensitivity.png",
    }
    for dst, src in mapping.items():
        shutil.copy2(KCI / src, WHITE / dst)
        pdf_src = KCI / src.replace(".png", ".pdf")
        if pdf_src.exists():
            shutil.copy2(pdf_src, WHITE / dst.replace(".png", ".pdf"))
        print(f"copied {src} -> {dst}")


def fig5_tft_attention():
    attn = np.load(OUTPUT / "TFT_attention.npy")
    # Expected shape: samples, heads, query_time, key_time
    mat = attn.mean(axis=(0, 1))
    k = mat.shape[0]

    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_title("TFT Attention Map (K=14)")
    ax.set_xlabel("Key snapshot")
    ax.set_ylabel("Query snapshot")
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(np.arange(1, k + 1))
    ax.set_yticklabels(np.arange(1, k + 1))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean attention weight")
    ax.spines[["top", "right"]].set_visible(False)
    save("fig5_tft_attention.png")


def fig5a_lstm_attention():
    viral = np.load(OUTPUT / "LSTM_K14_attn_viral.npy")
    nonviral = np.load(OUTPUT / "LSTM_K14_attn_nonviral.npy")
    std_v = np.load(OUTPUT / "LSTM_K14_attn_std_v.npy")
    std_nv = np.load(OUTPUT / "LSTM_K14_attn_std_nv.npy")
    x = np.arange(1, len(viral) + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(x, viral, marker="o", lw=2.2, label="High persistence", color="#1f77b4")
    ax.fill_between(x, viral - std_v, viral + std_v, color="#1f77b4", alpha=0.15)
    ax.plot(x, nonviral, marker="s", lw=2.2, label="Low persistence", color="#ff7f0e")
    ax.fill_between(x, nonviral - std_nv, nonviral + std_nv, color="#ff7f0e", alpha=0.15)
    ax.set_title("LSTM Attention by Early Snapshot")
    ax.set_xlabel("Early snapshot index")
    ax.set_ylabel("Attention weight")
    ax.set_xticks(x)
    ax.grid(alpha=0.75)
    ax.legend(frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    save("fig5a_lstm_attention.png")


def fig5b_roc_per_k():
    models = ["LSTM", "GRU", "Transformer", "TFT"]
    k_values = [5, 7, 10, 14]
    colors = {5: "#7f7f7f", 7: "#2ca02c", 10: "#ff7f0e", 14: "#1f77b4"}

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    for ax, model in zip(axes.flat, models):
        for k in k_values:
            path = OUTPUT / f"preds_{model}_K{k}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            fpr, tpr, _ = roc_curve(df["y_true"], df["y_pred"])
            ax.plot(fpr, tpr, lw=1.8, color=colors[k], label=f"K={k}, AUC={auc(fpr, tpr):.3f}")
        ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--")
        ax.set_title(model)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.grid(alpha=0.7)
        ax.legend(fontsize=8, frameon=True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("ROC Curves by Observation Length K", fontsize=15, y=1.02)
    save("fig5b_roc_per_k.png")


def fig5c_auc_heatmap():
    dl = pd.read_csv(OUTPUT / "dl_ablation_results.csv")
    pivot = dl.pivot(index="model", columns="K_EARLY", values="roc_auc").loc[
        ["LSTM", "GRU", "Transformer", "TFT"], [5, 7, 10, 14]
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0.74, vmax=0.84)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("K")
    ax.set_title("DL ROC-AUC Heatmap by K")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", color="#111111", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ROC-AUC")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    save("fig5c_auc_heatmap.png")


def fig6_pr_curve():
    names = ["CatBoost_v2", "XGB_v2", "Transformer_K14", "LSTM_K14", "TFT_K14", "GRU_K14"]
    display = {
        "CatBoost_v2": "CatBoost-Full (K=14)",
        "XGB_v2": "XGBoost-Full (K=14)",
        "Transformer_K14": "Transformer (K=14)",
        "LSTM_K14": "LSTM (K=14)",
        "TFT_K14": "TFT (K=14)",
        "GRU_K14": "GRU (K=14)",
    }
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#bcbd22", "#7f7f7f"]

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    for name, color in zip(names, colors):
        path = OUTPUT / f"preds_{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        precision, recall, _ = precision_recall_curve(df["y_true"], df["y_pred"])
        ap = average_precision_score(df["y_true"], df["y_pred"])
        ax.plot(recall, precision, lw=2, color=color, label=f"{display[name]} (AP={ap:.3f})")
    ax.set_title("Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.75)
    ax.legend(loc="upper right", frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    save("fig6_pr_curve.png")


def main():
    setup()
    copy_kci_figures()
    fig5_tft_attention()
    fig5a_lstm_attention()
    fig5b_roc_per_k()
    fig5c_auc_heatmap()
    fig6_pr_curve()


if __name__ == "__main__":
    main()
