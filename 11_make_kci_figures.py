"""
Create publication-ready figures for the KCI manuscript.

Inputs are the already generated experiment outputs in ./output.
Figures are saved to ./figures/kci as high-resolution PNG and PDF files.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score


BASE = Path(".")
OUTPUT = BASE / "output"
FIGDIR = BASE / "figures" / "kci"
FIGDIR.mkdir(parents=True, exist_ok=True)


DISPLAY_NAMES = {
    "CatBoost_v2": "CatBoost-Full (K=14)",
    "XGB_v2": "XGBoost-Full (K=14)",
    "LGBM_v2": "LightGBM-Full (K=14)",
    "RF_v2": "Random Forest-Full (K=14)",
    "MLP_v2": "MLP-Full (K=14)",
    "Transformer_K14": "Transformer (K=14)",
    "LSTM_K14": "LSTM (K=14)",
    "GRU_K14": "GRU (K=14)",
    "TFT_K14": "TFT (K=14)",
}

MODEL_COLORS = {
    "CatBoost_v2": "#1f77b4",
    "XGB_v2": "#2ca02c",
    "LGBM_v2": "#17becf",
    "RF_v2": "#9467bd",
    "MLP_v2": "#8c564b",
    "Transformer_K14": "#ff7f0e",
    "LSTM_K14": "#d62728",
    "GRU_K14": "#7f7f7f",
    "TFT_K14": "#bcbd22",
}


def setup_style():
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
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def savefig(name):
    png = FIGDIR / f"{name}.png"
    pdf = FIGDIR / f"{name}.pdf"
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    print(f"saved {png}")
    print(f"saved {pdf}")


def annotate_barh(ax, values, xpad=0.003, fmt="{:.3f}"):
    for idx, value in enumerate(values):
        ax.text(value + xpad, idx, fmt.format(value), va="center", fontsize=9, color="#111111")


def fig_model_auc_ci():
    df = pd.read_csv(OUTPUT / "bootstrap_ci.csv")
    keep = [
        "CatBoost_v2",
        "XGB_v2",
        "LGBM_v2",
        "RF_v2",
        "Transformer_K14",
        "LSTM_K14",
        "TFT_K14",
        "GRU_K14",
        "MLP_v2",
    ]
    df = df[df["model"].isin(keep)].copy()
    df["model"] = pd.Categorical(df["model"], keep, ordered=True)
    df = df.sort_values("auc", ascending=True)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    colors = [MODEL_COLORS.get(m, "#5B677A") for m in df["model"]]
    df["display_model"] = df["model"].map(DISPLAY_NAMES).fillna(df["model"].astype(str))
    ax.barh(df["display_model"], df["auc"], color=colors, alpha=0.92)
    xerr = np.vstack([df["auc"] - df["ci_lower"], df["ci_upper"] - df["auc"]])
    ax.errorbar(df["auc"], np.arange(len(df)), xerr=xerr, fmt="none", ecolor="#222222", capsize=3, lw=1)
    annotate_barh(ax, df["auc"].values)
    ax.set_xlim(0.75, 0.87)
    ax.set_xlabel("ROC-AUC with 95% bootstrap CI")
    ax.set_title("Main Model Comparison on K=14 Test Set")
    ax.grid(axis="x", alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    savefig("kci_fig1_model_auc_ci")


def fig_k_ablation():
    df = pd.read_csv(OUTPUT / "dl_ablation_results.csv")
    order = ["LSTM", "GRU", "Transformer", "TFT"]
    colors = {
        "LSTM": "#d62728",
        "GRU": "#7f7f7f",
        "Transformer": "#ff7f0e",
        "TFT": "#bcbd22",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for model in order:
        sub = df[df["model"] == model].sort_values("K_EARLY")
        ax.plot(
            sub["K_EARLY"],
            sub["roc_auc"],
            marker="o",
            lw=2.2,
            ms=5.5,
            label=model,
            color=colors[model],
        )
    ax.set_xticks([5, 7, 10, 14])
    ax.set_ylim(0.74, 0.84)
    ax.set_xlabel("Number of early trending snapshots (K)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Effect of Early Observation Length")
    ax.grid(alpha=0.75)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    savefig("kci_fig2_k_ablation")


def fig_bootstrap_diff():
    df = pd.read_csv(OUTPUT / "bootstrap_diff.csv")
    labels = [
        "ML best vs DL best",
        "CatBoost K14 vs K5",
        "GRU K14 vs K5",
        "LSTM vs GRU K14",
        "GRU vs Transformer K14",
        "CatBoost vs XGBoost",
    ]
    df = df.copy()
    df["label"] = labels[: len(df)]
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    y = np.arange(len(df))
    colors = ["#1f77b4" if s else "#A9B0BB" for s in df["significant"]]
    ax.barh(y, df["delta"], color=colors, alpha=0.9)
    xerr = np.vstack([df["delta"] - df["ci_lower"], df["ci_upper"] - df["delta"]])
    ax.errorbar(df["delta"], y, xerr=xerr, fmt="none", ecolor="#222222", capsize=3, lw=1)
    ax.axvline(0, color="#222222", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("AUC difference (A - B), 95% bootstrap CI")
    ax.set_title("Paired Bootstrap Model Comparisons")
    ax.grid(axis="x", alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    for idx, row in enumerate(df.itertuples()):
        ax.text(row.delta + (0.003 if row.delta >= 0 else -0.003), idx, f"{row.delta:+.3f}", va="center",
                ha="left" if row.delta >= 0 else "right", fontsize=9)
    savefig("kci_fig3_bootstrap_diff")


def fig_feature_importance():
    df = pd.read_csv(OUTPUT / "CatBoost_v2_feature_importance.csv").head(12)
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.barh(df["feature"], df["importance"], color="#4C78A8")
    ax.set_xlabel("CatBoost feature importance")
    ax.set_title("Top Predictors of Long-Duration Trending")
    ax.grid(axis="x", alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    for idx, value in enumerate(df["importance"]):
        ax.text(value + 0.05, idx, f"{value:.2f}", va="center", fontsize=9)
    savefig("kci_fig4_feature_importance")


def fig_ablation():
    df = pd.read_csv(OUTPUT / "ablation_baselines.csv")
    keep = [
        ("Static only", "CatBoost"),
        ("First snapshot only", "CatBoost"),
        ("Score: last rank", "Univariate score"),
        ("Delta/growth only", "CatBoost"),
        ("Basic sequence summaries", "CatBoost"),
        ("Full ML v2 features", "CatBoost"),
    ]
    rows = []
    for experiment, model in keep:
        row = df[(df["experiment"] == experiment) & (df["model"] == model)].iloc[0].copy()
        row["label"] = experiment
        rows.append(row)
    plot = pd.DataFrame(rows).sort_values("auc", ascending=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    colors = ["#A9B0BB" if m == "Univariate score" else "#1f77b4" for m in plot["model"]]
    ax.barh(plot["label"], plot["auc"], color=colors, alpha=0.92)
    annotate_barh(ax, plot["auc"].values)
    ax.set_xlim(0.68, 0.87)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Baseline and Feature-Set Ablation")
    ax.grid(axis="x", alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    savefig("kci_fig5_ablation_baselines")


def fig_threshold_sensitivity():
    df = pd.read_csv(OUTPUT / "threshold_sensitivity.csv")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(df["percentile"], df["auc"], marker="o", lw=2.2, label="ROC-AUC", color="#1f77b4")
    ax.plot(df["percentile"], df["average_precision"], marker="s", lw=2.0, label="Average precision", color="#2ca02c")
    ax.plot(df["percentile"], df["f1"], marker="^", lw=2.0, label="F1", color="#ff7f0e")
    ax.set_xticks(df["percentile"])
    ax.set_ylim(0.45, 0.88)
    ax.set_xlabel("Viral label percentile cutoff")
    ax.set_ylabel("Test metric")
    ax.set_title("Sensitivity to Viral-Label Threshold")
    ax.grid(alpha=0.75)
    ax.legend(loc="lower left", frameon=True, framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    savefig("kci_fig6_threshold_sensitivity")


def fig_temporal_validation():
    df = pd.read_csv(OUTPUT / "temporal_split_results.csv")
    df = df.sort_values("test_auc", ascending=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.barh(df["model"], df["test_auc"], color="#4C78A8", alpha=0.9, label="Temporal test AUC")
    annotate_barh(ax, df["test_auc"].values)
    ax.set_xlim(0.72, 0.80)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Chronological Robustness Check")
    ax.grid(axis="x", alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    note = "Positive rate shift: train 0.175 → test 0.481"
    ax.text(0.721, -0.75, note, fontsize=9.5, color="#333333")
    savefig("kci_fig7_temporal_validation")


def fig_curves():
    candidates = [
        "CatBoost_v2",
        "XGB_v2",
        "Transformer_K14",
        "LSTM_K14",
    ]
    preds = {}
    for name in candidates:
        path = OUTPUT / f"preds_{name}.csv"
        if path.exists():
            preds[name] = pd.read_csv(path)
    if not preds:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for name, df in preds.items():
        y = df["y_true"].values
        p = df["y_pred"].values
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, lw=2, label=f"{DISPLAY_NAMES.get(name, name)} ({auc(fpr, tpr):.3f})", color=MODEL_COLORS.get(name))
        axes[1].plot(recall, precision, lw=2, label=f"{DISPLAY_NAMES.get(name, name)} ({average_precision_score(y, p):.3f})",
                     color=MODEL_COLORS.get(name))

    axes[0].plot([0, 1], [0, 1], color="#888888", lw=1, linestyle="--")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC Curves")
    axes[0].grid(alpha=0.7)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower right", frameon=True, framealpha=0.95)

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves")
    axes[1].grid(alpha=0.7)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(loc="upper right", frameon=True, framealpha=0.95)
    savefig("kci_fig8_roc_pr_curves")


def main():
    setup_style()
    fig_model_auc_ci()
    fig_k_ablation()
    fig_bootstrap_diff()
    fig_feature_importance()
    fig_ablation()
    fig_threshold_sensitivity()
    fig_temporal_validation()
    fig_curves()


if __name__ == "__main__":
    main()


