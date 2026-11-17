"""
Create Korean, clean, white-background manuscript figures.

Outputs are saved to ./figures/korean. These are intended for the Korean KCI
manuscript body, while ./figures/white keeps the previous white-background set.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


BASE = Path(__file__).resolve().parents[2]
OUTPUT = BASE / "output"
KOR = BASE / "figures" / "korean"
KOR.mkdir(parents=True, exist_ok=True)


MODEL_DISPLAY = {
    "CatBoost_v2": "CatBoost-Full",
    "XGB_v2": "XGBoost-Full",
    "LGBM_v2": "LightGBM-Full",
    "RF_v2": "Random Forest-Full",
    "Transformer_K14": "Transformer",
    "LSTM_K14": "LSTM",
    "TFT_K14": "TFT",
    "GRU_K14": "GRU",
    "MLP_v2": "MLP-Full",
}

FEATURE_DISPLAY = {
    "category_id": "카테고리",
    "rank_last": "마지막 순위",
    "rank_mean": "평균 순위",
    "tag_count": "태그 수",
    "rank_max": "순위 최대값",
    "views_acceleration_last": "마지막 조회속도 가속도",
    "rank_delta_slope": "순위 변화 기울기",
    "publish_weekday_sin": "업로드 요일(sin)",
    "comment_view_ratio_slope": "댓글/조회 비율 기울기",
    "momentum_std": "모멘텀 표준편차",
    "views_per_hour_min": "시간당 조회수 최소값",
    "momentum_slope": "모멘텀 기울기",
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
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
        }
    )


def savefig(name):
    png = KOR / f"{name}.png"
    pdf = KOR / f"{name}.pdf"
    plt.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved {png}")
    print(f"saved {pdf}")


def fig1_model_comparison():
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
    ]
    df = df[df["model"].isin(keep)].copy()
    df["표시명"] = df["model"].map(MODEL_DISPLAY)
    df["계열"] = np.where(df["model"].isin(["CatBoost_v2", "XGB_v2", "LGBM_v2", "RF_v2"]), "트리 기반", "딥러닝")
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
        ax.text(value + 0.001, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor="#2F6BBA", label="트리 기반 모델"),
            Patch(facecolor="#9AA4B2", label="딥러닝 모델"),
        ],
        loc="lower right",
        frameon=False,
    )
    savefig("fig1_model_comparison")


def fig2_k_effect():
    ml = pd.read_csv(OUTPUT / "ml_k_ablation_results.csv")
    dl = pd.read_csv(OUTPUT / "dl_ablation_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), sharey=True)

    ml_order = ["CatBoost", "XGBoost", "LightGBM", "Random Forest"]
    ml_colors = ["#2F6BBA", "#2CA02C", "#17BECF", "#9467BD"]
    for model, color in zip(ml_order, ml_colors):
        sub = ml[ml["model"] == model].sort_values("K_EARLY")
        axes[0].plot(sub["K_EARLY"], sub["roc_auc"], marker="o", lw=2.2, color=color, label=model)
    axes[0].set_title("머신러닝 모델")
    axes[0].set_xlabel("초기 관측 길이 K")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].legend(frameon=False)

    dl_order = ["LSTM", "GRU", "Transformer", "TFT"]
    dl_colors = ["#D62728", "#7F7F7F", "#FF7F0E", "#BCBD22"]
    for model, color in zip(dl_order, dl_colors):
        sub = dl[dl["model"] == model].sort_values("K_EARLY")
        axes[1].plot(sub["K_EARLY"], sub["roc_auc"], marker="o", lw=2.2, color=color, label=model)
    axes[1].set_title("딥러닝 모델")
    axes[1].set_xlabel("초기 관측 길이 K")
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.set_xticks([5, 7, 10, 14])
        ax.set_ylim(0.74, 0.855)
        ax.grid(alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("초기 관측 길이 K에 따른 예측 성능 변화", fontsize=16, fontweight="bold", y=1.03)
    savefig("fig2_k_effect")


def fig3_feature_importance():
    df = pd.read_csv(OUTPUT / "CatBoost_v2_feature_importance.csv").head(12)
    df["표시명"] = df["feature"].map(FEATURE_DISPLAY).fillna(df["feature"])
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    bars = ax.barh(df["표시명"], df["importance"], color="#2F6BBA", height=0.65)
    ax.set_xlabel("특징 중요도")
    ax.set_title("CatBoost-Full 주요 특징 중요도", pad=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.75)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for bar, value in zip(bars, df["importance"]):
        ax.text(value + 0.05, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=9.5)
    savefig("fig3_feature_importance")


def fig4_ablation():
    df = pd.read_csv(OUTPUT / "ablation_baselines.csv")
    mapping = {
        "Static only": "정적 특징만",
        "First snapshot only": "첫 스냅샷만",
        "Score: last rank": "마지막 순위 단일점수",
        "Delta/growth only": "변화·성장 특징만",
        "Basic sequence summaries": "시계열 요약 특징",
        "Full ML v2 features": "전체 특징",
    }
    keep = [
        ("Static only", "CatBoost"),
        ("First snapshot only", "CatBoost"),
        ("Score: last rank", "Univariate score"),
        ("Delta/growth only", "CatBoost"),
        ("Basic sequence summaries", "CatBoost"),
        ("Full ML v2 features", "CatBoost"),
    ]
    rows = []
    for exp, model in keep:
        row = df[(df["experiment"] == exp) & (df["model"] == model)].iloc[0].copy()
        row["표시명"] = mapping[exp]
        row["모델"] = model
        rows.append(row)
    plot = pd.DataFrame(rows).sort_values("auc", ascending=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    colors = ["#A9B0BB" if m == "Univariate score" else "#2F6BBA" for m in plot["모델"]]
    bars = ax.barh(plot["표시명"], plot["auc"], color=colors, height=0.65)
    ax.set_xlim(0.68, 0.87)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("기준선 및 특징군 제거 실험", pad=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.75)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for bar, value in zip(bars, plot["auc"]):
        ax.text(value + 0.003, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=9.5)
    savefig("fig4_ablation")


def fig5_threshold():
    df = pd.read_csv(OUTPUT / "threshold_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(df["percentile"], df["auc"], marker="o", lw=2.4, label="ROC-AUC", color="#2F6BBA")
    ax.plot(df["percentile"], df["average_precision"], marker="s", lw=2.1, label="AP", color="#2CA02C")
    ax.plot(df["percentile"], df["f1"], marker="^", lw=2.1, label="F1", color="#FF7F0E")
    ax.set_xticks(df["percentile"])
    ax.set_ylim(0.45, 0.88)
    ax.set_xlabel("트렌딩 지속성 라벨 기준(백분위)")
    ax.set_ylabel("성능 지표")
    ax.set_title("라벨 임계값 민감도", pad=14, fontweight="bold")
    ax.grid(alpha=0.75)
    ax.legend(loc="lower left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    savefig("fig5_threshold_sensitivity")


def fig6_bootstrap():
    df = pd.read_csv(OUTPUT / "bootstrap_diff.csv")
    labels = [
        "CatBoost-Full vs Transformer",
        "CatBoost-Full vs K=5 기준선",
        "GRU K=14 vs K=5",
        "LSTM vs GRU",
        "GRU vs Transformer",
        "CatBoost-Full vs XGBoost-Full",
    ]
    df = df.copy()
    df["표시명"] = labels[: len(df)]
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    y = np.arange(len(df))
    colors = ["#2F6BBA" if s else "#A9B0BB" for s in df["significant"]]
    ax.barh(y, df["delta"], color=colors, height=0.62)
    xerr = np.vstack([df["delta"] - df["ci_lower"], df["ci_upper"] - df["delta"]])
    ax.errorbar(df["delta"], y, xerr=xerr, fmt="none", ecolor="#222222", capsize=3, lw=1)
    ax.axvline(0, color="#222222", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["표시명"])
    ax.set_xlabel("AUC 차이(95% bootstrap CI)")
    ax.set_title("paired bootstrap 기반 모델 비교", pad=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.75)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    savefig("fig6_bootstrap_diff")


def fig7_temporal():
    rows = [
        ("무작위 분할", 26.2, 26.4, 0.8444),
        ("시간순 분할", 17.5, 48.1, 0.7834),
    ]
    df = pd.DataFrame(rows, columns=["분할", "Train 양성률", "Test 양성률", "ROC-AUC"])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))

    x = np.arange(len(df))
    width = 0.35
    axes[0].bar(x - width / 2, df["Train 양성률"], width, label="Train", color="#2F6BBA")
    axes[0].bar(x + width / 2, df["Test 양성률"], width, label="Test", color="#FF7F0E")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["분할"])
    axes[0].set_ylabel("양성 클래스 비율(%)")
    axes[0].set_title("분할별 양성률")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.75)

    bars = axes[1].bar(df["분할"], df["ROC-AUC"], color=["#2F6BBA", "#9AA4B2"], width=0.52)
    axes[1].set_ylim(0.74, 0.86)
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title("CatBoost 성능 비교")
    axes[1].grid(axis="y", alpha=0.75)
    for bar, value in zip(bars, df["ROC-AUC"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.4f}", ha="center", fontsize=10)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("무작위 분할과 시간순 분할 비교", fontsize=16, fontweight="bold", y=1.03)
    savefig("fig7_temporal_split")


def fig8_pr_curve():
    names = ["CatBoost_v2", "XGB_v2", "Transformer_K14", "LSTM_K14", "TFT_K14", "GRU_K14"]
    colors = ["#2F6BBA", "#2CA02C", "#FF7F0E", "#D62728", "#BCBD22", "#7F7F7F"]
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    for name, color in zip(names, colors):
        path = OUTPUT / f"preds_{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        precision, recall, _ = precision_recall_curve(df["y_true"], df["y_pred"])
        ap = average_precision_score(df["y_true"], df["y_pred"])
        ax.plot(recall, precision, lw=2, color=color, label=f"{MODEL_DISPLAY[name]} (AP={ap:.3f})")
    ax.set_title("Precision-Recall 곡선", pad=14, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.75)
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    savefig("fig8_pr_curve")


def main():
    setup_style()
    fig1_model_comparison()
    fig2_k_effect()
    fig3_feature_importance()
    fig4_ablation()
    fig5_threshold()
    fig6_bootstrap()
    fig7_temporal()
    fig8_pr_curve()


if __name__ == "__main__":
    main()
