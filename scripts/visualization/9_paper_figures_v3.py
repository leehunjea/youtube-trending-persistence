# -*- coding: utf-8 -*-
"""
논문 그림 재생성 (누수수정본 · validation 임곗값 F1).
- 흰 배경, 300dpi, 흑백 인쇄 가독성(명도차 + 해칭 + 마커/선종류)
- 조건명은 표 10과 완전히 일치 (리뷰어3 지적8)
출력: figures_v3/fig1~fig4
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

OUT = "./figures_v3"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
    "grid.color": "#CCCCCC",
    "grid.linewidth": 0.6,
})

GREY = ["#2B2B2B", "#5A5A5A", "#8A8A8A", "#B4B4B4", "#D6D6D6"]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT}/{name}.png / .pdf")


# ── 그림 1. 주요 모델별 ROC-AUC (표 6) ────────────────────
def fig1_models():
    models = ["CatBoost", "LightGBM", "XGBoost", "Random\nForest",
              "LSTM", "GRU", "Transformer", "TFT", "MLP-\nTabular"]
    auc = [0.8416, 0.8385, 0.8361, 0.8318, 0.8296, 0.8266, 0.8254, 0.8236, 0.8235]
    lo = [0.8252, 0.8218, 0.8190, 0.8148, 0.8129, 0.8091, 0.8078, 0.8061, 0.8062]
    hi = [0.8575, 0.8548, 0.8526, 0.8484, 0.8465, 0.8435, 0.8423, 0.8405, 0.8404]
    kind = ["tree"] * 4 + ["dl"] * 4 + ["mlp"]

    err = [np.array(auc) - np.array(lo), np.array(hi) - np.array(auc)]
    face = {"tree": "#4A4A4A", "dl": "#A8A8A8", "mlp": "#DCDCDC"}
    hatch = {"tree": "", "dl": "///", "mlp": "xxx"}

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = np.arange(len(models))
    for i, (v, k) in enumerate(zip(auc, kind)):
        ax.bar(x[i], v, width=0.68, color=face[k], edgecolor="black",
               linewidth=0.8, hatch=hatch[k], zorder=3)
    ax.errorbar(x, auc, yerr=err, fmt="none", ecolor="black",
                elinewidth=1.1, capsize=4, capthick=1.1, zorder=4)

    for i, v in enumerate(auc):
        ax.text(x[i], hi[i] + 0.0022, f"{v:.4f}", ha="center", va="bottom",
                fontsize=9.5, zorder=5)

    ax.axhline(0.8017, color="black", linestyle="--", linewidth=1.2, zorder=2)
    ax.annotate("마지막 순위 단일 특징\n기준선 (0.8017)", xy=(1.004, 0.8017),
                xycoords=("axes fraction", "data"), ha="left", va="center",
                fontsize=9.5, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.792, 0.868)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=face["tree"], edgecolor="black", label="트리 기반"),
        plt.Rectangle((0, 0), 1, 1, facecolor=face["dl"], edgecolor="black", hatch="///", label="딥러닝"),
        plt.Rectangle((0, 0), 1, 1, facecolor=face["mlp"], edgecolor="black", hatch="xxx", label="MLP-Tabular"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, ncol=3,
              bbox_to_anchor=(1.0, 1.14))
    save(fig, "fig1_model_comparison")


# ── 그림 2. K별 성능 (표 8·9) ─────────────────────────────
def fig2_kablation():
    K = [5, 7, 10, 14]
    ml = {
        "CatBoost":      [0.7816, 0.8028, 0.8163, 0.8416],
        "XGBoost":       [0.7819, 0.7910, 0.8153, 0.8361],
        "LightGBM":      [0.7805, 0.7956, 0.8141, 0.8385],
        "Random Forest": [0.7671, 0.7818, 0.8066, 0.8318],
    }
    dl = {
        "LSTM":        [0.7589, 0.7761, 0.7925, 0.8296],
        "GRU":         [0.7613, 0.7735, 0.7976, 0.8266],
        "Transformer": [0.7574, 0.7682, 0.7965, 0.8254],
        "TFT":         [0.7599, 0.7682, 0.8008, 0.8236],
    }
    mk = ["o", "s", "^", "D"]
    ls = ["-", "--", "-.", ":"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, data, title in zip(axes, (ml, dl),
                               ("(a) 머신러닝 모델", "(b) 딥러닝 모델")):
        for i, (name, vals) in enumerate(data.items()):
            ax.plot(K, vals, marker=mk[i], linestyle=ls[i], color=GREY[i],
                    linewidth=1.9, markersize=7, markerfacecolor="white",
                    markeredgewidth=1.5, label=name, zorder=3)
        ax.set_title(title, pad=10)
        ax.set_xlabel("초기 관측 길이 K (스냅샷 개수)")
        ax.set_xticks(K)
        ax.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="lower right")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_ylim(0.75, 0.855)
    fig.tight_layout()
    save(fig, "fig2_k_ablation")


# ── 그림 3. 기준선 및 특징군 제거 (표 10) ─────────────────
def fig3_ablation():
    # 조건명은 표 10과 완전히 동일
    conds = ["전체 특징\n(71)", "정적+시계열\n요약 (40)", "변화량, 성장\n특징 (31)",
             "마지막 순위\n점수 (1)", "정적+첫\n스냅샷 (12)", "정적 특징\n(8)"]
    auc = [0.8416, 0.8415, 0.8121, 0.8017, 0.7567, 0.7227]
    ap  = [0.6294, 0.6355, 0.5602, 0.5411, 0.5391, 0.4974]

    x = np.arange(len(conds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10.2, 4.9))
    ax.bar(x - w / 2, auc, w, color="#4A4A4A", edgecolor="black",
           linewidth=0.8, label="ROC-AUC", zorder=3)
    ax.bar(x + w / 2, ap, w, color="#CFCFCF", edgecolor="black",
           linewidth=0.8, hatch="///", label="AP", zorder=3)

    for i in range(len(conds)):
        ax.text(x[i] - w / 2, auc[i] + 0.008, f"{auc[i]:.4f}", ha="center",
                va="bottom", fontsize=9.2)
        ax.text(x[i] + w / 2, ap[i] + 0.008, f"{ap[i]:.4f}", ha="center",
                va="bottom", fontsize=9.2)

    ax.axhline(0.5, color="black", linestyle=":", linewidth=1.2, zorder=2)
    ax.annotate("무작위 분류기 (ROC-AUC 0.5)", xy=(1.002, 0.5),
                xycoords=("axes fraction", "data"), ha="left", va="center",
                fontsize=9.5, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylabel("점수")
    ax.set_ylim(0.42, 0.93)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig3_feature_ablation")


# ── 그림 4. 라벨 임곗값 민감도 (표 12) ────────────────────
def fig4_threshold():
    labels = ["p65\n(55)", "p70\n(57)", "p75\n(60)", "p80\n(63)", "p85\n(66)"]
    auc = [0.8457, 0.8407, 0.8416, 0.8456, 0.8396]
    ap  = [0.7424, 0.6908, 0.6294, 0.5774, 0.4935]
    f1  = [0.7109, 0.6762, 0.6270, 0.5873, 0.5308]
    pos = [0.3591, 0.3129, 0.2558, 0.2028, 0.1528]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.bar(x, pos, width=0.55, color="#EAEAEA", edgecolor="#999999",
           linewidth=0.8, label="양성률", zorder=1)

    for vals, name, m, l, c in [
        (auc, "ROC-AUC", "o", "-", "#2B2B2B"),
        (ap, "AP", "s", "--", "#6E6E6E"),
        (f1, "F1", "^", "-.", "#A0A0A0"),
    ]:
        ax.plot(x, vals, marker=m, linestyle=l, color=c, linewidth=1.9,
                markersize=7, markerfacecolor="white", markeredgewidth=1.5,
                label=name, zorder=3)

    for i, v in enumerate(auc):
        ax.text(x[i], v + 0.018, f"{v:.4f}", ha="center", va="bottom", fontsize=9.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("라벨 임곗값 기준 (cutoff)")
    ax.set_ylabel("점수 / 양성률")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig4_threshold_sensitivity")


# ── 그림 5. CatBoost-Full 주요 특징 중요도 (표 11) ────────
def fig5_importance():
    # 표 11과 동일한 순서·값 (한글 표기 병기)
    feats = [
        ("카테고리 (category_id)", 4.8839),
        ("평균 순위 (rank_mean)", 4.5483),
        ("최저 순위 (rank_max)", 4.4893),
        ("태그 수 (tag_count)", 3.8885),
        ("조회 가속도 마지막값\n(views_acceleration_last)", 3.8703),
        ("마지막 순위 (rank_last)", 3.7495),
        ("순위 변화량 기울기\n(rank_delta_slope)", 2.6262),
        ("댓글-조회 비율 기울기\n(comment_view_ratio_slope)", 2.5419),
        ("모멘텀 표준편차\n(momentum_std)", 2.4404),
        ("업로드 시각 cos\n(publish_hour_cos)", 2.3844),
    ]
    names = [f[0] for f in feats][::-1]   # 위→아래 = 1위→10위
    vals = [f[1] for f in feats][::-1]

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    y = np.arange(len(names))
    # 순위 그룹별 명도 구분: 상위 3 진하게
    colors = ["#B4B4B4"] * len(vals)
    for i in range(len(vals) - 3, len(vals)):
        colors[i] = "#4A4A4A"
    ax.barh(y, vals, height=0.68, color=colors, edgecolor="black",
            linewidth=0.8, zorder=3)

    for i, v in enumerate(vals):
        ax.text(v + 0.08, y[i], f"{v:.4f}", va="center", ha="left", fontsize=9.5)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.8)
    ax.set_xlabel("특징 중요도")
    ax.set_xlim(0, 5.8)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "fig5_feature_importance")


if __name__ == "__main__":
    print("그림 생성 중...")
    fig1_models()
    fig2_kablation()
    fig3_ablation()
    fig4_threshold()
    fig5_importance()
    print("완료.")
