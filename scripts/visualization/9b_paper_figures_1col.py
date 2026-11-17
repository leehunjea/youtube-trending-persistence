# -*- coding: utf-8 -*-
"""
2단 조판 단칼럼(약 8.5cm) 그림 — 절제된 학술지 스타일.
설계 원칙
  · 수치는 표가 담당한다. 그림은 '패턴'만 보여준다(막대 위 숫자 나열 제거).
  · 색은 계열 구분에만 쓰고, 해칭·굵은 테두리 같은 중복 장식은 쓰지 않는다.
  · 범례는 그림 안쪽 여백에 조용히 놓는다.
출력: figures_1col/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

OUT = "./figures_1col"
os.makedirs(OUT, exist_ok=True)

W = 3.35  # inch ≈ 8.5cm

# 연구 프로세스 그림(그림 1)과 통일한 팔레트
NAVY  = "#1F3C6E"
BLUE  = "#3D7EBB"
TEAL  = "#2E9AA6"
CORAL = "#D6453C"
PALE  = "#C3D5E8"
INK   = "#1A1A1A"
MUTE  = "#6B6B6B"
RULE  = "#D8D8D8"

plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 8,
    "text.color": INK,
    "axes.labelsize": 8.5,
    "axes.labelcolor": INK,
    "axes.labelpad": 4,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.color": MUTE,
    "ytick.color": MUTE,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 7.5,
    "axes.edgecolor": "#9A9A9A",
    "axes.linewidth": 0.6,
    "grid.color": RULE,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.3,
})


def tidy(ax, grid_axis="y"):
    """공통 마감: 위·오른쪽 축선 제거, 옅은 격자, 눈금 색 정리."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9A9A9A")
    ax.grid(axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, pad=2.5)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)
    print(f"  {OUT}/{name}")


# ── 그림 2. 모델별 ROC-AUC ────────────────────────────────
def fig_models():
    names = ["CatBoost", "LightGBM", "XGBoost", "Random Forest",
             "LSTM", "GRU", "Transformer", "TFT", "MLP-Tabular"]
    auc = [0.8416, 0.8385, 0.8361, 0.8318, 0.8296, 0.8266, 0.8254, 0.8236, 0.8235]
    lo = [0.8252, 0.8218, 0.8190, 0.8148, 0.8129, 0.8091, 0.8078, 0.8061, 0.8062]
    hi = [0.8575, 0.8548, 0.8526, 0.8484, 0.8465, 0.8435, 0.8423, 0.8405, 0.8404]
    color = [NAVY] * 4 + [TEAL] * 4 + [PALE]

    y = np.arange(len(names))[::-1]
    err = [np.array(auc) - np.array(lo), np.array(hi) - np.array(auc)]

    fig, ax = plt.subplots(figsize=(W, 2.95))
    ax.barh(y, auc, height=0.62, color=color, linewidth=0, zorder=3)
    ax.errorbar(auc, y, xerr=err, fmt="none", ecolor="#4A4A4A",
                elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=4)

    ax.axvline(0.8017, color=CORAL, linestyle=(0, (4, 2)), linewidth=0.9, zorder=2)
    ax.text(0.8017, len(names) - 0.35, " 기준선 0.8017", color=CORAL,
            fontsize=7, va="bottom", ha="left")

    ax.set_yticks(y)
    ax.set_yticklabels(names, color=INK)
    ax.set_xlabel("ROC-AUC")
    ax.set_xlim(0.795, 0.862)
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    tidy(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)

    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=5.5,
                          markerfacecolor=c, markeredgecolor="none", label=l)
               for c, l in [(NAVY, "트리 기반"), (TEAL, "딥러닝"), (PALE, "MLP")]]
    ax.legend(handles=handles, loc="lower right", frameon=False,
              handletextpad=0.4, labelspacing=0.28, borderpad=0.3)
    save(fig, "fig2_model_comparison_1col")


# ── 그림 3. 관측 길이 K에 따른 성능 ───────────────────────
def fig_kablation():
    K = [5, 7, 10, 14]
    ml = {"CatBoost": [0.7816, 0.8028, 0.8163, 0.8416],
          "XGBoost": [0.7819, 0.7910, 0.8153, 0.8361],
          "LightGBM": [0.7805, 0.7956, 0.8141, 0.8385],
          "Random Forest": [0.7671, 0.7818, 0.8066, 0.8318]}
    dl = {"LSTM": [0.7589, 0.7761, 0.7925, 0.8296],
          "GRU": [0.7613, 0.7735, 0.7976, 0.8266],
          "Transformer": [0.7574, 0.7682, 0.7965, 0.8254],
          "TFT": [0.7599, 0.7682, 0.8008, 0.8236]}
    colors = [NAVY, BLUE, TEAL, "#9BB4CC"]
    mk = ["o", "s", "^", "D"]
    ls = ["-", "--", "-.", (0, (1, 1.6))]

    fig, axes = plt.subplots(2, 1, figsize=(W, 4.15), sharex=True)
    for ax, data, title in zip(axes, (ml, dl), ("(a) 머신러닝", "(b) 딥러닝")):
        for i, (name, vals) in enumerate(data.items()):
            ax.plot(K, vals, marker=mk[i], linestyle=ls[i], color=colors[i],
                    markersize=3.8, markerfacecolor="white",
                    markeredgewidth=1.0, label=name, zorder=3)
        ax.set_title(title, fontsize=8, color=INK, pad=5, loc="left")
        ax.set_xticks(K)
        ax.set_ylabel("ROC-AUC")
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        tidy(ax)
        ax.legend(frameon=False, loc="lower right", ncol=2,
                  handlelength=1.9, columnspacing=0.8, labelspacing=0.25,
                  handletextpad=0.4, borderpad=0.3)
    axes[0].set_ylim(0.758, 0.856)
    axes[1].set_ylim(0.750, 0.845)
    axes[1].set_xlabel("초기 관측 길이 K (스냅샷 개수)")
    fig.tight_layout(h_pad=1.2)
    save(fig, "fig3_k_ablation_1col")


# ── 그림 4. 기준선 및 특징군 제거 ─────────────────────────
def fig_ablation():
    conds = ["전체 특징 (71)", "정적＋시계열 요약 (40)", "변화량·성장 특징 (31)",
             "마지막 순위 점수 (1)", "정적＋첫 스냅샷 (12)", "정적 특징 (8)"]
    auc = [0.8416, 0.8415, 0.8121, 0.8017, 0.7567, 0.7227]
    ap = [0.6294, 0.6355, 0.5602, 0.5411, 0.5391, 0.4974]

    y = np.arange(len(conds))[::-1]
    h = 0.34
    fig, ax = plt.subplots(figsize=(W, 2.75))
    ax.barh(y + h / 2 + 0.02, auc, h, color=NAVY, linewidth=0,
            label="ROC-AUC", zorder=3)
    ax.barh(y - h / 2 - 0.02, ap, h, color=TEAL, linewidth=0,
            label="AP", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(conds, color=INK)
    ax.set_xlabel("점수")
    ax.set_xlim(0, 0.92)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    tidy(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0),
              ncol=2, handlelength=1.1, handleheight=0.9, handletextpad=0.45,
              columnspacing=1.0, borderpad=0.1)
    save(fig, "fig4_feature_ablation_1col")


# ── 그림 5. 특징 중요도 ───────────────────────────────────
def fig_importance():
    feats = [("category_id", 4.8839), ("rank_mean", 4.5483), ("rank_max", 4.4893),
             ("tag_count", 3.8885), ("views_acceleration_last", 3.8703),
             ("rank_last", 3.7495), ("rank_delta_slope", 2.6262),
             ("comment_view_ratio_slope", 2.5419), ("momentum_std", 2.4404),
             ("publish_hour_cos", 2.3844)]
    names = [f[0] for f in feats][::-1]
    vals = [f[1] for f in feats][::-1]

    fig, ax = plt.subplots(figsize=(W, 2.85))
    y = np.arange(len(names))
    # 모든 막대가 같은 값(특징 중요도)이므로 색을 나누지 않는다.
    # 순위는 정렬과 막대 길이가 이미 보여준다.
    ax.barh(y, vals, height=0.64, color=NAVY, linewidth=0, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.2, color=INK)
    ax.set_xlabel("특징 중요도")
    ax.set_xlim(0, 5.4)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    tidy(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)
    save(fig, "fig5_feature_importance_1col")


# ── 그림 6. 라벨 임곗값 민감도 ────────────────────────────
def fig_threshold():
    labels = ["p65", "p70", "p75", "p80", "p85"]
    auc = [0.8457, 0.8407, 0.8416, 0.8456, 0.8396]
    ap = [0.7424, 0.6908, 0.6294, 0.5774, 0.4935]
    f1 = [0.7109, 0.6762, 0.6270, 0.5873, 0.5308]
    pos = [0.3591, 0.3129, 0.2558, 0.2028, 0.1528]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(W, 2.6))
    ax.bar(x, pos, width=0.46, color=PALE, linewidth=0, label="양성률", zorder=1)
    for vals, name, m, l, c in [(auc, "ROC-AUC", "o", "-", NAVY),
                                (ap, "AP", "s", "--", TEAL),
                                (f1, "F1", "^", "-.", CORAL)]:
        ax.plot(x, vals, marker=m, linestyle=l, color=c, markersize=3.8,
                markerfacecolor="white", markeredgewidth=1.0, label=name, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=INK)
    ax.set_xlabel("라벨 임곗값 기준")
    ax.set_ylabel("점수 / 양성률")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    tidy(ax)
    ax.legend(frameon=False, ncol=2, loc="center left", bbox_to_anchor=(0.02, 0.42),
              handlelength=1.7, columnspacing=0.9, labelspacing=0.28,
              handletextpad=0.4, borderpad=0.3)
    save(fig, "fig6_threshold_1col")


if __name__ == "__main__":
    print("단칼럼 그림 생성 중...")
    fig_models()
    fig_kablation()
    fig_ablation()
    fig_importance()
    fig_threshold()
    print("완료.")
