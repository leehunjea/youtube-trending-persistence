"""
Fig 5a. LSTM Attention 1D (Viral vs Non-Viral)
Fig 5b. ROC Curve per K_EARLY
Fig 5c. K_EARLY 별 AUC 향상 누적 (논문 메인 주장)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import os

# ── 한글 폰트 설정 (Windows) ──────────────────────────
import matplotlib.font_manager as fm

# 방법 1: 맑은 고딕 직접 지정 (Windows 기본 탑재)
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_PATH = "./output"
FIGURE_PATH = "./figures"
os.makedirs(FIGURE_PATH, exist_ok=True)

DARK_BG  = '#0f1117'
PANEL_BG = '#1a1d2e'
GRID_COL = '#2e3250'
TEXT_COL = '#e8eaf6'

plt.rcParams.update({
    'figure.facecolor': DARK_BG, 'axes.facecolor': PANEL_BG,
    'axes.edgecolor': '#3d4166', 'axes.labelcolor': TEXT_COL,
    'xtick.color': TEXT_COL, 'ytick.color': TEXT_COL,
    'text.color': TEXT_COL, 'grid.color': GRID_COL,
    'grid.linewidth': 0.8, 'grid.alpha': 0.5,
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.labelsize': 12, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'legend.fontsize': 10,
    'legend.framealpha': 0.3, 'legend.edgecolor': '#3d4166',
})

def savefig(fig, name, dpi=180):
    path = f"{FIGURE_PATH}/{name}"
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✅ {path}")

# ════════════════════════════════════════════════════
# Fig 5a: LSTM Attention Weight — Viral vs Non-Viral
# ════════════════════════════════════════════════════
print("[1/3] LSTM Attention Figure...")

v_path  = f"{OUTPUT_PATH}/LSTM_K14_attn_viral.npy"
nv_path = f"{OUTPUT_PATH}/LSTM_K14_attn_nonviral.npy"

if os.path.exists(v_path) and os.path.exists(nv_path):
    mean_v  = np.load(v_path)
    mean_nv = np.load(nv_path)
    std_v   = np.load(f"{OUTPUT_PATH}/LSTM_K14_attn_std_v.npy")
    std_nv  = np.load(f"{OUTPUT_PATH}/LSTM_K14_attn_std_nv.npy")
    K = len(mean_v)
    source = "Real LSTM K=14 Attention"
else:
    # 현실적 시뮬
    K = 14
    t = np.arange(K)
    mean_v  = 0.04 + 0.005 * t + 0.003 * t**1.5
    mean_v  /= mean_v.sum()
    mean_nv = 0.12 - 0.004 * t
    mean_nv[mean_nv < 0.01] = 0.01
    mean_nv /= mean_nv.sum()
    std_v   = np.ones(K) * 0.012
    std_nv  = np.ones(K) * 0.014
    source  = "Simulated"
    print("  ⚠️  Simulated data 사용")

t_labels = [f't{i+1}' for i in range(K)]
t_idx    = np.arange(K)
diff     = mean_v - mean_nv

fig, axes = plt.subplots(2, 1, figsize=(11, 8),
                          gridspec_kw={'height_ratios': [2.5, 1]})

# 상단: Attention 분포
ax = axes[0]
ax.plot(t_idx, mean_v, 'o-', color='#4FC3F7', lw=2.5, ms=7,
        label='Viral (바이럴)', zorder=5)
ax.fill_between(t_idx,
                mean_v - std_v,
                mean_v + std_v,
                color='#4FC3F7', alpha=0.15, zorder=3)

ax.plot(t_idx, mean_nv, 's--', color='#EF9A9A', lw=2.5, ms=7,
        label='Non-Viral (비바이럴)', zorder=5)
ax.fill_between(t_idx,
                mean_nv - std_nv,
                mean_nv + std_nv,
                color='#EF9A9A', alpha=0.15, zorder=3)

# 최대 차이 지점 강조
peak_idx = np.abs(diff).argmax()
ax.axvline(peak_idx, color='#FFD54F', ls=':', lw=1.8, alpha=0.7,
           label=f'Max diff at t{peak_idx+1}')
ax.annotate(f'Δ={diff[peak_idx]:+.4f}',
            xy=(peak_idx, max(mean_v[peak_idx], mean_nv[peak_idx])),
            xytext=(peak_idx+0.8, max(mean_v[peak_idx],mean_nv[peak_idx])+0.004),
            color='#FFD54F', fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#FFD54F', lw=1.5))

ax.set_xticks(t_idx)
ax.set_xticklabels(t_labels)
ax.set_ylabel('Average Attention Weight')
ax.set_title(f'Fig. 5a  LSTM K=14 Attention: Viral vs Non-Viral\n'
             f'[{source}]  Mean ± Std across test set')
ax.legend(loc='upper left')
ax.grid(True, axis='y')

# 하단: 차이값
ax2 = axes[1]
bar_c = ['#4FC3F7' if d > 0 else '#EF9A9A' for d in diff]
ax2.bar(t_idx, diff, color=bar_c, width=0.65, zorder=3, alpha=0.85)
ax2.axhline(0, color='white', lw=1.2)
ax2.set_xticks(t_idx)
ax2.set_xticklabels(t_labels)
ax2.set_ylabel('Δ Attention\n(Viral − Non-Viral)')
ax2.set_xlabel('Snapshot Timestep')
ax2.grid(True, axis='y')
patches = [
    mpatches.Patch(color='#4FC3F7', label='Viral pays MORE attention'),
    mpatches.Patch(color='#EF9A9A', label='Non-Viral pays MORE attention'),
]
ax2.legend(handles=patches, loc='upper left', fontsize=9)

fig.tight_layout(h_pad=0.5)
savefig(fig, 'fig5a_lstm_attention.png')

# ════════════════════════════════════════════════════
# Fig 5b: ROC Curve per K_EARLY (GRU + LSTM)
# ════════════════════════════════════════════════════
print("[2/3] ROC Curve per K...")

K_configs = [
    (5,  'LSTM', '#b0bec5', '-',  1.5),
    (7,  'LSTM', '#80deea', '--', 1.8),
    (10, 'LSTM', '#4FC3F7', '-',  2.0),
    (14, 'LSTM', '#0288D1', '-',  2.8),
    (14, 'GRU',  '#81C784', '-.',  2.0),
    (14, 'Transformer', '#FFB74D', ':', 2.0),
]
ml_baselines = [
    ('CatBoost_v1', '#9e9e9e', '--', 1.5, 'CatBoost v1 (K=5)'),
    ('CatBoost_v2', '#FF6B00', '-',  2.2, 'CatBoost v2 (K=14)'),
]

fig, ax = plt.subplots(figsize=(9, 7.5))

for K, mn, color, ls, lw in K_configs:
    fname = f"{OUTPUT_PATH}/preds_{mn}_K{K}.csv"
    if not os.path.exists(fname):
        continue
    df = pd.read_csv(fname)
    fpr, tpr, _ = roc_curve(df['y_true'], df['y_pred'])
    roc_auc = auc(fpr, tpr)
    label = f'{mn} K={K}  (AUC={roc_auc:.4f})'
    ax.plot(fpr, tpr, color=color, lw=lw, ls=ls, label=label, alpha=0.9)

for fname_key, color, ls, lw, label_prefix in ml_baselines:
    fname = f"{OUTPUT_PATH}/preds_{fname_key}.csv"
    if not os.path.exists(fname):
        continue
    df = pd.read_csv(fname)
    fpr, tpr, _ = roc_curve(df['y_true'], df['y_pred'])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=lw, ls=ls,
            label=f'{label_prefix}  (AUC={roc_auc:.4f})', alpha=0.9)

ax.plot([0,1],[0,1], color='#444', lw=1.2, ls='--', label='Random (AUC=0.500)')
ax.set_xlabel('False Positive Rate (1 - Specificity)')
ax.set_ylabel('True Positive Rate (Sensitivity / Recall)')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_title('Fig. 5b  ROC Curve: K_EARLY Ablation + ML Baselines')
ax.legend(loc='lower right', fontsize=9.5, framealpha=0.4)
ax.grid(True, alpha=0.35)

# AUC 향상 화살표
ax.annotate('', xy=(0.2, 0.82), xytext=(0.2, 0.72),
            arrowprops=dict(arrowstyle='->', color='#FFD54F', lw=2))
ax.text(0.21, 0.77, 'K↑ → AUC↑', color='#FFD54F', fontsize=10, va='center')

fig.tight_layout()
savefig(fig, 'fig5b_roc_per_k.png')

# ════════════════════════════════════════════════════
# Fig 5c: K_EARLY × Model AUC Heatmap
# ════════════════════════════════════════════════════
print("[3/3] AUC Heatmap per K...")

auc_matrix = {
    'LSTM':        [0.7657, 0.7749, 0.7954, 0.8320],
    'GRU':         [0.7608, 0.7731, 0.7946, 0.8260],
    'Transformer': [0.7599, 0.7702, 0.7976, 0.8317],
    'TFT':         [0.7539, 0.7704, 0.7966, 0.8263],
}
K_list   = [5, 7, 10, 14]
models   = list(auc_matrix.keys())
mat      = np.array([auc_matrix[m] for m in models])  # (4, 4)

cmap = LinearSegmentedColormap.from_list(
    'auc', ['#1a1d2e','#0d47a1','#1565c0','#1976d2','#42a5f5','#80d8ff'])

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(mat, cmap=cmap, aspect='auto',
               vmin=0.745, vmax=0.840)

for i in range(len(models)):
    for j in range(len(K_list)):
        val = mat[i, j]
        color = 'white' if val < 0.80 else '#0f1117'
        ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                fontsize=12, fontweight='bold', color=color)

ax.set_xticks(range(len(K_list)))
ax.set_xticklabels([f'K={k}' for k in K_list], fontsize=12)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=12)
ax.set_xlabel('K_EARLY (Number of Early Snapshots)', fontsize=12)
ax.set_ylabel('DL Model', fontsize=12)
ax.set_title('Fig. 5c  ROC-AUC Heatmap: Model × K_EARLY')

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('ROC-AUC', fontsize=11)

# 열 평균 추가
col_means = mat.mean(axis=0)
for j, cm in enumerate(col_means):
    ax.text(j, len(models)-0.35, f'avg\n{cm:.4f}',
            ha='center', va='bottom', fontsize=9,
            color='#FFD54F', fontweight='bold')

fig.tight_layout()
savefig(fig, 'fig5c_auc_heatmap.png')

# ════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("  모든 Figure 생성 완료!")
print(f"{'='*55}")
for f in sorted(os.listdir(FIGURE_PATH)):
    if f.endswith('.png'):
        sz = os.path.getsize(f"{FIGURE_PATH}/{f}") // 1024
        print(f"  {f:<40} {sz:>5} KB")
