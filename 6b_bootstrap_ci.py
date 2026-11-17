import numpy as np
import pandas as pd
import os, glob
from datetime import datetime
from sklearn.metrics import roc_auc_score, f1_score

OUTPUT_PATH  = "./output"
N_BOOTSTRAP  = 10000
ALPHA        = 0.05
np.random.seed(42)

print("=" * 60)
print(f"  Bootstrap CI  N={N_BOOTSTRAP:,}  α={ALPHA} (95%)")
print("=" * 60)

# ── CSV 로드 ─────────────────────────────────────────
preds = {}
for fpath in sorted(glob.glob(f"{OUTPUT_PATH}/preds_*.csv")):
    name = os.path.basename(fpath).replace("preds_","").replace(".csv","")
    if name.endswith("_temporal"):
        print(f"  제외: {name:<22} (temporal split은 별도 표로 보고)")
        continue
    df   = pd.read_csv(fpath)

    if 'video_id' not in df.columns:
        # 과거 예측 CSV 호환: 재생성 전에는 위치 기반 비교만 가능하다.
        df = df.copy()
        df['video_id'] = np.arange(len(df)).astype(str)
        print(f"  ⚠️  {name:<22} video_id 없음 → 위치 기반 임시 ID 사용")
    else:
        df['video_id'] = df['video_id'].astype(str)

    preds[name] = df[['video_id', 'y_true', 'y_pred']].copy()
    auc  = roc_auc_score(df['y_true'], df['y_pred'])
    print(f"  로드: {name:<22} AUC={auc:.4f}  N={len(df)}")

# ── Bootstrap CI 함수 ────────────────────────────────
def boot_ci(y_true, y_pred, n=N_BOOTSTRAP):
    idx  = np.arange(len(y_true))
    aucs = []
    for _ in range(n):
        s = np.random.choice(idx, len(idx), replace=True)
        if len(np.unique(y_true[s])) < 2: continue
        aucs.append(roc_auc_score(y_true[s], y_pred[s]))
    aucs = np.array(aucs)
    return aucs, np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

def align_pair(df_a, df_b):
    """동일 video_id만 남겨 paired comparison용 배열을 만든다."""
    merged = df_a.merge(
        df_b,
        on='video_id',
        how='inner',
        suffixes=('_a', '_b'),
    )
    if merged.empty:
        raise ValueError("공통 video_id가 없습니다.")

    mismatch = (merged['y_true_a'].values != merged['y_true_b'].values).sum()
    if mismatch > 0:
        raise ValueError(f"공통 video_id 중 y_true 불일치 {mismatch}건")

    return (
        merged['y_true_a'].values,
        merged['y_pred_a'].values,
        merged['y_pred_b'].values,
        len(merged),
    )

def boot_diff(y_true, pa, pb, n=N_BOOTSTRAP):
    idx  = np.arange(len(y_true))
    diff = []
    for _ in range(n):
        s = np.random.choice(idx, len(idx), replace=True)
        if len(np.unique(y_true[s])) < 2: continue
        diff.append(roc_auc_score(y_true[s], pa[s]) -
                    roc_auc_score(y_true[s], pb[s]))
    diff = np.array(diff)
    obs  = roc_auc_score(y_true, pa) - roc_auc_score(y_true, pb)
    # 양측 paired bootstrap p-value: 0을 기준으로 관측 차이와 반대쪽 꼬리까지 고려
    p_left  = (diff <= 0).mean()
    p_right = (diff >= 0).mean()
    pval = min(1.0, 2 * min(p_left, p_right))
    return obs, pval, np.percentile(diff, 2.5), np.percentile(diff, 97.5)

# ── 개별 CI 계산 ─────────────────────────────────────
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ▶ 개별 Bootstrap CI...")
ci_rows = []
for name, df in preds.items():
    yt = df['y_true'].values
    yp = df['y_pred'].values
    obs = roc_auc_score(yt, yp)
    yb  = (yp >= 0.5).astype(int)
    _, lo, hi = boot_ci(yt, yp)
    ci_rows.append({
        'model':    name,
        'auc':      round(obs, 4),
        'ci_lower': round(lo,  4),
        'ci_upper': round(hi,  4),
        'f1':       round(f1_score(yt, yb), 4),
    })

df_ci = pd.DataFrame(ci_rows).sort_values('auc', ascending=False).reset_index(drop=True)
df_ci.to_csv(f"{OUTPUT_PATH}/bootstrap_ci.csv", index=False, encoding='utf-8-sig')

print(f"\n{'Model':<22} {'AUC':>6}  {'95% CI':^17}  {'F1':>6}")
print("-" * 60)
for _, r in df_ci.iterrows():
    print(f"  {r['model']:<20} {r['auc']:.4f}  [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]  {r['f1']:.4f}")

# ── 핵심 쌍 비교 ─────────────────────────────────────
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ▶ 쌍별 유의성 검정...")

# 공통 test set 크기 맞추기용 — K=14 기준
pairs = [
    ('CatBoost_v2', 'Transformer_K14', 'ML Best vs DL Best (K=14)'),
    ('CatBoost_v2', 'CatBoost_v1',  'CatBoost: K=14 vs K=5 (Delta 효과)'),
    ('GRU_K14',     'GRU_K5',       'GRU: K=14 vs K=5 (K 효과)'),
    ('LSTM_K14',    'GRU_K14',      'LSTM vs GRU (K=14)'),
    ('GRU_K14',     'Transformer_K14', 'GRU vs Transformer (K=14)'),
    ('CatBoost_v2', 'XGB_v2',       'CatBoost vs XGBoost (ML v2)'),
]

diff_rows = []
for na, nb, desc in pairs:
    if na not in preds or nb not in preds:
        print(f"  ⚠️  {na} 또는 {nb} 없음 → skip")
        continue
    try:
        ya, pa, pb, n_common = align_pair(preds[na], preds[nb])
    except ValueError as e:
        print(f"  ⚠️  {desc} skip: {e}")
        continue

    obs, pval, lo, hi = boot_diff(ya, pa, pb)
    sig = "✅ p<0.05" if pval < ALPHA else f"❌ p={pval:.3f}"

    diff_rows.append({
        'comparison': desc, 'A': na, 'B': nb,
        'n_common': n_common,
        'auc_A': round(roc_auc_score(ya,pa),4),
        'auc_B': round(roc_auc_score(ya,pb),4),
        'delta': round(obs,4),
        'ci_lower': round(lo,4), 'ci_upper': round(hi,4),
        'p_value': round(pval,4),
        'significant': pval < ALPHA,
    })
    print(f"\n  [{desc}]")
    print(f"  N_common={n_common:,}")
    print(f"  {na}={roc_auc_score(ya,pa):.4f}  {nb}={roc_auc_score(ya,pb):.4f}")
    print(f"  Δ={obs:+.4f}  95%CI [{lo:.4f},{hi:.4f}]  {sig}")

df_diff = pd.DataFrame(diff_rows)
df_diff.to_csv(f"{OUTPUT_PATH}/bootstrap_diff.csv", index=False, encoding='utf-8-sig')

# ── 논문용 요약 ──────────────────────────────────────
print(f"\n\n{'='*60}")
print("  📋 논문용 통계 요약")
print(f"{'='*60}")

# K 효과 (GRU 기준)
for k in [5,7,10,14]:
    key = f'GRU_K{k}'
    if key in preds:
        yt = preds[key]['y_true'].values
        yp = preds[key]['y_pred'].values
        print(f"  GRU K={k:2d}: AUC={roc_auc_score(yt,yp):.4f}")

print(f"\n  ML v1 Best  : {df_ci[df_ci['model'].str.contains('v1')]['auc'].max():.4f}")
print(f"  DL Best K14 : {df_ci[df_ci['model'].str.contains('K14')]['auc'].max():.4f}")
print(f"  ML v2 Best  : {df_ci[df_ci['model'].str.contains('v2')]['auc'].max():.4f}")

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✅ 완료!")
print(f"  → output/bootstrap_ci.csv")
print(f"  → output/bootstrap_diff.csv")
