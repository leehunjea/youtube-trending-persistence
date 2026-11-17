import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pickle

OUTPUT_PATH = "./output"

# ── 핵심 설계 변경 ─────────────────────────────────
# 시계열 Feature (동적 변화 포착)
SEQ_FEATURES = [
    'rank',               # 현재 순위
    'views_per_hour',     # 시간당 조회수
    'comment_view_ratio', # 댓글 비율
    'hours_since_publish',# 업로드 후 경과 시간
    'rank_delta',         # ✅ 순위 변화 (음수=순위 상승)
    'views_acceleration', # ✅ 조회수 가속도
    'momentum',           # ✅ 초기 대비 현재 성장률
]

# 정적 Feature (중복 제거 — DL/ML 공정 비교)
STATIC_FEATURES = [
    'category_id',
    'tag_count',
    'publish_hour_sin', 'publish_hour_cos',
    'publish_weekday_sin', 'publish_weekday_cos',
    'hours_to_trend',       # 업로드→트렌딩 진입 시간
    'relative_first_views', # 카테고리 대비 초기 조회수
]

# Ablation 대상 K_EARLY 목록
K_EARLY_LIST = [5, 7, 10, 14]
K_EARLY_DEFAULT = 10  # 기본값

print(f"[{datetime.now().strftime('%H:%M:%S')}] ▶ 데이터 로드 중...")

df_snap  = pd.read_csv(f"{OUTPUT_PATH}/kr_snapshot.csv",      encoding='utf-8-sig')
df_video = pd.read_csv(f"{OUTPUT_PATH}/kr_video_labeled.csv", encoding='utf-8-sig')

df_snap['collection_date'] = pd.to_datetime(df_snap['collection_date'])
df_snap['published_at']    = pd.to_datetime(df_snap['published_at'])

# 기본 Feature 재계산
if 'views_per_hour' not in df_snap.columns:
    df_snap['hours_since_publish'] = (
        df_snap['collection_date'] - df_snap['published_at']
    ).dt.total_seconds() / 3600
    df_snap['views_per_hour']      = df_snap['view_count'] / (df_snap['hours_since_publish'] + 1)
    df_snap['comment_view_ratio']  = df_snap['comment_count'] / (df_snap['view_count'] + 1)

print(f"  ✅ 스냅샷: {len(df_snap):,}행 | 영상: {len(df_video):,}개")

# ── Delta Feature 생성 ───────────────────────────────
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ▶ Delta Feature 생성 중...")

df_snap = df_snap.sort_values(['video_id', 'collection_date']).copy()

# rank_delta: 이전 스냅샷 대비 순위 변화 (음수=상승)
df_snap['rank_delta'] = df_snap.groupby('video_id')['rank'].diff().fillna(0)

# views_acceleration: 이전 대비 views_per_hour 증가량
df_snap['views_acceleration'] = df_snap.groupby('video_id')['views_per_hour'].diff().fillna(0)

# momentum: 현재 views_per_hour / 해당 영상 첫 스냅샷 views_per_hour
first_vph = df_snap.groupby('video_id')['views_per_hour'].transform('first')
df_snap['momentum'] = df_snap['views_per_hour'] / (first_vph + 1)

print(f"  ✅ rank_delta, views_acceleration, momentum 생성 완료")
print(f"  rank_delta 통계: mean={df_snap['rank_delta'].mean():.2f}, std={df_snap['rank_delta'].std():.2f}")
print(f"  momentum 통계  : mean={df_snap['momentum'].mean():.2f}, std={df_snap['momentum'].std():.2f}")

# ── 유효 영상 필터링 (최대 K_EARLY 기준) ─────────────
snap_count = df_snap.groupby('video_id')['collection_date'].nunique()
valid_ids  = snap_count[snap_count >= max(K_EARLY_LIST)].index
df_video_f = df_video[df_video['video_id'].isin(valid_ids)].reset_index(drop=True)

print(f"\n  K_EARLY max={max(K_EARLY_LIST)} 기준 유효 영상: {len(df_video_f):,}개")
print(f"  (전체 {len(df_video):,}개 중 {len(df_video_f)/len(df_video)*100:.1f}%)")

# ── Expert Feature 추출 함수 (ML과 동일 로직) ────────────────
def extract_expert_features(df_snap_vid, K):
    """시퀀스에서 전문가 설계 피처 추출"""
    # 1. 시계열 통계 (각 SEQ Feature별)
    expert_feat = {}
    for col_name in SEQ_FEATURES:
        col = df_snap_vid[col_name].values[:K]
        expert_feat[f'{col_name}_slope'] = np.polyfit(range(len(col)), col, 1)[0] if len(col) > 1 else 0
        expert_feat[f'{col_name}_mean']  = col.mean()
        expert_feat[f'{col_name}_std']   = col.std()

    # 2. 바이럴 핵심 신호 (복합 지표)
    rank_col = df_snap_vid['rank'].values[:K]
    vph_col  = df_snap_vid['views_per_hour'].values[:K]
    mom_col  = df_snap_vid['momentum'].values[:K]
    rd_col   = df_snap_vid['rank_delta'].values[:K]
    va_col   = df_snap_vid['views_acceleration'].values[:K]

    expert_feat['rank_improvement']    = rank_col[0] - rank_col[-1]
    expert_feat['peak_momentum']       = mom_col.max()
    expert_feat['momentum_trend']      = mom_col[-1] - mom_col[0]
    expert_feat['positive_rank_delta'] = (rd_col < 0).sum()
    expert_feat['max_acceleration']    = va_col.max()
    expert_feat['sustained_growth']    = (va_col > 0).sum() / K
    expert_feat['vph_growth_rate']     = vph_col[-1] / (vph_col[0] + 1e-6)

    return expert_feat

# ── 각 K_EARLY별 시퀀스 생성 함수 ────────────────────
def make_sequences(df_snap, df_video, K_EARLY, seq_scaler, static_scaler, expert_scaler):
    """K_EARLY 개 스냅샷으로 텐서 생성 (Expert Feature 포함)"""

    # 기본 정적 피처 스케일링
    static_scaled = static_scaler.transform(df_video[STATIC_FEATURES].fillna(0))

    # 시퀀스 데이터 준비
    df_snap_scaled = df_snap.copy()
    df_snap_scaled[SEQ_FEATURES] = seq_scaler.transform(df_snap[SEQ_FEATURES].fillna(0))

    video_ids  = df_video['video_id'].values
    labels_cls = df_video['is_viral'].values
    labels_reg = df_video['label_regression'].values
    splits     = df_video['split'].values

    N           = len(video_ids)
    X_seq       = np.zeros((N, K_EARLY, len(SEQ_FEATURES)), dtype=np.float32)
    mask        = np.zeros((N, K_EARLY), dtype=np.float32)
    expert_list = []

    grouped = df_snap.sort_values(['video_id','collection_date']).groupby('video_id')
    grouped_scaled = df_snap_scaled.sort_values(['video_id','collection_date']).groupby('video_id')

    print(f"  [{K_EARLY}] 시퀀스 및 Expert Feature 생성 중...")
    for i, vid in enumerate(video_ids):
        # 1. 시퀀스 데이터 (Scaled)
        grp_s = grouped_scaled.get_group(vid)
        vals  = grp_s[SEQ_FEATURES].values[:K_EARLY]
        L     = len(vals)
        X_seq[i, :L, :] = vals
        mask[i, :L]      = 1.0

        # 2. Expert Feature (Raw -> Scaler 나중에 적용)
        grp_raw = grouped.get_group(vid)
        expert_list.append(extract_expert_features(grp_raw, K_EARLY))

    # Expert Feature를 DF로 변환 후 스케일링
    df_expert = pd.DataFrame(expert_list)
    expert_scaled = expert_scaler.transform(df_expert.fillna(0))

    # 최종 Static 결합: [기존 정적 피처 + Expert 피처]
    X_static_combined = np.hstack([static_scaled, expert_scaled]).astype(np.float32)

    train_idx = np.where(splits == 'train')[0]
    val_idx   = np.where(splits == 'val')[0]
    test_idx  = np.where(splits == 'test')[0]

    return X_seq, X_static_combined, mask, labels_cls, labels_reg, train_idx, val_idx, test_idx, video_ids, df_expert.columns.tolist()

# ── Scaler 학습 (Train + K_EARLY_DEFAULT 기준) ───────
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ▶ Scaler 학습 중...")

train_ids  = df_video_f[df_video_f['split'] == 'train']['video_id'].values
df_snap_tr = df_snap[df_snap['video_id'].isin(train_ids)]
df_snap_tr_early = df_snap_tr.groupby('video_id').head(K_EARLY_DEFAULT)

# 1. Seq Scaler
seq_scaler = StandardScaler()
seq_scaler.fit(df_snap_tr_early[SEQ_FEATURES].fillna(0))

# 2. Static Scaler
static_scaler = StandardScaler()
static_scaler.fit(df_video_f[df_video_f['split'] == 'train'][STATIC_FEATURES].fillna(0))

# 3. Expert Scaler (추가)
expert_tr = []
for vid in train_ids:
    grp = df_snap_tr[df_snap_tr['video_id'] == vid].head(K_EARLY_DEFAULT)
    expert_tr.append(extract_expert_features(grp, K_EARLY_DEFAULT))

expert_scaler = StandardScaler()
expert_scaler.fit(pd.DataFrame(expert_tr).fillna(0))

print(f"  ✅ SEQ Scaler: {len(SEQ_FEATURES)}개")
print(f"  ✅ Static Scaler: {len(STATIC_FEATURES)}개")
print(f"  ✅ Expert Scaler: {len(expert_tr[0])}개 피처")

# ── K_EARLY별 시퀀스 생성 & 저장 ─────────────────────
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ▶ K_EARLY Ablation 시퀀스 생성 중...")

for K in K_EARLY_LIST:
    df_video_k = df_video[df_video['video_id'].isin(
        snap_count[snap_count >= K].index
    )].reset_index(drop=True)

    X_seq, X_static, mask, y_cls, y_reg, tr_idx, val_idx, te_idx, vids, expert_cols = \
        make_sequences(df_snap, df_video_k, K, seq_scaler, static_scaler, expert_scaler)

    fname = f"{OUTPUT_PATH}/kr_sequences_K{K}.npz"
    np.savez(fname,
        X_seq=X_seq, X_static=X_static, mask=mask,
        y_cls=y_cls, y_reg=y_reg,
        train_idx=tr_idx, val_idx=val_idx, test_idx=te_idx,
        video_ids=vids.astype(str),
        K_EARLY=np.array([K]),
        expert_features=np.array(expert_cols)
    )
    # ... (이후 동일)
    viral_r = y_cls.mean() * 100
    print(f"  K={K:2d} | 영상: {len(df_video_k):,} | 바이럴: {viral_r:.1f}% | "
          f"Shape: {X_seq.shape} → {fname}")

# 기본값(K=10)을 메인 파일로 복사
import shutil
shutil.copy(f"{OUTPUT_PATH}/kr_sequences_K{K_EARLY_DEFAULT}.npz",
            f"{OUTPUT_PATH}/kr_sequences.npz")
print(f"\n  ✅ 기본 파일: kr_sequences.npz = K_EARLY={K_EARLY_DEFAULT}")

# Scaler 저장
with open(f"{OUTPUT_PATH}/scalers.pkl", 'wb') as f:
    pickle.dump({
        'seq_scaler': seq_scaler, 
        'static_scaler': static_scaler,
        'expert_scaler': expert_scaler, # 추가
        'seq_features': SEQ_FEATURES, 
        'static_features': STATIC_FEATURES,
        'expert_features': expert_cols, # 추가
        'K_EARLY_default': K_EARLY_DEFAULT,
        'K_EARLY_list': K_EARLY_LIST,
    }, f)

print(f"  ✅ scalers.pkl 저장 완료")
print(f"\n  📌 SEQ Feature {len(SEQ_FEATURES)}개:")
for f in SEQ_FEATURES:
    flag = "✅ NEW" if f in ['rank_delta','views_acceleration','momentum'] else "   "
    print(f"  {flag} {f}")
print(f"\n  📌 STATIC Feature {len(STATIC_FEATURES)}개 (중복 제거 완료):")
for f in STATIC_FEATURES:
    print(f"     {f}")
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✅ 4단계 완료!")
