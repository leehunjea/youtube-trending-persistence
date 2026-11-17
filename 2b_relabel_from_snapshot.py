"""
Leakage-fixed regeneration of kr_video_labeled.csv from the EXISTING
output/kr_snapshot.csv, avoiding a full re-parse of the 77GB/52GB raw CSVs.

This reproduces steps 6-11 of 2_preprocess.py verbatim. kr_snapshot.csv is the
KR-filtered per-snapshot table that 2_preprocess.py saves at step 5; steps 6-11
operate purely on that table, so regenerating from it is identical to a full run
(only relative_first_views is recomputed train-only, the leakage fix).
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

OUTPUT_PATH = "./output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── 0. 기존 스냅샷 로드 (raw 재파싱 회피) ─────────────
print(f"[{datetime.now().strftime('%H:%M:%S')}] ▶ kr_snapshot.csv 로드 중...")
df_kr = pd.read_csv(f"{OUTPUT_PATH}/kr_snapshot.csv", encoding="utf-8-sig", low_memory=False)
df_kr['collection_date'] = pd.to_datetime(df_kr['collection_date'], errors='coerce')
df_kr['published_at']    = pd.to_datetime(df_kr['published_at'], errors='coerce')
print(f"  ✅ 스냅샷: {len(df_kr):,}행 × {df_kr.shape[1]}열")

# ── 6. 전체 집계 (레이블용) ───────────────────────────
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ▶ 전체 집계 중...")

all_agg = df_kr.groupby('video_id').agg(
    max_views_per_hour  = ('views_per_hour', 'max'),
    best_rank           = ('rank',           'min'),
    trending_snapshots  = ('collection_date','nunique'),  # ✅ 레이블
    first_trending_date = ('collection_date','min'),
).reset_index()

# ── 7. 첫 스냅샷 추출 (ML Feature용) ─────────────────
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ▶ 첫 스냅샷 추출 중...")

first_snap = df_kr.sort_values(
    ['video_id', 'collection_date']
).groupby('video_id').first().reset_index()

first_snap = first_snap.rename(columns={
    'rank':               'first_rank',
    'view_count':         'first_views',
    'comment_count':      'first_comments',
    'views_per_hour':     'first_views_per_hour',
    'comment_view_ratio': 'first_comment_view_ratio',
    'hours_since_publish':'first_hours_since_publish',
})

# ── 8. 병합 ───────────────────────────────────────────
video_df = first_snap[[
    'video_id', 'title', 'channel_title', 'channel_id',
    'category_id', 'published_at',
    'publish_hour', 'publish_weekday',
    'publish_hour_sin', 'publish_hour_cos',
    'publish_weekday_sin', 'publish_weekday_cos',
    'first_rank', 'first_views', 'first_comments',
    'first_views_per_hour', 'first_comment_view_ratio',
    'first_hours_since_publish', 'tag_count',
]].merge(all_agg, on='video_id', how='left')

# 업로드 → 첫 트렌딩 도달 시간
video_df['hours_to_trend'] = (
    video_df['first_trending_date'] - video_df['published_at']
).dt.total_seconds() / 3600
video_df['hours_to_trend'] = video_df['hours_to_trend'].clip(lower=0)

# 카테고리 대비 상대적 초기 조회수는 split 이후 train 기준으로 재계산한다.
video_df['category_median_views'] = np.nan
video_df['relative_first_views'] = np.nan

print(f"  ✅ {len(video_df):,}개 고유 영상")

# ── 9. ✅ 레이블 재정의 (trending_snapshots 기반) ─────
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ▶ 레이블 재정의 중...")

snap_stats = video_df['trending_snapshots']
print(f"\n  📊 trending_snapshots 분포:")
print(f"  min    : {snap_stats.min():.0f}")
print(f"  25%    : {snap_stats.quantile(0.25):.0f}")
print(f"  중앙값 : {snap_stats.median():.0f}")
print(f"  75%    : {snap_stats.quantile(0.75):.0f}")
print(f"  90%    : {snap_stats.quantile(0.90):.0f}")
print(f"  max    : {snap_stats.max():.0f}")

# 회귀 레이블 (TFT 용)
video_df['label_regression'] = video_df['trending_snapshots']

# 분류 레이블 — 중앙값 기준 상위 = 강한 바이럴
median_snaps = snap_stats.median()
p75_snaps    = snap_stats.quantile(0.75)

video_df['is_viral'] = (
    video_df['trending_snapshots'] >= p75_snaps
).astype(int)

# 3단계 등급
video_df['viral_grade'] = pd.cut(
    video_df['trending_snapshots'],
    bins   = [0, median_snaps, p75_snaps, np.inf],
    labels = ['단기(일반)', '중기(준바이럴)', '장기(바이럴)'],
    right  = True
)

print(f"\n  📊 레이블 분포 (trending_snapshots >= {p75_snaps:.0f}회):")
print(f"  바이럴(1)   : {video_df['is_viral'].sum():,}개 ({video_df['is_viral'].mean()*100:.1f}%)")
print(f"  비바이럴(0) : {(video_df['is_viral']==0).sum():,}개")

# ── 10. ✅ Train / Val / Test 3분할 ──────────────────
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ▶ Train/Val/Test 분할 중...")

from sklearn.model_selection import train_test_split

idx = np.arange(len(video_df))
y   = video_df['is_viral'].values

# 70 / 15 / 15 분할
idx_train, idx_temp, y_train, y_temp = train_test_split(
    idx, y, test_size=0.30, random_state=42, stratify=y
)
idx_val, idx_test, _, _ = train_test_split(
    idx_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

video_df['split'] = 'train'
video_df.iloc[idx_val,  video_df.columns.get_loc('split')]  = 'val'
video_df.iloc[idx_test, video_df.columns.get_loc('split')]  = 'test'

print(f"  Train : {(video_df['split']=='train').sum():,}개 (70%)")
print(f"  Val   : {(video_df['split']=='val').sum():,}개   (15%)")
print(f"  Test  : {(video_df['split']=='test').sum():,}개  (15%)")

# ── 10-1. Train 기준 카테고리 중앙값으로 상대 초기 조회수 계산 ──
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ▶ Train-only category median 계산 중...")
train_category_median = (
    video_df[video_df['split'] == 'train']
    .groupby('category_id')['first_views']
    .median()
)
global_train_median = video_df.loc[video_df['split'] == 'train', 'first_views'].median()

video_df['category_median_views'] = (
    video_df['category_id']
    .map(train_category_median)
    .fillna(global_train_median)
)
video_df['relative_first_views'] = video_df['first_views'] / (
    video_df['category_median_views'] + 1
)
print("  ✅ relative_first_views: train split 기준으로 재계산 완료")

# ── 11. 저장 ──────────────────────────────────────────
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ▶ 저장 중...")

SAVE_COLS = [
    'video_id', 'title', 'channel_title', 'channel_id', 'split',
    'category_id', 'tag_count',
    'publish_hour', 'publish_weekday',
    'publish_hour_sin', 'publish_hour_cos',
    'publish_weekday_sin', 'publish_weekday_cos',
    'first_rank', 'first_views', 'first_comments',
    'first_views_per_hour', 'first_comment_view_ratio',
    'first_hours_since_publish',
    'hours_to_trend', 'relative_first_views',
    'trending_snapshots', 'best_rank', 'max_views_per_hour',
    'label_regression', 'is_viral', 'viral_grade'
]

video_df[SAVE_COLS].to_csv(
    f"{OUTPUT_PATH}/kr_video_labeled.csv",
    index=False, encoding='utf-8-sig'
)

meta = {
    'label_definition':    f'trending_snapshots >= {p75_snaps:.0f} (75th percentile)',
    'regression_target':   'trending_snapshots (raw count)',
    'split_ratio':         'Train 70 / Val 15 / Test 15',
    'sin_cos_encoding':    'publish_hour(24), publish_weekday(7)',
    'total_videos':        len(video_df),
    'viral_count':         int(video_df['is_viral'].sum()),
    'median_snapshots':    float(median_snaps),
    'p75_snapshots':       float(p75_snaps),
    'created_at':          datetime.now().strftime('%Y-%m-%d %H:%M'),
    'regen_source':        'kr_snapshot.csv (raw re-parse skipped)',
}
pd.DataFrame([meta]).to_csv(
    f"{OUTPUT_PATH}/label_metadata.csv",
    index=False, encoding='utf-8-sig'
)

print(f"  ✅ output/kr_video_labeled.csv 저장 완료 ({len(video_df):,}행)")
print(f"  ✅ output/label_metadata.csv 저장 완료")
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✅ 재라벨 완료!")
