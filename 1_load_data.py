import pandas as pd
import os
from datetime import datetime
# 데이터 로드 + 저장

# ── 경로 설정 ─────────────────────────────────────────
DATA_PATH   = "./data/youtube_trends"
OUTPUT_PATH = "./output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── 플랫폼/데이터셋 식별자 (나중에 트위터 등 추가 시 활용) ──
PLATFORM   = "youtube"
DATASET_ID = "illinois_trending"

def load_csv_with_progress(filepath, label):
    """청크 단위 로드 + 진행 상황 출력"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📂 {label} 로드 중...")
    
    chunks = []
    chunk_count = 0
    total_rows  = 0
    
    for chunk in pd.read_csv(filepath, encoding='utf-8-sig', chunksize=200_000):
        chunks.append(chunk)
        chunk_count += 1
        total_rows  += len(chunk)
        print(f"  ... {chunk_count}번째 청크 완료 ({total_rows:,}행 누적)", end='\r')
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"\n  ✅ {label}: {df.shape[0]:,}행 × {df.shape[1]}열 로드 완료")
    return df

def validate_columns(df, required_cols, label):
    """필수 컬럼 존재 여부 확인"""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  ⚠️  [{label}] 누락 컬럼: {missing}")
    else:
        print(f"  ✅ [{label}] 필수 컬럼 모두 확인")

# ── 필수 컬럼 정의 ────────────────────────────────────
# 나중에 트위터/다른 YouTube 데이터 추가 시 각각 정의
REQUIRED_COLS_POPULAR = [
    'collection_date', 'region_code', 'rank',
    'video_id', 'published_at', 'category_id',
    'view_count', 'comment_count'
]
REQUIRED_COLS_TAG = ['video_id']

# ── 데이터 로드 ───────────────────────────────────────
print("=" * 55)
print(f"  플랫폼  : {PLATFORM}")
print(f"  데이터셋: {DATASET_ID}")
print(f"  경로    : {DATA_PATH}")
print("=" * 55)

df_popular = load_csv_with_progress(
    f"{DATA_PATH}/most_popular.csv", "most_popular"
)
df_tag = load_csv_with_progress(
    f"{DATA_PATH}/tag.csv", "tag"
)

# ── 컬럼 검증 ─────────────────────────────────────────
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔍 컬럼 검증 중...")
validate_columns(df_popular, REQUIRED_COLS_POPULAR, "most_popular")
validate_columns(df_tag,     REQUIRED_COLS_TAG,     "tag")

# ── 기본 정보 출력 ────────────────────────────────────
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 📊 데이터 기본 정보")
print("\n=== most_popular 컬럼 ===")
for i, col in enumerate(df_popular.columns):
    null_pct = df_popular[col].isnull().mean() * 100
    print(f"  {i:02d}. {col:<30} dtype: {str(df_popular[col].dtype):<10} null: {null_pct:.1f}%")

print("\n=== tag 컬럼 ===")
for i, col in enumerate(df_tag.columns):
    null_pct = df_tag[col].isnull().mean() * 100
    print(f"  {i:02d}. {col:<30} dtype: {str(df_tag[col].dtype):<10} null: {null_pct:.1f}%")

# ── 샘플 출력 ─────────────────────────────────────────
print(f"\n=== most_popular 샘플 (3행) ===")
print(df_popular.head(3).to_string())
print(f"\n=== tag 샘플 (3행) ===")
print(df_tag.head(3).to_string())

# ── 국가별 분포 ───────────────────────────────────────
print(f"\n=== 국가별 데이터 분포 (Top 15) ===")
print(df_popular['region_code'].value_counts().head(15).to_string())

# ── 날짜 범위 ─────────────────────────────────────────
df_popular['collection_date'] = pd.to_datetime(
    df_popular['collection_date'], errors='coerce'
)
date_min = df_popular['collection_date'].min()
date_max = df_popular['collection_date'].max()
print(f"\n=== 데이터 기간 ===")
print(f"  시작: {date_min}")
print(f"  종료: {date_max}")

# ── 로드 결과 저장 (2번 파일이 바로 읽을 수 있도록) ──
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 💾 로드 결과 저장 중...")

# 용량 절약을 위해 description 제거 후 저장
cols_to_save = [c for c in df_popular.columns if c != 'description']
df_popular[cols_to_save].to_csv(
    f"{OUTPUT_PATH}/raw_popular.csv",
    index=False, encoding='utf-8-sig'
)
df_tag.to_csv(
    f"{OUTPUT_PATH}/raw_tag.csv",
    index=False, encoding='utf-8-sig'
)

print(f"  ✅ output/raw_popular.csv 저장 완료")
print(f"  ✅ output/raw_tag.csv 저장 완료")
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✅ 1단계 완료! → 2_preprocess.py 실행하세요")
