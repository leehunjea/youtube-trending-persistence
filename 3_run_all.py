import subprocess
import sys
import os
from datetime import datetime

PYTHON = sys.executable
BASE   = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    ("3a_rf.py",       "Random Forest"),
    ("3b_xgb.py",      "XGBoost"),
    ("3c_lgbm.py",     "LightGBM"),
    ("3d_catboost.py", "CatBoost"),
    ("3e_mlp.py",      "MLP"),
]

results       = []
total         = len(SCRIPTS)
pipeline_start = datetime.now()

print("=" * 55)
print("  모델 학습 파이프라인 시작")
print(f"  시작 시각: {pipeline_start.strftime('%H:%M:%S')}")
print("=" * 55)

# 파일 존재 여부 먼저 확인
print("\n📁 파일 존재 여부 확인:")
for script, label in SCRIPTS:
    path   = os.path.join(BASE, script)
    exists = "✅" if os.path.exists(path) else "❌ 없음"
    print(f"  {exists}  {script}")

print("\n계속 진행하려면 Enter...")
input()

for i, (script, label) in enumerate(SCRIPTS, 1):
    script_path = os.path.join(BASE, script)
    start_time  = datetime.now()

    # 파일 없으면 스킵
    if not os.path.exists(script_path):
        print(f"\n[{i}/{total}] ❌ {script} 파일 없음 → 스킵")
        results.append({'label': label, 'status': '❌ 파일없음', 'elapsed': 0})
        continue

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [{i}/{total}] {label}")
    print("-" * 55)

    # ✅ 에러 출력 캡처해서 표시
    result = subprocess.run(
        [PYTHON, script_path],
        capture_output=False,
        text=True
    )

    elapsed = (datetime.now() - start_time).seconds
    status  = "✅ 완료" if result.returncode == 0 else "❌ 실패"

    results.append({
        'label':   label,
        'status':  status,
        'elapsed': elapsed
    })
    print(f"\n  → {status} ({elapsed}초 소요)")

# ── 요약 ──────────────────────────────────────────────
total_elapsed = (datetime.now() - pipeline_start).seconds
print("\n" + "=" * 55)
print("  학습 완료 요약")
print("=" * 55)
for r in results:
    print(f"  {r['status']}  {r['label']:<20} {r['elapsed']}초")

success = sum(1 for r in results if '✅' in r['status'])
fail    = sum(1 for r in results if '❌' in r['status'])
print(f"\n  성공: {success}개 | 실패: {fail}개")
print(f"  총 소요시간: {total_elapsed}초")
