import numpy as np
import pandas as pd
import os, pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, f1_score,
    precision_score, recall_score, classification_report)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# 경로 설정
OUTPUT_PATH = "./output"
MODEL_PATH  = "./models"
os.makedirs(MODEL_PATH, exist_ok=True)

print("=" * 60)
print("  ML Fair Experiment v2 (Strict Fixed Split for Paper)")
print("=" * 60)

# ── 1. 데이터 로드 ────────────────────────────────────
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ▶ 데이터 로드 중...")

data      = np.load(f"{OUTPUT_PATH}/kr_sequences_K14.npz", allow_pickle=True)
X_seq     = data['X_seq']      # (N, 14, 7)
X_static  = data['X_static']  # (N, 8+)
y_cls     = data['y_cls']
train_idx = data['train_idx']
val_idx   = data['val_idx']
test_idx  = data['test_idx']

with open(f"{OUTPUT_PATH}/scalers.pkl", 'rb') as f:
    scalers = pickle.load(f)
SEQ_FEATURES    = scalers['seq_features']
STATIC_FEATURES = scalers['static_features']

# ── 2. Feature 엔지니어링 ─────────────────────────────
print(f"[{datetime.now().strftime('%H:%M:%S')}] ▶ ML용 Feature 추출 중...")

def extract_ml_features(X_seq, X_static, seq_features, static_features):
    N, K, F = X_seq.shape
    rows = []
    for i in range(N):
        seq = X_seq[i]; st = X_static[i]
        feat = {}
        for j, name in enumerate(static_features): feat[name] = st[j]
        for j, name in enumerate(seq_features):
            col = seq[:, j]
            feat[f'{name}_mean']  = col.mean()
            feat[f'{name}_std']   = col.std()
            feat[f'{name}_slope'] = np.polyfit(range(K), col, 1)[0] if K > 1 else 0
            feat[f'{name}_last']  = col[-1]
        
        # 핵심 도메인 피처
        rank_col = seq[:, seq_features.index('rank')]
        vph_col  = seq[:, seq_features.index('views_per_hour')]
        feat['rank_improvement'] = rank_col[0] - rank_col[-1]
        feat['vph_growth_rate']  = vph_col[-1] / (vph_col[0] + 1e-6)
        rows.append(feat)
    return pd.DataFrame(rows)

X_ml_all = extract_ml_features(X_seq, X_static, SEQ_FEATURES, STATIC_FEATURES)
X_train, y_train = X_ml_all.iloc[train_idx].values, y_cls[train_idx]
X_val,   y_val   = X_ml_all.iloc[val_idx].values,   y_cls[val_idx]
X_test,  y_test  = X_ml_all.iloc[test_idx].values,  y_cls[test_idx]

print(f"  Data Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# ── 3. 모델 정의 (Early Stopping 설정 포함) ──────────
MODELS = {
    'RandomForest': RandomForestClassifier(
        n_estimators=500, class_weight='balanced', n_jobs=-1, random_state=42
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=6,
        scale_pos_weight=scale_pos_weight, eval_metric='auc',
        early_stopping_rounds=50, random_state=42, verbosity=0
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, num_leaves=63,
        scale_pos_weight=scale_pos_weight, random_state=42,
        early_stopping_rounds=50, verbosity=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=1000, learning_rate=0.03, depth=6,
        scale_pos_weight=scale_pos_weight, early_stopping_rounds=50,
        random_seed=42, verbose=0
    ),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(256, 128), max_iter=500,
        early_stopping=True, validation_fraction=0.15, random_state=42
    ),
}

# ── 4. 학습 및 평가 ──────────────────────────────────
results = []
for name, model in MODELS.items():
    print(f"\n>>> [{name}] Training...")
    t0 = datetime.now()
    
    if name in ['XGBoost', 'LightGBM', 'CatBoost']:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    else:
        model.fit(X_train, y_train)
    
    elapsed = (datetime.now() - t0).seconds
    
    # 예측 및 평가
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
    preds = (probs >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, probs)
    f1  = f1_score(y_test, preds)
    
    res = {'model': name, 'auc': round(auc, 4), 'f1': round(f1, 4), 'elapsed': elapsed}
    results.append(res)
    print(f"    Result -> AUC: {auc:.4f}, F1: {f1:.4f} ({elapsed}s)")

    # 모델 저장
    with open(f"{MODEL_PATH}/{name}_v2_model.pkl", 'wb') as f:
        pickle.dump({'model': model, 'feature_names': X_ml_all.columns.tolist()}, f)

# ── 5. 요약 ──────────────────────────────────────────
df_res = pd.DataFrame(results).sort_values('auc', ascending=False)
df_res.to_csv(f"{OUTPUT_PATH}/ml_v2_results.csv", index=False)
print("\n" + "="*60)
print("  FINAL ML RESULTS (Test Set)")
print("="*60)
print(df_res.to_string(index=False))
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✅ 실험 완료. 이제 scripts/evaluation/6b_bootstrap_ci.py를 실행하세요.")
