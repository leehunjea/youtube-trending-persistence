import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                             f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

OUTPUT_PATH = "./output"
MODEL_PATH  = "./models"
os.makedirs(MODEL_PATH,  exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── 공통 설정 ─────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5

# ✅ 첫 스냅샷 Feature만 사용 (누수 없음)
# 근거: 트렌딩 첫 진입 시점에 알 수 있는 정보만 사용
#       → 미래 바이럴 강도를 예측하는 진짜 예측 모델
FEATURES = [
    'category_id',
    'tag_count',
    # sin/cos 인코딩 (기존 publish_hour, publish_weekday 대체)
    'publish_hour_sin',      # ← 새로 추가
    'publish_hour_cos',      # ← 새로 추가
    'publish_weekday_sin',   # ← 새로 추가
    'publish_weekday_cos',   # ← 새로 추가
    # 기존 Feature
    'first_rank',
    'first_views',
    'first_comments',
    'first_views_per_hour',
    'first_comment_view_ratio',
    'first_hours_since_publish',
    'hours_to_trend',
    'relative_first_views',
]

# ✅ 절대 기준 레이블
# 근거: Feature와 독립적인 최종 성과 기준
#       views_per_hour>=100,000 AND rank<=20 AND snapshots>=4
LABEL = 'is_viral'

# ── 데이터 로드 ───────────────────────────────────────
def load_data(label=LABEL):
    df = pd.read_csv(f"{OUTPUT_PATH}/kr_video_labeled.csv", encoding='utf-8-sig')

    # Feature 존재 여부 확인
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"누락된 Feature: {missing}\nscripts/preprocessing/2_preprocess.py 먼저 실행하세요.")

    X = df[FEATURES].fillna(0)
    y = df[label]

    neg, pos         = (y == 0).sum(), (y == 1).sum()
    scale_pos_weight = neg / pos

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y
    )

    sw_train = compute_sample_weight('balanced', y_train)
    sw_test  = compute_sample_weight('balanced', y_test)

    print(f"  전체 데이터      : {len(df):,}개")
    print(f"  레이블           : {label}")
    print(f"  바이럴(1)        : {pos:,}개 ({y.mean()*100:.1f}%)")
    print(f"  비바이럴(0)      : {neg:,}개 ({(1-y.mean())*100:.1f}%)")
    print(f"  scale_pos_weight : {scale_pos_weight:.2f}")
    print(f"  Train            : {len(X_train):,}개")
    print(f"  Test             : {len(X_test):,}개")
    print(f"  Features ({len(FEATURES)}개)  : {FEATURES}")

    return X_train, X_test, y_train, y_test, sw_train, sw_test, scale_pos_weight, df

# ── 공통 평가 ─────────────────────────────────────────
def evaluate(model, X_test, y_test, model_name, X_all=None, y_all=None):
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'model':           model_name,
        'roc_auc':         round(roc_auc_score(y_test, y_pred_prob), 4),
        'f1':              round(f1_score(y_test, y_pred),            4),
        'precision':       round(precision_score(y_test, y_pred),     4),
        'recall':          round(recall_score(y_test, y_pred),        4),
        'cv_roc_auc_mean': None,
        'cv_roc_auc_std':  None,
    }

    # 5-Fold Stratified CV
    if X_all is not None and y_all is not None:
        cv = StratifiedKFold(
            n_splits     = CV_FOLDS,
            shuffle      = True,
            random_state = RANDOM_STATE
        )
        cv_scores = cross_val_score(
            model, X_all, y_all,
            cv      = cv,
            scoring = 'roc_auc',
            n_jobs  = -1
        )
        metrics['cv_roc_auc_mean'] = round(cv_scores.mean(), 4)
        metrics['cv_roc_auc_std']  = round(cv_scores.std(),  4)

    print(f"\n  {'='*45}")
    print(f"  📊 {model_name} 성능")
    print(f"  {'='*45}")
    print(f"  ROC-AUC          : {metrics['roc_auc']}")
    print(f"  F1               : {metrics['f1']}")
    print(f"  Precision        : {metrics['precision']}")
    print(f"  Recall           : {metrics['recall']}")
    if metrics['cv_roc_auc_mean']:
        print(f"  CV ROC-AUC       : {metrics['cv_roc_auc_mean']} ± {metrics['cv_roc_auc_std']}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['비바이럴', '바이럴'])}")

    return metrics, y_pred, y_pred_prob

# ── 모델 저장 ─────────────────────────────────────────
def save_model(model, model_name, metrics, scaler=None):
    meta = {
        'model':      model,
        'model_name': model_name,
        'features':   FEATURES,
        'label':      LABEL,
        'metrics':    metrics,
        'scaler':     scaler,
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M')
    }
    with open(f"{MODEL_PATH}/{model_name}_model.pkl", 'wb') as f:
        pickle.dump(meta, f)

    pd.DataFrame([metrics]).to_csv(
        f"{OUTPUT_PATH}/{model_name}_result.csv",
        index=False, encoding='utf-8-sig'
    )
    print(f"  ✅ models/{model_name}_model.pkl 저장 완료")

# ════════════════════════════════════════════════════════
#  모델 구조 정의
# ════════════════════════════════════════════════════════

def get_random_forest():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators  = 300,
        max_depth     = 15,
        min_samples_leaf = 5,
        class_weight  = 'balanced',
        random_state  = RANDOM_STATE,
        n_jobs        = -1
    )

def get_xgboost(scale_pos_weight=1):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators      = 300,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale_pos_weight,
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        eval_metric       = 'logloss',
        verbosity         = 0
    )

def get_lightgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators  = 300,
        max_depth     = 6,
        learning_rate = 0.05,
        num_leaves    = 63,
        subsample     = 0.8,
        class_weight  = 'balanced',
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
        verbose       = -1
    )

def get_catboost(cat_features=None):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations        = 300,
        depth             = 6,
        learning_rate     = 0.05,
        cat_features      = cat_features or [],
        auto_class_weights= 'Balanced',
        random_seed       = RANDOM_STATE,
        verbose           = 0
    )

def get_mlp():
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(
        hidden_layer_sizes  = (256, 128, 64, 32),
        activation          = 'relu',
        learning_rate       = 'adaptive',
        learning_rate_init  = 0.001,
        alpha               = 0.001,
        max_iter            = 500,
        random_state        = RANDOM_STATE,
        early_stopping      = True,
        validation_fraction = 0.1,
        verbose             = False
    )
