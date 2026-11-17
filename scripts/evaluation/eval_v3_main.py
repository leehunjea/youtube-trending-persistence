# -*- coding: utf-8 -*-
"""
논문 재현 + F1 임곗값 방식 교정 (리뷰어3 지적4·9).
- 특징: 8_temporal_validation.py의 make_ml_features (71개: 8요약통계×7 + 정적8 + 도메인7)
- 모델: 논문 표4 설정 (CatBoost 500/0.05/depth6 등)
- F1: (a) 0.5 고정, (b) train 최적 임곗값, (c) validation 최적 임곗값  ← 교정안
"""
import numpy as np, pandas as pd, pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_recall_curve)

OUT = "./output"; K = 14

def make_ml_features(X_seq, X_static, seq_features, static_features):
    rows = []; k = X_seq.shape[1]
    for i in range(len(X_seq)):
        seq = X_seq[i]; st = X_static[i]; feat = {}
        for j, name in enumerate(static_features): feat[name] = st[j]
        for j, name in enumerate(seq_features):
            col = seq[:, j]
            feat[f"{name}_first"]=col[0]; feat[f"{name}_last"]=col[-1]
            feat[f"{name}_mean"]=col.mean(); feat[f"{name}_std"]=col.std()
            feat[f"{name}_max"]=col.max(); feat[f"{name}_min"]=col.min()
            feat[f"{name}_range"]=col.max()-col.min()
            feat[f"{name}_slope"]=np.polyfit(range(k),col,1)[0] if k>1 else 0
        rank_col=seq[:,seq_features.index("rank")]; vph_col=seq[:,seq_features.index("views_per_hour")]
        mom_col=seq[:,seq_features.index("momentum")]; rd_col=seq[:,seq_features.index("rank_delta")]
        va_col=seq[:,seq_features.index("views_acceleration")]
        feat["rank_improvement"]=rank_col[0]-rank_col[-1]
        feat["peak_momentum"]=mom_col.max(); feat["momentum_trend"]=mom_col[-1]-mom_col[0]
        feat["positive_rank_delta"]=(rd_col<0).sum(); feat["max_acceleration"]=va_col.max()
        feat["sustained_growth"]=(va_col>0).sum()/k; feat["vph_growth_rate"]=vph_col[-1]/(vph_col[0]+1e-6)
        rows.append(feat)
    return pd.DataFrame(rows)

def best_f1_threshold(y_ref, p_ref):
    prec, rec, thr = precision_recall_curve(y_ref, p_ref)
    f1 = 2*prec*rec/(prec+rec+1e-12)
    # thr has len-1 vs prec/rec; align
    best = np.nanargmax(f1[:-1]) if len(thr)>0 else 0
    return thr[best] if len(thr)>0 else 0.5

def models_paper(spw):
    m = {
        "CatBoost": None, "XGBoost": None, "LightGBM": None,
        "RandomForest": RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                                n_jobs=-1, random_state=42),
        "MLP-Tabular": MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=300,
                                     early_stopping=True, random_state=42),
    }
    from catboost import CatBoostClassifier
    m["CatBoost"] = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                                       scale_pos_weight=spw, random_seed=42, verbose=0)
    import xgboost as xgb
    m["XGBoost"] = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                                     scale_pos_weight=spw, eval_metric="auc",
                                     random_state=42, verbosity=0)
    import lightgbm as lgb
    m["LightGBM"] = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                       scale_pos_weight=spw, random_state=42, verbosity=-1)
    return m

def main():
    data = np.load(f"{OUT}/kr_sequences_K{K}.npz", allow_pickle=True)
    X_seq, X_static, y = data["X_seq"], data["X_static"], data["y_cls"]
    tr, va, te = data["train_idx"], data["val_idx"], data["test_idx"]
    vids = data["video_ids"].astype(str)
    sc = pickle.load(open(f"{OUT}/scalers.pkl","rb"))
    seqf, statf = sc["seq_features"], sc["static_features"]

    X = make_ml_features(X_seq, X_static, seqf, statf)
    print(f"features: {X.shape[1]}  | train {len(tr)} val {len(va)} test {len(te)}")
    Xtr,ytr = X.iloc[tr].values, y[tr]
    Xva,yva = X.iloc[va].values, y[va]
    Xte,yte = X.iloc[te].values, y[te]
    spw = (ytr==0).sum()/max((ytr==1).sum(),1)
    print(f"test positive rate = {yte.mean():.4f}")

    rows=[]; preds={}
    for name, model in models_paper(spw).items():
        model.fit(Xtr, ytr)
        ptr = model.predict_proba(Xtr)[:,1]
        pva = model.predict_proba(Xva)[:,1]
        pte = model.predict_proba(Xte)[:,1]
        auc = roc_auc_score(yte, pte); ap = average_precision_score(yte, pte)
        thr_tr = best_f1_threshold(ytr, ptr); thr_va = best_f1_threshold(yva, pva)
        f1_05 = f1_score(yte, (pte>=0.5).astype(int))
        f1_tr = f1_score(yte, (pte>=thr_tr).astype(int))
        f1_va = f1_score(yte, (pte>=thr_va).astype(int))
        rows.append({"model":name,"ROC_AUC":round(auc,4),"AP":round(ap,4),
                     "F1@0.5":round(f1_05,4),"F1@train_thr":round(f1_tr,4),
                     "F1@val_thr":round(f1_va,4),"thr_tr":round(float(thr_tr),3),
                     "thr_va":round(float(thr_va),3)})
        preds[name]=pd.DataFrame({"video_id":vids[te],"y_true":yte,"y_pred":pte})
    res = pd.DataFrame(rows).sort_values("ROC_AUC",ascending=False)
    print("\n"+"="*90)
    print(res.to_string(index=False))
    print("="*90)
    print("\n[논문 표6 대조] CatBoost ROC-AUC 0.8444 / AP 0.6483 / F1 0.6321(표6),0.6281(표9)")
    res.to_csv(f"{OUT}/eval_v3_main_K14.csv", index=False, encoding="utf-8-sig")
    for n,d in preds.items():
        d.to_csv(f"{OUT}/predsv3_{n}_K14.csv", index=False, encoding="utf-8-sig")
    print(f"\nsaved: {OUT}/eval_v3_main_K14.csv  +  predsv3_*_K14.csv")

if __name__=="__main__":
    main()
