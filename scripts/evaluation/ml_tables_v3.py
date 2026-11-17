# -*- coding: utf-8 -*-
"""
누수수정본 단일 파이프라인으로 ML 표 전체 재산출 (validation 임곗값 F1 통일).
- 표6(ML) + bootstrap CI, 표7(K-ablation), 표9(특징군), 표10(중요도),
  표11(임곗값 민감도), 표12(temporal, CatBoost)
"""
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_recall_curve)
from catboost import CatBoostClassifier
import xgboost as xgb, lightgbm as lgb

OUT = "./output"; RNG = np.random.RandomState(42)
STAT = ["category_id","tag_count","publish_hour_sin","publish_hour_cos",
        "publish_weekday_sin","publish_weekday_cos","hours_to_trend","relative_first_views"]
BASIC = ["rank","views_per_hour","comment_view_ratio","hours_since_publish"]
DERIV = ["rank_delta","views_acceleration","momentum"]
STATS = ["first","last","mean","std","max","min","range","slope"]
DOMAIN = ["rank_improvement","peak_momentum","momentum_trend","positive_rank_delta",
          "max_acceleration","sustained_growth","vph_growth_rate"]

def make_feats(X_seq, X_static, seqf, statf):
    rows=[]; k=X_seq.shape[1]
    for i in range(len(X_seq)):
        seq=X_seq[i]; st=X_static[i]; f={}
        for j,n in enumerate(statf): f[n]=st[j]
        for j,n in enumerate(seqf):
            c=seq[:,j]
            f[f"{n}_first"]=c[0]; f[f"{n}_last"]=c[-1]; f[f"{n}_mean"]=c.mean()
            f[f"{n}_std"]=c.std(); f[f"{n}_max"]=c.max(); f[f"{n}_min"]=c.min()
            f[f"{n}_range"]=c.max()-c.min(); f[f"{n}_slope"]=np.polyfit(range(k),c,1)[0] if k>1 else 0
        rc=seq[:,seqf.index("rank")]; vc=seq[:,seqf.index("views_per_hour")]
        mc=seq[:,seqf.index("momentum")]; rd=seq[:,seqf.index("rank_delta")]; va=seq[:,seqf.index("views_acceleration")]
        f["rank_improvement"]=rc[0]-rc[-1]; f["peak_momentum"]=mc.max(); f["momentum_trend"]=mc[-1]-mc[0]
        f["positive_rank_delta"]=(rd<0).sum(); f["max_acceleration"]=va.max()
        f["sustained_growth"]=(va>0).sum()/k; f["vph_growth_rate"]=vc[-1]/(vc[0]+1e-6)
        rows.append(f)
    return pd.DataFrame(rows)

def best_thr(y,p):
    pr,rc,th=precision_recall_curve(y,p); f1=2*pr*rc/(pr+rc+1e-12)
    return th[np.nanargmax(f1[:-1])] if len(th)>0 else 0.5

def cat(spw): return CatBoostClassifier(iterations=500,learning_rate=0.05,depth=6,
                                        scale_pos_weight=spw,random_seed=42,verbose=0)
def models(spw):
    return {"CatBoost":cat(spw),
            "XGBoost":xgb.XGBClassifier(n_estimators=500,learning_rate=0.05,max_depth=6,
                     scale_pos_weight=spw,eval_metric="auc",random_state=42,verbosity=0),
            "LightGBM":lgb.LGBMClassifier(n_estimators=500,learning_rate=0.05,
                     scale_pos_weight=spw,random_state=42,verbosity=-1),
            "RandomForest":RandomForestClassifier(n_estimators=500,class_weight="balanced",
                     n_jobs=-1,random_state=42),
            "MLP-Tabular":MLPClassifier(hidden_layer_sizes=(256,128,64),max_iter=300,
                     early_stopping=True,random_state=42)}

def metrics(model, Xtr,ytr,Xva,yva,Xte,yte):
    model.fit(Xtr,ytr)
    pva=model.predict_proba(Xva)[:,1]; pte=model.predict_proba(Xte)[:,1]
    thr=best_thr(yva,pva)
    return (roc_auc_score(yte,pte), average_precision_score(yte,pte),
            f1_score(yte,(pte>=thr).astype(int)), pte, model)

def boot_ci(y,p,n=10000):
    idx=np.arange(len(y)); aucs=np.empty(n)
    for b in range(n):
        s=RNG.randint(0,len(y),len(y))
        if y[s].sum()==0 or y[s].sum()==len(s): aucs[b]=np.nan; continue
        aucs[b]=roc_auc_score(y[s],p[s])
    return np.nanpercentile(aucs,2.5), np.nanpercentile(aucs,97.5)

# ── 데이터 로드 ──
sc=pickle.load(open(f"{OUT}/scalers.pkl","rb")); seqf,statf=sc["seq_features"],sc["static_features"]
def load(K):
    d=np.load(f"{OUT}/kr_sequences_K{K}.npz",allow_pickle=True)
    X=make_feats(d["X_seq"],d["X_static"],seqf,statf)
    return X,d["y_cls"],d["y_reg"],d["train_idx"],d["val_idx"],d["test_idx"],d["video_ids"].astype(str)

print("로딩 K=14..."); X14,y14,yr14,tr,va,te,vid14=load(14)
spw=(y14[tr]==0).sum()/max((y14[tr]==1).sum(),1)

# ════════ 표6 (ML) + bootstrap CI ════════
print("\n[표6] K=14 ML 모델 + bootstrap CI")
rows6=[]
for name,m in models(spw).items():
    auc,ap,f1,pte,_=metrics(m,X14.iloc[tr].values,y14[tr],X14.iloc[va].values,y14[va],X14.iloc[te].values,y14[te])
    lo,hi=boot_ci(y14[te],pte)
    rows6.append({"model":name,"ROC_AUC":round(auc,4),"CI_low":round(lo,4),"CI_high":round(hi,4),
                  "F1_valthr":round(f1,4),"AP":round(ap,4)})
    print(f"  {name:13s} AUC={auc:.4f} [{lo:.4f},{hi:.4f}] AP={ap:.4f} F1={f1:.4f}")
pd.DataFrame(rows6).to_csv(f"{OUT}/tab6_ml_v3.csv",index=False,encoding="utf-8-sig")

# ════════ 표7 (ML K-ablation, ROC-AUC) ════════
print("\n[표7] ML K-ablation")
rows7=[]
for K in [5,7,10,14]:
    X,y,_,a,b,c,_=load(K); sp=(y[a]==0).sum()/max((y[a]==1).sum(),1)
    r={"K":K}
    for name,m in models(sp).items():
        if name=="MLP-Tabular": continue
        auc,_,_,_,_=metrics(m,X.iloc[a].values,y[a],X.iloc[b].values,y[b],X.iloc[c].values,y[c])
        r[name]=round(auc,4)
    rows7.append(r); print(f"  K={K}: {r}")
pd.DataFrame(rows7).to_csv(f"{OUT}/tab7_ml_kabl_v3.csv",index=False,encoding="utf-8-sig")

# ════════ 표9 (특징군 제거, CatBoost K=14) ════════
print("\n[표9] 특징군 제거 (CatBoost K=14)")
def cols_summary(seqs): return [f"{n}_{s}" for n in seqs for s in STATS]
conds={
 "전체 특징(71)": list(X14.columns),
 "정적+시계열요약(40)": STAT+cols_summary(BASIC),
 "변화량·성장(31)": cols_summary(DERIV)+DOMAIN,
 "마지막 순위(1)": ["rank_last"],
 "정적+첫스냅샷(12)": STAT+[f"{n}_first" for n in BASIC],
 "정적 특징(8)": STAT,
}
rows9=[]
for cname,cols in conds.items():
    Xc=X14[cols]
    auc,ap,f1,_,_=metrics(cat(spw),Xc.iloc[tr].values,y14[tr],Xc.iloc[va].values,y14[va],Xc.iloc[te].values,y14[te])
    rows9.append({"조건":cname,"특징수":len(cols),"ROC_AUC":round(auc,4),"AP":round(ap,4),"F1_valthr":round(f1,4)})
    print(f"  {cname:18s} n={len(cols):2d} AUC={auc:.4f} AP={ap:.4f} F1={f1:.4f}")
pd.DataFrame(rows9).to_csv(f"{OUT}/tab9_featabl_v3.csv",index=False,encoding="utf-8-sig")

# ════════ 표10 (CatBoost-Full 중요도) ════════
print("\n[표10] CatBoost-Full 특징 중요도 top-10")
mfull=cat(spw); mfull.fit(X14.iloc[tr].values,y14[tr])
imp=pd.DataFrame({"feature":X14.columns,"importance":mfull.get_feature_importance()}).sort_values("importance",ascending=False).head(10)
imp["importance"]=imp["importance"].round(4)
print(imp.to_string(index=False)); imp.to_csv(f"{OUT}/tab10_importance_v3.csv",index=False,encoding="utf-8-sig")

# ════════ 표11 (임곗값 민감도, CatBoost K=14) ════════
print("\n[표11] 라벨 임곗값 민감도")
full=pd.read_csv(f"{OUT}/kr_video_labeled.csv",encoding="utf-8-sig",usecols=["trending_snapshots"])
ts_full=full["trending_snapshots"].values
rows11=[]
for pct,cut in [("p65",55),("p70",57),("p75",60),("p80",63),("p85",66)]:
    yb=(yr14>=cut).astype(int)
    sp=(yb[tr]==0).sum()/max((yb[tr]==1).sum(),1)
    auc,ap,f1,_,_=metrics(cat(sp),X14.iloc[tr].values,yb[tr],X14.iloc[va].values,yb[va],X14.iloc[te].values,yb[te])
    posrate=float((ts_full>=cut).mean())
    rows11.append({"기준":pct,"cutoff":cut,"양성률":round(posrate,4),"ROC_AUC":round(auc,4),"AP":round(ap,4),"F1_valthr":round(f1,4)})
    print(f"  {pct}(cut={cut}) 양성률={posrate:.4f} AUC={auc:.4f} AP={ap:.4f} F1={f1:.4f}")
pd.DataFrame(rows11).to_csv(f"{OUT}/tab11_thresh_v3.csv",index=False,encoding="utf-8-sig")

print("\n저장 완료: tab6/7/9/10/11_*_v3.csv")
print("표12(temporal)은 scripts/evaluation/ml_tables_v3_temporal.py에서 별도 처리")
