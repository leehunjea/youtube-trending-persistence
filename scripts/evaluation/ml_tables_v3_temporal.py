# -*- coding: utf-8 -*-
"""표12: CatBoost stratified vs temporal split (validation 임곗값 F1). CatBoost는
트리모델이라 affine 스케일링에 불변 → 사전 스케일된 npz 특징 사용 무방."""
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from catboost import CatBoostClassifier
from ml_tables_v3 import make_feats, best_thr

OUT="./output"
sc=pickle.load(open(f"{OUT}/scalers.pkl","rb")); seqf,statf=sc["seq_features"],sc["static_features"]
d=np.load(f"{OUT}/kr_sequences_K14.npz",allow_pickle=True)
X=make_feats(d["X_seq"],d["X_static"],seqf,statf)
y=d["y_cls"]; vids=d["video_ids"].astype(str)
tr,va,te=d["train_idx"],d["val_idx"],d["test_idx"]

def cat(spw): return CatBoostClassifier(iterations=500,learning_rate=0.05,depth=6,
                                        scale_pos_weight=spw,random_seed=42,verbose=0)
def run(Xtr,ytr,Xva,yva,Xte,yte):
    sp=(ytr==0).sum()/max((ytr==1).sum(),1); m=cat(sp); m.fit(Xtr,ytr)
    pva=m.predict_proba(Xva)[:,1]; pte=m.predict_proba(Xte)[:,1]; thr=best_thr(yva,pva)
    return (roc_auc_score(yte,pte),average_precision_score(yte,pte),
            f1_score(yte,(pte>=thr).astype(int)),ytr.mean(),yte.mean())

# stratified
s=run(X.iloc[tr].values,y[tr],X.iloc[va].values,y[va],X.iloc[te].values,y[te])
print(f"Stratified : trainPos={s[3]:.3f} testPos={s[4]:.3f} AUC={s[0]:.4f} AP={s[1]:.4f} F1={s[2]:.4f}")

# temporal: first_trending_date 기준 정렬
print("kr_snapshot 로드(최초 진입시각)...")
snap=pd.read_csv(f"{OUT}/kr_snapshot.csv",encoding="utf-8-sig",usecols=["video_id","collection_date"])
snap["video_id"]=snap["video_id"].astype(str); snap["collection_date"]=pd.to_datetime(snap["collection_date"],errors="coerce")
fd=snap.groupby("video_id")["collection_date"].min()
order_df=pd.DataFrame({"video_id":vids,"idx":np.arange(len(vids))})
order_df["fd"]=order_df["video_id"].map(fd)
order_df=order_df.sort_values(["fd","video_id"]).reset_index(drop=True)
n=len(order_df); e1,e2=int(n*0.70),int(n*0.85)
tr2=order_df["idx"].values[:e1]; va2=order_df["idx"].values[e1:e2]; te2=order_df["idx"].values[e2:]
t=run(X.iloc[tr2].values,y[tr2],X.iloc[va2].values,y[va2],X.iloc[te2].values,y[te2])
print(f"Temporal   : trainPos={t[3]:.3f} testPos={t[4]:.3f} AUC={t[0]:.4f} AP={t[1]:.4f} F1={t[2]:.4f}")
print(f"  train {order_df['fd'].iloc[0].date()}~{order_df['fd'].iloc[e1-1].date()} n={len(tr2)}")
print(f"  val   {order_df['fd'].iloc[e1].date()}~{order_df['fd'].iloc[e2-1].date()} n={len(va2)}")
print(f"  test  {order_df['fd'].iloc[e2].date()}~{order_df['fd'].iloc[-1].date()} n={len(te2)}")

pd.DataFrame([
 {"분할":"Stratified random split","Train양성률":round(s[3],3),"Test양성률":round(s[4],3),
  "ROC_AUC":round(s[0],4),"AP":round(s[1],4),"F1_valthr":round(s[2],4)},
 {"분할":"Temporal split","Train양성률":round(t[3],3),"Test양성률":round(t[4],3),
  "ROC_AUC":round(t[0],4),"AP":round(t[1],4),"F1_valthr":round(t[2],4)},
]).to_csv(f"{OUT}/tab12_temporal_v3.csv",index=False,encoding="utf-8-sig")
print("\n저장: tab12_temporal_v3.csv")
