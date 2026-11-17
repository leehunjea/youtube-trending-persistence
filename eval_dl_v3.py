# -*- coding: utf-8 -*-
"""DL 지표 재산출: 저장된 val/test 예측확률 → AUC/AP/F1(val임곗값) + bootstrap CI.
표6 DL행, 표8(DL K-ablation)."""
import numpy as np, pandas as pd, os
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve

OUT="./output"; RNG=np.random.RandomState(42)
MODELS=["LSTM","GRU","Transformer","TFT"]; KS=[5,7,10,14]

def best_thr(y,p):
    pr,rc,th=precision_recall_curve(y,p); f1=2*pr*rc/(pr+rc+1e-12)
    return th[np.nanargmax(f1[:-1])] if len(th)>0 else 0.5
def boot_ci(y,p,n=10000):
    a=np.empty(n)
    for b in range(n):
        s=RNG.randint(0,len(y),len(y))
        a[b]=np.nan if y[s].sum() in (0,len(s)) else roc_auc_score(y[s],p[s])
    return np.nanpercentile(a,2.5),np.nanpercentile(a,97.5)

def load(model,K,split):
    f=f"{OUT}/predsv3_{model}_K{K}_{split}.csv"
    if not os.path.exists(f): return None
    d=pd.read_csv(f,encoding="utf-8-sig"); return d["y_true"].values,d["y_pred"].values

# 표8: DL K-ablation (ROC-AUC)
rows8=[]
for m in MODELS:
    r={"model":m}
    for K in KS:
        te=load(m,K,"test")
        r[f"K={K}"]=round(roc_auc_score(*te),4) if te else None
    rows8.append(r)
tab8=pd.DataFrame(rows8); print("[표8] DL K-ablation ROC-AUC"); print(tab8.to_string(index=False))
tab8.to_csv(f"{OUT}/tab8_dl_kabl_v3.csv",index=False,encoding="utf-8-sig")

# 표6 DL행 (K=14): AUC + CI + AP + F1(val임곗값)
print("\n[표6-DL] K=14")
rows6dl=[]
for m in MODELS:
    va=load(m,14,"val"); te=load(m,14,"test")
    if te is None: print(f"  {m}: (아직 없음)"); continue
    yv,pv=va; yt,pt=te; thr=best_thr(yv,pv)
    auc=roc_auc_score(yt,pt); ap=average_precision_score(yt,pt); f1=f1_score(yt,(pt>=thr).astype(int))
    lo,hi=boot_ci(yt,pt)
    rows6dl.append({"model":m,"ROC_AUC":round(auc,4),"CI_low":round(lo,4),"CI_high":round(hi,4),
                    "F1_valthr":round(f1,4),"AP":round(ap,4)})
    print(f"  {m:12s} AUC={auc:.4f} [{lo:.4f},{hi:.4f}] AP={ap:.4f} F1={f1:.4f}")
pd.DataFrame(rows6dl).to_csv(f"{OUT}/tab6_dl_v3.csv",index=False,encoding="utf-8-sig")
print("\n저장: tab8_dl_kabl_v3.csv, tab6_dl_v3.csv")
