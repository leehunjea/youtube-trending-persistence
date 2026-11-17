# -*- coding: utf-8 -*-
"""강건성 검증: CatBoost vs 타모델 paired bootstrap (동일 test, video_id 정렬)."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
OUT="./output"; RNG=np.random.RandomState(42)

def load(path):
    d=pd.read_csv(path,encoding="utf-8-sig"); d["video_id"]=d["video_id"].astype(str)
    return d[["video_id","y_true","y_pred"]]

base=load(f"{OUT}/predsv3_CatBoost_K14.csv")
comp={"XGBoost":f"{OUT}/predsv3_XGBoost_K14.csv",
      "LightGBM":f"{OUT}/predsv3_LightGBM_K14.csv",
      "LSTM":f"{OUT}/predsv3_LSTM_K14_test.csv",
      "Transformer":f"{OUT}/predsv3_Transformer_K14_test.csv"}

def paired(A,B,n=10000):
    m=A.merge(B,on="video_id",suffixes=("_a","_b"))
    assert (m["y_true_a"]==m["y_true_b"]).all(), "라벨 불일치"
    y=m["y_true_a"].values; pa=m["y_pred_a"].values; pb=m["y_pred_b"].values
    dA=roc_auc_score(y,pa)-roc_auc_score(y,pb)
    d=np.empty(n)
    for i in range(n):
        s=RNG.randint(0,len(y),len(y))
        if y[s].sum() in (0,len(s)): d[i]=np.nan; continue
        d[i]=roc_auc_score(y[s],pa[s])-roc_auc_score(y[s],pb[s])
    d=d[~np.isnan(d)]
    lo,hi=np.percentile(d,2.5),np.percentile(d,97.5)
    p=2*min((d<=0).mean(),(d>=0).mean())  # two-sided
    return dA,lo,hi,p,len(y)

rows=[]
for name,path in comp.items():
    B=load(path); dA,lo,hi,p,n=paired(base,B)
    sig = p<0.05
    rows.append({"비교":f"CatBoost vs {name}","ΔAUC":round(dA,4),
                 "CI_low":round(lo,4),"CI_high":round(hi,4),"p_value":round(p,4),
                 "유의(0.05)":sig,"n":n})
    print(f"  CatBoost vs {name:12s} ΔAUC={dA:+.4f} CI[{lo:+.4f},{hi:+.4f}] p={p:.4f} {'유의' if sig else '비유의'} (n={n})")
pd.DataFrame(rows).to_csv(f"{OUT}/paired_boot_v3.csv",index=False,encoding="utf-8-sig")
print("\n저장: paired_boot_v3.csv")
