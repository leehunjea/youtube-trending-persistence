import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import os, pickle
from datetime import datetime
import pandas as pd

OUTPUT_PATH = "./output"
MODEL_PATH  = "./models"
os.makedirs(MODEL_PATH, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {DEVICE}")

# ── 재현성: 시드 고정 (리뷰어 재현성 요구) ────────────
SEED = 42
import random as _random
_random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── 공통 하이퍼파라미터 ──────────────────────────────
BATCH_SIZE = 64
EPOCHS     = 60
PATIENCE   = 10
DROPOUT    = 0.3

# ── Dataset ──────────────────────────────────────────
class ViralDataset(Dataset):
    def __init__(self, X_seq, X_static, mask, y, idx):
        self.X_seq = X_seq; self.X_static = X_static
        self.mask  = mask;  self.y = y;  self.idx = idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        n = self.idx[i]
        return self.X_seq[n], self.X_static[n], self.mask[n], self.y[n]

# ── 모델 정의 ─────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, seq_dim, static_dim, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.lstm    = nn.LSTM(seq_dim, hidden, layers, batch_first=True,
                               dropout=dropout if layers>1 else 0, bidirectional=True)
        self.attn_fc = nn.Linear(hidden*2, 1)
        self.static_fc = nn.Sequential(
            nn.Linear(static_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout))
        self.clf = nn.Sequential(
            nn.Linear(hidden*2+64, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, xs, xst, m):
        o,_ = self.lstm(xs)
        a   = self.attn_fc(o).squeeze(-1).masked_fill(m==0, -1e9)
        w   = torch.softmax(a, -1)
        p   = (o * w.unsqueeze(-1)).sum(1)
        return self.clf(torch.cat([p, self.static_fc(xst)], -1)).squeeze(1)

class GRUClassifier(nn.Module):
    def __init__(self, seq_dim, static_dim, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.gru     = nn.GRU(seq_dim, hidden, layers, batch_first=True,
                              dropout=dropout if layers>1 else 0, bidirectional=True)
        self.attn_fc = nn.Linear(hidden*2, 1)
        self.static_fc = nn.Sequential(
            nn.Linear(static_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout))
        self.clf = nn.Sequential(
            nn.Linear(hidden*2+64, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, xs, xst, m):
        o,_ = self.gru(xs)
        a   = self.attn_fc(o).squeeze(-1).masked_fill(m==0, -1e9)
        w   = torch.softmax(a, -1)
        p   = (o * w.unsqueeze(-1)).sum(1)
        return self.clf(torch.cat([p, self.static_fc(xst)], -1)).squeeze(1)

class TransformerClassifier(nn.Module):
    def __init__(self, seq_dim, static_dim, d_model=64, nhead=4, layers=2, dropout=0.2):
        super().__init__()
        self.cls_tok  = nn.Parameter(torch.zeros(1,1,d_model))
        self.inp_proj = nn.Linear(seq_dim, d_model)
        # Positional Encoding
        pe = torch.zeros(200, d_model)
        pos= torch.arange(200).unsqueeze(1).float()
        div= torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
        enc = nn.TransformerEncoderLayer(d_model,nhead,d_model*4,dropout,batch_first=True,norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, layers)
        self.static_fc   = nn.Sequential(
            nn.Linear(static_dim,64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout))
        self.clf = nn.Sequential(
            nn.Linear(d_model+64,256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, xs, xst, m):
        B  = xs.shape[0]
        x  = self.inp_proj(xs) + self.pe[:,:xs.shape[1],:]
        cls= self.cls_tok.expand(B,-1,-1)
        x  = torch.cat([cls,x],1)
        km = (torch.cat([torch.ones(B,1,device=m.device),m],-1)==0)
        x  = self.transformer(x, src_key_padding_mask=km)
        return self.clf(torch.cat([x[:,0,:], self.static_fc(xst)],-1)).squeeze(1)

class GRN(nn.Module):
    def __init__(self, i, h, o, d=0.1):
        super().__init__()
        self.fc1=nn.Linear(i,h); self.fc2=nn.Linear(h,o)
        self.gate=nn.Linear(h,o); self.ln=nn.LayerNorm(o)
        self.drop=nn.Dropout(d)
        self.skip=nn.Linear(i,o) if i!=o else nn.Identity()
    def forward(self,x):
        h=torch.relu(self.fc1(x)); h=self.drop(h)
        return self.ln(self.fc2(h)*torch.sigmoid(self.gate(h))+self.skip(x))

class TFTClassifier(nn.Module):
    def __init__(self, seq_dim, static_dim, hidden=128, nhead=4, dropout=0.2):
        super().__init__()
        self.static_grn = GRN(static_dim, hidden, hidden, dropout)
        self.ctx_h = nn.Linear(hidden, hidden)
        self.ctx_c = nn.Linear(hidden, hidden)
        self.ctx_e = nn.Linear(hidden, hidden)
        self.seq_proj  = nn.Linear(seq_dim, hidden)
        self.lstm      = nn.LSTM(hidden, hidden, batch_first=True)
        self.lstm_gate = nn.Sequential(nn.Linear(hidden,hidden), nn.Sigmoid())
        self.lstm_ln   = nn.LayerNorm(hidden)
        self.attn      = nn.MultiheadAttention(hidden, nhead, dropout=dropout, batch_first=True)
        self.attn_gate = nn.Sequential(nn.Linear(hidden,hidden), nn.Sigmoid())
        self.attn_ln   = nn.LayerNorm(hidden)
        self.pos_grn   = GRN(hidden, hidden*2, hidden, dropout)
        self.pos_gate  = nn.Sequential(nn.Linear(hidden,hidden), nn.Sigmoid())
        self.pos_ln    = nn.LayerNorm(hidden)
        self.clf = nn.Sequential(
            nn.Linear(hidden*2,256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, xs, xst, m):
        s   = self.static_grn(xst)
        c_h = self.ctx_h(s).unsqueeze(0)
        c_c = self.ctx_c(s).unsqueeze(0)
        c_e = self.ctx_e(s)
        xe  = self.seq_proj(xs)
        lo,_= self.lstm(xe,(c_h,c_c))
        lg  = self.lstm_gate(lo)*lo
        lo  = self.lstm_ln(lg+xe)
        ao,_= self.attn(lo,lo,lo,key_padding_mask=(m==0))
        ag  = self.attn_gate(ao)*ao
        ao  = self.attn_ln(ag+lo)
        B,T,H=ao.shape
        po  = self.pos_grn(ao.reshape(B*T,H)).reshape(B,T,H)
        pg  = self.pos_gate(po)*po
        po  = self.pos_ln(pg+ao)
        me  = m.unsqueeze(-1)
        pool= (po*me).sum(1)/me.sum(1).clamp(min=1)
        return self.clf(torch.cat([pool,c_e],-1)).squeeze(1)

MODELS = {
    'LSTM':        LSTMClassifier,
    'GRU':         GRUClassifier,
    'Transformer': TransformerClassifier,
    'TFT':         TFTClassifier,
}

def run_one(model_name, K_EARLY, npz_path, results_list):
    print(f"\n{'='*55}")
    print(f"  [{model_name}] K_EARLY={K_EARLY}")
    print(f"{'='*55}")

    data      = np.load(npz_path, allow_pickle=True)
    X_seq     = torch.tensor(data['X_seq'],    dtype=torch.float32)
    X_static  = torch.tensor(data['X_static'], dtype=torch.float32)
    mask      = torch.tensor(data['mask'],     dtype=torch.float32)
    y_cls     = torch.tensor(data['y_cls'],    dtype=torch.float32)
    train_idx = data['train_idx']
    val_idx   = data['val_idx']
    test_idx  = data['test_idx']

    SEQ_DIM    = X_seq.shape[2]
    STATIC_DIM = X_static.shape[1]

    train_loader = DataLoader(ViralDataset(X_seq,X_static,mask,y_cls,train_idx),
                              BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(ViralDataset(X_seq,X_static,mask,y_cls,val_idx),
                              BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(ViralDataset(X_seq,X_static,mask,y_cls,test_idx),
                              BATCH_SIZE, shuffle=False)

    pos_weight = torch.tensor([(y_cls==0).sum()/(y_cls==1).sum()]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model     = MODELS[model_name](SEQ_DIM, STATIC_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc = 0; patience_c = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        for xs,xst,mb,y in train_loader:
            xs,xst,mb,y = xs.to(DEVICE),xst.to(DEVICE),mb.to(DEVICE),y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xs,xst,mb), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval(); vp,vl=[],[]
        with torch.no_grad():
            for xs,xst,mb,y in val_loader:
                xs,xst,mb,y = xs.to(DEVICE),xst.to(DEVICE),mb.to(DEVICE),y.to(DEVICE)
                vp += torch.sigmoid(model(xs,xst,mb)).cpu().tolist()
                vl += y.cpu().tolist()
        val_auc = roc_auc_score(np.array(vl), np.array(vp))
        print(f"  Epoch {epoch:3d}/{EPOCHS} | Val AUC: {val_auc:.4f}", end='\r')

        if val_auc > best_auc:
            best_auc = val_auc; patience_c = 0
            torch.save(model.state_dict(),
                f"{MODEL_PATH}/{model_name}_K{K_EARLY}_best.pt")
        else:
            patience_c += 1
            if patience_c >= PATIENCE:
                print(f"\n  ⏹ Early Stopping (epoch={epoch})")
                break

    # 최종 평가
    model.load_state_dict(
        torch.load(f"{MODEL_PATH}/{model_name}_K{K_EARLY}_best.pt"))
    model.eval(); tp,tl=[],[]
    with torch.no_grad():
        for xs,xst,mb,y in test_loader:
            xs,xst,mb,y = xs.to(DEVICE),xst.to(DEVICE),mb.to(DEVICE),y.to(DEVICE)
            tp += torch.sigmoid(model(xs,xst,mb)).cpu().tolist()
            tl += y.cpu().tolist()

    # val 예측확률도 저장 (validation 임곗값 F1 산출용) — 순서 보존 위해 shuffle 없음
    vp2, vl2 = [], []
    with torch.no_grad():
        for xs,xst,mb,y in val_loader:
            xs,xst,mb,y = xs.to(DEVICE),xst.to(DEVICE),mb.to(DEVICE),y.to(DEVICE)
            vp2 += torch.sigmoid(model(xs,xst,mb)).cpu().tolist()
            vl2 += y.cpu().tolist()
    vids_all = data['video_ids'].astype(str)
    pd.DataFrame({'video_id':vids_all[val_idx],'y_true':np.array(vl2),
                  'y_pred':np.array(vp2)}).to_csv(
        f"{OUTPUT_PATH}/predsv3_{model_name}_K{K_EARLY}_val.csv",
        index=False, encoding='utf-8-sig')
    pd.DataFrame({'video_id':vids_all[test_idx],'y_true':np.array(tl),
                  'y_pred':np.array(tp)}).to_csv(
        f"{OUTPUT_PATH}/predsv3_{model_name}_K{K_EARLY}_test.csv",
        index=False, encoding='utf-8-sig')

    tp, tl = np.array(tp), np.array(tl)
    tb = (tp >= 0.5).astype(int)

    result = {
        'model':        model_name,
        'K_EARLY':      K_EARLY,
        'roc_auc':      round(roc_auc_score(tl, tp),   4),
        'f1':           round(f1_score(tl, tb),          4),
        'precision':    round(precision_score(tl, tb),   4),
        'recall':       round(recall_score(tl, tb),      4),
        'best_val_auc': round(best_auc, 4),
        'cv_roc_auc_mean': None,
        'cv_roc_auc_std':  None,
    }
    print(f"\n  ✅ AUC={result['roc_auc']} | F1={result['f1']} | "
          f"P={result['precision']} | R={result['recall']}")
    results_list.append(result)

    # 모델별 개별 결과 저장
    pd.DataFrame([result]).to_csv(
        f"{OUTPUT_PATH}/{model_name}_K{K_EARLY}_result.csv",
        index=False, encoding='utf-8-sig')

# ── 전체 실험 실행 ────────────────────────────────────
K_LIST = [5, 7, 10, 14]
all_results = []

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ▶ 전체 실험 시작")
print(f"  모델: {list(MODELS.keys())}")
print(f"  K_EARLY: {K_LIST}")
print(f"  총 {len(MODELS)*len(K_LIST)}개 실험")

for K in K_LIST:
    npz = f"{OUTPUT_PATH}/kr_sequences_K{K}.npz"
    if not os.path.exists(npz):
        print(f"  ⚠️  {npz} 없음 → skip")
        continue
    for model_name in MODELS.keys():
        run_one(model_name, K, npz, all_results)

# ── 결과 저장 ─────────────────────────────────────────
df_results = pd.DataFrame(all_results)
df_results.to_csv(f"{OUTPUT_PATH}/dl_ablation_results.csv",
                  index=False, encoding='utf-8-sig')

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✅ 전체 실험 완료!")
print(f"\n{'='*55}")
print(f"  📊 K_EARLY별 최고 성능 (AUC 기준)")
print(f"{'='*55}")
best_per_K = df_results.loc[df_results.groupby('K_EARLY')['roc_auc'].idxmax()]
print(best_per_K[['K_EARLY','model','roc_auc','f1']].to_string(index=False))

print(f"\n{'='*55}")
print(f"  📊 모델별 최고 성능 (AUC 기준)")
print(f"{'='*55}")
best_per_model = df_results.loc[df_results.groupby('model')['roc_auc'].idxmax()]
print(best_per_model[['model','K_EARLY','roc_auc','f1']].to_string(index=False))
