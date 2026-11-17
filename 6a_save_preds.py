"""
모든 모델의 test 예측값을 CSV로 저장
→ 6b_bootstrap_ci.py에서 클래스 정의 없이 로드 가능
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import os, pickle
from sklearn.metrics import roc_auc_score

OUTPUT_PATH = "./output"
MODEL_PATH  = "./models"
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ════════════════════════════════════════════════════
# DL 모델 클래스 정의 (5_run_dl.py와 동일)
# ════════════════════════════════════════════════════
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

DL_CLASSES = {
    'LSTM': LSTMClassifier, 'GRU': GRUClassifier,
    'Transformer': TransformerClassifier, 'TFT': TFTClassifier,
}

# ════════════════════════════════════════════════════
# ML Feature 생성 함수
# ════════════════════════════════════════════════════
def make_ml_features(X_seq, X_static, SEQ_F, STA_F):
    rows = []
    K = X_seq.shape[1]
    for i in range(len(X_seq)):
        seq = X_seq[i]; st = X_static[i]
        feat = {}
        for j, name in enumerate(STA_F):
            feat[name] = st[j]
        for j, name in enumerate(SEQ_F):
            col = seq[:, j]
            feat[f'{name}_first']  = col[0]
            feat[f'{name}_last']   = col[-1]
            feat[f'{name}_mean']   = col.mean()
            feat[f'{name}_std']    = col.std()
            feat[f'{name}_max']    = col.max()
            feat[f'{name}_min']    = col.min()
            feat[f'{name}_range']  = col.max() - col.min()
            feat[f'{name}_slope']  = np.polyfit(range(K), col, 1)[0] if K > 1 else 0
        rank_col = seq[:, SEQ_F.index('rank')]
        vph_col  = seq[:, SEQ_F.index('views_per_hour')]
        mom_col  = seq[:, SEQ_F.index('momentum')]
        rd_col   = seq[:, SEQ_F.index('rank_delta')]
        va_col   = seq[:, SEQ_F.index('views_acceleration')]
        feat['rank_improvement']    = rank_col[0] - rank_col[-1]
        feat['peak_momentum']       = mom_col.max()
        feat['momentum_trend']      = mom_col[-1] - mom_col[0]
        feat['positive_rank_delta'] = (rd_col < 0).sum()
        feat['max_acceleration']    = va_col.max()
        feat['sustained_growth']    = (va_col > 0).sum() / K
        feat['vph_growth_rate']     = vph_col[-1] / (vph_col[0] + 1e-6)
        rows.append(feat)
    return pd.DataFrame(rows)

# ════════════════════════════════════════════════════
# 예측값 수집 및 저장
# ════════════════════════════════════════════════════
print("\n▶ 예측값 수집 시작...")

with open(f"{OUTPUT_PATH}/scalers.pkl", 'rb') as f:
    scalers = pickle.load(f)
SEQ_F = scalers['seq_features']
STA_F = scalers['static_features']

all_preds = []   # {name, y_true, y_pred} 리스트

# ── DL 모델 (K=5,7,10,14 전부) ───────────────────────
from torch.utils.data import Dataset, DataLoader

class DS(Dataset):
    def __init__(self, X_seq, X_static, mask, idx):
        self.X_seq=X_seq; self.X_static=X_static
        self.mask=mask; self.idx=idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        n=self.idx[i]
        return self.X_seq[n], self.X_static[n], self.mask[n]

for K in [5, 7, 10, 14]:
    data     = np.load(f"{OUTPUT_PATH}/kr_sequences_K{K}.npz", allow_pickle=True)
    X_seq    = torch.tensor(data['X_seq'],    dtype=torch.float32)
    X_static = torch.tensor(data['X_static'], dtype=torch.float32)
    mask     = torch.tensor(data['mask'],     dtype=torch.float32)
    y_cls    = data['y_cls']
    test_idx = data['test_idx']
    SEQ_DIM  = X_seq.shape[2]
    STA_DIM  = X_static.shape[1]

    loader = DataLoader(DS(X_seq, X_static, mask, test_idx), 256, shuffle=False)

    for mn, ModelClass in DL_CLASSES.items():
        pt_path = f"{MODEL_PATH}/{mn}_K{K}_best.pt"
        if not os.path.exists(pt_path):
            print(f"  ⚠️  {pt_path} 없음 → skip")
            continue
        model = ModelClass(SEQ_DIM, STA_DIM).to(DEVICE)
        model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
        model.eval()
        preds = []
        with torch.no_grad():
            for xs, xst, mb in loader:
                xs,xst,mb = xs.to(DEVICE), xst.to(DEVICE), mb.to(DEVICE)
                out = model(xs, xst, mb)
                if isinstance(out, tuple): out = out[0]
                preds += torch.sigmoid(out).cpu().tolist()
        y_true = y_cls[test_idx]
        auc = roc_auc_score(y_true, preds)
        all_preds.append({
            'name': f'{mn}_K{K}',
            'video_id': data['video_ids'][test_idx].astype(str),
            'y_true': y_true,
            'y_pred': np.array(preds),
        })
        print(f"  ✅ {mn}_K{K:<3} AUC={auc:.4f}")

# ── ML v2 (CatBoost만 성공했으니 전체 재생성) ─────────
data14   = np.load(f"{OUTPUT_PATH}/kr_sequences_K14.npz", allow_pickle=True)
X_seq14  = data14['X_seq']
X_sta14  = data14['X_static']
y14      = data14['y_cls']
te14     = data14['test_idx']
tr14     = data14['train_idx']
val14    = data14['val_idx']
trval14  = np.concatenate([tr14, val14])

X_all_ml = make_ml_features(X_seq14, X_sta14, SEQ_F, STA_F)
X_te_ml  = X_all_ml.iloc[te14].values
X_tr_ml  = X_all_ml.iloc[trval14].values
y_tr_ml  = y14[trval14]
y_te_ml  = y14[te14]

scale_pos = (y_tr_ml==0).sum() / (y_tr_ml==1).sum()
print(f"\n  ML v2 재학습 (K=14, {X_all_ml.shape[1]}개 Feature, N={len(X_tr_ml):,})")

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

ML_MODELS = {
    'RF_v2':      RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                          n_jobs=-1, random_state=42),
    'XGB_v2':     xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                                      scale_pos_weight=scale_pos, verbosity=0, random_state=42),
    'LGBM_v2':    lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                       scale_pos_weight=scale_pos, verbosity=-1, random_state=42),
    'CatBoost_v2':CatBoostClassifier(iterations=500, learning_rate=0.05,
                                      scale_pos_weight=scale_pos, verbose=0, random_seed=42),
    'MLP_v2':     MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=300,
                                 early_stopping=True, random_state=42),
}

for name, model in ML_MODELS.items():
    model.fit(X_tr_ml, y_tr_ml)
    probs = model.predict_proba(X_te_ml)[:,1]
    auc   = roc_auc_score(y_te_ml, probs)
    all_preds.append({
        'name': name,
        'video_id': data14['video_ids'][te14].astype(str),
        'y_true': y_te_ml,
        'y_pred': probs,
    })
    # 모델 저장 (덮어쓰기)
    with open(f"{MODEL_PATH}/{name}_model.pkl", 'wb') as f:
        pickle.dump({'model': model}, f)
    print(f"  ✅ {name:<15} AUC={auc:.4f}")

# ── ML v1 (K=5 기준 재현) ────────────────────────────
data5    = np.load(f"{OUTPUT_PATH}/kr_sequences_K5.npz", allow_pickle=True)
X_seq5   = data5['X_seq'];  X_sta5 = data5['X_static']
y5       = data5['y_cls'];  te5    = data5['test_idx']
tr5      = data5['train_idx']; val5 = data5['val_idx']
trval5   = np.concatenate([tr5, val5])

# K=5용 구버전 Feature (SEQ 4개만, Delta 없음)
SEQ_F_V1 = ['rank','views_per_hour','comment_view_ratio','hours_since_publish']

def make_ml_features_v1(X_seq, X_static, SEQ_F, STA_F):
    rows = []
    K = X_seq.shape[1]
    for i in range(len(X_seq)):
        seq = X_seq[i]; st = X_static[i]
        feat = {}
        for j, name in enumerate(STA_F):
            feat[name] = st[j]
        for j, name in enumerate(SEQ_F):
            col = seq[:, j]
            feat[f'{name}_first'] = col[0]
            feat[f'{name}_mean']  = col.mean()
            feat[f'{name}_slope'] = np.polyfit(range(K), col, 1)[0] if K > 1 else 0
        rows.append(feat)
    return pd.DataFrame(rows)

# Static Feature 수 맞추기 (K=5 데이터는 Static 8개)
X_all_v1 = make_ml_features_v1(X_seq5, X_sta5, SEQ_F_V1, STA_F)
X_te_v1  = X_all_v1.iloc[te5].values
X_tr_v1  = X_all_v1.iloc[trval5].values
y_tr_v1  = y5[trval5];  y_te_v1 = y5[te5]
sp_v1    = (y_tr_v1==0).sum()/(y_tr_v1==1).sum()

ML_V1 = {
    'RF_v1':       RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                           n_jobs=-1, random_state=42),
    'CatBoost_v1': CatBoostClassifier(iterations=300, learning_rate=0.05,
                                       scale_pos_weight=sp_v1, verbose=0, random_seed=42),
}
for name, model in ML_V1.items():
    model.fit(X_tr_v1, y_tr_v1)
    probs = model.predict_proba(X_te_v1)[:,1]
    auc   = roc_auc_score(y_te_v1, probs)
    all_preds.append({
        'name': name,
        'video_id': data5['video_ids'][te5].astype(str),
        'y_true': y_te_v1,
        'y_pred': probs,
    })
    print(f"  ✅ {name:<15} AUC={auc:.4f}")

# ── 예측값 CSV 저장 ──────────────────────────────────
print("\n▶ 예측값 저장 중...")
for item in all_preds:
    df = pd.DataFrame({
        'video_id': item['video_id'],
        'y_true': item['y_true'],
        'y_pred': item['y_pred'],
    })
    df.to_csv(f"{OUTPUT_PATH}/preds_{item['name']}.csv", index=False)
    print(f"  ✅ preds_{item['name']}.csv")

print(f"\n총 {len(all_preds)}개 모델 예측값 저장 완료")
print("→ python 6b_bootstrap_ci.py 실행하세요")
