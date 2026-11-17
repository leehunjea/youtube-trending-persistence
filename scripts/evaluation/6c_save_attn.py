"""
LSTM K=14 Attention Weight 저장
→ 바이럴/비바이럴별 시간축 집중 패턴 추출
"""
import numpy as np
import torch
import torch.nn as nn
import os, pickle
from torch.utils.data import Dataset, DataLoader

OUTPUT_PATH = "./output"
MODEL_PATH  = "./models"
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── 모델 클래스 (5_run_dl.py와 동일) ─────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, seq_dim, static_dim, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.lstm    = nn.LSTM(seq_dim, hidden, layers, batch_first=True,
                               dropout=dropout if layers>1 else 0,
                               bidirectional=True)
        self.attn_fc = nn.Linear(hidden*2, 1)
        self.static_fc = nn.Sequential(
            nn.Linear(static_dim, 64), nn.LayerNorm(64),
            nn.ReLU(), nn.Dropout(dropout))
        self.clf = nn.Sequential(
            nn.Linear(hidden*2+64, 256), nn.LayerNorm(256),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward_with_attn(self, xs, xst, m):
        """Attention weight도 함께 반환하는 forward"""
        o, _ = self.lstm(xs)                               # (B, K, H*2)
        a    = self.attn_fc(o).squeeze(-1)                 # (B, K)
        a    = a.masked_fill(m == 0, -1e9)
        w    = torch.softmax(a, -1)                        # (B, K) ← 핵심
        p    = (o * w.unsqueeze(-1)).sum(1)                # (B, H*2)
        out  = self.clf(torch.cat([p, self.static_fc(xst)], -1)).squeeze(1)
        return out, w                                       # 예측값 + Attention

class DS(Dataset):
    def __init__(self, X_seq, X_static, mask, idx):
        self.X_seq=X_seq; self.X_static=X_static
        self.mask=mask; self.idx=idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        n = self.idx[i]
        return self.X_seq[n], self.X_static[n], self.mask[n]

# ── 데이터 로드 ───────────────────────────────────────
data     = np.load(f"{OUTPUT_PATH}/kr_sequences_K14.npz", allow_pickle=True)
X_seq    = torch.tensor(data['X_seq'],    dtype=torch.float32)
X_static = torch.tensor(data['X_static'], dtype=torch.float32)
mask     = torch.tensor(data['mask'],     dtype=torch.float32)
y_cls    = data['y_cls']
test_idx = data['test_idx']

SEQ_DIM = X_seq.shape[2]
STA_DIM = X_static.shape[1]
K       = X_seq.shape[1]
print(f"K={K}, SEQ_DIM={SEQ_DIM}, STA_DIM={STA_DIM}")
print(f"Test: {len(test_idx):,}개")

loader = DataLoader(DS(X_seq, X_static, mask, test_idx),
                    batch_size=256, shuffle=False)

# ── 모델 로드 ─────────────────────────────────────────
model = LSTMClassifier(SEQ_DIM, STA_DIM).to(DEVICE)
model.load_state_dict(torch.load(
    f"{MODEL_PATH}/LSTM_K14_best.pt", map_location=DEVICE))
model.eval()
print("LSTM K=14 모델 로드 완료")

# ── Attention 추출 ────────────────────────────────────
all_attn  = []   # (N, K) : 각 영상의 K개 타임스텝 가중치
all_preds = []
all_true  = []

with torch.no_grad():
    for xs, xst, mb in loader:
        xs  = xs.to(DEVICE)
        xst = xst.to(DEVICE)
        mb  = mb.to(DEVICE)
        out, w = model.forward_with_attn(xs, xst, mb)
        all_attn.append(w.cpu().numpy())                   # (B, K)
        all_preds.append(torch.sigmoid(out).cpu().numpy())
        all_true.append(mb.sum(dim=1).cpu().numpy())       # mask 합 = 실제 길이

attn_matrix = np.concatenate(all_attn,  axis=0)           # (N_test, K)
y_true      = y_cls[test_idx]                              # (N_test,)

print(f"\n  Attention shape: {attn_matrix.shape}")
print(f"  바이럴:   {(y_true==1).sum():,}개")
print(f"  비바이럴: {(y_true==0).sum():,}개")

# ── 통계 분석 ─────────────────────────────────────────
viral_attn    = attn_matrix[y_true == 1]   # (N_viral,   K)
nonviral_attn = attn_matrix[y_true == 0]   # (N_nonviral, K)

mean_viral    = viral_attn.mean(axis=0)    # (K,)
std_viral     = viral_attn.std(axis=0)
mean_nonviral = nonviral_attn.mean(axis=0)
std_nonviral  = nonviral_attn.std(axis=0)
diff          = mean_viral - mean_nonviral

print(f"\n  Viral    집중 스냅샷: t{mean_viral.argmax()+1}   "
      f"(weight={mean_viral.max():.4f})")
print(f"  NonViral 집중 스냅샷: t{mean_nonviral.argmax()+1} "
      f"(weight={mean_nonviral.max():.4f})")
print(f"  최대 차이 스냅샷:     t{np.abs(diff).argmax()+1}  "
      f"(Δ={diff[np.abs(diff).argmax()]:+.4f})")

# ── 저장 ─────────────────────────────────────────────
np.save(f"{OUTPUT_PATH}/LSTM_K14_attn_viral.npy",    mean_viral)
np.save(f"{OUTPUT_PATH}/LSTM_K14_attn_nonviral.npy", mean_nonviral)
np.save(f"{OUTPUT_PATH}/LSTM_K14_attn_std_v.npy",   std_viral)
np.save(f"{OUTPUT_PATH}/LSTM_K14_attn_std_nv.npy",  std_nonviral)
np.save(f"{OUTPUT_PATH}/LSTM_K14_attn_matrix.npy",  attn_matrix)

print(f"\n  ✅ LSTM_K14_attn_viral.npy    저장")
print(f"  ✅ LSTM_K14_attn_nonviral.npy 저장")
print(f"  ✅ LSTM_K14_attn_matrix.npy   저장")
print(f"\n→ python scripts/visualization/7b_fig5_replace.py 실행하세요")
