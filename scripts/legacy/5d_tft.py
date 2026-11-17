import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, f1_score,
    precision_score, recall_score, classification_report)
import os, pickle
from datetime import datetime
import pandas as pd

OUTPUT_PATH = "./output"
MODEL_PATH  = "./models"
os.makedirs(MODEL_PATH, exist_ok=True)

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = "TFT"
print(f"  Device: {DEVICE}")

BATCH_SIZE = 64
EPOCHS     = 50
LR         = 5e-4
HIDDEN_DIM = 128
NHEAD      = 4
DROPOUT    = 0.2
PATIENCE   = 10

data      = np.load(f"{OUTPUT_PATH}/kr_sequences.npz", allow_pickle=True)
X_seq     = torch.tensor(data['X_seq'],    dtype=torch.float32)
X_static  = torch.tensor(data['X_static'], dtype=torch.float32)
mask      = torch.tensor(data['mask'],     dtype=torch.float32)
y_cls     = torch.tensor(data['y_cls'],    dtype=torch.float32)
train_idx = data['train_idx']
val_idx   = data['val_idx']
test_idx  = data['test_idx']

SEQ_DIM    = X_seq.shape[2]
STATIC_DIM = X_static.shape[1]

print(f"  X_seq    : {tuple(X_seq.shape)}")
print(f"  X_static : {tuple(X_static.shape)}")
print(f"  바이럴 비율: {y_cls.mean().item()*100:.1f}%")

class ViralDataset(Dataset):
    def __init__(self, idx):
        self.idx = idx
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        n = self.idx[i]
        return X_seq[n], X_static[n], mask[n], y_cls[n]

train_loader = DataLoader(ViralDataset(train_idx), batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_loader   = DataLoader(ViralDataset(val_idx),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(ViralDataset(test_idx),  batch_size=BATCH_SIZE, shuffle=False)

class GRN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.fc1   = nn.Linear(in_dim, hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, out_dim)
        self.gate  = nn.Linear(hidden_dim, out_dim)
        self.ln    = nn.LayerNorm(out_dim)
        self.drop  = nn.Dropout(dropout)
        self.skip  = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        h   = torch.relu(self.fc1(x))
        h   = self.drop(h)
        out = self.fc2(h) * torch.sigmoid(self.gate(h))
        return self.ln(out + self.skip(x))

class TFTClassifier(nn.Module):
    def __init__(self, seq_dim, static_dim, hidden_dim, nhead, dropout):
        super().__init__()
        # Static 처리
        self.static_grn   = GRN(static_dim, hidden_dim, hidden_dim, dropout)
        self.static_ctx_h = nn.Linear(hidden_dim, hidden_dim)
        self.static_ctx_c = nn.Linear(hidden_dim, hidden_dim)
        self.static_ctx_e = nn.Linear(hidden_dim, hidden_dim)

        # Sequence 처리
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.lstm     = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # LSTM Gate & Norm
        self.lstm_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.lstm_ln   = nn.LayerNorm(hidden_dim)

        # Self-Attention
        self.attn      = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.attn_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attn_ln   = nn.LayerNorm(hidden_dim)

        # Position-wise GRN
        self.pos_grn  = GRN(hidden_dim, hidden_dim * 2, hidden_dim, dropout)
        self.pos_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.pos_ln   = nn.LayerNorm(hidden_dim)

        # 분류기
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x_seq, x_static, mask):
        # Static 인코딩
        s   = self.static_grn(x_static)                   # (B, H)
        c_h = self.static_ctx_h(s).unsqueeze(0)           # (1, B, H)
        c_c = self.static_ctx_c(s).unsqueeze(0)           # (1, B, H)
        c_e = self.static_ctx_e(s)                        # (B, H)

        # Sequence 처리
        x_emb           = self.seq_proj(x_seq)            # (B, K, H)
        lstm_out, _     = self.lstm(x_emb, (c_h, c_c))   # (B, K, H)
        lstm_gated      = self.lstm_gate(lstm_out) * lstm_out
        lstm_out        = self.lstm_ln(lstm_gated + x_emb)

        # Self-Attention
        key_padding_mask = (mask == 0)
        attn_out, attn_w = self.attn(lstm_out, lstm_out, lstm_out,
                                      key_padding_mask=key_padding_mask)
        attn_gated  = self.attn_gate(attn_out) * attn_out
        attn_out    = self.attn_ln(attn_gated + lstm_out)

        # Position-wise GRN
        B, T, H     = attn_out.shape
        pos_out     = self.pos_grn(attn_out.reshape(B * T, H)).reshape(B, T, H)
        pos_gated   = self.pos_gate(pos_out) * pos_out
        pos_out     = self.pos_ln(pos_gated + attn_out)

        # 마스크 적용 평균 풀링
        mask_exp    = mask.unsqueeze(-1)
        pooled      = (pos_out * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)

        # Static context 결합
        combined    = torch.cat([pooled, c_e], dim=1)    # (B, H*2)
        return self.classifier(combined).squeeze(1), attn_w

pos_weight = torch.tensor([(y_cls == 0).sum() / (y_cls == 1).sum()]).to(DEVICE)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
print(f"  pos_weight: {pos_weight.item():.4f}")

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for x_seq, x_static, mask_b, y in loader:
        x_seq, x_static, mask_b, y = (
            x_seq.to(DEVICE), x_static.to(DEVICE),
            mask_b.to(DEVICE), y.to(DEVICE)
        )
        optimizer.zero_grad()
        logits, _ = model(x_seq, x_static, mask_b)
        loss      = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader):
    model.eval()
    total_loss, all_preds, all_labels, all_attn = 0, [], [], []
    with torch.no_grad():
        for x_seq, x_static, mask_b, y in loader:
            x_seq, x_static, mask_b, y = (
                x_seq.to(DEVICE), x_static.to(DEVICE),
                mask_b.to(DEVICE), y.to(DEVICE)
            )
            logits, attn = model(x_seq, x_static, mask_b)
            loss         = criterion(logits, y)
            probs        = torch.sigmoid(logits)
            total_loss  += loss.item()
            all_preds   += probs.cpu().numpy().tolist()
            all_labels  += y.cpu().numpy().tolist()
            all_attn.append(attn.cpu().numpy())
    return (total_loss / len(loader),
            np.array(all_preds), np.array(all_labels),
            np.concatenate(all_attn, axis=0))

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ▶ {MODEL_NAME} 학습 시작...")
model     = TFTClassifier(SEQ_DIM, STATIC_DIM, HIDDEN_DIM, NHEAD, DROPOUT).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_auc = 0; patience_count = 0; history = []

for epoch in range(1, EPOCHS + 1):
    train_loss                          = train_epoch(model, train_loader, optimizer)
    val_loss, val_preds, val_true, _    = eval_epoch(model, val_loader)
    val_auc                             = roc_auc_score(val_true, val_preds)
    scheduler.step()
    history.append({'epoch': epoch, 'train_loss': round(train_loss,4),
                    'val_loss': round(val_loss,4), 'val_auc': round(val_auc,4)})
    print(f"  Epoch {epoch:3d}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | AUC: {val_auc:.4f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc; patience_count = 0
        torch.save(model.state_dict(), f"{MODEL_PATH}/{MODEL_NAME}_best.pt")
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"  ⏹ Early Stopping (epoch={epoch})")
            break

model.load_state_dict(torch.load(f"{MODEL_PATH}/{MODEL_NAME}_best.pt"))
_, test_preds, test_true, test_attn = eval_epoch(model, test_loader)
test_binary = (test_preds >= 0.5).astype(int)

metrics = {
    'model': MODEL_NAME,
    'roc_auc':   round(roc_auc_score(test_true, test_preds),   4),
    'f1':        round(f1_score(test_true, test_binary),        4),
    'precision': round(precision_score(test_true, test_binary), 4),
    'recall':    round(recall_score(test_true, test_binary),    4),
    'best_val_auc': round(best_val_auc, 4),
    'cv_roc_auc_mean': None, 'cv_roc_auc_std': None,
}
print(classification_report(test_true, test_binary, target_names=['비바이럴','바이럴']))
np.save(f"{OUTPUT_PATH}/{MODEL_NAME}_attention.npy", test_attn)
pd.DataFrame([metrics]).to_csv(f"{OUTPUT_PATH}/{MODEL_NAME}_result.csv", index=False, encoding='utf-8-sig')
pd.DataFrame(history).to_csv(f"{OUTPUT_PATH}/{MODEL_NAME}_history.csv", index=False, encoding='utf-8-sig')
with open(f"{MODEL_PATH}/{MODEL_NAME}_model.pkl", 'wb') as f:
    pickle.dump({'model': model, 'metrics': metrics, 'history': history}, f)
print(f"  ✅ output/TFT_attention.npy 저장 완료 → 논문 Figure 활용")
print(f"  ✅ models/{MODEL_NAME}_model.pkl 저장 완료")
print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {MODEL_NAME} 완료!")
