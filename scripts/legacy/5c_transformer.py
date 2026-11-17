import numpy as np
import torch
import torch.nn as nn
import math
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
MODEL_NAME = "Transformer"
print(f"  Device: {DEVICE}")

BATCH_SIZE = 64
EPOCHS     = 50
LR         = 5e-4
D_MODEL    = 64
NHEAD      = 4
NUM_LAYERS = 2
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class TransformerClassifier(nn.Module):
    def __init__(self, seq_dim, static_dim, d_model, nhead, num_layers, dropout):
        super().__init__()
        # ✅ CLS 토큰 추가
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.input_proj = nn.Linear(seq_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.static_fc  = nn.Sequential(
            nn.Linear(static_dim, 128), nn.LayerNorm(128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 64, 256),
            nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x_seq, x_static, mask):
        B = x_seq.shape[0]
        x = self.input_proj(x_seq)               # (B, K, d_model)
        x = self.pos_enc(x)

        # CLS 토큰 앞에 추가
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)          # (B, K+1, d_model)

        # CLS 위치는 마스크 1로 추가
        cls_mask = torch.ones(B, 1, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)
        key_padding_mask = (full_mask == 0)

        x   = self.transformer(x, src_key_padding_mask=key_padding_mask)
        cls_out    = x[:, 0, :]                   # CLS 토큰 출력
        static_out = self.static_fc(x_static)
        combined   = torch.cat([cls_out, static_out], dim=1)
        return self.classifier(combined).squeeze(1)

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
        logits = model(x_seq, x_static, mask_b)
        loss   = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for x_seq, x_static, mask_b, y in loader:
            x_seq, x_static, mask_b, y = (
                x_seq.to(DEVICE), x_static.to(DEVICE),
                mask_b.to(DEVICE), y.to(DEVICE)
            )
            logits      = model(x_seq, x_static, mask_b)
            loss        = criterion(logits, y)
            probs       = torch.sigmoid(logits)
            total_loss += loss.item()
            all_preds  += probs.cpu().numpy().tolist()
            all_labels += y.cpu().numpy().tolist()
    return total_loss / len(loader), np.array(all_preds), np.array(all_labels)

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ▶ {MODEL_NAME} 학습 시작...")
model     = TransformerClassifier(SEQ_DIM, STATIC_DIM, D_MODEL, NHEAD, NUM_LAYERS, DROPOUT).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_auc = 0; patience_count = 0; history = []

for epoch in range(1, EPOCHS + 1):
    train_loss                    = train_epoch(model, train_loader, optimizer)
    val_loss, val_preds, val_true = eval_epoch(model, val_loader)
    val_auc                       = roc_auc_score(val_true, val_preds)
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
_, test_preds, test_true = eval_epoch(model, test_loader)
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
pd.DataFrame([metrics]).to_csv(f"{OUTPUT_PATH}/{MODEL_NAME}_result.csv", index=False, encoding='utf-8-sig')
pd.DataFrame(history).to_csv(f"{OUTPUT_PATH}/{MODEL_NAME}_history.csv", index=False, encoding='utf-8-sig')
with open(f"{MODEL_PATH}/{MODEL_NAME}_model.pkl", 'wb') as f:
    pickle.dump({'model': model, 'metrics': metrics, 'history': history}, f)
print(f"  ✅ models/{MODEL_NAME}_model.pkl 저장 완료")
print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {MODEL_NAME} 완료!")
