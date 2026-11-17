"""
Hybrid Transformer + tabular-summary model for revision.

The model uses:
- K=14 sequence matrix (B, K, 7) for temporal pattern learning.
- 71 ML summary/domain features for tabular signal learning.

This addresses the review concern that tree models and sequence deep learning
models used different feature representations.
"""
from pathlib import Path
import math
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
MODELS = ROOT / "models"
RANDOM_STATE = 42

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 45
PATIENCE = 8
LR = 5e-4
D_MODEL = 64
NHEAD = 4
LAYERS = 2
DROPOUT = 0.25


def make_ml_features(X_seq, X_static, seq_features, static_features):
    rows = []
    k = X_seq.shape[1]
    available_seq = seq_features[: X_seq.shape[2]]
    suffixes = ["first", "last", "mean", "std", "max", "min", "range", "slope"]

    for i in range(len(X_seq)):
        seq = X_seq[i]
        st = X_static[i]
        feat = {}
        for j, name in enumerate(static_features):
            feat[name] = st[j]
        for j, name in enumerate(available_seq):
            col = seq[:, j]
            values = {
                "first": col[0],
                "last": col[-1],
                "mean": col.mean(),
                "std": col.std(),
                "max": col.max(),
                "min": col.min(),
                "range": col.max() - col.min(),
                "slope": np.polyfit(range(k), col, 1)[0] if k > 1 else 0,
            }
            for suffix in suffixes:
                feat[f"{name}_{suffix}"] = values[suffix]

        rank_col = seq[:, available_seq.index("rank")]
        vph_col = seq[:, available_seq.index("views_per_hour")]
        mom_col = seq[:, available_seq.index("momentum")]
        rd_col = seq[:, available_seq.index("rank_delta")]
        va_col = seq[:, available_seq.index("views_acceleration")]
        feat["rank_improvement"] = rank_col[0] - rank_col[-1]
        feat["peak_momentum"] = mom_col.max()
        feat["momentum_trend"] = mom_col[-1] - mom_col[0]
        feat["positive_rank_delta"] = (rd_col < 0).sum()
        feat["max_acceleration"] = va_col.max()
        feat["sustained_growth"] = (va_col > 0).sum() / k
        feat["vph_growth_rate"] = vph_col[-1] / (vph_col[0] + 1e-6)
        rows.append(feat)

    return pd.DataFrame(rows)


class HybridDataset(Dataset):
    def __init__(self, x_seq, x_tab, mask, y, idx):
        self.x_seq = x_seq
        self.x_tab = x_tab
        self.mask = mask
        self.y = y
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        n = self.idx[i]
        return self.x_seq[n], self.x_tab[n], self.mask[n], self.y[n]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.shape[1], :]


class HybridTransformerTabular(nn.Module):
    def __init__(self, seq_dim, tab_dim):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.seq_proj = nn.Linear(seq_dim, D_MODEL)
        self.pos = PositionalEncoding(D_MODEL)
        enc = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=D_MODEL * 4,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=LAYERS)
        self.tabular = nn.Sequential(
            nn.Linear(tab_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(D_MODEL + 64, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x_seq, x_tab, mask):
        bsz = x_seq.shape[0]
        x = self.pos(self.seq_proj(x_seq))
        cls = self.cls.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)
        full_mask = torch.cat([torch.ones(bsz, 1, device=mask.device), mask], dim=1)
        x = self.encoder(x, src_key_padding_mask=(full_mask == 0))
        seq_repr = x[:, 0, :]
        tab_repr = self.tabular(x_tab)
        return self.head(torch.cat([seq_repr, tab_repr], dim=1)).squeeze(1)


def evaluate(model, loader, criterion):
    model.eval()
    losses = []
    probs = []
    labels = []
    with torch.no_grad():
        for x_seq, x_tab, mask, y in loader:
            x_seq = x_seq.to(DEVICE)
            x_tab = x_tab.to(DEVICE)
            mask = mask.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x_seq, x_tab, mask)
            losses.append(criterion(logits, y).item())
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())
    return float(np.mean(losses)), np.array(probs), np.array(labels)


def main():
    data = np.load(OUTPUT / "kr_sequences_K14.npz", allow_pickle=True)
    with open(OUTPUT / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    seq_features = scalers["seq_features"]
    static_features = scalers["static_features"]

    x_tab = make_ml_features(data["X_seq"], data["X_static"], seq_features, static_features).values.astype("float32")
    x_seq = data["X_seq"].astype("float32")
    mask = data["mask"].astype("float32")
    y = data["y_cls"].astype("float32")
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    x_seq_t = torch.tensor(x_seq)
    x_tab_t = torch.tensor(x_tab)
    mask_t = torch.tensor(mask)
    y_t = torch.tensor(y)

    train_loader = DataLoader(HybridDataset(x_seq_t, x_tab_t, mask_t, y_t, train_idx), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(HybridDataset(x_seq_t, x_tab_t, mask_t, y_t, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(HybridDataset(x_seq_t, x_tab_t, mask_t, y_t, test_idx), batch_size=BATCH_SIZE, shuffle=False)

    pos_weight = torch.tensor([(y[train_idx] == 0).sum() / max((y[train_idx] == 1).sum(), 1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = HybridTransformerTabular(x_seq.shape[2], x_tab.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"Device={DEVICE}, X_seq={x_seq.shape}, X_tab={x_tab.shape}, pos_weight={pos_weight.item():.4f}")
    best_auc = -1.0
    patience = 0
    history = []
    best_path = MODELS / "HybridTransformerTabular_K14_best.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for x_seq_b, x_tab_b, mask_b, y_b in train_loader:
            x_seq_b = x_seq_b.to(DEVICE)
            x_tab_b = x_tab_b.to(DEVICE)
            mask_b = mask_b.to(DEVICE)
            y_b = y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x_seq_b, x_tab_b, mask_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        val_loss, val_prob, val_true = evaluate(model, val_loader, criterion)
        val_auc = roc_auc_score(val_true, val_prob)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "val_auc": val_auc,
        }
        history.append(row)
        print(f"Epoch {epoch:02d}/{EPOCHS} train={row['train_loss']:.4f} val={val_loss:.4f} val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    _, test_prob, test_true = evaluate(model, test_loader, criterion)
    test_pred = (test_prob >= 0.5).astype(int)
    result = {
        "model": "Hybrid Transformer+Tabular",
        "K_EARLY": 14,
        "roc_auc": roc_auc_score(test_true, test_prob),
        "ap": average_precision_score(test_true, test_prob),
        "f1": f1_score(test_true, test_pred),
        "precision": precision_score(test_true, test_pred, zero_division=0),
        "recall": recall_score(test_true, test_pred, zero_division=0),
        "best_val_auc": best_auc,
        "n_tabular_features": x_tab.shape[1],
        "n_sequence_features": x_seq.shape[2],
    }
    pd.DataFrame([result]).to_csv(OUTPUT / "hybrid_transformer_tabular_K14_result.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(OUTPUT / "hybrid_transformer_tabular_K14_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"y_true": test_true.astype(int), "y_pred": test_prob}).to_csv(
        OUTPUT / "preds_HybridTransformerTabular_K14.csv", index=False, encoding="utf-8-sig"
    )
    print(pd.DataFrame([result]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
