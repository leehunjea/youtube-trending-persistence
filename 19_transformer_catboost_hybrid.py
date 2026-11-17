"""
Transformer encoder + CatBoost hybrid model for revision.

Stage 1: Train a Transformer sequence encoder on the K=14 snapshot sequence.
Stage 2: Extract the Transformer's CLS embedding and concatenate it with the
         71 summary/domain tabular features.
Stage 3: Train CatBoost on the hybrid feature vector and evaluate on test set.
"""
from pathlib import Path
import math
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
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


class SeqDataset(Dataset):
    def __init__(self, x_seq, mask, y, idx):
        self.x_seq = x_seq
        self.mask = mask
        self.y = y
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        n = self.idx[i]
        return self.x_seq[n], self.mask[n], self.y[n]


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


class SequenceTransformer(nn.Module):
    def __init__(self, seq_dim):
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
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 1),
        )

    def encode(self, x_seq, mask):
        bsz = x_seq.shape[0]
        x = self.pos(self.seq_proj(x_seq))
        cls = self.cls.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)
        full_mask = torch.cat([torch.ones(bsz, 1, device=mask.device), mask], dim=1)
        x = self.encoder(x, src_key_padding_mask=(full_mask == 0))
        return x[:, 0, :]

    def forward(self, x_seq, mask):
        return self.head(self.encode(x_seq, mask)).squeeze(1)


def eval_encoder(model, loader, criterion):
    model.eval()
    losses, probs, labels = [], [], []
    with torch.no_grad():
        for x_seq, mask, y in loader:
            x_seq = x_seq.to(DEVICE)
            mask = mask.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x_seq, mask)
            losses.append(criterion(logits, y).item())
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())
    return float(np.mean(losses)), np.array(probs), np.array(labels)


def extract_embeddings(model, x_seq_t, mask_t):
    model.eval()
    loader = DataLoader(
        torch.utils.data.TensorDataset(x_seq_t, mask_t),
        batch_size=512,
        shuffle=False,
    )
    embs = []
    with torch.no_grad():
        for x_seq, mask in loader:
            emb = model.encode(x_seq.to(DEVICE), mask.to(DEVICE)).cpu().numpy()
            embs.append(emb)
    return np.vstack(embs)


def main():
    data = np.load(OUTPUT / "kr_sequences_K14.npz", allow_pickle=True)
    with open(OUTPUT / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    x_seq = data["X_seq"].astype("float32")
    mask = data["mask"].astype("float32")
    y = data["y_cls"].astype("float32")
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    trainval_idx = np.concatenate([train_idx, val_idx])

    x_tab_df = make_ml_features(
        data["X_seq"],
        data["X_static"],
        scalers["seq_features"],
        scalers["static_features"],
    )
    x_tab = x_tab_df.values.astype("float32")

    x_seq_t = torch.tensor(x_seq)
    mask_t = torch.tensor(mask)
    y_t = torch.tensor(y)

    train_loader = DataLoader(SeqDataset(x_seq_t, mask_t, y_t, train_idx), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(SeqDataset(x_seq_t, mask_t, y_t, val_idx), batch_size=BATCH_SIZE, shuffle=False)

    pos_weight = torch.tensor([(y[train_idx] == 0).sum() / max((y[train_idx] == 1).sum(), 1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    encoder = SequenceTransformer(seq_dim=x_seq.shape[2]).to(DEVICE)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_path = MODELS / "TransformerEncoderForCatBoost_K14_best.pt"

    print(f"Device={DEVICE}, X_seq={x_seq.shape}, X_tab={x_tab.shape}, pos_weight={pos_weight.item():.4f}")
    best_auc = -1.0
    patience = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        train_losses = []
        for x_seq_b, mask_b, y_b in train_loader:
            x_seq_b = x_seq_b.to(DEVICE)
            mask_b = mask_b.to(DEVICE)
            y_b = y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = encoder(x_seq_b, mask_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        val_loss, val_prob, val_true = eval_encoder(encoder, val_loader, criterion)
        val_auc = roc_auc_score(val_true, val_prob)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
                "val_auc": val_auc,
            }
        )
        print(f"Epoch {epoch:02d}/{EPOCHS} train={np.mean(train_losses):.4f} val={val_loss:.4f} val_auc={val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            torch.save(encoder.state_dict(), best_path)
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    encoder.load_state_dict(torch.load(best_path, map_location=DEVICE))
    seq_emb = extract_embeddings(encoder, x_seq_t, mask_t)
    hybrid = np.hstack([x_tab, seq_emb]).astype("float32")

    scale_pos = (y[trainval_idx] == 0).sum() / max((y[trainval_idx] == 1).sum(), 1)
    cb = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        verbose=0,
        random_seed=RANDOM_STATE,
    )
    cb.fit(hybrid[trainval_idx], y[trainval_idx].astype(int))
    prob = cb.predict_proba(hybrid[test_idx])[:, 1]
    y_test = y[test_idx].astype(int)
    pred = (prob >= 0.5).astype(int)

    result = {
        "model": "Transformer-CatBoost Hybrid",
        "K_EARLY": 14,
        "roc_auc": roc_auc_score(y_test, prob),
        "ap": average_precision_score(y_test, prob),
        "f1": f1_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "best_encoder_val_auc": best_auc,
        "n_tabular_features": x_tab.shape[1],
        "n_transformer_embedding": seq_emb.shape[1],
        "n_hybrid_features": hybrid.shape[1],
    }
    pd.DataFrame([result]).to_csv(OUTPUT / "transformer_catboost_hybrid_K14_result.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(OUTPUT / "transformer_catboost_hybrid_K14_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"y_true": y_test, "y_pred": prob}).to_csv(OUTPUT / "preds_TransformerCatBoostHybrid_K14.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(seq_emb, columns=[f"tr_emb_{i:02d}" for i in range(seq_emb.shape[1])]).to_csv(
        OUTPUT / "transformer_K14_cls_embeddings.csv", index=False, encoding="utf-8-sig"
    )

    print(pd.DataFrame([result]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
