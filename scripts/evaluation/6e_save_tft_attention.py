import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

OUTPUT_PATH = "./output"
MODEL_PATH = "./models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRN(nn.Module):
    def __init__(self, i, h, o, d=0.1):
        super().__init__()
        self.fc1 = nn.Linear(i, h)
        self.fc2 = nn.Linear(h, o)
        self.gate = nn.Linear(h, o)
        self.ln = nn.LayerNorm(o)
        self.drop = nn.Dropout(d)
        self.skip = nn.Linear(i, o) if i != o else nn.Identity()

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = self.drop(h)
        return self.ln(self.fc2(h) * torch.sigmoid(self.gate(h)) + self.skip(x))


class TFTClassifier(nn.Module):
    def __init__(self, seq_dim, static_dim, hidden=128, nhead=4, dropout=0.2):
        super().__init__()
        self.static_grn = GRN(static_dim, hidden, hidden, dropout)
        self.ctx_h = nn.Linear(hidden, hidden)
        self.ctx_c = nn.Linear(hidden, hidden)
        self.ctx_e = nn.Linear(hidden, hidden)
        self.seq_proj = nn.Linear(seq_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.lstm_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.lstm_ln = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, nhead, dropout=dropout, batch_first=True)
        self.attn_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.attn_ln = nn.LayerNorm(hidden)
        self.pos_grn = GRN(hidden, hidden * 2, hidden, dropout)
        self.pos_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.pos_ln = nn.LayerNorm(hidden)
        self.clf = nn.Sequential(
            nn.Linear(hidden * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward_with_attn(self, xs, xst, m):
        s = self.static_grn(xst)
        c_h = self.ctx_h(s).unsqueeze(0)
        c_c = self.ctx_c(s).unsqueeze(0)
        c_e = self.ctx_e(s)
        xe = self.seq_proj(xs)
        lo, _ = self.lstm(xe, (c_h, c_c))
        lg = self.lstm_gate(lo) * lo
        lo = self.lstm_ln(lg + xe)
        ao, attn_w = self.attn(
            lo,
            lo,
            lo,
            key_padding_mask=(m == 0),
            need_weights=True,
            average_attn_weights=False,
        )
        ag = self.attn_gate(ao) * ao
        ao = self.attn_ln(ag + lo)
        b, t, h = ao.shape
        po = self.pos_grn(ao.reshape(b * t, h)).reshape(b, t, h)
        pg = self.pos_gate(po) * po
        po = self.pos_ln(pg + ao)
        me = m.unsqueeze(-1)
        pool = (po * me).sum(1) / me.sum(1).clamp(min=1)
        logits = self.clf(torch.cat([pool, c_e], -1)).squeeze(1)
        return logits, attn_w


class DS(Dataset):
    def __init__(self, X_seq, X_static, mask, idx):
        self.X_seq = X_seq
        self.X_static = X_static
        self.mask = mask
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        n = self.idx[i]
        return self.X_seq[n], self.X_static[n], self.mask[n]


def main():
    print(f"Device: {DEVICE}")
    data = np.load(f"{OUTPUT_PATH}/kr_sequences_K14.npz", allow_pickle=True)
    X_seq = torch.tensor(data["X_seq"], dtype=torch.float32)
    X_static = torch.tensor(data["X_static"], dtype=torch.float32)
    mask = torch.tensor(data["mask"], dtype=torch.float32)
    test_idx = data["test_idx"]

    model = TFTClassifier(X_seq.shape[2], X_static.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(f"{MODEL_PATH}/TFT_K14_best.pt", map_location=DEVICE))
    model.eval()

    loader = DataLoader(DS(X_seq, X_static, mask, test_idx), batch_size=256, shuffle=False)
    attn_all = []
    preds = []
    with torch.no_grad():
        for xs, xst, mb in loader:
            xs = xs.to(DEVICE)
            xst = xst.to(DEVICE)
            mb = mb.to(DEVICE)
            logits, attn_w = model.forward_with_attn(xs, xst, mb)
            attn_all.append(attn_w.cpu().numpy())
            preds.append(torch.sigmoid(logits).cpu().numpy())

    attn = np.concatenate(attn_all, axis=0)
    pred = np.concatenate(preds, axis=0)
    np.save(f"{OUTPUT_PATH}/TFT_attention.npy", attn)
    np.save(f"{OUTPUT_PATH}/TFT_K14_pred_from_attention.npy", pred)
    print(f"TFT attention saved: {attn.shape}")
    print(f"Saved: {OUTPUT_PATH}/TFT_attention.npy")


if __name__ == "__main__":
    main()
