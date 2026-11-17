from pathlib import Path

import pandas as pd

p = Path(__file__).resolve().parent / "output" / "kr_snapshot.csv"
df_head = pd.read_csv(p, nrows=5)
print("columns:", df_head.columns.tolist())
print(df_head.to_string(index=False))

usecols = None
date_candidates = [
    "trending_date",
    "snapshot_date",
    "publishedAt",
    "published_at",
    "publish_time",
    "collected_at",
]
cols = df_head.columns.tolist()
usecols = [c for c in date_candidates if c in cols]
if usecols:
    df = pd.read_csv(p, usecols=usecols)
    for c in usecols:
        s = pd.to_datetime(df[c], errors="coerce")
        print(c, "min=", s.min(), "max=", s.max(), "nunique=", s.dt.date.nunique())
else:
    print("no candidate date columns found")
