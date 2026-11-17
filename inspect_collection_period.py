from pathlib import Path
import pandas as pd

p = Path(__file__).resolve().parent / "output" / "kr_snapshot.csv"
df = pd.read_csv(p, usecols=["collection_date", "published_at", "video_id"])
cd = pd.to_datetime(df["collection_date"], errors="coerce")
pa = pd.to_datetime(df["published_at"], errors="coerce")

print("collection_min", cd.min())
print("collection_max", cd.max())
print("collection_unique_dates", cd.dt.date.nunique())
print("published_min", pa.min())
print("published_max", pa.max())
print("published_unique_dates", pa.dt.date.nunique())
print("snapshots", len(df))
print("videos", df["video_id"].nunique())
