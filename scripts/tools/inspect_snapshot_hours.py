from pathlib import Path
import pandas as pd

df = pd.read_csv(Path(__file__).resolve().parents[2] / "output" / "kr_snapshot.csv", usecols=["collection_date"])
s = pd.to_datetime(df["collection_date"], errors="coerce")
print("hours", sorted(s.dt.hour.dropna().unique().tolist()))
print(s.dt.hour.value_counts().sort_index().to_string())
