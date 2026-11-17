from pathlib import Path
import numpy as np

out = Path(__file__).resolve().parents[2] / "output"
for name in ["kr_sequences_K14.npz", "kr_sequences_K5.npz"]:
    data = np.load(out / name, allow_pickle=True)
    print("==", name, "==")
    print(data.files)
    for key in data.files:
        arr = data[key]
        print(key, getattr(arr, "shape", None), arr.dtype)
        if arr.shape == () or (hasattr(arr, "shape") and len(arr.shape) == 1 and len(arr) < 20):
            try:
                print(" ", arr)
            except Exception:
                pass
