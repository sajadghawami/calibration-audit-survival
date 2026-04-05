import h5py
import torch
import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv("patient_index.csv")
output_dir = Path("SurvPath/data/WSI_features/TCGA-BRCA")
output_dir.mkdir(parents=True, exist_ok=True)

for i, row in df.iterrows():
    paths = [p.strip() for p in str(row["h5_paths"]).split(";")]
    patches = []
    for path in paths:
        with h5py.File(path, "r") as f:
            feats = f["features"][:]           # (1, n_patches, 1536)
            feats = np.squeeze(feats, axis=0)  # (n_patches, 1536)
            patches.append(feats)

    combined = np.concatenate(patches, axis=0)  # (total_patches, 1536)
    tensor = torch.tensor(combined, dtype=torch.float32)

    out_path = output_dir / f"{row['patient_id']}.pt"
    torch.save(tensor, out_path)

    if (i + 1) % 100 == 0:
        print(f"[{i+1}/{len(df)}] saved")

print(f"Saved {len(df)} .pt files")
