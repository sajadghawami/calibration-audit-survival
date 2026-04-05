import os
import csv
from collections import defaultdict

GDC_DIR = "gdc_data"
H5_DIR = "uni2_features/TCGA-BRCA"
OUTPUT = "patient_index.csv"

# Scan RNA-seq files: extract patient ID from filename prefix
rna_map = {}
for fname in os.listdir(GDC_DIR):
    if not fname.endswith(".tsv"):
        continue
    patient_id = fname.split("_")[0]  # e.g. TCGA-BH-A18H
    rna_map[patient_id] = os.path.join(GDC_DIR, fname)

# Scan h5 files: extract patient ID (first 3 segments of TCGA barcode)
h5_map = defaultdict(list)
for fname in os.listdir(H5_DIR):
    if not fname.endswith(".h5"):
        continue
    parts = fname.split("-")
    patient_id = "-".join(parts[:3])  # e.g. TCGA-BH-A18H
    h5_map[patient_id].append(os.path.join(H5_DIR, fname))

# Find intersection
rna_patients = set(rna_map.keys())
h5_patients = set(h5_map.keys())
shared = sorted(rna_patients & h5_patients)

# Write index
with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["patient_id", "rna_seq_path", "h5_paths"])
    for pid in shared:
        writer.writerow([pid, rna_map[pid], ";".join(sorted(h5_map[pid]))])

# Summary
multi_slide = sum(1 for pid in shared if len(h5_map[pid]) > 1)
print(f"RNA-seq patients:  {len(rna_patients)}")
print(f"H5 patients:       {len(h5_patients)}")
print(f"Overlap (paired):  {len(shared)}")
print(f"Multi-slide:       {multi_slide}")

rna_only = sorted(rna_patients - h5_patients)
h5_only = sorted(h5_patients - rna_patients)
if rna_only:
    print(f"\nRNA-only ({len(rna_only)}): {rna_only[:5]}{'...' if len(rna_only) > 5 else ''}")
if h5_only:
    print(f"H5-only ({len(h5_only)}): {h5_only[:5]}{'...' if len(h5_only) > 5 else ''}")

print(f"\nSaved to {OUTPUT}")
