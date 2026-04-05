from huggingface_hub import hf_hub_download

REPO = "MahmoodLab/UNI2-h-features"
LOCAL_DIR = "uni2_features"

files = [
    "TCGA/TCGA-BRCA_IDC.tar.gz",
    "TCGA/TCGA-BRCA_OTHERS.tar.gz",
]

for f in files:
    print(f"Downloading {f}...")
    path = hf_hub_download(
        repo_id=REPO,
        filename=f,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
    )
    print(f"  Saved to {path}")

print("Done. Extract with: tar -xzf <file>")
