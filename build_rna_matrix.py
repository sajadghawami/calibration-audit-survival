import pandas as pd
import numpy as np

df = pd.read_csv("patient_index.csv")

all_rna = {}
for i, row in df.iterrows():
    rna = pd.read_csv(
        row["rna_seq_path"],
        sep="\t",
        comment="#",
        index_col=1,  # gene_name column
    )
    # Skip QC rows (N_unmapped, etc.) and keep only tpm_unstranded
    rna = rna[rna["gene_type"].notna()]
    tpm = rna["tpm_unstranded"]
    # Deduplicate gene names (keep first occurrence)
    tpm = tpm[~tpm.index.duplicated(keep="first")]
    all_rna[row["patient_id"]] = tpm

    if (i + 1) % 100 == 0:
        print(f"[{i+1}/{len(df)}] read")

rna_matrix = pd.DataFrame(all_rna).T  # (n_patients, n_genes)
rna_matrix.index.name = "case_id"

# Log1p transform (standard for TPM)
rna_matrix = np.log1p(rna_matrix)

rna_matrix.to_csv("SurvPath/data/rna_clean.csv")
print(f"RNA matrix: {rna_matrix.shape}")
