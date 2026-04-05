import requests
import json

# Get RNA-seq TPM files for TCGA-BRCA — no auth needed
filters = {
    "op": "and",
    "content": [
        {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
        {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
        {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
        {"op": "=", "content": {"field": "access", "value": "open"}},
    ],
}

params = {
    "filters": json.dumps(filters),
    "fields": "file_id,file_name,cases.submitter_id",
    "format": "TSV",
    "size": "2000",
}

print("Querying GDC API for TCGA-BRCA RNA-seq files...")
response = requests.get("https://api.gdc.cancer.gov/files", params=params)
response.raise_for_status()

with open("manifest.tsv", "w") as f:
    f.write(response.text)

lines = response.text.strip().split("\n")
print(f"Got {len(lines) - 1} files. Saved to manifest.tsv")
