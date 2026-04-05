import csv
import os
import sys
import time
import requests

MANIFEST = "manifest.tsv"
OUTPUT_DIR = "gdc_data"
API_BASE = "https://api.gdc.cancer.gov/data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(MANIFEST) as f:
    reader = csv.DictReader(f, delimiter="\t")
    entries = list(reader)

total = len(entries)
print(f"Downloading {total} files to {OUTPUT_DIR}/")

for i, entry in enumerate(entries, 1):
    file_id = entry["file_id"]
    file_name = entry["file_name"]
    submitter_id = entry.get("cases.0.submitter_id", "unknown")
    out_path = os.path.join(OUTPUT_DIR, f"{submitter_id}_{file_name}")

    if os.path.exists(out_path):
        print(f"[{i}/{total}] Skipping {submitter_id} (already exists)")
        continue

    for attempt in range(3):
        try:
            r = requests.get(f"{API_BASE}/{file_id}", stream=True, timeout=60)
            r.raise_for_status()
            with open(out_path, "wb") as out:
                for chunk in r.iter_content(chunk_size=8192):
                    out.write(chunk)
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f"[{i}/{total}] {submitter_id} — {size_mb:.1f} MB")
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"[{i}/{total}] FAILED {submitter_id}: {e}", file=sys.stderr)

print("Done.")
