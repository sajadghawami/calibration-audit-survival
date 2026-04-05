import requests
import json
import csv

# Step 1: Fetch clinical data from GDC API
filters = {
    "op": "=",
    "content": {"field": "project.project_id", "value": "TCGA-BRCA"},
}

params = {
    "filters": json.dumps(filters),
    "fields": "submitter_id,demographic.vital_status,demographic.days_to_death,diagnoses.days_to_last_follow_up",
    "size": "2000",
    "format": "JSON",
}

print("Fetching clinical data from GDC API...")
response = requests.get("https://api.gdc.cancer.gov/cases", params=params)
response.raise_for_status()

hits = response.json()["data"]["hits"]
print(f"Got {len(hits)} cases")

with open("clinical.json", "w") as f:
    json.dump(hits, f, indent=2)

# Step 2: Parse survival labels
rows = []
for case in hits:
    pid = case["submitter_id"]
    demo = case.get("demographic", {})
    vital = demo.get("vital_status", None)
    days_death = demo.get("days_to_death", None)

    diagnoses = case.get("diagnoses", [{}])
    days_followup = diagnoses[0].get("days_to_last_follow_up", None) if diagnoses else None

    if vital == "Dead" and days_death is not None:
        survival_time = days_death
        event = 1
    elif days_followup is not None:
        survival_time = days_followup
        event = 0
    else:
        survival_time = None
        event = None

    rows.append({
        "patient_id": pid,
        "survival_time": survival_time,
        "event": event,
        "vital_status": vital,
    })

# Drop missing or invalid (negative survival time)
valid = [r for r in rows if r["survival_time"] is not None and r["event"] is not None and r["survival_time"] > 0]
print(f"{len(valid)} patients with valid survival labels ({len(rows) - len(valid)} dropped)")

with open("survival_labels.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["patient_id", "survival_time", "event", "vital_status"])
    writer.writeheader()
    writer.writerows(valid)

# Summary stats
times = [r["survival_time"] for r in valid]
events = [r["event"] for r in valid]
times.sort()
print(f"\nSurvival stats:")
print(f"  Median time: {times[len(times)//2]:.0f} days")
print(f"  Range: {min(times):.0f} — {max(times):.0f} days")
print(f"  Deaths: {sum(events)} ({100*sum(events)/len(events):.1f}%)")
print(f"  Censored: {len(events)-sum(events)} ({100*(len(events)-sum(events))/len(events):.1f}%)")

print(f"\nSaved to survival_labels.csv")
