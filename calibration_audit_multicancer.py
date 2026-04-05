"""
Multi-Cancer Calibration Audit — MCAT ICCV Pre-computed Results

Extends calibration_audit.py to evaluate MCAT ICCV results across:
- 5 cancer types: BLCA, BRCA, GBMLGG, LUAD, UCEC
- 11 architectures: MCATsm, AMILsm (x3), DSsm (x3), MIFCNsm (x3), SNNsm

Uses per-fold evaluation with training data for IPCW weights (validated methodology).
"""

import pickle
import zipfile
import warnings
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sksurv.metrics import concordance_index_censored
from SurvivalEVAL import SurvivalEvaluator

warnings.filterwarnings("ignore")

N_TIME_POINTS = 20

ARCHITECTURES = [
    "MCATsm_nll_surv_a0.0_5foldcv_gc32_concat",
    "AMILsm_nll_surv_a0.0_5foldcv_gc32_sig",
    "AMILsm_nll_surv_a0.0_5foldcv_gc32_sig_bilinear",
    "AMILsm_nll_surv_a0.0_5foldcv_gc32_sig_concat",
    "DSsm_nll_surv_a0.0_5foldcv_gc32_sig",
    "DSsm_nll_surv_a0.0_5foldcv_gc32_sig_bilinear",
    "DSsm_nll_surv_a0.0_5foldcv_gc32_sig_concat",
    "MIFCNsm_nll_surv_a0.0_5foldcv_gc32_sig",
    "MIFCNsm_nll_surv_a0.0_5foldcv_gc32_sig_bilinear",
    "MIFCNsm_nll_surv_a0.0_5foldcv_gc32_sig_concat",
    "SNNsm_nll_surv_a0.0_reg1e-04_5foldcv_gc32_sig",
]

CANCER_TYPES = ["blca", "brca", "gbmlgg", "luad", "ucec"]

# Short names for display
ARCH_SHORT = {
    "MCATsm_nll_surv_a0.0_5foldcv_gc32_concat": "MCAT",
    "AMILsm_nll_surv_a0.0_5foldcv_gc32_sig": "AMIL-sig",
    "AMILsm_nll_surv_a0.0_5foldcv_gc32_sig_bilinear": "AMIL-bilin",
    "AMILsm_nll_surv_a0.0_5foldcv_gc32_sig_concat": "AMIL-cat",
    "DSsm_nll_surv_a0.0_5foldcv_gc32_sig": "DS-sig",
    "DSsm_nll_surv_a0.0_5foldcv_gc32_sig_bilinear": "DS-bilin",
    "DSsm_nll_surv_a0.0_5foldcv_gc32_sig_concat": "DS-cat",
    "MIFCNsm_nll_surv_a0.0_5foldcv_gc32_sig": "MIFCN-sig",
    "MIFCNsm_nll_surv_a0.0_5foldcv_gc32_sig_bilinear": "MIFCN-bilin",
    "MIFCNsm_nll_surv_a0.0_5foldcv_gc32_sig_concat": "MIFCN-cat",
    "SNNsm_nll_surv_a0.0_reg1e-04_5foldcv_gc32_sig": "SNN",
}

# Cache for dataset CSVs (avoid re-reading zips)
_csv_cache = {}


def _load_dataset_csv(cancer_type):
    """Load and cache the MCAT dataset CSV for a cancer type."""
    if cancer_type not in _csv_cache:
        zip_path = f"MCAT/dataset_csv/tcga_{cancer_type}_all_clean.csv.zip"
        with zipfile.ZipFile(zip_path) as z:
            df = pd.read_csv(z.open(z.namelist()[0]))
        # Deduplicate by case_id (multiple slides per patient)
        df = df.drop_duplicates(subset="case_id", keep="first")
        _csv_cache[cancer_type] = df
    return _csv_cache[cancer_type]


def load_iccv_train_data(fold, cancer_type):
    """Load training fold survival data from MCAT splits + dataset CSV."""
    split = pd.read_csv(f"MCAT/splits/5foldcv/tcga_{cancer_type}/splits_{fold}.csv")
    train_ids = set(split["train"].dropna())
    df = _load_dataset_csv(cancer_type)
    train = df[df["case_id"].isin(train_ids)]
    # MCAT convention: censorship=0 means event, censorship=1 means censored
    times = train["survival_months"].values.astype(float)
    events = (1 - train["censorship"].values).astype(int)
    return times, events


def load_iccv_fold(fold, architecture, cancer_type):
    """Load ICCV prediction results for a single fold."""
    base = f"MCAT/results/ICCV/{architecture}/tcga_{cancer_type}_{architecture}_s1"
    pkl_path = f"{base}/split_latest_val_{fold}_results.pkl"
    try:
        with open(pkl_path, "rb") as f:
            results = pickle.load(f)
    except FileNotFoundError:
        return pd.DataFrame()

    records = []
    for pid, data in results.items():
        records.append({
            "patient_id": pid,
            "risk": float(data["risk"]),
            "time": float(data["survival"]),      # months
            "event": 1 - float(data["censorship"]),  # 1=dead, 0=censored
        })
    return pd.DataFrame(records)


def risk_to_survival_breslow(risks, train_times, train_events, time_coords):
    """Convert risk scores to survival curves using Breslow baseline from TRAINING data."""
    km = KaplanMeierFitter()
    km.fit(train_times, event_observed=train_events)
    baseline_surv = np.array([km.predict(t) for t in time_coords])
    median_risk = np.median(risks)
    surv_curves = []
    for r in risks:
        hr = np.exp(r - median_risk)
        patient_surv = np.clip(np.power(baseline_surv, hr), 0, 1)
        surv_curves.append(patient_surv)
    return np.array(surv_curves)


def evaluate_fold(val_df, train_times, train_events):
    """Run calibration metrics for a single fold (risk-score models, no logits)."""
    val_times = val_df["time"].values
    val_events = val_df["event"].values.astype(bool)
    val_risks = val_df["risk"].values

    event_times_only = val_times[val_events]
    if len(event_times_only) < 5:
        return {"ci": np.nan, "dcal": np.nan, "onecal": np.nan, "ibs": np.nan, "n_val": len(val_df), "n_events": int(val_events.sum())}

    # Time coordinates from training event times
    train_event_times = train_times[train_events > 0]
    if len(train_event_times) < 5:
        return {"ci": np.nan, "dcal": np.nan, "onecal": np.nan, "ibs": np.nan, "n_val": len(val_df), "n_events": int(val_events.sum())}

    time_coords = np.percentile(train_event_times, np.linspace(5, 95, N_TIME_POINTS))
    time_coords = np.sort(np.unique(time_coords))

    # Breslow baseline from training data
    surv_curves = risk_to_survival_breslow(val_risks, train_times, train_events, time_coords)

    # C-index
    ci = concordance_index_censored(val_events, val_times, val_risks)[0]

    # SurvivalEVAL
    evaluator = SurvivalEvaluator(
        pred_survs=surv_curves,
        time_coordinates=time_coords,
        event_times=val_times,
        event_indicators=val_events.astype(int),
        train_event_times=train_times,
        train_event_indicators=train_events.astype(int),
    )

    try:
        dcal = evaluator.d_calibration()[0]
    except Exception:
        dcal = np.nan
    try:
        median_t = float(np.median(event_times_only))
        onecal = evaluator.one_calibration(target_time=median_t)[0]
    except Exception:
        onecal = np.nan
    try:
        ibs = evaluator.integrated_brier_score()
    except Exception:
        ibs = np.nan

    return {"ci": ci, "dcal": dcal, "onecal": onecal, "ibs": ibs, "n_val": len(val_df), "n_events": int(val_events.sum())}


if __name__ == "__main__":
    all_results = {}

    # ============================================================
    # Per-architecture, per-cancer evaluation
    # ============================================================
    for arch in ARCHITECTURES:
        short = ARCH_SHORT[arch]
        print(f"\n{'='*70}")
        print(f"  {short} ({arch})")
        print(f"{'='*70}")

        for cancer in CANCER_TYPES:
            print(f"\n  {cancer.upper()}")
            print(f"  {'Fold':>4} {'N':>5} {'Evts':>5} {'C-idx':>7} {'1-cal p':>8} {'IBS':>7}")
            print(f"  {'-'*42}")

            fold_results = []
            for fold in range(5):
                val_df = load_iccv_fold(fold, arch, cancer)
                if val_df.empty:
                    continue
                train_t, train_e = load_iccv_train_data(fold, cancer)
                result = evaluate_fold(val_df, train_t, train_e)
                fold_results.append(result)

                status = "PASS" if result["onecal"] > 0.05 else "FAIL"
                print(f"  {fold:>4} {result['n_val']:>5} {result['n_events']:>5} "
                      f"{result['ci']:>7.3f} {result['onecal']:>7.4f} {status} "
                      f"{result['ibs']:>7.3f}")

            if fold_results:
                key = (short, cancer.upper())
                cis = [r["ci"] for r in fold_results if not np.isnan(r["ci"])]
                onecals = [r["onecal"] for r in fold_results if not np.isnan(r["onecal"])]
                ibss = [r["ibs"] for r in fold_results if not np.isnan(r["ibs"])]
                all_results[key] = {
                    "ci_mean": np.mean(cis) if cis else np.nan,
                    "ci_std": np.std(cis) if cis else np.nan,
                    "onecal_fail": sum(1 for o in onecals if o < 0.05),
                    "onecal_total": len(onecals),
                    "ibs_mean": np.mean(ibss) if ibss else np.nan,
                    "ibs_std": np.std(ibss) if ibss else np.nan,
                }

    # ============================================================
    # Summary table
    # ============================================================
    print(f"\n\n{'='*90}")
    print(f"  SUMMARY — Multi-Cancer 1-Calibration Audit (MCAT ICCV pre-computed)")
    print(f"{'='*90}")
    print(f"\n{'Model':<14} {'Cancer':<8} {'C-index':>14} {'1-cal fail':>10} {'IBS':>14}")
    print(f"{'-'*62}")
    for (model, cancer), s in sorted(all_results.items()):
        ci_str = f"{s['ci_mean']:.3f}+/-{s['ci_std']:.3f}"
        ibs_str = f"{s['ibs_mean']:.3f}+/-{s['ibs_std']:.3f}" if not np.isnan(s['ibs_mean']) else "N/A"
        print(f"{model:<14} {cancer:<8} {ci_str:>14} {s['onecal_fail']:>5}/{s['onecal_total']}    {ibs_str:>14}")

    # ============================================================
    # Cross-cancer summary (MCAT only — main comparison)
    # ============================================================
    print(f"\n\n{'='*70}")
    print(f"  MCAT Cross-Cancer Summary")
    print(f"{'='*70}")
    print(f"\n{'Cancer':<10} {'N (avg)':>8} {'C-index':>14} {'1-cal fail':>10} {'IBS':>14}")
    print(f"{'-'*58}")
    for cancer in CANCER_TYPES:
        key = ("MCAT", cancer.upper())
        if key in all_results:
            s = all_results[key]
            ci_str = f"{s['ci_mean']:.3f}+/-{s['ci_std']:.3f}"
            ibs_str = f"{s['ibs_mean']:.3f}+/-{s['ibs_std']:.3f}"
            print(f"{cancer.upper():<10} {'':>8} {ci_str:>14} {s['onecal_fail']:>5}/{s['onecal_total']}    {ibs_str:>14}")

    # ============================================================
    # All-architecture miscalibration rate
    # ============================================================
    total_configs = len(all_results)
    total_failing = sum(1 for s in all_results.values() if s["onecal_fail"] >= 3)
    total_folds = sum(s["onecal_total"] for s in all_results.values())
    total_fold_fails = sum(s["onecal_fail"] for s in all_results.values())

    print(f"\n\n{'='*70}")
    print(f"  Overall Miscalibration Rate")
    print(f"{'='*70}")
    print(f"  Configurations evaluated: {total_configs}")
    print(f"  Configs with majority folds failing 1-cal: {total_failing}/{total_configs} ({100*total_failing/total_configs:.0f}%)")
    print(f"  Individual fold failures: {total_fold_fails}/{total_folds} ({100*total_fold_fails/total_folds:.0f}%)")
