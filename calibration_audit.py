"""
Calibration Audit for Multimodal Survival Models (v2 — per-fold, fixed)

Fixes from validation audit:
1. Per-fold evaluation (no pooling artifact)
2. Interpolated survival curves for SurvPath (20 time points, not 5)
3. Training data used for IPCW weights and Breslow baseline
4. Proper positive/negative controls
"""

import pickle
import glob
import warnings
import numpy as np
import pandas as pd
import torch
from lifelines import KaplanMeierFitter
from sksurv.metrics import concordance_index_censored
from SurvivalEVAL import SurvivalEvaluator

warnings.filterwarnings("ignore")

PATIENT_INDEX = "patient_index.csv"
N_TIME_POINTS = 20


# ============================================================
# Training data loaders (for IPCW weights + Breslow baseline)
# ============================================================
def load_survpath_train_data(fold):
    """Load training fold survival data for SurvPath/MCAT (same splits)."""
    split = pd.read_csv(f"SurvPath/splits/5foldcv/tcga_brca/splits_{fold}.csv")
    train_ids = set(split["train"].dropna().replace("", pd.NA).dropna())
    pi = pd.read_csv(PATIENT_INDEX)
    train = pi[pi["patient_id"].isin(train_ids)]
    # SurvPath uses months
    return train["survival_time"].values / 30.44, train["event"].values.astype(int)


def load_mmp_train_data(fold):
    """Load training fold survival data for MMP (its own splits)."""
    train = pd.read_csv(f"MMP/src/splits/survival/TCGA_BRCA_overall_survival_k={fold}/train.csv")
    # MMP uses days in os_survival_days
    return train["os_survival_days"].values, (1 - train["os_censorship"].values).astype(int)


# ============================================================
# Survival curve construction
# ============================================================
def logits_to_survival_interpolated(logits, train_event_times, n_points=N_TIME_POINTS):
    """Convert 4-bin hazard logits to interpolated survival curve with n_points."""
    hazards = 1.0 / (1.0 + np.exp(-logits))
    coarse_surv = np.concatenate([[1.0], np.cumprod(1 - hazards)])

    # Bin edges from TRAINING data quartiles
    bin_edges = np.percentile(train_event_times[train_event_times > 0], [25, 50, 75, 100])
    coarse_times = np.concatenate([[0], bin_edges])

    # Interpolate to finer grid
    fine_times = np.linspace(coarse_times[0], coarse_times[-1], n_points)
    fine_surv = np.interp(fine_times, coarse_times, coarse_surv)
    # Ensure monotonically non-increasing
    fine_surv = np.minimum.accumulate(fine_surv)

    return fine_surv, fine_times


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


# ============================================================
# Per-fold result loaders
# ============================================================
def load_survpath_fold(fold):
    base = glob.glob("SurvPath/results/results/brca_run1/tcga_brca*")[0]
    with open(f"{base}/split_{fold}_results.pkl", "rb") as f:
        results = pickle.load(f)
    records = []
    for pid, data in results.items():
        records.append({
            "patient_id": pid,
            "risk": data["risk"],
            "time": data["time"],
            "event": 1 - data["censorship"],
            "logits": data["logits"],
        })
    return pd.DataFrame(records)


def load_mcat_fold(fold):
    base = "MCAT/results/brca_run1/5foldcv/MCAT_nll_surv_a0.0_5foldcv_gc32_concat/tcga_brca_MCAT_nll_surv_a0.0_5foldcv_gc32_concat_s42"
    with open(f"{base}/split_latest_val_{fold}_results.pkl", "rb") as f:
        results = pickle.load(f)
    records = []
    for pid, data in results.items():
        records.append({
            "patient_id": pid,
            "risk": data["risk"],
            "time": data["survival"],
            "event": 1 - data["censorship"],
        })
    return pd.DataFrame(records)


def load_mmp_fold(fold):
    pkls = glob.glob(f"MMP/results/brca_run1/tcga_brca/k={fold}/*/*/test_results.pkl")
    if not pkls:
        return pd.DataFrame()
    with open(pkls[0], "rb") as f:
        results = pickle.load(f)
    records = []
    risks = results["all_risk_scores"]
    censorships = results["all_censorships"]
    times = results["all_event_times"]
    ids = results.get("sample_ids", list(range(len(risks))))
    for i in range(len(risks)):
        records.append({
            "patient_id": ids[i] if isinstance(ids[i], str) else f"fold{fold}_p{i}",
            "risk": float(risks[i]),
            "time": float(times[i]),
            "event": 1 - float(censorships[i]),
        })
    return pd.DataFrame(records)


# ============================================================
# Evaluate one fold
# ============================================================
def evaluate_fold(val_df, train_times, train_events, has_logits=False):
    """Run calibration metrics for a single fold."""
    val_times = val_df["time"].values
    val_events = val_df["event"].values.astype(bool)
    val_risks = val_df["risk"].values

    event_times_only = val_times[val_events]
    if len(event_times_only) < 5:
        return {"ci": np.nan, "dcal": np.nan, "onecal": np.nan, "ibs": np.nan}

    # Build survival curves
    if has_logits:
        # Use training event times for bin edges
        train_event_times = train_times[train_events > 0]
        curves_list = []
        for _, row in val_df.iterrows():
            surv, fine_times = logits_to_survival_interpolated(row["logits"], train_event_times)
            curves_list.append(surv)
        surv_curves = np.array(curves_list)
        time_coords = fine_times
    else:
        # Time coordinates from training event times
        train_event_times = train_times[train_events > 0]
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

    return {"ci": ci, "dcal": dcal, "onecal": onecal, "ibs": ibs}


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    models = {
        "SurvPath": {"loader": load_survpath_fold, "train_loader": load_survpath_train_data, "has_logits": True},
        "MCAT": {"loader": load_mcat_fold, "train_loader": load_survpath_train_data, "has_logits": False},
        "MMP": {"loader": load_mmp_fold, "train_loader": load_mmp_train_data, "has_logits": False},
    }

    print(f"{'Model':<12} {'Fold':>4} {'C-idx':>7} {'D-cal p':>8} {'1-cal p':>8} {'IBS':>7}")
    print("-" * 52)

    summary = {}

    for model_name, cfg in models.items():
        fold_results = []
        n_folds = 5

        for fold in range(n_folds):
            val_df = cfg["loader"](fold)
            if val_df.empty:
                continue
            train_times, train_events = cfg["train_loader"](fold)
            result = evaluate_fold(val_df, train_times, train_events, has_logits=cfg["has_logits"])
            fold_results.append(result)

            status_d = "PASS" if result["dcal"] > 0.05 else "FAIL"
            status_1 = "PASS" if result["onecal"] > 0.05 else "FAIL"
            print(f"{model_name:<12} {fold:>4} {result['ci']:>7.4f} {result['dcal']:>7.4f} {status_d} {result['onecal']:>7.4f} {status_1} {result['ibs']:>7.4f}")

        # Aggregate
        if fold_results:
            cis = [r["ci"] for r in fold_results]
            dcals = [r["dcal"] for r in fold_results if not np.isnan(r["dcal"])]
            onecals = [r["onecal"] for r in fold_results if not np.isnan(r["onecal"])]
            ibss = [r["ibs"] for r in fold_results if not np.isnan(r["ibs"])]

            summary[model_name] = {
                "ci": f"{np.mean(cis):.4f}±{np.std(cis):.4f}",
                "dcal_mean": np.mean(dcals),
                "dcal_fail": sum(1 for d in dcals if d < 0.05),
                "onecal_mean": np.mean(onecals),
                "onecal_fail": sum(1 for o in onecals if o < 0.05),
                "ibs": f"{np.mean(ibss):.4f}±{np.std(ibss):.4f}",
            }
        print()

    # Summary
    print("=" * 60)
    print("  SUMMARY — Calibration Audit (per-fold, corrected)")
    print("=" * 60)
    print(f"{'Model':<12} {'C-index':>14} {'D-cal fail':>10} {'1-cal fail':>10} {'IBS':>14}")
    print("-" * 62)
    for name, s in summary.items():
        print(f"{name:<12} {s['ci']:>14} {s['dcal_fail']:>5}/5      {s['onecal_fail']:>5}/5      {s['ibs']:>14}")

    # Controls
    print("\n" + "=" * 60)
    print("  CONTROLS")
    print("=" * 60)

    # Positive control: Cox-PH
    from lifelines import CoxPHFitter
    rna = pd.read_csv("SurvPath/data/rna_clean.csv", index_col=0)
    top_genes = rna.var(axis=0).sort_values(ascending=False).head(20).index.tolist()
    pi = pd.read_csv(PATIENT_INDEX)

    split = pd.read_csv("SurvPath/splits/5foldcv/tcga_brca/splits_0.csv")
    train_ids = set(split["train"].dropna().replace("", pd.NA).dropna())
    val_ids = set(split["val"].dropna().replace("", pd.NA).dropna())

    rna_sub = rna[top_genes].copy()
    rna_sub["patient_id"] = rna_sub.index
    rna_sub = rna_sub.reset_index(drop=True)
    merged = pi.merge(rna_sub, on="patient_id")
    train = merged[merged["patient_id"].isin(train_ids)]
    val = merged[merged["patient_id"].isin(val_ids)]

    cox_train = train[top_genes + ["survival_time", "event"]].copy()
    cox_train["survival_time"] = cox_train["survival_time"].astype(float)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_train, duration_col="survival_time", event_col="event")

    cox_val = val[top_genes].copy()
    surv_fn = cph.predict_survival_function(cox_val)
    tc = surv_fn.index.values
    max_event_t = val[val["event"] == 1]["survival_time"].max()
    mask = tc <= max_event_t
    tc_f = tc[mask]
    ps_f = surv_fn.T.values[:, mask]

    val_times = val["survival_time"].values.astype(float)
    val_events = val["event"].values.astype(int).astype(bool)
    train_times = train["survival_time"].values.astype(float)
    train_events = train["event"].values.astype(int)

    ev = SurvivalEvaluator(pred_survs=ps_f, time_coordinates=tc_f, event_times=val_times,
                           event_indicators=val_events.astype(int), train_event_times=train_times,
                           train_event_indicators=train_events)
    cox_dcal = ev.d_calibration()[0]
    cox_1cal = ev.one_calibration(target_time=float(np.median(val_times[val_events])))[0]
    print(f"  Cox-PH (positive): D-cal p={cox_dcal:.4f} {'PASS' if cox_dcal > 0.05 else 'FAIL'}, "
          f"1-cal p={cox_1cal:.4f} {'PASS' if cox_1cal > 0.05 else 'FAIL'}")

    # Negative control: shuffled SurvPath fold 0
    sp0 = load_survpath_fold(0)
    train_t, train_e = load_survpath_train_data(0)
    train_et = train_t[train_e > 0]

    logits_all = np.array(list(sp0["logits"]))
    curves = []
    for l in logits_all:
        s, ft = logits_to_survival_interpolated(l, train_et)
        curves.append(s)
    curves = np.array(curves)

    np.random.seed(42)
    shuffled = curves[np.random.permutation(len(curves))]

    val_t = sp0["time"].values
    val_e = sp0["event"].values.astype(bool)

    for label, c in [("Original", curves), ("Shuffled", shuffled)]:
        ev2 = SurvivalEvaluator(pred_survs=c, time_coordinates=ft, event_times=val_t,
                                event_indicators=val_e.astype(int), train_event_times=train_t,
                                train_event_indicators=train_e)
        d = ev2.d_calibration()[0]
        o = ev2.one_calibration(target_time=float(np.median(val_t[val_e])))[0]
        ibs = ev2.integrated_brier_score()
        print(f"  {label:>10}: D-cal p={d:.4f} {'PASS' if d > 0.05 else 'FAIL'}, "
              f"1-cal p={o:.4f} {'PASS' if o > 0.05 else 'FAIL'}, IBS={ibs:.4f}")
