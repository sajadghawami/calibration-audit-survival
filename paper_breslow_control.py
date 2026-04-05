"""
Task 2: Breslow Control — validates that KM-shift survival curve construction
does not itself introduce miscalibration.

Three methods compared on Cox-PH fold 0 BRCA:
A. Lifelines predict_survival_function (proper Breslow)
B. Our KM-shift method (same as used for MCAT/MMP)
C. Random risk scores + KM-shift (should be trivially calibrated)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from sksurv.metrics import concordance_index_censored
from SurvivalEVAL import SurvivalEvaluator

from calibration_audit import risk_to_survival_breslow

warnings.filterwarnings("ignore")

PATIENT_INDEX = "patient_index.csv"
N_TIME_POINTS = 20
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']


def fit_cox_and_split():
    """Fit Cox-PH on fold 0, return model + train/val data."""
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

    return cph, top_genes, train, val


def run_1cal(surv_curves, time_coords, val_times, val_events, train_times, train_events):
    """Run SurvivalEvaluator and return 1-cal p-value, IBS."""
    ev = SurvivalEvaluator(
        pred_survs=surv_curves,
        time_coordinates=time_coords,
        event_times=val_times,
        event_indicators=val_events.astype(int),
        train_event_times=train_times,
        train_event_indicators=train_events.astype(int),
    )
    median_t = float(np.median(val_times[val_events.astype(bool)]))
    onecal = ev.one_calibration(target_time=median_t)[0]
    ibs = ev.integrated_brier_score()
    return onecal, ibs, median_t


def compute_deciles(surv_curves, time_coords, val_times, val_events, t_eval, n_bins=10):
    """Compute decile calibration data."""
    predicted = np.array([np.interp(t_eval, time_coords, c) for c in surv_curves])
    bin_edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (predicted >= lo) & (predicted <= hi)
        else:
            mask = (predicted >= lo) & (predicted < hi)
        if mask.sum() < 2:
            continue
        mean_pred = predicted[mask].mean()
        km = KaplanMeierFitter()
        km.fit(val_times[mask], event_observed=val_events[mask])
        obs = km.predict(t_eval)
        rows.append({"decile": i, "mean_predicted": mean_pred, "km_observed": obs, "n": int(mask.sum())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Fitting Cox-PH on fold 0 BRCA...")
    cph, top_genes, train, val = fit_cox_and_split()

    cox_val = val[top_genes].copy()
    val_times = val["survival_time"].values.astype(float)
    val_events = val["event"].values.astype(int)
    train_times = train["survival_time"].values.astype(float)
    train_events = train["event"].values.astype(int)

    results = {}

    # === Method A: Lifelines proper Breslow ===
    print("Method A: Lifelines predict_survival_function...")
    surv_fn = cph.predict_survival_function(cox_val)
    tc_a = surv_fn.index.values
    max_event_t = val_times[val_events.astype(bool)].max()
    mask_a = tc_a <= max_event_t
    tc_a = tc_a[mask_a]
    curves_a = surv_fn.T.values[:, mask_a]
    onecal_a, ibs_a, t_eval = run_1cal(curves_a, tc_a, val_times, val_events, train_times, train_events)
    print(f"  1-cal p={onecal_a:.4f}, IBS={ibs_a:.4f}")
    results["Lifelines (proper)"] = {"onecal_p": onecal_a, "ibs": ibs_a, "t_eval": t_eval}

    # === Method B: KM-shift (our method) ===
    print("Method B: KM-shift (our approximation)...")
    partial_hazards = cph.predict_partial_hazard(cox_val).values.flatten()
    risks_b = np.log(partial_hazards)
    train_event_times = train_times[train_events > 0]
    tc_b = np.percentile(train_event_times, np.linspace(5, 95, N_TIME_POINTS))
    tc_b = np.sort(np.unique(tc_b))
    curves_b = risk_to_survival_breslow(risks_b, train_times, train_events, tc_b)
    onecal_b, ibs_b, _ = run_1cal(curves_b, tc_b, val_times, val_events, train_times, train_events)
    print(f"  1-cal p={onecal_b:.4f}, IBS={ibs_b:.4f}")
    results["KM-shift (ours)"] = {"onecal_p": onecal_b, "ibs": ibs_b, "t_eval": t_eval}

    # === Method C: Random risks + KM-shift ===
    print("Method C: Random risks + KM-shift...")
    np.random.seed(42)
    random_risks = np.random.randn(len(val))
    curves_c = risk_to_survival_breslow(random_risks, train_times, train_events, tc_b)
    onecal_c, ibs_c, _ = run_1cal(curves_c, tc_b, val_times, val_events, train_times, train_events)
    print(f"  1-cal p={onecal_c:.4f}, IBS={ibs_c:.4f}")
    results["Random + KM-shift"] = {"onecal_p": onecal_c, "ibs": ibs_c, "t_eval": t_eval}

    # === Save CSV ===
    rows = []
    for method, r in results.items():
        rows.append({"method": method, **r})
    pd.DataFrame(rows).to_csv("controls/breslow_validation.csv", index=False)

    # === Compute deciles for figure ===
    dec_a = compute_deciles(curves_a, tc_a, val_times, val_events, t_eval)
    dec_a["method"] = "Lifelines (proper)"
    dec_b = compute_deciles(curves_b, tc_b, val_times, val_events, t_eval)
    dec_b["method"] = "KM-shift (ours)"
    dec_c = compute_deciles(curves_c, tc_b, val_times, val_events, t_eval)
    dec_c["method"] = "Random + KM-shift"

    # === Figure ===
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")

    for i, (dec, label, color) in enumerate([
        (dec_a, f"Lifelines (p={onecal_a:.3f})", COLORS[0]),
        (dec_b, f"KM-shift (p={onecal_b:.3f})", COLORS[1]),
        (dec_c, f"Random (p={onecal_c:.3f})", COLORS[2]),
    ]):
        if not dec.empty:
            ax.plot(dec["mean_predicted"], dec["km_observed"], "o-", color=color,
                    ms=4, lw=1.2, label=label)

    ax.set_xlabel("Predicted survival probability", fontsize=12)
    ax.set_ylabel("Observed survival proportion", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right")
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig("figures/breslow_control.pdf", dpi=300, bbox_inches="tight")
    fig.savefig("figures/breslow_control.png", dpi=300, bbox_inches="tight")
    plt.close()

    # === Summary ===
    print(f"\n{'='*60}")
    print(f"  Breslow Control Summary")
    print(f"{'='*60}")
    print(f"  {'Method':<25} {'1-cal p':>10} {'IBS':>10}")
    print(f"  {'-'*47}")
    for method, r in results.items():
        status = "PASS" if r["onecal_p"] > 0.05 else "FAIL"
        print(f"  {method:<25} {r['onecal_p']:>9.4f} {status} {r['ibs']:>8.4f}")

    print(f"\n  t_eval = {t_eval:.1f} days")
    print(f"\n  Saved: figures/breslow_control.{{pdf,png}}")
    print(f"  Saved: controls/breslow_validation.csv")
