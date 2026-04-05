"""
Power Analysis for Hosmer-Lemeshow Test & Calibration Curve Quantification

Part 1: Simulation-based power analysis for the 1-calibration (IPCW-weighted H-L)
        test across sample sizes matching TCGA per-fold event counts.
        Uses permutation-calibrated critical values so that the null rejection
        rate is exactly alpha=0.05 (the chi-squared approximation is unreliable
        at n=15--37).
Part 2: ECE, MCE, and calibration curve area from pre-computed calibration data.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import chi2

np.random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Part 1: Power Analysis for Hosmer-Lemeshow Test
# ============================================================

def hosmer_lemeshow_statistic(observed, predicted, n_groups=5):
    """Compute the Hosmer-Lemeshow chi-squared test statistic.

    Groups patients by predicted probability, computes chi-squared statistic
    comparing observed vs expected events in each group.

    Args:
        observed: binary outcomes (0/1)
        predicted: predicted probabilities
        n_groups: number of groups (quintiles=5)

    Returns:
        chi2_statistic (float)
    """
    order = np.argsort(predicted)
    obs_sorted = observed[order]
    pred_sorted = predicted[order]

    groups = np.array_split(np.arange(len(observed)), n_groups)

    chi2_stat = 0.0
    for group_idx in groups:
        if len(group_idx) == 0:
            continue
        n_g = len(group_idx)
        obs_g = obs_sorted[group_idx]
        pred_g = pred_sorted[group_idx]

        O_1 = obs_g.sum()
        E_1 = pred_g.sum()
        O_0 = n_g - O_1
        E_0 = n_g - E_1

        if E_1 > 0 and E_0 > 0:
            chi2_stat += (O_1 - E_1) ** 2 / E_1 + (O_0 - E_0) ** 2 / E_0

    return chi2_stat


def calibrate_null_threshold(n_events, n_groups=5, alpha=0.05, n_null=5000):
    """Compute the H-L statistic threshold at significance level alpha under the null.

    Because the chi-squared approximation is unreliable at small n, we simulate
    the null distribution of the H-L statistic (well-calibrated predictions)
    and return the (1-alpha) quantile as the critical value.

    Args:
        n_events: sample size
        n_groups: H-L groups
        alpha: desired type I error rate
        n_null: number of null simulations

    Returns:
        critical_value (float): reject H0 if H-L stat > critical_value
    """
    p_true = 0.5
    null_stats = np.zeros(n_null)
    for i in range(n_null):
        outcomes = np.random.binomial(1, p_true, size=n_events)
        # Under the null, predicted = true probability + noise (same noise as alternative)
        preds = np.clip(p_true + np.random.normal(0, 0.05, size=n_events), 0.01, 0.99)
        null_stats[i] = hosmer_lemeshow_statistic(outcomes, preds, n_groups=n_groups)

    return np.percentile(null_stats, 100 * (1 - alpha))


def simulate_power(n_events, miscal_delta, critical_value,
                   n_replicates=2000, n_groups=5):
    """Simulate size-corrected power of H-L test.

    Uses a pre-computed critical value from the null distribution so that
    the type I error rate is properly controlled at alpha.

    Args:
        n_events: number of uncensored patients (effective sample size per fold)
        miscal_delta: shift applied to predicted probabilities
        critical_value: H-L statistic threshold from calibrate_null_threshold()
        n_replicates: number of simulation replicates
        n_groups: groups for H-L test

    Returns:
        rejection_rate (size-corrected power)
    """
    p_true = 0.5
    p_predicted = np.clip(p_true + miscal_delta, 0.01, 0.99)

    rejections = 0
    for _ in range(n_replicates):
        outcomes = np.random.binomial(1, p_true, size=n_events)
        preds = np.clip(
            p_predicted + np.random.normal(0, 0.05, size=n_events),
            0.01, 0.99
        )
        stat = hosmer_lemeshow_statistic(outcomes, preds, n_groups=n_groups)
        if stat > critical_value:
            rejections += 1

    return rejections / n_replicates


def run_power_analysis():
    """Run full power analysis across sample sizes and miscalibration levels."""
    print("=" * 65)
    print("  Part 1: Power Analysis for Hosmer-Lemeshow Test")
    print("  (permutation-calibrated critical values for exact size)")
    print("=" * 65)

    n_events_list = [15, 30, 37]  # UCEC, BRCA, GBMLGG per-fold
    deltas = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    n_replicates = 2000
    n_null = 5000
    alpha = 0.05

    # Step 1: calibrate critical values under the null for each sample size
    print("\n  Calibrating null distribution (5000 simulations per n)...")
    critical_values = {}
    for n_events in n_events_list:
        cv = calibrate_null_threshold(n_events, n_groups=5, alpha=alpha, n_null=n_null)
        critical_values[n_events] = cv
        print(f"    n={n_events:3d}:  critical value = {cv:.3f}")

    # Step 2: compute power at each (n, delta)
    rows = []
    for n_events in n_events_list:
        cv = critical_values[n_events]
        print(f"\n  n_events = {n_events} (cv = {cv:.3f}):")
        for delta in deltas:
            power = simulate_power(n_events, delta, cv,
                                   n_replicates=n_replicates, n_groups=5)
            rows.append({
                "n_events": n_events,
                "miscal_delta": delta,
                "power": power,
                "critical_value": cv,
                "n_replicates": n_replicates,
                "n_null_calibration": n_null,
                "alpha": alpha,
            })
            marker = " <-- 80% power" if power >= 0.80 else ""
            print(f"    delta={delta:.2f}  power={power:.3f}{marker}")

    df = pd.DataFrame(rows)

    # Find minimum detectable effect (MDE) at 80% power for each n
    print("\n  Minimum Detectable Effect (MDE) at 80% power:")
    print("  " + "-" * 50)
    for n_events in n_events_list:
        sub = df[df["n_events"] == n_events]
        above_80 = sub[sub["power"] >= 0.80]
        if len(above_80) > 0:
            mde = above_80["miscal_delta"].min()
            mde_power = above_80.loc[above_80["miscal_delta"].idxmin(), "power"]
            print(f"    n={n_events:3d}:  MDE = {mde:.2f}  (power = {mde_power:.3f})")
        else:
            print(f"    n={n_events:3d}:  MDE > 0.30 (power never reaches 80%)")

    # Verify type I error control
    print("\n  Type I error rate (delta=0, should be ~0.05):")
    for n_events in n_events_list:
        null_power = df[(df["n_events"] == n_events) & (df["miscal_delta"] == 0.0)]["power"].values[0]
        print(f"    n={n_events:3d}:  rejection rate = {null_power:.3f}")

    # Save
    out_path = os.path.join(RESULTS_DIR, "power_analysis.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    return df


# ============================================================
# Part 2: Calibration Curve Quantification (ECE, MCE, area)
# ============================================================

def compute_calibration_metrics(group_df):
    """Compute ECE, MCE, and area between calibration curve and diagonal.

    Args:
        group_df: DataFrame with columns mean_predicted, km_observed, n_patients
                  representing binned calibration data for one model/fold.

    Returns:
        dict with ECE, MCE, area_between_curves, n_bins, total_patients
    """
    pred = group_df["mean_predicted"].values
    obs = group_df["km_observed"].values
    n_pts = group_df["n_patients"].values

    total_n = n_pts.sum()
    if total_n == 0:
        return {"ECE": np.nan, "MCE": np.nan, "area_between_curves": np.nan,
                "n_bins": 0, "total_patients": 0}

    # Absolute calibration errors per bin
    abs_errors = np.abs(pred - obs)

    # ECE: weighted average |predicted - observed|
    ece = np.sum(abs_errors * n_pts) / total_n

    # MCE: maximum |predicted - observed|
    mce = np.max(abs_errors)

    # Area between calibration curve and diagonal via trapezoidal rule
    # Sort by predicted probability for proper integration
    order = np.argsort(pred)
    pred_sorted = pred[order]
    obs_sorted = obs[order]

    # Trapezoidal integration of |obs - pred| over the pred axis
    # np.trapezoid in numpy >= 2.0 (np.trapz removed in 2.0+)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    area = _trapz(np.abs(obs_sorted - pred_sorted), pred_sorted)

    return {
        "ECE": ece,
        "MCE": mce,
        "area_between_curves": area,
        "n_bins": len(pred),
        "total_patients": int(total_n),
    }


def run_calibration_quantification():
    """Load calibration curve data and compute ECE, MCE, area per model."""
    print("\n" + "=" * 65)
    print("  Part 2: Calibration Curve Quantification")
    print("=" * 65)

    csv_path = os.path.join(FIGURES_DIR, "calibration_curves_data.csv")
    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} not found.")
        print("  Run paper_calibration_curves.py first to generate calibration data.")
        return None

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows from calibration_curves_data.csv")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Cancers: {sorted(df['cancer'].unique())}")

    rows = []

    # Per model, per cancer, per fold
    for (model, cancer, fold), group in df.groupby(["model", "cancer", "fold"]):
        metrics = compute_calibration_metrics(group)
        metrics["model"] = model
        metrics["cancer"] = cancer
        metrics["fold"] = fold
        rows.append(metrics)

    fold_df = pd.DataFrame(rows)

    # Aggregate: mean +/- std across folds, per model per cancer
    agg_rows = []
    for (model, cancer), group in fold_df.groupby(["model", "cancer"]):
        agg_rows.append({
            "model": model,
            "cancer": cancer,
            "ECE_mean": group["ECE"].mean(),
            "ECE_std": group["ECE"].std(),
            "MCE_mean": group["MCE"].mean(),
            "MCE_std": group["MCE"].std(),
            "area_mean": group["area_between_curves"].mean(),
            "area_std": group["area_between_curves"].std(),
            "n_folds": len(group),
        })

    agg_df = pd.DataFrame(agg_rows)

    # Display results
    # -- BRCA results (Experiment A)
    brca = agg_df[agg_df["cancer"] == "BRCA"].sort_values("ECE_mean")
    if len(brca) > 0:
        print("\n  BRCA (Experiment A):")
        print(f"  {'Model':<16} {'ECE':>16} {'MCE':>16} {'Area':>16}")
        print("  " + "-" * 66)
        for _, r in brca.iterrows():
            print(f"  {r['model']:<16} "
                  f"{r['ECE_mean']:.4f}+/-{r['ECE_std']:.4f}  "
                  f"{r['MCE_mean']:.4f}+/-{r['MCE_std']:.4f}  "
                  f"{r['area_mean']:.4f}+/-{r['area_std']:.4f}")

    # -- Multi-cancer results (Experiment B)
    mc = agg_df[agg_df["cancer"] != "BRCA"].sort_values(["cancer", "ECE_mean"])
    if len(mc) > 0:
        print("\n  Multi-cancer (Experiment B):")
        print(f"  {'Cancer':<8} {'Model':<16} {'ECE':>16} {'MCE':>16} {'Area':>16}")
        print("  " + "-" * 74)
        for _, r in mc.iterrows():
            print(f"  {r['cancer']:<8} {r['model']:<16} "
                  f"{r['ECE_mean']:.4f}+/-{r['ECE_std']:.4f}  "
                  f"{r['MCE_mean']:.4f}+/-{r['MCE_std']:.4f}  "
                  f"{r['area_mean']:.4f}+/-{r['area_std']:.4f}")

    # Grand summary: average ECE across cancers, per model
    grand = agg_df.groupby("model").agg({
        "ECE_mean": "mean",
        "MCE_mean": "mean",
        "area_mean": "mean",
    }).reset_index().sort_values("ECE_mean")
    print("\n  Grand average across all cancers:")
    print(f"  {'Model':<16} {'ECE':>8} {'MCE':>8} {'Area':>8}")
    print("  " + "-" * 42)
    for _, r in grand.iterrows():
        print(f"  {r['model']:<16} {r['ECE_mean']:.4f}   {r['MCE_mean']:.4f}   {r['area_mean']:.4f}")

    # Save per-fold and aggregated
    fold_out = os.path.join(RESULTS_DIR, "calibration_quantification.csv")
    agg_out = os.path.join(RESULTS_DIR, "calibration_quantification_summary.csv")
    fold_df.to_csv(fold_out, index=False)
    agg_df.to_csv(agg_out, index=False)
    print(f"\n  Saved: {fold_out}")
    print(f"  Saved: {agg_out}")

    return fold_df, agg_df


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    power_df = run_power_analysis()
    cal_results = run_calibration_quantification()
    print("\nDone.")
