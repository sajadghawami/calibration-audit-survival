"""
Multi-Horizon Calibration Analysis

Extends the single-horizon 1-calibration audit to:
1. Multiple clinically relevant time horizons (1, 3, 5 years + median)
2. MACE (Mean Absolute Calibration Error) — continuous calibration metric
3. Censoring-aware effective sample size (n_eff) per fold
4. Platt scaling transfer: scaler fit at median, evaluated at other horizons

Uses existing code patterns from calibration_audit.py and paper_recalibration_v2.py.
"""

import warnings
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sklearn.linear_model import LogisticRegression
from SurvivalEVAL import SurvivalEvaluator
from statsmodels.stats.multitest import multipletests

from calibration_audit import (
    logits_to_survival_interpolated, risk_to_survival_breslow,
    load_survpath_fold, load_mcat_fold, load_mmp_fold,
    load_survpath_train_data, load_mmp_train_data,
    N_TIME_POINTS,
)

warnings.filterwarnings("ignore")

# ============================================================
# Time horizons per model (different time units)
# ============================================================
# SurvPath and MCAT operate in months; MMP operates in days
HORIZONS_MONTHS = {
    "1yr": 12.0,
    "3yr": 36.0,
    "5yr": 60.0,
}

HORIZONS_DAYS = {
    "1yr": 365.25,
    "3yr": 1095.75,
    "5yr": 1826.25,
}


def get_horizons(model_name):
    """Return named time horizons dict in the correct units for the model."""
    if model_name == "MMP":
        return HORIZONS_DAYS
    return HORIZONS_MONTHS


# ============================================================
# Survival curve builders (copied from paper_recalibration_v2)
# ============================================================
def build_survival_curves(val_df, train_times, train_events, has_logits):
    """Build full survival curves for validation patients.
    Returns (surv_curves, time_coords)."""
    if has_logits:
        train_event_times = train_times[train_events > 0]
        curves_list = []
        for _, row in val_df.iterrows():
            surv, fine_times = logits_to_survival_interpolated(
                row["logits"], train_event_times
            )
            curves_list.append(surv)
        return np.array(curves_list), fine_times
    else:
        train_event_times = train_times[train_events > 0]
        time_coords = np.percentile(
            train_event_times, np.linspace(5, 95, N_TIME_POINTS)
        )
        time_coords = np.sort(np.unique(time_coords))
        curves = risk_to_survival_breslow(
            val_df["risk"].values, train_times, train_events, time_coords
        )
        return curves, time_coords


# ============================================================
# 1. Multi-horizon 1-calibration
# ============================================================
def evaluate_one_cal_at_horizon(surv_curves, time_coords, val_times,
                                val_events, train_times, train_events,
                                target_time):
    """Run SurvivalEVAL one_calibration at a specific time horizon.
    Returns (p_value, observed_probs, expected_probs)."""
    evaluator = SurvivalEvaluator(
        pred_survs=surv_curves,
        time_coordinates=time_coords,
        event_times=val_times,
        event_indicators=val_events.astype(int),
        train_event_times=train_times,
        train_event_indicators=train_events.astype(int),
    )
    try:
        result = evaluator.one_calibration(target_time=target_time)
        # SurvivalEvaluator.one_calibration returns (p_value, observed, expected)
        p_val = result[0]
        obs = result[1]
        exp = result[2]
        return p_val, obs, exp
    except Exception as e:
        return np.nan, [], []


# ============================================================
# 2. MACE (Mean Absolute Calibration Error)
# ============================================================
def compute_mace(surv_curves, time_coords, val_times, val_events,
                 target_time, n_bins=10):
    """Compute Mean Absolute Calibration Error at a specific time horizon.

    Bins predicted S(t) into n_bins equal-frequency groups.
    For each group, computes |mean predicted S(t) - KM estimate of S(t)|.
    Averages across groups, weighted by group size.

    Returns (mace, per_bin_details_dict).
    """
    # Get predicted survival probabilities at target_time
    pred_st = np.array([np.interp(target_time, time_coords, c) for c in surv_curves])

    # Sort by predicted probability and bin into equal-frequency groups
    sorted_idx = np.argsort(pred_st)
    bins = np.array_split(sorted_idx, n_bins)

    total_n = 0
    weighted_abs_err = 0.0
    bin_details = []

    for b, bin_idx in enumerate(bins):
        if len(bin_idx) == 0:
            continue

        bin_pred = pred_st[bin_idx]
        bin_times = val_times[bin_idx]
        bin_events = val_events[bin_idx].astype(int)

        mean_pred = np.mean(bin_pred)

        # Observed: KM estimate of S(target_time) in this bin
        km = KaplanMeierFitter()
        km.fit(bin_times, event_observed=bin_events)
        observed_st = km.predict(target_time)

        abs_err = abs(mean_pred - observed_st)
        n_bin = len(bin_idx)
        weighted_abs_err += abs_err * n_bin
        total_n += n_bin

        bin_details.append({
            "bin": b,
            "n": n_bin,
            "mean_pred_St": mean_pred,
            "km_observed_St": observed_st,
            "abs_error": abs_err,
        })

    mace = weighted_abs_err / total_n if total_n > 0 else np.nan
    return mace, bin_details


# ============================================================
# 3. Censoring-aware effective sample size
# ============================================================
def compute_censoring_neff(val_times, val_events, train_times, train_events,
                           target_time):
    """Compute IPCW-style effective sample size for a 1-calibration evaluation.

    The DN method used by SurvivalEVAL relies on per-bin KM estimates. The
    reliability of these estimates degrades with censoring. We compute the
    IPCW effective sample size to quantify how many "effective" observations
    the calibration test has:

        n_eff = (sum w_i)^2 / sum(w_i^2)

    where w_i = 1/G_hat(min(T_i, target_time)^-) for each patient, and
    G_hat is the KM estimate of the censoring distribution from training data.

    Also returns the nominal number of patients and events at risk.
    """
    # Fit censoring distribution from training data
    km_cens = KaplanMeierFitter()
    km_cens.fit(train_times, event_observed=(1 - train_events))

    eps = 1e-6
    n_total = len(val_times)
    events_bool = val_events.astype(bool)

    # For each patient, compute IPCW weight at min(T_i, target_time)
    eval_times = np.minimum(val_times, target_time)
    g_values = km_cens.predict(eval_times - eps).values
    g_values = np.maximum(g_values, 0.1)  # floor to prevent explosion
    weights = 1.0 / g_values

    n_eff = (np.sum(weights)) ** 2 / np.sum(weights ** 2)

    # Count events before target_time
    n_events_before_t = int(np.sum(events_bool & (val_times <= target_time)))
    # Count patients at risk at target_time (alive and not censored before t)
    n_at_risk = int(np.sum(val_times >= target_time))

    return {
        "n_total": n_total,
        "n_events_before_t": n_events_before_t,
        "n_at_risk_at_t": n_at_risk,
        "n_eff": n_eff,
        "max_weight": float(np.max(weights)),
        "weight_cv": float(np.std(weights) / np.mean(weights)),
    }


# ============================================================
# 4. Platt scaling at non-fitted horizons
# ============================================================
def fit_platt_scaler(fold_data, eval_fold, t_fit):
    """Fit a Platt scaler on all folds except eval_fold, at time t_fit.

    Uses the same cross-fold approach as paper_recalibration_v2.py.
    Returns a fitted LogisticRegression.
    """
    X_train, y_train = [], []
    for f2, d2 in fold_data.items():
        if f2 == eval_fold:
            continue
        s_at_t = np.array(
            [np.interp(t_fit, d2["time_coords"], c) for c in d2["curves"]]
        )
        vt2 = d2["val_times"]
        ve2 = d2["val_events"]
        # Only include uncensored or observed past t_fit
        mask = (ve2 == 1) | (vt2 >= t_fit)
        X_train.append(s_at_t[mask])
        y_train.append((vt2[mask] >= t_fit).astype(float))

    X_tr = np.concatenate(X_train)
    y_tr = np.concatenate(y_train)

    if len(np.unique(y_tr)) < 2:
        return None

    cal = LogisticRegression(max_iter=1000)
    cal.fit(X_tr.reshape(-1, 1), y_tr)
    return cal


def recalibrate_curves(original_curves, calibrator):
    """Apply Platt calibrator to every element of the survival matrix.
    Enforces monotonicity afterward."""
    flat = original_curves.flatten().reshape(-1, 1)
    recal_flat = calibrator.predict_proba(flat)[:, 1]
    recal = recal_flat.reshape(original_curves.shape)
    for i in range(recal.shape[0]):
        recal[i] = np.minimum.accumulate(recal[i])
    return np.clip(recal, 0.0, 1.0)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    models_cfg = {
        "SurvPath": {
            "loader": load_survpath_fold,
            "train_loader": load_survpath_train_data,
            "has_logits": True,
        },
        "MCAT": {
            "loader": load_mcat_fold,
            "train_loader": load_survpath_train_data,
            "has_logits": False,
        },
        "MMP": {
            "loader": load_mmp_fold,
            "train_loader": load_mmp_train_data,
            "has_logits": False,
        },
    }

    all_results = []

    for model_name, cfg in models_cfg.items():
        print(f"\n{'='*70}")
        print(f"  {model_name}")
        print(f"{'='*70}")

        horizons = get_horizons(model_name)

        # ── Collect data for all 5 folds ──
        fold_data = {}
        for fold in range(5):
            val_df = cfg["loader"](fold)
            if val_df.empty:
                continue
            train_t, train_e = cfg["train_loader"](fold)
            curves, tc = build_survival_curves(
                val_df, train_t, train_e, cfg["has_logits"]
            )

            val_times = val_df["time"].values
            val_events = val_df["event"].values.astype(int)
            event_times_only = val_times[val_events.astype(bool)]
            median_event_t = (
                float(np.median(event_times_only))
                if len(event_times_only) >= 1
                else None
            )

            fold_data[fold] = {
                "curves": curves,
                "time_coords": tc,
                "val_times": val_times,
                "val_events": val_events,
                "train_times": train_t,
                "train_events": train_e,
                "median_event_t": median_event_t,
            }

        if len(fold_data) < 5:
            print(f"  Only {len(fold_data)} folds available, skipping")
            continue

        # ── Add median to horizons for this model ──
        # Use the average of per-fold median event times
        median_event_ts = [
            d["median_event_t"]
            for d in fold_data.values()
            if d["median_event_t"] is not None
        ]
        model_median_t = np.mean(median_event_ts)
        all_horizons = dict(horizons)
        all_horizons["median"] = model_median_t

        units = "days" if model_name == "MMP" else "months"
        print(f"  Time horizons ({units}):")
        for hname, ht in all_horizons.items():
            print(f"    {hname}: {ht:.1f}")
        print()

        # ── Check which horizons are within the survival curve's time range ──
        tc_example = fold_data[0]["time_coords"]
        print(f"  Survival curve time range: [{tc_example[0]:.1f}, {tc_example[-1]:.1f}] {units}")

        # ──────────────────────────────────────────────
        # PART 1: Multi-horizon 1-calibration
        # ──────────────────────────────────────────────
        print(f"\n  --- Part 1: Multi-horizon 1-calibration ---")
        print(f"  {'Horizon':<10} {'Fold':>4} {'p-value':>10} {'Status':>6}")
        print(f"  {'-'*34}")

        horizon_pvals = {h: [] for h in all_horizons}

        for hname, ht in all_horizons.items():
            for fold in range(5):
                d = fold_data[fold]

                # Check if horizon is within range (with some tolerance)
                if ht > d["time_coords"][-1] * 1.05:
                    p_val = np.nan
                    status = "OOR"  # out of range
                else:
                    p_val, obs, exp = evaluate_one_cal_at_horizon(
                        d["curves"], d["time_coords"],
                        d["val_times"], d["val_events"],
                        d["train_times"], d["train_events"],
                        target_time=ht,
                    )
                    status = "PASS" if (not np.isnan(p_val) and p_val > 0.05) else "FAIL"
                    if np.isnan(p_val):
                        status = "ERR"

                horizon_pvals[hname].append(p_val)
                print(f"  {hname:<10} {fold:>4} {p_val:>10.4f} {status:>6}")

                all_results.append({
                    "model": model_name,
                    "analysis": "1-calibration",
                    "horizon": hname,
                    "horizon_t": ht,
                    "fold": fold,
                    "p_value": p_val,
                    "metric_value": np.nan,
                    "method": "original",
                })

        # BH correction across all folds per horizon
        print(f"\n  BH-corrected failure counts (alpha=0.05):")
        for hname in all_horizons:
            pvals = [p for p in horizon_pvals[hname] if not np.isnan(p)]
            if len(pvals) == 0:
                print(f"    {hname:<10}: no valid p-values")
                continue
            rejected, corrected, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
            n_fail_raw = sum(1 for p in pvals if p < 0.05)
            n_fail_bh = int(sum(rejected))
            print(f"    {hname:<10}: raw fails={n_fail_raw}/5, BH fails={n_fail_bh}/5")

        # ──────────────────────────────────────────────
        # PART 2: MACE
        # ──────────────────────────────────────────────
        print(f"\n  --- Part 2: MACE (Mean Absolute Calibration Error) ---")
        print(f"  {'Horizon':<10} {'Fold':>4} {'MACE':>8}")
        print(f"  {'-'*24}")

        for hname, ht in all_horizons.items():
            for fold in range(5):
                d = fold_data[fold]

                if ht > d["time_coords"][-1] * 1.05:
                    mace = np.nan
                else:
                    mace, _ = compute_mace(
                        d["curves"], d["time_coords"],
                        d["val_times"], d["val_events"],
                        target_time=ht,
                    )

                print(f"  {hname:<10} {fold:>4} {mace:>8.4f}")

                all_results.append({
                    "model": model_name,
                    "analysis": "MACE",
                    "horizon": hname,
                    "horizon_t": ht,
                    "fold": fold,
                    "p_value": np.nan,
                    "metric_value": mace,
                    "method": "original",
                })

        # ──────────────────────────────────────────────
        # PART 3: Censoring-aware effective sample size
        # ──────────────────────────────────────────────
        print(f"\n  --- Part 3: Censoring-Aware Effective Sample Size ---")
        print(f"  {'Horizon':<10} {'Fold':>4} {'n_total':>8} {'n_evt<t':>8} {'n_at_risk':>10} {'n_eff':>8} {'wt_CV':>7}")
        print(f"  {'-'*60}")

        for hname, ht in all_horizons.items():
            for fold in range(5):
                d = fold_data[fold]

                neff_info = compute_censoring_neff(
                    d["val_times"], d["val_events"],
                    d["train_times"], d["train_events"],
                    target_time=ht,
                )

                print(
                    f"  {hname:<10} {fold:>4} "
                    f"{neff_info['n_total']:>8} "
                    f"{neff_info['n_events_before_t']:>8} "
                    f"{neff_info['n_at_risk_at_t']:>10} "
                    f"{neff_info['n_eff']:>8.1f} "
                    f"{neff_info['weight_cv']:>7.3f}"
                )

                all_results.append({
                    "model": model_name,
                    "analysis": "n_eff",
                    "horizon": hname,
                    "horizon_t": ht,
                    "fold": fold,
                    "p_value": np.nan,
                    "metric_value": neff_info["n_eff"],
                    "method": f"n_total={neff_info['n_total']},n_evt={neff_info['n_events_before_t']},n_risk={neff_info['n_at_risk_at_t']},wt_cv={neff_info['weight_cv']:.3f}",
                })

        # ──────────────────────────────────────────────
        # PART 4: Platt scaling transfer
        # ──────────────────────────────────────────────
        print(f"\n  --- Part 4: Platt Scaling Transfer ---")
        print(f"  Platt scaler fit at median event time ({model_median_t:.1f} {units})")
        print(f"  Evaluated at all horizons:")
        print(f"  {'Horizon':<10} {'Fold':>4} {'Orig p':>10} {'Platt p':>10} {'Orig':>6} {'Platt':>6}")
        print(f"  {'-'*50}")

        for hname, ht in all_horizons.items():
            for fold in range(5):
                d = fold_data[fold]

                if ht > d["time_coords"][-1] * 1.05:
                    orig_p = np.nan
                    platt_p = np.nan
                else:
                    # Original
                    orig_p, _, _ = evaluate_one_cal_at_horizon(
                        d["curves"], d["time_coords"],
                        d["val_times"], d["val_events"],
                        d["train_times"], d["train_events"],
                        target_time=ht,
                    )

                    # Platt: fit at median, apply to full curves, evaluate at ht
                    platt_cal = fit_platt_scaler(
                        fold_data, fold, t_fit=model_median_t
                    )
                    if platt_cal is not None:
                        recal_curves = recalibrate_curves(d["curves"], platt_cal)
                        platt_p, _, _ = evaluate_one_cal_at_horizon(
                            recal_curves, d["time_coords"],
                            d["val_times"], d["val_events"],
                            d["train_times"], d["train_events"],
                            target_time=ht,
                        )
                    else:
                        platt_p = np.nan

                orig_s = "PASS" if (not np.isnan(orig_p) and orig_p > 0.05) else "FAIL"
                platt_s = "PASS" if (not np.isnan(platt_p) and platt_p > 0.05) else "FAIL"
                if np.isnan(orig_p):
                    orig_s = "OOR"
                if np.isnan(platt_p):
                    platt_s = "OOR"

                print(
                    f"  {hname:<10} {fold:>4} "
                    f"{orig_p:>10.4f} {platt_p:>10.4f} "
                    f"{orig_s:>6} {platt_s:>6}"
                )

                all_results.append({
                    "model": model_name,
                    "analysis": "platt_transfer",
                    "horizon": hname,
                    "horizon_t": ht,
                    "fold": fold,
                    "p_value": platt_p,
                    "metric_value": np.nan,
                    "method": f"platt_fit_at_median({model_median_t:.1f})",
                })

    # ============================================================
    # Save CSV
    # ============================================================
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/multi_horizon_calibration.csv", index=False)
    print(f"\n\nSaved {len(results_df)} rows to results/multi_horizon_calibration.csv")

    # ============================================================
    # Summary tables
    # ============================================================
    print(f"\n{'='*80}")
    print(f"  SUMMARY: Multi-Horizon 1-Calibration (p < 0.05 = miscalibrated)")
    print(f"{'='*80}")
    print(f"  {'Model':<10} {'1yr fail':>10} {'3yr fail':>10} {'5yr fail':>10} {'median fail':>12}")
    print(f"  {'-'*56}")

    for model in ["SurvPath", "MCAT", "MMP"]:
        fails = {}
        for hname in ["1yr", "3yr", "5yr", "median"]:
            sub = results_df[
                (results_df["model"] == model)
                & (results_df["analysis"] == "1-calibration")
                & (results_df["horizon"] == hname)
            ]
            valid_p = sub["p_value"].dropna()
            n_fail = int((valid_p < 0.05).sum())
            n_valid = len(valid_p)
            fails[hname] = f"{n_fail}/{n_valid}"
        print(
            f"  {model:<10} {fails['1yr']:>10} {fails['3yr']:>10} "
            f"{fails['5yr']:>10} {fails['median']:>12}"
        )

    print(f"\n{'='*80}")
    print(f"  SUMMARY: MACE by Model and Horizon (mean +/- std across folds)")
    print(f"{'='*80}")
    print(f"  {'Model':<10} {'1yr':>14} {'3yr':>14} {'5yr':>14} {'median':>14}")
    print(f"  {'-'*60}")

    for model in ["SurvPath", "MCAT", "MMP"]:
        vals = {}
        for hname in ["1yr", "3yr", "5yr", "median"]:
            sub = results_df[
                (results_df["model"] == model)
                & (results_df["analysis"] == "MACE")
                & (results_df["horizon"] == hname)
            ]
            v = sub["metric_value"].dropna()
            if len(v) > 0:
                vals[hname] = f"{v.mean():.4f}+/-{v.std():.4f}"
            else:
                vals[hname] = "N/A"
        print(
            f"  {model:<10} {vals['1yr']:>14} {vals['3yr']:>14} "
            f"{vals['5yr']:>14} {vals['median']:>14}"
        )

    print(f"\n{'='*80}")
    print(f"  SUMMARY: Effective Sample Size (mean n_eff across folds)")
    print(f"{'='*80}")
    print(f"  {'Model':<10} {'1yr':>10} {'3yr':>10} {'5yr':>10} {'median':>10}")
    print(f"  {'-'*44}")

    for model in ["SurvPath", "MCAT", "MMP"]:
        vals = {}
        for hname in ["1yr", "3yr", "5yr", "median"]:
            sub = results_df[
                (results_df["model"] == model)
                & (results_df["analysis"] == "n_eff")
                & (results_df["horizon"] == hname)
            ]
            v = sub["metric_value"].dropna()
            if len(v) > 0:
                vals[hname] = f"{v.mean():.1f}"
            else:
                vals[hname] = "N/A"
        print(
            f"  {model:<10} {vals['1yr']:>10} {vals['3yr']:>10} "
            f"{vals['5yr']:>10} {vals['median']:>10}"
        )

    print(f"\n{'='*80}")
    print(f"  SUMMARY: Platt Transfer — failure counts (raw)")
    print(f"{'='*80}")
    print(f"  {'Model':<10} {'1yr fail':>10} {'3yr fail':>10} {'5yr fail':>10} {'median fail':>12}")
    print(f"  {'-'*56}")

    for model in ["SurvPath", "MCAT", "MMP"]:
        fails = {}
        for hname in ["1yr", "3yr", "5yr", "median"]:
            sub = results_df[
                (results_df["model"] == model)
                & (results_df["analysis"] == "platt_transfer")
                & (results_df["horizon"] == hname)
            ]
            valid_p = sub["p_value"].dropna()
            n_fail = int((valid_p < 0.05).sum())
            n_valid = len(valid_p)
            fails[hname] = f"{n_fail}/{n_valid}"
        print(
            f"  {model:<10} {fails['1yr']:>10} {fails['3yr']:>10} "
            f"{fails['5yr']:>10} {fails['median']:>12}"
        )

    print("\nDone.")
