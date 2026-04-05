"""
Recalibration Experiment v2 — uses SurvivalEVAL one_calibration() throughout
for consistency with the main audit.

Cross-fold recalibration: for fold k, Platt/isotonic is fit on the union of
validation predictions from the other 4 folds, then applied to fold k's full
survival curves and evaluated with SurvivalEVAL.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sksurv.metrics import concordance_index_censored
from SurvivalEVAL import SurvivalEvaluator

from calibration_audit import (
    logits_to_survival_interpolated, risk_to_survival_breslow,
    load_survpath_fold, load_mcat_fold, load_mmp_fold,
    load_survpath_train_data, load_mmp_train_data,
    N_TIME_POINTS,
)

warnings.filterwarnings("ignore")

COLORS = ['#0072B2', '#D55E00', '#009E73']


def build_survival_curves(val_df, train_times, train_events, has_logits):
    """Build full survival curves for validation patients.
    Returns (surv_curves, time_coords, t_eval)."""
    val_times = val_df["time"].values
    val_events = val_df["event"].values.astype(bool)
    event_times_only = val_times[val_events]
    t_eval = float(np.median(event_times_only)) if len(event_times_only) >= 1 else None

    if has_logits:
        train_event_times = train_times[train_events > 0]
        curves_list = []
        for _, row in val_df.iterrows():
            surv, fine_times = logits_to_survival_interpolated(row["logits"], train_event_times)
            curves_list.append(surv)
        return np.array(curves_list), fine_times, t_eval
    else:
        train_event_times = train_times[train_events > 0]
        time_coords = np.percentile(train_event_times, np.linspace(5, 95, N_TIME_POINTS))
        time_coords = np.sort(np.unique(time_coords))
        curves = risk_to_survival_breslow(
            val_df["risk"].values, train_times, train_events, time_coords)
        return curves, time_coords, t_eval


def evaluate_curves(surv_curves, time_coords, val_times, val_events,
                    train_times, train_events, t_eval):
    """Evaluate survival curves using SurvivalEVAL — same as main audit."""
    evaluator = SurvivalEvaluator(
        pred_survs=surv_curves,
        time_coordinates=time_coords,
        event_times=val_times,
        event_indicators=val_events.astype(int),
        train_event_times=train_times,
        train_event_indicators=train_events.astype(int),
    )
    try:
        onecal = evaluator.one_calibration(target_time=t_eval)[0]
    except Exception:
        onecal = np.nan
    try:
        ibs = evaluator.integrated_brier_score()
    except Exception:
        ibs = np.nan
    return onecal, ibs


def recalibrate_curves(original_curves, time_coords, calibrator):
    """Apply a fitted calibrator to every time point of every survival curve.
    Returns recalibrated curves with monotonicity enforced."""
    flat = original_curves.flatten().reshape(-1, 1)
    recal_flat = calibrator.predict(flat.ravel()) if hasattr(calibrator, 'predict') and not hasattr(calibrator, 'predict_proba') else calibrator.predict_proba(flat)[:, 1]
    recal = recal_flat.reshape(original_curves.shape)
    # Enforce monotonically non-increasing along time axis
    for i in range(recal.shape[0]):
        recal[i] = np.minimum.accumulate(recal[i])
    # Clip to valid range
    recal = np.clip(recal, 0.0, 1.0)
    return recal


if __name__ == "__main__":
    models_cfg = {
        "SurvPath": {"loader": load_survpath_fold, "train_loader": load_survpath_train_data, "has_logits": True},
        "MCAT":     {"loader": load_mcat_fold,     "train_loader": load_survpath_train_data, "has_logits": False},
        "MMP":      {"loader": load_mmp_fold,      "train_loader": load_mmp_train_data,      "has_logits": False},
    }

    all_results = []

    for model_name, cfg in models_cfg.items():
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        # ── Collect full survival curves for all 5 folds ──
        fold_data = {}
        for fold in range(5):
            val_df = cfg["loader"](fold)
            if val_df.empty:
                continue
            train_t, train_e = cfg["train_loader"](fold)
            curves, tc, t_eval = build_survival_curves(
                val_df, train_t, train_e, cfg["has_logits"])
            if t_eval is None:
                continue
            fold_data[fold] = {
                "curves": curves,
                "time_coords": tc,
                "t_eval": t_eval,
                "val_times": val_df["time"].values,
                "val_events": val_df["event"].values.astype(int),
                "val_risks": val_df["risk"].values,
                "train_times": train_t,
                "train_events": train_e,
            }

        if len(fold_data) < 5:
            print(f"  Only {len(fold_data)} folds, skipping")
            continue

        print(f"  {'Fold':>4} {'Method':<10} {'1-cal p':>9} {'':>5} {'IBS':>8} {'C-idx':>7}")
        print(f"  {'-'*48}")

        for fold in range(5):
            d = fold_data[fold]
            vt = d["val_times"]
            ve = d["val_events"]
            tt = d["train_times"]
            te = d["train_events"]
            tc = d["time_coords"]
            t_eval = d["t_eval"]
            curves_orig = d["curves"]
            risks = d["val_risks"]

            # C-index (same for all methods — recalibration preserves ranking)
            ci = concordance_index_censored(ve.astype(bool), vt, risks)[0]

            # ── Original ──
            onecal_orig, ibs_orig = evaluate_curves(
                curves_orig, tc, vt, ve, tt, te, t_eval)
            status = "PASS" if onecal_orig > 0.05 else "FAIL"
            print(f"  {fold:>4} {'Original':<10} {onecal_orig:>8.4f} {status} {ibs_orig:>8.4f} {ci:>7.4f}")
            all_results.append({"model": model_name, "fold": fold, "method": "Original",
                                "onecal_p": onecal_orig, "ibs": ibs_orig, "ci": ci})

            # ── Fit Platt & Isotonic on other folds ──
            for method_name, make_calibrator in [
                ("Platt", lambda: LogisticRegression(max_iter=1000)),
                ("Isotonic", lambda: IsotonicRegression(out_of_bounds="clip",
                                                        y_min=0.001, y_max=0.999)),
            ]:
                # Training data: S(t_eval) from other folds + binary outcome
                X_train, y_train = [], []
                for f2 in fold_data:
                    if f2 == fold:
                        continue
                    d2 = fold_data[f2]
                    # Extract S(t_eval) for each patient in fold f2
                    s_at_t = np.array([np.interp(d2["t_eval"], d2["time_coords"], c)
                                       for c in d2["curves"]])
                    vt2 = d2["val_times"]
                    ve2 = d2["val_events"]
                    # Exclude censored before t_eval
                    mask = (ve2 == 1) | (vt2 >= d2["t_eval"])
                    X_train.append(s_at_t[mask])
                    y_train.append((vt2[mask] >= d2["t_eval"]).astype(float))

                X_tr = np.concatenate(X_train)
                y_tr = np.concatenate(y_train)

                # Fit
                cal = make_calibrator()
                if method_name == "Platt":
                    cal.fit(X_tr.reshape(-1, 1), y_tr)
                    recal_curves = recalibrate_curves(curves_orig, tc, cal)
                else:
                    cal.fit(X_tr, y_tr)
                    recal_curves = recalibrate_curves(curves_orig, tc, cal)

                # Evaluate with SurvivalEVAL
                onecal_r, ibs_r = evaluate_curves(
                    recal_curves, tc, vt, ve, tt, te, t_eval)
                status = "PASS" if onecal_r > 0.05 else "FAIL"
                print(f"  {fold:>4} {method_name:<10} {onecal_r:>8.4f} {status} {ibs_r:>8.4f} {ci:>7.4f}")
                all_results.append({"model": model_name, "fold": fold, "method": method_name,
                                    "onecal_p": onecal_r, "ibs": ibs_r, "ci": ci})

        # Print example curves for last fold
        print(f"\n  Example curves (fold {fold}, first 3 patients):")
        print(f"  {'Time pt':>8} {'Orig[0]':>8} {'Platt[0]':>8}  {'Orig[1]':>8} {'Platt[1]':>8}  {'Orig[2]':>8} {'Platt[2]':>8}")
        # Recompute Platt for the last fold for display
        cal_disp = LogisticRegression(max_iter=1000)
        X_d, y_d = [], []
        for f2 in fold_data:
            if f2 == fold:
                continue
            d2 = fold_data[f2]
            s_at_t = np.array([np.interp(d2["t_eval"], d2["time_coords"], c) for c in d2["curves"]])
            vt2, ve2 = d2["val_times"], d2["val_events"]
            mask = (ve2 == 1) | (vt2 >= d2["t_eval"])
            X_d.append(s_at_t[mask])
            y_d.append((vt2[mask] >= d2["t_eval"]).astype(float))
        cal_disp.fit(np.concatenate(X_d).reshape(-1, 1), np.concatenate(y_d))
        rc = recalibrate_curves(fold_data[fold]["curves"], fold_data[fold]["time_coords"], cal_disp)
        oc = fold_data[fold]["curves"]
        tcs = fold_data[fold]["time_coords"]
        for ti in range(0, len(tcs), max(1, len(tcs) // 6)):
            print(f"  {tcs[ti]:>8.1f} {oc[0, ti]:>8.4f} {rc[0, ti]:>8.4f}  "
                  f"{oc[1, ti]:>8.4f} {rc[1, ti]:>8.4f}  "
                  f"{oc[2, ti]:>8.4f} {rc[2, ti]:>8.4f}")

    # ── Save CSV ──
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/recalibration_surveval.csv", index=False)

    # ── Summary table ──
    print(f"\n\n{'='*80}")
    print(f"  RECALIBRATION SUMMARY (SurvivalEVAL one_calibration)")
    print(f"{'='*80}")
    print(f"  {'Model':<10} {'Orig fail':>10} {'Platt fail':>11} {'Iso fail':>10}"
          f" {'Orig IBS':>14} {'Platt IBS':>14} {'Iso IBS':>14}")
    print(f"  {'-'*75}")

    for model in ["SurvPath", "MCAT", "MMP"]:
        for method, label in [("Original", "Orig"), ("Platt", "Platt"), ("Isotonic", "Iso")]:
            sub = results_df[(results_df["model"] == model) & (results_df["method"] == method)]
            n_fail = int((sub["onecal_p"] < 0.05).sum())
            ibs_vals = sub["ibs"].dropna()
            ibs_str = f"{ibs_vals.mean():.3f}+/-{ibs_vals.std():.3f}" if len(ibs_vals) > 0 else "N/A"
            if method == "Original":
                orig_fail = n_fail
                orig_ibs = ibs_str
            elif method == "Platt":
                platt_fail = n_fail
                platt_ibs = ibs_str
            elif method == "Isotonic":
                iso_fail = n_fail
                iso_ibs = ibs_str
        print(f"  {model:<10} {orig_fail:>5}/5     {platt_fail:>5}/5      {iso_fail:>5}/5"
              f"     {orig_ibs:>14} {platt_ibs:>14} {iso_ibs:>14}")

    # ── Consistency check vs main audit ──
    print(f"\n  --- Consistency check vs main audit ---")
    print(f"  (Original 1-cal p should match calibration_audit.py results)")
    main_audit = {
        "SurvPath": [0.018, 0.033, 0.000, 0.000, 0.001],
        "MCAT":     [0.000, 0.000, 0.000, 0.000, 0.000],
        "MMP":      [0.233, 0.037, 0.012, 0.000, 0.000],
    }
    for model in ["SurvPath", "MCAT", "MMP"]:
        sub = results_df[(results_df["model"] == model) & (results_df["method"] == "Original")]
        for fold in range(5):
            row = sub[sub["fold"] == fold]
            if row.empty:
                continue
            this_p = row["onecal_p"].values[0]
            ref_p = main_audit[model][fold]
            match = "OK" if abs(this_p - ref_p) < 0.01 else "MISMATCH"
            if match == "MISMATCH":
                print(f"  {match}: {model} fold {fold}: this={this_p:.4f} vs main={ref_p:.3f}")
    print(f"  (Only mismatches shown; silence = all match)")

    print(f"\n  Saved: results/recalibration_surveval.csv")
