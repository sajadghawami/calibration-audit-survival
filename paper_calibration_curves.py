"""
Task 1: Calibration Curves — predicted vs observed survival probability.

Figure A: BRCA (Experiment A) — SurvPath, MCAT, MMP + Cox-PH control
Figure B: Multi-cancer (Experiment B) — MCAT, AMIL-sig, AMIL-concat, SNN
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from SurvivalEVAL import SurvivalEvaluator

from calibration_audit import (
    logits_to_survival_interpolated, risk_to_survival_breslow,
    load_survpath_fold, load_mcat_fold, load_mmp_fold,
    load_survpath_train_data, load_mmp_train_data,
    N_TIME_POINTS,
)
from calibration_audit_multicancer import (
    load_iccv_fold, load_iccv_train_data,
)

warnings.filterwarnings("ignore")

PATIENT_INDEX = "patient_index.csv"
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#E69F00']
N_BINS = 10


def get_survival_curves_and_teval(val_df, train_times, train_events, has_logits=False):
    """Build survival curves and compute t_eval. Returns (curves, time_coords, t_eval)."""
    val_events = val_df["event"].values.astype(bool)
    event_times_only = val_df["time"].values[val_events]
    t_eval = float(np.median(event_times_only)) if len(event_times_only) > 0 else None

    if has_logits:
        train_event_times = train_times[train_events > 0]
        curves_list, time_coords = [], None
        for _, row in val_df.iterrows():
            surv, tc = logits_to_survival_interpolated(row["logits"], train_event_times)
            curves_list.append(surv)
            time_coords = tc
        return np.array(curves_list), time_coords, t_eval
    else:
        train_event_times = train_times[train_events > 0]
        time_coords = np.percentile(train_event_times, np.linspace(5, 95, N_TIME_POINTS))
        time_coords = np.sort(np.unique(time_coords))
        curves = risk_to_survival_breslow(val_df["risk"].values, train_times, train_events, time_coords)
        return curves, time_coords, t_eval


def compute_decile_calibration(surv_curves, time_coords, val_times, val_events, t_eval, n_bins=N_BINS):
    """Compute decile-level calibration data with 95% CI from Greenwood."""
    predicted = np.array([np.interp(t_eval, time_coords, c) for c in surv_curves])
    # Use quantile-based bins for equal-sized groups
    try:
        bin_edges = np.percentile(predicted, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] -= 0.001
        bin_edges[-1] += 0.001
    except Exception:
        return pd.DataFrame()

    rows = []
    for i in range(n_bins):
        mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (predicted >= bin_edges[i]) & (predicted <= bin_edges[i + 1])
        n = int(mask.sum())
        if n < 2:
            continue
        mean_pred = float(predicted[mask].mean())
        km = KaplanMeierFitter()
        km.fit(val_times[mask], event_observed=val_events[mask])
        obs = float(km.predict(t_eval))

        # 95% CI from Greenwood (lifelines)
        ci = km.confidence_interval_survival_function_
        idx_closest = np.argmin(np.abs(ci.index.values - t_eval))
        ci_at_t = ci.iloc[idx_closest]
        ci_lower = float(ci_at_t.iloc[0])
        ci_upper = float(ci_at_t.iloc[1])

        rows.append({
            "mean_predicted": mean_pred,
            "km_observed": obs,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_patients": n,
        })
    return pd.DataFrame(rows)


# ============================================================
# Figure A: BRCA (Experiment A)
# ============================================================
def build_brca_calibration_data():
    """Compute calibration deciles for 3 models + Cox-PH, pooled across folds."""
    all_deciles = []

    models = {
        "SurvPath": {"loader": load_survpath_fold, "train_loader": load_survpath_train_data, "has_logits": True},
        "MCAT":     {"loader": load_mcat_fold,     "train_loader": load_survpath_train_data, "has_logits": False},
        "MMP":      {"loader": load_mmp_fold,      "train_loader": load_mmp_train_data,      "has_logits": False},
    }

    for model_name, cfg in models.items():
        print(f"  {model_name}...", end="", flush=True)
        for fold in range(5):
            val_df = cfg["loader"](fold)
            if val_df.empty:
                continue
            train_t, train_e = cfg["train_loader"](fold)
            curves, tc, t_eval = get_survival_curves_and_teval(
                val_df, train_t, train_e, has_logits=cfg["has_logits"])
            if t_eval is None:
                continue
            val_times = val_df["time"].values
            val_events = val_df["event"].values.astype(bool)
            dec = compute_decile_calibration(curves, tc, val_times, val_events, t_eval)
            if not dec.empty:
                dec["model"] = model_name
                dec["cancer"] = "BRCA"
                dec["fold"] = fold
                all_deciles.append(dec)
        print(" done")

    # Cox-PH control (fold 0 only)
    print("  Cox-PH...", end="", flush=True)
    rna = pd.read_csv("SurvPath/data/rna_clean.csv", index_col=0)
    top_genes = rna.var(axis=0).sort_values(ascending=False).head(20).index.tolist()
    pi = pd.read_csv(PATIENT_INDEX)
    split = pd.read_csv("SurvPath/splits/5foldcv/tcga_brca/splits_0.csv")
    train_ids = set(split["train"].dropna().replace("", pd.NA).dropna())
    rna_sub = rna[top_genes].copy()
    rna_sub["patient_id"] = rna_sub.index
    rna_sub = rna_sub.reset_index(drop=True)
    merged = pi.merge(rna_sub, on="patient_id")
    train = merged[merged["patient_id"].isin(train_ids)]
    val = merged[~merged["patient_id"].isin(train_ids)]

    cox_train = train[top_genes + ["survival_time", "event"]].copy()
    cox_train["survival_time"] = cox_train["survival_time"].astype(float)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_train, duration_col="survival_time", event_col="event")

    surv_fn = cph.predict_survival_function(val[top_genes])
    tc_cox = surv_fn.index.values
    curves_cox = surv_fn.T.values
    val_times = val["survival_time"].values.astype(float)
    val_events = val["event"].values.astype(bool)
    t_eval_cox = float(np.median(val_times[val_events]))

    # Mask to valid range
    mask = tc_cox <= val_times[val_events].max()
    tc_cox = tc_cox[mask]
    curves_cox = curves_cox[:, mask]

    dec = compute_decile_calibration(curves_cox, tc_cox, val_times, val_events, t_eval_cox)
    if not dec.empty:
        dec["model"] = "Cox-PH"
        dec["cancer"] = "BRCA"
        dec["fold"] = 0
        all_deciles.append(dec)
    print(" done")

    return pd.concat(all_deciles, ignore_index=True)


def plot_brca_figure(df):
    """Plot calibration curves for BRCA models."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)

    model_order = ["Cox-PH", "SurvPath", "MCAT", "MMP"]
    for i, model in enumerate(model_order):
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        # Average across folds per decile rank
        grouped = sub.groupby(sub.groupby("fold").cumcount()).agg({
            "mean_predicted": "mean",
            "km_observed": "mean",
            "ci_lower": "mean",
            "ci_upper": "mean",
        }).reset_index(drop=True)

        ax.plot(grouped["mean_predicted"], grouped["km_observed"],
                "o-", color=COLORS[i], ms=4, lw=1.3, label=model, zorder=3)
        ax.fill_between(grouped["mean_predicted"],
                        grouped["ci_lower"], grouped["ci_upper"],
                        color=COLORS[i], alpha=0.15, zorder=2)

    ax.set_xlabel("Predicted survival probability", fontsize=12)
    ax.set_ylabel("Observed survival proportion", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig("figures/calibration_curves_brca.pdf", dpi=300, bbox_inches="tight")
    fig.savefig("figures/calibration_curves_brca.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/calibration_curves_brca.{pdf,png}")


# ============================================================
# Figure B: Multi-cancer (Experiment B)
# ============================================================
def build_multicancer_calibration_data():
    """Compute calibration deciles for 4 architectures x 5 cancers."""
    arch_keys = {
        "MCAT": "MCATsm_nll_surv_a0.0_5foldcv_gc32_concat",
        "AMIL-sig": "AMILsm_nll_surv_a0.0_5foldcv_gc32_sig",
        "AMIL-concat": "AMILsm_nll_surv_a0.0_5foldcv_gc32_sig_concat",
        "SNN": "SNNsm_nll_surv_a0.0_reg1e-04_5foldcv_gc32_sig",
    }
    cancers = ["blca", "brca", "gbmlgg", "luad", "ucec"]
    all_deciles = []

    for short_name, arch in arch_keys.items():
        for cancer in cancers:
            print(f"  {short_name} / {cancer.upper()}...", end="", flush=True)
            for fold in range(5):
                val_df = load_iccv_fold(fold, arch, cancer)
                if val_df.empty:
                    continue
                train_t, train_e = load_iccv_train_data(fold, cancer)
                curves, tc, t_eval = get_survival_curves_and_teval(
                    val_df, train_t, train_e, has_logits=False)
                if t_eval is None:
                    continue
                val_times = val_df["time"].values
                val_events = val_df["event"].values.astype(bool)
                dec = compute_decile_calibration(curves, tc, val_times, val_events, t_eval)
                if not dec.empty:
                    dec["model"] = short_name
                    dec["cancer"] = cancer.upper()
                    dec["fold"] = fold
                    all_deciles.append(dec)
            print(" done")

    return pd.concat(all_deciles, ignore_index=True)


def plot_multicancer_figure(df):
    """Plot 2x3 grid of calibration curves per cancer type."""
    cancers = ["BLCA", "BRCA", "GBMLGG", "LUAD", "UCEC"]
    models = ["MCAT", "AMIL-sig", "AMIL-concat", "SNN"]

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.0))
    axes_flat = axes.flatten()

    for idx, cancer in enumerate(cancers):
        ax = axes_flat[idx]
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)

        for i, model in enumerate(models):
            sub = df[(df["model"] == model) & (df["cancer"] == cancer)]
            if sub.empty:
                continue
            grouped = sub.groupby(sub.groupby("fold").cumcount()).agg({
                "mean_predicted": "mean",
                "km_observed": "mean",
                "ci_lower": "mean",
                "ci_upper": "mean",
            }).reset_index(drop=True)

            ax.plot(grouped["mean_predicted"], grouped["km_observed"],
                    "o-", color=COLORS[i], ms=3, lw=1.0, label=model)
            ax.fill_between(grouped["mean_predicted"],
                            grouped["ci_lower"], grouped["ci_upper"],
                            color=COLORS[i], alpha=0.12)

        ax.set_title(cancer, fontsize=11, fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)
        if idx >= 3:
            ax.set_xlabel("Predicted S(t)", fontsize=9)
        if idx % 3 == 0:
            ax.set_ylabel("Observed S(t)", fontsize=9)

    # Legend in empty 6th panel
    axes_flat[5].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[5].legend(handles, labels, fontsize=9, loc="center", frameon=False)

    fig.tight_layout()
    fig.savefig("figures/calibration_curves_multicancer.pdf", dpi=300, bbox_inches="tight")
    fig.savefig("figures/calibration_curves_multicancer.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/calibration_curves_multicancer.{pdf,png}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Building BRCA calibration data (Experiment A)...")
    brca_df = build_brca_calibration_data()
    plot_brca_figure(brca_df)

    print("\nBuilding multi-cancer calibration data (Experiment B)...")
    mc_df = build_multicancer_calibration_data()
    plot_multicancer_figure(mc_df)

    # Save combined CSV
    all_df = pd.concat([brca_df, mc_df], ignore_index=True)
    all_df.to_csv("figures/calibration_curves_data.csv", index=False)
    print(f"\n  Saved: figures/calibration_curves_data.csv ({len(all_df)} rows)")
