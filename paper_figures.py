"""
Publication-quality figures for MLHC 2026 paper.
All figures use consistent styling and SurvivalEVAL-based evaluation.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator

warnings.filterwarnings("ignore")

# ── Global style ──
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
})

# ── Colors ──
C = {
    "Cox-PH":      "#1f77b4",
    "SurvPath":    "#ff7f0e",
    "MCAT":        "#2ca02c",
    "MMP":         "#d62728",
    "SNN":         "#9467bd",
    "AMIL-sig":    "#ff7f0e",
    "AMIL-concat": "#d62728",
    "Original":    "#1f77b4",
    "Platt":       "#ff7f0e",
}
DIAG_COLOR = "#888888"


def setup_cal_ax(ax):
    """Standard calibration axis setup."""
    ax.plot([0, 1], [0, 1], "--", color=DIAG_COLOR, lw=1.0, zorder=0)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(False)


MARKERS = {"Cox-PH": "o", "SurvPath": "s", "MCAT": "D", "MMP": "^",
           "AMIL-sig": "s", "AMIL-concat": "^", "SNN": "v",
           "Original": "o", "Platt": "s"}
LSTYLES = {"Cox-PH": "-", "SurvPath": "-", "MCAT": "--", "MMP": "-.",
           "AMIL-sig": "-", "AMIL-concat": "--", "SNN": "-.",
           "Original": "-", "Platt": "--"}


def plot_model_on_ax(ax, df_model, color, label, thick=2.5, thin=0.8, alpha_thin=0.3):
    """Plot mean + per-fold calibration curves for one model."""
    marker = MARKERS.get(label, "o")
    lstyle = LSTYLES.get(label, "-")
    folds = sorted(df_model["fold"].unique())
    fold_curves = []
    for fold in folds:
        sub = df_model[df_model["fold"] == fold].sort_values("mean_predicted")
        if len(sub) < 3:
            continue
        ax.plot(sub["mean_predicted"], sub["km_observed"],
                "-", color=color, lw=thin, alpha=alpha_thin, zorder=1)
        fold_curves.append(sub)

    if fold_curves:
        # Mean curve: average across folds by decile rank
        all_x, all_y = [], []
        min_len = min(len(fc) for fc in fold_curves)
        for fc in fold_curves:
            fc_sorted = fc.sort_values("mean_predicted").reset_index(drop=True)
            all_x.append(fc_sorted["mean_predicted"].values[:min_len])
            all_y.append(fc_sorted["km_observed"].values[:min_len])
        mean_x = np.mean(all_x, axis=0)
        mean_y = np.mean(all_y, axis=0)
        ax.plot(mean_x, mean_y, marker=marker, linestyle=lstyle, color=color,
                lw=thick, ms=5, label=label, zorder=2)


# ================================================================
# Figure 1: BRCA 2x2
# ================================================================
def fig1_brca():
    print("Generating Figure 1: BRCA calibration curves...")
    df = pd.read_csv("figures/calibration_curves_data.csv")
    brca = df[df["cancer"] == "BRCA"]

    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    panels = [
        ("Cox-PH", "Cox-PH control (1-cal: PASS)", C["Cox-PH"]),
        ("SurvPath", "SurvPath (1-cal: 5/5 fail)", C["SurvPath"]),
        ("MCAT", "MCAT (1-cal: 5/5 fail)", C["MCAT"]),
        ("MMP", "MMP (1-cal: 4/5 fail)", C["MMP"]),
    ]

    for idx, (model, title, color) in enumerate(panels):
        ax = axes.flat[idx]
        setup_cal_ax(ax)
        sub = brca[brca["model"] == model]
        plot_model_on_ax(ax, sub, color, model)
        ax.set_title(title, fontsize=12, fontweight="bold")

        if idx >= 2:
            ax.set_xlabel("Predicted S(t)")
        if idx % 2 == 0:
            ax.set_ylabel("Observed S(t)")

    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    fig.savefig("figures/fig1_calibration_brca.pdf")
    fig.savefig("figures/fig1_calibration_brca.png")
    plt.close()
    print("  Saved: figures/fig1_calibration_brca.{pdf,png}")


# ================================================================
# Figure 2: Multi-cancer 5 panels
# ================================================================
def fig2_multicancer():
    print("Generating Figure 2: Multi-cancer calibration curves...")
    df = pd.read_csv("figures/calibration_curves_data.csv")
    mc = df[df["cancer"] != "BRCA"].copy()
    # Also include BRCA from multicancer models (not Cox-PH)
    mc_brca = df[(df["cancer"] == "BRCA") & (df["model"].isin(["MCAT", "AMIL-sig", "AMIL-concat", "SNN"]))]
    mc = pd.concat([mc, mc_brca], ignore_index=True)

    cancers = ["BLCA", "BRCA", "GBMLGG", "LUAD", "UCEC"]
    models = [
        ("MCAT",        C["MCAT"]),
        ("AMIL-sig",    C["AMIL-sig"]),
        ("AMIL-concat", C["AMIL-concat"]),
        ("SNN",         C["SNN"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5))

    for idx, cancer in enumerate(cancers):
        ax = axes.flat[idx]
        setup_cal_ax(ax)
        for model, color in models:
            sub = mc[(mc["model"] == model) & (mc["cancer"] == cancer)]
            if sub.empty:
                continue
            plot_model_on_ax(ax, sub, color, model, thick=2.5, thin=0.7, alpha_thin=0.2)
        ax.set_title(cancer, fontsize=13, fontweight="bold")
        if idx >= 3:
            ax.set_xlabel("Predicted S(t)")
        if idx % 3 == 0:
            ax.set_ylabel("Observed S(t)")

    # Legend in empty 6th panel
    ax_leg = axes.flat[5]
    ax_leg.axis("off")
    handles = [plt.Line2D([0], [0], color=c, lw=2.5, marker=MARKERS.get(m, "o"),
                          linestyle=LSTYLES.get(m, "-"), ms=5, label=m)
               for m, c in models]
    ax_leg.legend(handles=handles, loc="center", fontsize=11, frameon=False)

    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    fig.savefig("figures/fig2_calibration_multicancer.pdf")
    fig.savefig("figures/fig2_calibration_multicancer.png")
    plt.close()
    print("  Saved: figures/fig2_calibration_multicancer.{pdf,png}")


# ================================================================
# Figure 3: Recalibration (Original vs Platt only)
# ================================================================
def fig3_recalibration():
    print("Generating Figure 3: Recalibration results...")
    # Need per-fold calibration curve data for Original and Platt.
    # Recompute from survival curves since we only have p-values in CSV.
    from calibration_audit import (
        logits_to_survival_interpolated, risk_to_survival_breslow,
        load_survpath_fold, load_mcat_fold, load_mmp_fold,
        load_survpath_train_data, load_mmp_train_data,
        N_TIME_POINTS,
    )
    from sklearn.linear_model import LogisticRegression
    from lifelines import KaplanMeierFitter

    models_cfg = {
        "SurvPath": {"loader": load_survpath_fold, "train_loader": load_survpath_train_data, "has_logits": True},
        "MCAT":     {"loader": load_mcat_fold,     "train_loader": load_survpath_train_data, "has_logits": False},
        "MMP":      {"loader": load_mmp_fold,      "train_loader": load_mmp_train_data,      "has_logits": False},
    }
    improvement = {"SurvPath": "5/5 \u2192 1/5 fail", "MCAT": "5/5 \u2192 2/5 fail", "MMP": "4/5 \u2192 3/5 fail"}

    def build_curves(val_df, train_t, train_e, has_logits):
        if has_logits:
            te = train_t[train_e > 0]
            cl, tc = [], None
            for _, row in val_df.iterrows():
                s, tc = logits_to_survival_interpolated(row["logits"], te)
                cl.append(s)
            return np.array(cl), tc
        else:
            te = train_t[train_e > 0]
            tc = np.percentile(te, np.linspace(5, 95, N_TIME_POINTS))
            tc = np.sort(np.unique(tc))
            curves = risk_to_survival_breslow(val_df["risk"].values, train_t, train_e, tc)
            return curves, tc

    def deciles_from_curves(curves, tc, val_times, val_events, t_eval, n_bins=10):
        predicted = np.array([np.interp(t_eval, tc, c) for c in curves])
        try:
            be = np.percentile(predicted, np.linspace(0, 100, n_bins + 1))
            be[0] -= 0.001; be[-1] += 0.001
        except Exception:
            return [], []
        xs, ys = [], []
        for i in range(n_bins):
            mask = (predicted >= be[i]) & (predicted < be[i + 1]) if i < n_bins - 1 else (predicted >= be[i]) & (predicted <= be[i + 1])
            if mask.sum() < 2:
                continue
            km = KaplanMeierFitter()
            km.fit(val_times[mask], event_observed=val_events[mask])
            xs.append(predicted[mask].mean())
            ys.append(float(km.predict(t_eval)))
        return xs, ys

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    for idx, model in enumerate(["SurvPath", "MCAT", "MMP"]):
        ax = axes[idx]
        setup_cal_ax(ax)
        cfg = models_cfg[model]

        # Build fold data
        fp, ft_d, fe_d, fte_d, curves_d, tc_d = {}, {}, {}, {}, {}, {}
        for f in range(5):
            vd = cfg["loader"](f)
            if vd.empty: continue
            tt, te = cfg["train_loader"](f)
            c, tc = build_curves(vd, tt, te, cfg["has_logits"])
            vt = vd["time"].values
            ve = vd["event"].values.astype(int)
            et = vt[ve.astype(bool)]
            tev = float(np.median(et)) if len(et) > 0 else None
            if tev is None: continue
            s_at_t = np.array([np.interp(tev, tc, cc) for cc in c])
            fp[f] = s_at_t; ft_d[f] = vt; fe_d[f] = ve; fte_d[f] = tev
            curves_d[f] = c; tc_d[f] = tc

        # Plot per-fold + mean for Original and Platt
        for method, color, mk, ls in [("Original", C["Original"], "o", "-"),
                                       ("Platt", C["Platt"], "s", "--")]:
            fold_xs, fold_ys = [], []
            for f in range(5):
                if f not in fp: continue
                if method == "Original":
                    pred_curves = curves_d[f]
                else:
                    # Fit Platt on other folds
                    Xt, yt = [], []
                    for f2 in fp:
                        if f2 == f: continue
                        mask = (fe_d[f2] == 1) | (ft_d[f2] >= fte_d[f2])
                        Xt.append(fp[f2][mask])
                        yt.append((ft_d[f2][mask] >= fte_d[f2]).astype(float))
                    lr = LogisticRegression(max_iter=1000)
                    lr.fit(np.concatenate(Xt).reshape(-1, 1), np.concatenate(yt))
                    flat = curves_d[f].flatten().reshape(-1, 1)
                    recal = lr.predict_proba(flat)[:, 1].reshape(curves_d[f].shape)
                    for i in range(recal.shape[0]):
                        recal[i] = np.minimum.accumulate(recal[i])
                    pred_curves = np.clip(recal, 0, 1)

                xs, ys = deciles_from_curves(
                    pred_curves, tc_d[f], ft_d[f], fe_d[f].astype(bool), fte_d[f])
                if xs:
                    ax.plot(xs, ys, "-", color=color, lw=0.7, alpha=0.25, zorder=1)
                    fold_xs.append(xs); fold_ys.append(ys)

            if fold_xs:
                ml = min(len(x) for x in fold_xs)
                mx = np.mean([x[:ml] for x in fold_xs], axis=0)
                my = np.mean([y[:ml] for y in fold_ys], axis=0)
                ax.plot(mx, my, marker=mk, linestyle=ls, color=color,
                        lw=2.5, ms=5, label=method, zorder=2)

        ax.set_title(model, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted S(t)")
        if idx == 0:
            ax.set_ylabel("Observed S(t)")
        ax.text(0.95, 0.08, improvement[model], transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        if idx == 2:
            ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout(w_pad=1.0)
    fig.savefig("figures/fig3_recalibration.pdf")
    fig.savefig("figures/fig3_recalibration.png")
    plt.close()
    print("  Saved: figures/fig3_recalibration.{pdf,png}")


# ================================================================
# Figure 4: Breslow control
# ================================================================
def fig4_breslow():
    print("Generating Figure 4: Breslow control...")
    bv = pd.read_csv("controls/breslow_validation.csv")

    # Recompute decile data (from paper_breslow_control.py logic)
    from calibration_audit import risk_to_survival_breslow
    from lifelines import KaplanMeierFitter, CoxPHFitter

    rna = pd.read_csv("SurvPath/data/rna_clean.csv", index_col=0)
    top_genes = rna.var(axis=0).sort_values(ascending=False).head(20).index.tolist()
    pi = pd.read_csv("patient_index.csv")
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

    vt = val["survival_time"].values.astype(float)
    ve = val["event"].values.astype(bool)
    tt = train["survival_time"].values.astype(float)
    te = train["event"].values.astype(int)
    t_eval = float(np.median(vt[ve]))

    def deciles(predicted, vt, ve, t_eval, n=10):
        be = np.percentile(predicted, np.linspace(0, 100, n + 1))
        be[0] -= 0.001; be[-1] += 0.001
        xs, ys = [], []
        for i in range(n):
            mask = (predicted >= be[i]) & (predicted < be[i + 1]) if i < n - 1 else (predicted >= be[i]) & (predicted <= be[i + 1])
            if mask.sum() < 2: continue
            km = KaplanMeierFitter()
            km.fit(vt[mask], event_observed=ve[mask])
            xs.append(predicted[mask].mean())
            ys.append(float(km.predict(t_eval)))
        return xs, ys

    # Lifelines
    sf = cph.predict_survival_function(val[top_genes])
    tc_l = sf.index.values
    curves_l = sf.T.values
    pred_l = np.array([np.interp(t_eval, tc_l, c) for c in curves_l])
    xl, yl = deciles(pred_l, vt, ve, t_eval)

    # KM-shift
    ph = cph.predict_partial_hazard(val[top_genes]).values.flatten()
    risks = np.log(ph)
    te_t = tt[te > 0]
    tc_k = np.percentile(te_t, np.linspace(5, 95, 20))
    tc_k = np.sort(np.unique(tc_k))
    curves_k = risk_to_survival_breslow(risks, tt, te, tc_k)
    pred_k = np.array([np.interp(t_eval, tc_k, c) for c in curves_k])
    xk, yk = deciles(pred_k, vt, ve, t_eval)

    # Random
    np.random.seed(42)
    rr = np.random.randn(len(val))
    curves_r = risk_to_survival_breslow(rr, tt, te, tc_k)
    pred_r = np.array([np.interp(t_eval, tc_k, c) for c in curves_r])
    xr, yr = deciles(pred_r, vt, ve, t_eval)

    p_l = bv[bv["method"] == "Lifelines (proper)"]["onecal_p"].values[0]
    p_k = bv[bv["method"] == "KM-shift (ours)"]["onecal_p"].values[0]
    p_r = bv[bv["method"] == "Random + KM-shift"]["onecal_p"].values[0]

    fig, ax = plt.subplots(figsize=(4, 3.5))
    setup_cal_ax(ax)
    ax.plot(xl, yl, "o-", color="#1f77b4", lw=3.0, ms=9, label=f"Lifelines (p={p_l:.3f})", zorder=2)
    ax.plot(xk, yk, "s--", color="#ff7f0e", lw=2.0, ms=5, label=f"KM-shift (p={p_k:.3f})", zorder=3,
            markeredgecolor="#ff7f0e", markerfacecolor="white", markeredgewidth=2.0)
    ax.plot(xr, yr, "^-", color="#2ca02c", lw=2.5, ms=6, label=f"Random (p={p_r:.3f})", zorder=1)

    ax.set_xlabel("Predicted survival probability")
    ax.set_ylabel("Observed survival proportion")
    ax.legend(fontsize=9, loc="lower right")

    # Annotation
    ax.text(0.05, 0.92, f"\u0394p = {abs(p_l - p_k):.3f}",
            transform=ax.transAxes, fontsize=9, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="gray", alpha=0.8))

    fig.tight_layout()
    fig.savefig("figures/fig4_breslow_control.pdf")
    fig.savefig("figures/fig4_breslow_control.png")
    plt.close()
    print("  Saved: figures/fig4_breslow_control.{pdf,png}")


# ================================================================
# Figure 5: Heatmap (NEW)
# ================================================================
def fig5_heatmap():
    print("Generating Figure 5: Architecture x Cancer heatmap...")

    archs = ["MCAT", "AMIL-sig", "AMIL-bilin", "AMIL-concat",
             "DS-sig", "DS-bilin", "DS-concat",
             "MIFCN-sig", "MIFCN-bilin", "MIFCN-concat", "SNN"]
    cancers = ["BLCA", "BRCA", "GBMLGG", "LUAD", "UCEC"]

    # Derive BH-corrected rejection counts from CSV
    csv_name_map = {"AMIL-cat": "AMIL-concat", "DS-cat": "DS-concat",
                    "MIFCN-cat": "MIFCN-concat"}
    df = pd.read_csv("results/multiple_testing.csv")
    df = df[df["experiment"] == "B"].copy()
    df["model"] = df["model"].replace(csv_name_map)
    counts = df.groupby(["model", "cancer"])["reject_bh"].sum().astype(int)
    data = np.array([[counts.get((a, c), 0) for c in cancers] for a in archs])

    # Colormap: white → yellow → orange → dark red
    cmap_colors = ["#ffffff", "#fff7bc", "#fec44f", "#fe9929", "#d95f0e", "#993404"]
    cmap = mcolors.LinearSegmentedColormap.from_list("cal_fail", cmap_colors, N=6)

    fig, ax = plt.subplots(figsize=(6, 5.5))

    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=5, aspect="auto")

    # Cell text
    for i in range(len(archs)):
        for j in range(len(cancers)):
            val = data[i, j]
            color = "white" if val >= 4 else "black"
            ax.text(j, i, f"{val}/5", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    # Labels
    ax.set_xticks(range(len(cancers)))
    ax.set_xticklabels(cancers, fontsize=11, fontweight="bold", rotation=0)
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels(archs, fontsize=10)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", pad=4)

    # Group separators
    for y in [0.5, 3.5, 6.5, 9.5]:
        ax.axhline(y=y, color="gray", lw=0.8)

    # Family labels on right — skip to avoid overlapping colorbar


    # Summary row
    totals = data.sum(axis=0)
    for j, t in enumerate(totals):
        ax.text(j, len(archs) - 0.5 + 0.7, f"{t}/55", ha="center", va="top",
                fontsize=10, fontweight="bold", color="#993404")
    ax.text(-0.8, len(archs) - 0.5 + 0.7, "Total", ha="center", va="top",
            fontsize=10, fontstyle="italic", color="gray")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label("Folds failed (out of 5)", fontsize=11)
    cbar.set_ticks([0, 1, 2, 3, 4, 5])

    ax.set_xlim(-0.5, len(cancers) - 0.5)
    ax.set_ylim(len(archs) - 0.5 + 0.6, -0.5)

    fig.tight_layout()
    fig.savefig("figures/fig5_heatmap.pdf")
    fig.savefig("figures/fig5_heatmap.png")
    plt.close()
    print("  Saved: figures/fig5_heatmap.{pdf,png}")


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    fig1_brca()
    fig2_multicancer()
    fig3_recalibration()
    fig4_breslow()
    fig5_heatmap()

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Figure | Size      | Location                                | Section")
    print(f"  -------|-----------|---------------------------------------|----------")
    print(f"  Fig 1  | 7x6\"     | figures/fig1_calibration_brca.pdf      | Main")
    print(f"  Fig 2  | 7.5x5\"   | figures/fig2_calibration_multicancer.pdf | Main")
    print(f"  Fig 3  | 7.5x2.8\" | figures/fig3_recalibration.pdf         | Main")
    print(f"  Fig 4  | 4x3.5\"   | figures/fig4_breslow_control.pdf       | Appendix")
    print(f"  Fig 5  | 5x5\"     | figures/fig5_heatmap.pdf               | Main")
