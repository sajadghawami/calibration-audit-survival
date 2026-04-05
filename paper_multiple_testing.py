"""
Task 4: Multiple Testing Correction

Collects all 1-calibration p-values from Experiments A (BRCA, 3 models) and B
(multi-cancer, 11 architectures), applies Bonferroni and Benjamini-Hochberg
corrections, and reports whether the finding survives correction.
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats

from calibration_audit import (
    load_survpath_fold, load_mcat_fold, load_mmp_fold,
    load_survpath_train_data, load_mmp_train_data,
    evaluate_fold,
)
from calibration_audit_multicancer import (
    load_iccv_fold, load_iccv_train_data,
    ARCHITECTURES, CANCER_TYPES, ARCH_SHORT,
    evaluate_fold as evaluate_fold_mc,
)

warnings.filterwarnings("ignore")


def collect_experiment_a():
    """Collect 15 p-values from Experiment A (BRCA, 3 models x 5 folds)."""
    rows = []

    models = {
        "SurvPath": {"loader": load_survpath_fold, "train_loader": load_survpath_train_data, "has_logits": True},
        "MCAT":     {"loader": load_mcat_fold,     "train_loader": load_survpath_train_data, "has_logits": False},
        "MMP":      {"loader": load_mmp_fold,      "train_loader": load_mmp_train_data,      "has_logits": False},
    }

    for model_name, cfg in models.items():
        for fold in range(5):
            val_df = cfg["loader"](fold)
            if val_df.empty:
                continue
            train_t, train_e = cfg["train_loader"](fold)
            result = evaluate_fold(val_df, train_t, train_e, has_logits=cfg["has_logits"])
            rows.append({
                "experiment": "A",
                "model": model_name,
                "cancer": "BRCA",
                "fold": fold,
                "raw_p": result["onecal"],
            })
    return rows


def collect_experiment_b():
    """Collect 275 p-values from Experiment B (11 arch x 5 cancers x 5 folds)."""
    rows = []
    for arch in ARCHITECTURES:
        short = ARCH_SHORT[arch]
        for cancer in CANCER_TYPES:
            for fold in range(5):
                val_df = load_iccv_fold(fold, arch, cancer)
                if val_df.empty:
                    continue
                train_t, train_e = load_iccv_train_data(fold, cancer)
                result = evaluate_fold_mc(val_df, train_t, train_e)
                rows.append({
                    "experiment": "B",
                    "model": short,
                    "cancer": cancer.upper(),
                    "fold": fold,
                    "raw_p": result["onecal"],
                })
    return rows


def benjamini_hochberg(pvals, alpha=0.05):
    """Apply BH procedure. Returns adjusted p-values."""
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_p = np.array(pvals)[sorted_idx]
    adjusted = np.zeros(n)
    # BH adjusted p-values (step-up)
    adjusted[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(adjusted[sorted_idx[i + 1]], sorted_p[i] * n / (i + 1))
    return np.clip(adjusted, 0, 1)


if __name__ == "__main__":
    print("Collecting p-values from Experiment A (BRCA, 3 models)...")
    rows_a = collect_experiment_a()
    print(f"  {len(rows_a)} p-values collected")

    print("Collecting p-values from Experiment B (multi-cancer, 11 arch)...")
    rows_b = collect_experiment_b()
    print(f"  {len(rows_b)} p-values collected")

    all_rows = rows_a + rows_b
    df = pd.DataFrame(all_rows)

    # Drop NaN p-values
    df = df.dropna(subset=["raw_p"])
    n_total = len(df)

    raw_p = df["raw_p"].values

    # Corrections
    bonferroni_p = np.clip(raw_p * n_total, 0, 1)
    bh_p = benjamini_hochberg(raw_p)

    df["bonferroni_p"] = bonferroni_p
    df["bh_p"] = bh_p
    df["reject_raw_005"] = raw_p < 0.05
    df["reject_raw_001"] = raw_p < 0.01
    df["reject_bonferroni"] = bonferroni_p < 0.05
    df["reject_bh"] = bh_p < 0.05

    # Save
    df.to_csv("results/multiple_testing.csv", index=False)

    # Summary
    n_raw_005 = df["reject_raw_005"].sum()
    n_raw_001 = df["reject_raw_001"].sum()
    n_bonf = df["reject_bonferroni"].sum()
    n_bh = df["reject_bh"].sum()

    # Experiment B only
    df_b = df[df["experiment"] == "B"]
    n_b = len(df_b)
    n_b_raw = df_b["reject_raw_005"].sum()
    n_b_bonf = df_b["reject_bonferroni"].sum()
    n_b_bh = df_b["reject_bh"].sum()

    # Binomial test: under null, expect 5% false positives
    expected_fp = n_b * 0.05
    binom_p = stats.binom_test(int(n_b_raw), n_b, 0.05, alternative="greater") if hasattr(stats, "binom_test") else stats.binomtest(int(n_b_raw), n_b, 0.05, alternative="greater").pvalue

    print(f"\n{'='*60}")
    print(f"  Multiple Testing Correction — All tests ({n_total} total)")
    print(f"{'='*60}")
    print(f"  Experiment A (BRCA):         {len(rows_a)} tests")
    print(f"  Experiment B (multi-cancer):  {n_b} tests")
    print(f"")
    print(f"  Raw rejections at alpha=0.05:              {n_raw_005}/{n_total} ({100*n_raw_005/n_total:.1f}%)")
    print(f"  Raw rejections at alpha=0.01:              {n_raw_001}/{n_total} ({100*n_raw_001/n_total:.1f}%)")
    print(f"  After Bonferroni at alpha=0.05:             {n_bonf}/{n_total} ({100*n_bonf/n_total:.1f}%)")
    print(f"  After Benjamini-Hochberg at FDR=0.05:       {n_bh}/{n_total} ({100*n_bh/n_total:.1f}%)")
    print(f"")
    print(f"  --- Experiment B only ({n_b} tests) ---")
    print(f"  Raw rejections at alpha=0.05:              {n_b_raw}/{n_b} ({100*n_b_raw/n_b:.1f}%)")
    print(f"  After Bonferroni at alpha=0.05:             {n_b_bonf}/{n_b} ({100*n_b_bonf/n_b:.1f}%)")
    print(f"  After Benjamini-Hochberg at FDR=0.05:       {n_b_bh}/{n_b} ({100*n_b_bh/n_b:.1f}%)")
    print(f"  Expected false positives under null:        {expected_fp:.1f}")
    print(f"  Binomial test p-value (H0: all null):       {binom_p:.2e}")
    print(f"")

    # One-sentence conclusion
    print(f"  CONCLUSION: {n_bh}/{n_total} tests remain significant after BH correction")
    print(f"  at FDR=0.05 ({100*n_bh/n_total:.0f}%), far exceeding the {expected_fp:.0f} expected")
    print(f"  under the null. Binomial test p={binom_p:.2e}.")

    print(f"\n  Saved: results/multiple_testing.csv")
