# Good Rankings, Wrong Probabilities

**A Calibration Audit of Multimodal Cancer Survival Models**

Sajad Ghawami | [sajad@ghawami.io](mailto:sajad@ghawami.io)

Submitted to MLHC 2026

---

Multimodal deep learning models that fuse whole-slide histopathology images with genomic data achieve strong concordance indices (0.60--0.82) for cancer survival prediction. But do the survival probabilities they produce actually mean anything? We conduct the first systematic fold-level 1-calibration audit of multimodal WSI-genomics survival architectures. Across 290 fold-level tests (11 architectures, 5 TCGA cancer types), 166 reject the null of correct calibration after Benjamini-Hochberg correction. Models with the highest discrimination are among the most miscalibrated. Post-hoc Platt scaling recovers much of the lost calibration without affecting ranking accuracy.

## Pre-computed results

All figures and results are included in this repository:

- **`figures/`** -- Figures 1-5 from the paper (PDF + PNG)
- **`results/`** -- Analysis CSVs (multiple testing, recalibration, power analysis, calibration quantification)
- **`controls/`** -- Breslow validation control results

## Repository structure

| File | Description |
|------|-------------|
| `calibration_audit.py` | Experiment A: 3 models on TCGA-BRCA (5-fold) |
| `calibration_audit_multicancer.py` | Experiment B: 11 architectures across 5 TCGA cancer types |
| `multi_horizon_calibration.py` | Time-dependent calibration at 1yr, 3yr, 5yr horizons |
| `power_and_calibration_quant.py` | Hosmer-Lemeshow power analysis, ECE/MCE quantification |
| `paper_multiple_testing.py` | Benjamini-Hochberg + Bonferroni correction |
| `paper_recalibration_v2.py` | Cross-fold Platt scaling and isotonic recalibration |
| `paper_calibration_curves.py` | Calibration curves with Greenwood confidence intervals |
| `paper_breslow_control.py` | Breslow baseline validation controls |
| `paper_figures.py` | Generate all 5 publication figures |
| `fetch_manifest.py` | Download GDC RNA-seq file manifest |
| `download_files.py` | Download RNA-seq files from GDC |
| `fetch_clinical.py` | Parse survival labels from GDC API |
| `download_features.py` | Download UNI2-h WSI embeddings (gated, HuggingFace) |
| `build_index.py` | Build patient index (1,004 TCGA-BRCA patients) |
| `build_rna_matrix.py` | Build RNA expression matrix |
| `convert_h5_to_pt.py` | Convert h5 feature files to PyTorch tensors |

## Full reproducibility

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- ~130 GB disk space (RNA-seq data + UNI2-h WSI features)
- HuggingFace account with access to [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) (gated dataset)

### Step 1: Install dependencies

```bash
uv sync
```

### Step 2: Set up model repositories

The audit evaluates three models from the [Mahmood Lab](https://github.com/mahmoodlab): SurvPath, MCAT, and MMP. We provide patch files that apply our modifications (Apple Silicon support, UNI2-h 1536-dim features, minor fixes) on top of pinned upstream commits.

```bash
bash patches/setup_models.sh
```

### Step 3: Data acquisition

Run the data pipeline scripts in order:

```bash
python fetch_manifest.py          # GDC RNA-seq manifest
python download_files.py          # Download RNA-seq (~4.9 GB)
python fetch_clinical.py          # Parse survival labels
python download_features.py       # Download UNI2-h embeddings (~65 GB, gated)
python build_index.py             # Build patient_index.csv (1,004 patients)
python build_rna_matrix.py        # Build RNA expression matrix
python convert_h5_to_pt.py        # Convert h5 -> PyTorch tensors for SurvPath
```

### Step 4: Train models

Train each model with 5-fold stratified cross-validation (seed 42). See the respective model repositories for training commands. Pre-trained fold models should be placed in `SurvPath/results/`, `MCAT/results/`, and `MMP/results/`.

### Step 5: Run experiments

```bash
python calibration_audit.py                # Experiment A
python calibration_audit_multicancer.py    # Experiment B
python multi_horizon_calibration.py        # Time-dependent analysis
python power_and_calibration_quant.py      # Power analysis + ECE/MCE
python paper_multiple_testing.py           # Multiple testing correction
python paper_recalibration_v2.py           # Post-hoc recalibration
python paper_calibration_curves.py         # Calibration curves with CIs
python paper_breslow_control.py            # Breslow validation controls
```

### Step 6: Generate figures

```bash
python paper_figures.py                    # Generates figures/fig{1-5}_*.{pdf,png}
```

## Citation

```bibtex
@inproceedings{ghawami2026calibration,
  title={Good Rankings, Wrong Probabilities: A Calibration Audit of Multimodal Cancer Survival Models},
  author={Ghawami, Sajad},
  booktitle={Machine Learning for Healthcare},
  year={2026}
}
```

## License

MIT -- see [LICENSE](LICENSE). The model repositories (SurvPath, MCAT, MMP) have their own licenses; see each repository for details.
