#!/bin/bash
# Clone and patch the three model repositories used in the calibration audit.
# Each repo is pinned to a specific commit and patched for compatibility
# (Apple Silicon / MPS support, UNI2-h 1536-dim features, minor fixes).
#
# Usage: bash patches/setup_models.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Setting up model repositories..."

# ── SurvPath ──────────────────────────────────────────────────
echo ""
echo "==> Cloning SurvPath..."
git clone https://github.com/mahmoodlab/SurvPath "$REPO_ROOT/SurvPath"
cd "$REPO_ROOT/SurvPath"
git checkout 3f73ddd
git apply "$SCRIPT_DIR/survpath.patch"
echo "    SurvPath patched OK"

# ── MCAT ──────────────────────────────────────────────────────
echo ""
echo "==> Cloning MCAT..."
git clone https://github.com/mahmoodlab/MCAT "$REPO_ROOT/MCAT"
cd "$REPO_ROOT/MCAT"
git checkout b9cca63
git apply "$SCRIPT_DIR/mcat.patch"
echo "    MCAT patched OK"

# ── MMP (PANTHER) ────────────────────────────────────────────
echo ""
echo "==> Cloning MMP..."
git clone https://github.com/mahmoodlab/MMP "$REPO_ROOT/MMP"
cd "$REPO_ROOT/MMP"
git checkout 1fd75e3
git apply "$SCRIPT_DIR/mmp.patch"
echo "    MMP patched OK"

echo ""
echo "All model repositories set up. See README.md for data preparation steps."
