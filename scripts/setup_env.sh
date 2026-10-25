#!/usr/bin/env bash
# ==============================================================================
# setup_env.sh
# Strategy 1 Clean Migration: Environment Setup and Preflight Verifier
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================================================="
echo " Conformal Query Processing (CQP) — Strategy 1 Environment Setup"
echo " Package Root: $PACKAGE_DIR"
echo "=============================================================================="

# 1. Verify Rust toolchain & build release engine binaries
echo ">>> [1/4] Verifying Rust toolchain..."
if ! command -v cargo &> /dev/null; then
    echo "ERROR: 'cargo' is required but not found in PATH." >&2
    exit 1
fi
echo "Rust version: $(cargo --version)"

echo ">>> Building Mode U Query Engine (Release profile)..."
(cd "$PACKAGE_DIR/engine" && cargo build --release)
echo "✓ Mode U Query Engine release binaries built successfully."

# 2. Verify Python environment
echo ">>> [2/4] Verifying Python environment..."
if [ -f "$PACKAGE_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$PACKAGE_DIR/.venv/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "ERROR: '$PYTHON_BIN' is required but not found in PATH." >&2
    exit 1
fi
echo "Python version: $($PYTHON_BIN --version)"

echo "Checking required Python packages..."
if ! "$PYTHON_BIN" -c "
import numpy
import scipy
import pandas
import matplotlib
import seaborn
" 2>/dev/null; then
    echo "Notice: Some required Python packages are missing in current environment."
    if [ ! -d "$PACKAGE_DIR/.venv" ]; then
        echo "Creating localized virtual environment at .venv..."
        "$PYTHON_BIN" -m venv "$PACKAGE_DIR/.venv" || python3 -m venv "$PACKAGE_DIR/.venv"
    fi
    PYTHON_BIN="$PACKAGE_DIR/.venv/bin/python"
    echo "Installing requirements into .venv using $PACKAGE_DIR/.venv/bin/pip..."
    "$PACKAGE_DIR/.venv/bin/pip" install -q --upgrade pip
    "$PACKAGE_DIR/.venv/bin/pip" install -q -r "$PACKAGE_DIR/python/requirements.txt"
fi
echo "✓ All core Python packages verified (NumPy, SciPy, Pandas, Matplotlib, Seaborn)."

# 3. Verify Cached Data Assets
echo ">>> [3/4] Checking frozen base tables and calibration artifacts..."
MISSING_DATA=0
for table in "coco_animals_calibrated_softmax.json" "jigsaw_toxicity_platt.json"; do
    if [ ! -f "$PACKAGE_DIR/data/cached_tables/$table" ]; then
        echo "ERROR: Missing expected base table: data/cached_tables/$table" >&2
        MISSING_DATA=1
    fi
done

for artifact in "coco_platt_model.json" "jigsaw_platt_model.json"; do
    if [ ! -f "$PACKAGE_DIR/data/calibration_artifacts/$artifact" ]; then
        echo "ERROR: Missing calibration artifact: data/calibration_artifacts/$artifact" >&2
        MISSING_DATA=1
    fi
done

if [ "$MISSING_DATA" -eq 1 ]; then
    echo "Data preflight check failed. Please ensure base tables are populated." >&2
    exit 1
fi
echo "✓ All required base data tables and calibration artifacts verified."

# 4. Summary
echo ">>> [4/4] Setup complete!"
echo "Ready for verification: execute 'scripts/run_smoke_test.sh' for smoke tests."
echo "=============================================================================="
