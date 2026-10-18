#!/usr/bin/env bash
set -e
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PACKAGE_DIR="$( dirname "$SCRIPT_DIR" )"
cd "$PACKAGE_DIR"

PY_BIN="python3"
if [[ -n "$VIRTUAL_ENV" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PY_BIN="$VIRTUAL_ENV/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
    PY_BIN=".venv/bin/python"
elif [[ -x "../.venv/bin/python" ]]; then
    PY_BIN="../.venv/bin/python"
elif [[ -x "python_env/bin/python" ]]; then
    PY_BIN="python_env/bin/python"
elif [[ -x "../python_env/bin/python" ]]; then
    PY_BIN="../python_env/bin/python"
elif [[ -x "$HOME/pdb_benchmarks/.venv/bin/python" ]]; then
    PY_BIN="$HOME/pdb_benchmarks/.venv/bin/python"
fi

# Preflight dependency verification: Ensure numpy, matplotlib, scipy, pandas are installed
if ! "$PY_BIN" -c "import pandas, numpy, scipy, matplotlib" &>/dev/null; then
    echo "Notice: Required Python dependencies (e.g. pandas) missing in '$PY_BIN'."
    echo "Running setup_env.sh to install/update isolated environment..."
    bash setup_env.sh
    if [[ -x ".venv/bin/python" ]]; then
        PY_BIN=".venv/bin/python"
    elif [[ -x "python_env/bin/python" ]]; then
        PY_BIN="python_env/bin/python"
    fi
fi

echo "=========================================================================="
echo " Executing Targeted Part d: TPC-H Scale 1.0 (100% scale, M=20, k=3, z=1.0)"
echo " Estimated Runtime: ~3.5 - 4.0 minutes (100 cal + 25 eval seeds across Q3 & Q4)"
echo " Python:            $PY_BIN"
echo "=========================================================================="

SCALE_OUT="results/scale_1_0"
RESULTS_BASE="results/canonical_v3"
FIGURES_BASE="results/canonical_v3/figures_v3"

if [[ -d "prod_sweep_results_oct13_3pm/paper_focused_quick_results_oct13_3pm/results" ]]; then
    SCALE_OUT="prod_sweep_results_oct13_3pm/paper_focused_quick_results_oct13_3pm/results/scale_1_0"
    RESULTS_BASE="prod_sweep_results_oct13_3pm/paper_focused_quick_results_oct13_3pm/results/paper_focused_quick"
    FIGURES_BASE="prod_sweep_results_oct13_3pm/paper_focused_quick_results_oct13_3pm/results/paper_focused_quick/figures_v3"
fi

"$PY_BIN" runners/run_scale_comparison_tpch.py \
    --scale 1.0 \
    --M 20 \
    --k 3 \
    --z 1.0 \
    --c 1 \
    --alpha 0.05 \
    --queries q3 q4 \
    --output-dir "$SCALE_OUT"

echo ">>> Re-rendering Canonical V3 Figures with new scale 1.0 data..."
"$PY_BIN" generate_paper_charts_v3.py \
    --results-dir "$RESULTS_BASE" \
    --figures-dir "$FIGURES_BASE" \
    --supplementary-dirs "$SCALE_OUT"

echo "=========================================================================="
echo "✓ Targeted Part d Scale Comparison Complete!"
echo "=========================================================================="
