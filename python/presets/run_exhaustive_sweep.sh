#!/usr/bin/env bash
set -e
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PACKAGE_DIR="$( dirname "$SCRIPT_DIR" )"
cd "$PACKAGE_DIR"

# Resolve Python binary with virtual environment resolution cascade
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
echo " Executing Exhaustive Thesis Sweep (1000 cal / 250 eval seeds)"
echo " Python: $PY_BIN"
echo "=========================================================================="

"$PY_BIN" run_production_sweep.py \
    --preset exhaustive_thesis \
    --config sweep_config.json \
    --output-dir results/exhaustive_thesis \
    --cache-dir ../experiment_datasets/production_cache

echo "=========================================================================="
echo "✓ Exhaustive Thesis Sweep Complete!"
echo "  Check results/exhaustive_thesis/figures/ for all figures."
echo "=========================================================================="
