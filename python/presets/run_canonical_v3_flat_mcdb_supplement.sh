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
echo " Executing Canonical V3 Q1a & Q1b Flat MCDB Supplementary Sweep"
echo " - Queries:           Q1a (COCO Flat), Q1b (Jigsaw Flat)"
echo " - Evaluations:       MCDB-1000 Baseline (1,000 worlds bitvec simulation)"
echo " - Cluster Parameter: c=1 strictly fixed"
echo " - Skew Parameters:   z in {0.5, 1.0, 2.0}"
echo " - Python Binary:     $PY_BIN"
echo "=========================================================================="

# 1. Compile Rust engine release binaries if not yet built
if [ -d "mode_u_engine" ]; then
    echo "Ensuring Rust engine release binaries are compiled..."
    (cd mode_u_engine && cargo build --release --bin benchmark_flat_conformal)
elif [ -d "../mode_u_engine" ]; then
    echo "Ensuring Rust engine release binaries are compiled..."
    (cd ../mode_u_engine && cargo build --release --bin benchmark_flat_conformal)
fi

CACHE_DIR="../experiment_datasets/production_cache"
if [ ! -d "../experiment_datasets" ]; then
    CACHE_DIR="cache/production_cache"
fi

SUPP_DIR="results/canonical_v3_flat_mcdb_supplement"
BASE_DIR="results/canonical_v3"

# Phase 1: Run Q1a and Q1b Flat Evaluations
echo "\n>>> Phase 1: Running Flat Evaluations for Q1a and Q1b (including MCDB-1000)..."
"$PY_BIN" run_production_sweep.py \
    --preset canonical_v3 \
    --config sweep_config.json \
    --queries q1a,q1b \
    --output-dir "$SUPP_DIR" \
    --cache-dir "$CACHE_DIR"

# Phase 2: Merge Supplementary Records into Primary Canonical V3 Results
if [ -f "$BASE_DIR/unified_sweep_results.jsonl.gz" ]; then
    echo "\n>>> Phase 2: Merging Supplementary MCDB Records into Primary Canonical V3 Dataset..."
    "$PY_BIN" merge_sweep_results.py \
        --base-dir "$BASE_DIR" \
        --supplement-dir "$SUPP_DIR" \
        --output-dir "$BASE_DIR"

    # Phase 3: Re-render Publication Figures
    echo "\n>>> Phase 3: Re-generating Canonical V3 Figures with Flat MCDB-1000 Included..."
    "$PY_BIN" generate_paper_charts_v3.py \
        --results-dir "$BASE_DIR" \
        --figures-dir "$BASE_DIR/figures_v3"
else
    echo "\n>>> Note: Primary canonical results file ($BASE_DIR/unified_sweep_results.jsonl.gz) not found."
    echo "    Supplementary results are saved at $SUPP_DIR/unified_sweep_results.jsonl.gz"
fi

echo "=========================================================================="
echo "✓ Flat MCDB Supplement Complete!"
echo "=========================================================================="
