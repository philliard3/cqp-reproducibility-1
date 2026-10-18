#!/usr/bin/env bash
set -e
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PACKAGE_DIR="$( dirname "$SCRIPT_DIR" )"
cd "$PACKAGE_DIR"

# Resolve Python binary with virtual environment resolution cascade
PY_BIN="python3"
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PY_BIN="$VIRTUAL_ENV/bin/python"
elif [[ -x "$PACKAGE_DIR/.venv/bin/python" ]]; then
    PY_BIN="$PACKAGE_DIR/.venv/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
    PY_BIN=".venv/bin/python"
elif [[ -x "../.venv/bin/python" ]]; then
    PY_BIN="../.venv/bin/python"
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
echo " Executing Q3 Extended Top-k Scaling Sweep (User 2 Specifications)"
echo " - Query:             Q3 (TPC-H 2-Table Join)"
echo " - Candidate Space:   M = 20 (Scale 0.5, 25,000 stream tuples)"
echo " - Target Ranks:      k in {1, 3, 5, 7, 9, 10, 11, 13, 15, 17, 19, 20}"
echo " - Skew Parameters:   z in {1.0, 2.0}"
echo " - Cluster Parameter: c = 1 strictly fixed"
echo " - Alphas:            alpha = 0.05"
echo " - Strategies:        Baseline (Naive) and CQP (In-Operator Stop)"
echo " - Python Binary:     $PY_BIN"
echo "=========================================================================="

# 1. Compile Rust engine release binaries if not yet built
if [ -d "mode_u_engine" ]; then
    echo "Ensuring Rust engine release binaries are compiled..."
    (cd mode_u_engine && cargo build --release --bin benchmark_tpch_q3)
elif [ -d "../mode_u_engine" ]; then
    echo "Ensuring Rust engine release binaries are compiled..."
    (cd ../mode_u_engine && cargo build --release --bin benchmark_tpch_q3)
fi

CACHE_DIR="../experiment_datasets/production_cache"
if [ ! -d "../experiment_datasets" ]; then
    CACHE_DIR="cache/production_cache"
fi

SUPP_DIR="results/q3_extended_k_supplement"
BASE_DIR="results/canonical_v3"

# Phase 1: Run Q3 Extended Sweep across k in {1, 3, 5, 7, 10, 12, 15}
echo -e "\n>>> Phase 1: Running Q3 Extended Top-k Scaling Benchmarks..."
"$PY_BIN" run_production_sweep.py \
    --preset q3_extended_k_sweep \
    --config sweep_config.json \
    --queries q3 \
    --output-dir "$SUPP_DIR" \
    --cache-dir "$CACHE_DIR"

if [ -f "$BASE_DIR/unified_sweep_results.jsonl.gz" ]; then
    # Phase 2: Merge Supplementary Records into Primary Canonical V3 Dataset
    if [ -d "results/canonical_v3_flat_mcdb_supplement" ]; then
        echo -e "\n>>> Phase 2a: Merging Flat MCDB Supplementary Records into Canonical V3..."
        "$PY_BIN" merge_sweep_results.py \
            --base-dir "$BASE_DIR" \
            --supplement-dir "results/canonical_v3_flat_mcdb_supplement" \
            --output-dir "$BASE_DIR"
    fi
    echo -e "\n>>> Phase 2b: Merging Supplementary Q3 Records into Primary Canonical V3 Dataset..."
    "$PY_BIN" merge_sweep_results.py \
        --base-dir "$BASE_DIR" \
        --supplement-dir "$SUPP_DIR" \
        --output-dir "$BASE_DIR"

    # Phase 3: Re-render Publication Figures
    echo -e "\n>>> Phase 3: Re-generating Canonical V3 Figures with Extended Q3 Top-k Suite..."
    "$PY_BIN" generate_paper_charts_v3.py \
        --results-dir "$BASE_DIR" \
        --figures-dir "$BASE_DIR/figures_v3"
else
    echo -e "\n>>> Note: Primary canonical results file ($BASE_DIR/unified_sweep_results.jsonl.gz) not found."
    echo "    Supplementary results are saved at $SUPP_DIR/unified_sweep_results.jsonl.gz"
fi

echo "=========================================================================="
echo "✓ Q3 Extended Top-k Sweep Complete!"
echo "=========================================================================="
