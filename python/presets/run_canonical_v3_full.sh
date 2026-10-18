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
    echo "Notice: Required Python dependencies missing in '$PY_BIN'."
    echo "Running setup_env.sh to install/update isolated environment..."
    bash "$PACKAGE_DIR/../scripts/setup_env.sh" || true
fi

echo "=========================================================================="
echo " Executing Canonical V3 Full Suite (User 2 Specifications)"
echo " - Cluster Parameter: c=1 strictly fixed (no cluster variation)"
echo " - Distributions:     z in {0.5, 1.0, 2.0}"
echo " - Alphas:            alpha in {0.01, 0.05, 0.10}"
echo " - Seeds:             100 calibration / 25 evaluation seeds"
echo " - Q3 Scaling:        Includes Extended M=20 Top-k Sweep"
echo " Python:              $PY_BIN"
echo "=========================================================================="

CACHE_DIR="$PACKAGE_DIR/../data/cached_tables"
if [ ! -d "$CACHE_DIR" ]; then
    CACHE_DIR="cache/production_cache"
fi

# Ensure Rust engine binaries are compiled
ENGINE_DIR="$PACKAGE_DIR/../engine"
if [ -d "$ENGINE_DIR" ]; then
    echo "Ensuring Rust engine release binaries are compiled..."
    (cd "$ENGINE_DIR" && cargo build --release --bins)
elif [ -d "engine" ]; then
    (cd engine && cargo build --release --bins)
fi

BASE_DIR="results/canonical_v3"
SUPP_DIR="results/q3_extended_k_supplement"

# Phase 1: Canonical Primary Sweep (all 6 queries at c=1)
echo -e "\n>>> Phase 1: Running Primary Canonical V3 Benchmarks..."
"$PY_BIN" run_production_sweep.py \
    --preset canonical_v3 \
    --config sweep_config.json \
    --output-dir "$BASE_DIR" \
    --cache-dir "$CACHE_DIR"

# Phase 1b: Extended Q3 Top-k Scaling Sweep (M=20)
echo -e "\n>>> Phase 1b: Running Extended Q3 M=20 Top-k Scaling Benchmarks..."
"$PY_BIN" run_production_sweep.py \
    --preset q3_extended_k_sweep \
    --config sweep_config.json \
    --queries q3 \
    --output-dir "$SUPP_DIR" \
    --cache-dir "$CACHE_DIR"

# Phase 1c: Merge Supplementary Records into Primary Dataset
echo -e "\n>>> Phase 1c: Merging Q3 Extended Records into Canonical Dataset..."
"$PY_BIN" merge_sweep_results.py \
    --base-dir "$BASE_DIR" \
    --supplement-dir "$SUPP_DIR" \
    --output-dir "$BASE_DIR"

# Phase 2: Render Canonical V3 Paper and Presentation Charts
echo -e "\n>>> Phase 2: Generating Canonical V3 Publication Figures..."
"$PY_BIN" generate_paper_charts_v3.py \
    --results-dir "$BASE_DIR" \
    --figures-dir "$BASE_DIR/figures_v3"

echo "=========================================================================="
echo "✓ Full Canonical V3 Suite Complete!"
echo "  All results aggregated in: $BASE_DIR/unified_sweep_results.jsonl.gz"
echo "  Artifacts saved in:        $BASE_DIR/figures_v3/"
echo "=========================================================================="

