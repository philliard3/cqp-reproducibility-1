#!/usr/bin/env bash
# ==============================================================================
# generate_paper_charts.sh
# Camera-Ready Paper Chart and Table Generator (Strategy 1 Clean Migration)
#
# Generates publication-grade figures (Figures 1a-e) and Table 1
# from completed production sweep outputs.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_DIR="$PACKAGE_DIR/python"

INPUT_DIR="${1:-$PACKAGE_DIR/data/reference_results/canonical_v3}"
OUTPUT_DIR="${2:-$PACKAGE_DIR/figures_v3}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Results directory '$INPUT_DIR' not found." >&2
    echo "Usage: scripts/generate_paper_charts.sh [path_to_results] [output_fig_dir]" >&2
    exit 1
fi

echo "=============================================================================="
echo " Rendering Camera-Ready Figures and Tables"
echo " Input Data:  $INPUT_DIR"
echo " Output Figs: $OUTPUT_DIR"
echo "=============================================================================="

mkdir -p "$OUTPUT_DIR"

# 1. Generate Table 1 Statistical Coverage & Recall Validation
if [ -f "$INPUT_DIR/unified_sweep_results.jsonl.gz" ]; then
    echo ">>> Generating Table 1 Coverage & Recall Validation..."
    LOOKUP_ARG=""
    if [ -f "$INPUT_DIR/schema_lookup.json" ]; then
        LOOKUP_ARG="--lookup $INPUT_DIR/schema_lookup.json"
    fi
    python3 "$PYTHON_DIR/analyze_results.py" \
        --results "$INPUT_DIR/unified_sweep_results.jsonl.gz" \
        $LOOKUP_ARG \
        --output-dir "$OUTPUT_DIR/tables" || true
fi

# 2. Render Figures across all parts and profiles (paper & presentation)
python3 "$PYTHON_DIR/generate_paper_charts_v3.py" \
    --results-dir "$INPUT_DIR" \
    --figures-dir "$OUTPUT_DIR"

echo "✓ Paper charts successfully generated in: $OUTPUT_DIR"

