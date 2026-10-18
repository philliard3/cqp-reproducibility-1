#!/usr/bin/env bash
# ==============================================================================
# run_production_sweep.sh
# Master Production Experimental Sweep Runner
#
# RESTRICTION: THIS SCRIPT IS STRICTLY FOR HUMAN OPERATOR EXECUTION ONLY.
# AUTONOMOUS AI AGENTS ARE FORBIDDEN FROM EXECUTING FULL EXPERIMENTAL SWEEPS.
# ==============================================================================
set -euo pipefail

# 1. STRICT AGENT PROHIBITION ENFORCEMENT
if [ "${AI_AGENT:-0}" = "1" ] || [ "${AGENT_MODE:-0}" = "1" ] || [ -n "${AGENT_RUN:-}" ]; then
    echo "=============================================================================="
    echo " [CRITICAL POLICY VIOLATION]"
    echo " Autonomous AI agents are strictly forbidden from running longform production sweeps."
    echo " Refer to AGENT_GUIDELINES.md."
    echo " Use 'scripts/run_smoke_test.sh' for automated validation."
    echo "=============================================================================="
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_DIR="$PACKAGE_DIR/python"
ENGINE_DIR="$PACKAGE_DIR/engine"
DATA_DIR="$PACKAGE_DIR/data"

PRESET="${1:-paper_focused}"
OUTPUT_DIR="${2:-$PACKAGE_DIR/results_production_$(date +%Y%m%d_%H%M%S)}"

echo "=============================================================================="
echo " Conformal Query Processing (CQP) — Production Experimental Sweep"
echo " (Human Operator Mode: Interactive Production Orchestration)"
echo " Preset:     $PRESET"
echo " Output Dir: $OUTPUT_DIR"
echo "=============================================================================="

# 2. Environment Preflight
echo ">>> Running Environment Preflight Verification..."
bash "$SCRIPT_DIR/setup_env.sh"

# 3. Confirmation Prompt (Human safeguard)
echo ""
echo "NOTICE: Running preset '$PRESET' will generate full experimental workloads."
echo "Press Ctrl+C within 5 seconds to cancel, or wait to continue..."
sleep 5

# 4. Dispatch Production Orchestrator
echo ">>> Starting master production sweep orchestrator..."
python3 "$PYTHON_DIR/run_production_sweep.py" \
    --preset "$PRESET" \
    --config "$PYTHON_DIR/sweep_config.json" \
    --output-dir "$OUTPUT_DIR" \
    --cache-dir "$DATA_DIR/cached_tables" \
    --engine-bin-dir "$ENGINE_DIR/target/release"

echo "=============================================================================="
echo " Production Sweep Completed Successfully!"
echo " Results written to: $OUTPUT_DIR"
echo " To render paper charts, run: 'scripts/generate_paper_charts.sh $OUTPUT_DIR'"
echo "=============================================================================="
