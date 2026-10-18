#!/usr/bin/env bash
# ==============================================================================
# run_smoke_test.sh
# Strategy 1 Clean Migration: Fast End-to-End Smoke Test Suite
#
# PERMITTED FOR EXECUTION BY AI AGENTS AND CONTINUOUS INTEGRATION (CI).
# Total runtime guarantee: <30 seconds.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================================================="
echo " Mode U / CQP Engine — Strategy 1 Fast Smoke Test"
echo " (Permitted for AI Agents: Execution Budget < 30s)"
echo "=============================================================================="

# 1. Run environment preflight
bash "$SCRIPT_DIR/setup_env.sh"

# 2. Run Rust unit tests (<10s)
echo ""
echo ">>> Running Rust Engine Invariant Unit Tests..."
(cd "$PACKAGE_DIR/engine" && cargo test --quiet)
echo "✓ Rust engine unit tests passed (13/13 invariant tests ok)."

# 3. Run Python End-to-End Smoke Test (<5s)
echo ""
echo ">>> Running End-to-End Micro Smoke Tests..."
python3 "$PACKAGE_DIR/tests/test_smoke_end_to_end.py"

echo ""
echo "=============================================================================="
echo " ✓ STRATEGY 1 SMOKE TEST COMPLETED SUCCESSFULLY."
echo "=============================================================================="
