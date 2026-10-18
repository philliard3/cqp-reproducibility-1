#!/usr/bin/env python3
"""
run_scale_comparison_tpch.py

Standalone runner script for TPC-H Database Scale-Up (Part d of User 2 requirements):
- Queries: Q3 (2-Table Join) and Q4 (4-Table Join).
- Fixed Parameters: k = 3, z = 1.0, c = 1, alpha = 0.05.
- Scaled Parameter: dataset_scale = 1.0 (100% scale, 50,000 stream tuples).
- Scaled Candidate Pool: M = 20 (scaled up for larger database sizing).
- Evaluates:
    1. Naive (Full stream scan Baseline)
    2. Conformal_InOperator_Stop (Indep) (CQP)

IMPORTANT: The AI agent prepares this script for USER manual deployment.
The agent is strictly forbidden from executing benchmark sweeps.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any

RUNNER_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = RUNNER_DIR.parent
REPO_ROOT = PACKAGE_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))
sys.path.insert(0, str(PACKAGE_DIR / "lib"))
sys.path.insert(0, str(REPO_ROOT))

from lib.schema_codec import SchemaCodec, read_jsonl_gz, write_jsonl_gz
from lib.profiler import SweepProfiler
from runners.run_relational_tpch import run_relational_tpch_sweep


def main():
    parser = argparse.ArgumentParser(description="Run TPC-H Scale Comparison Sweep (Part d: scale 1.0, M=20, k=3, z=1.0, c=1).")
    parser.add_argument("--scale", type=float, default=1.0, help="Dataset scale factor (e.g. 1.0 for 100%% scale)")
    parser.add_argument("--M", type=int, default=20, help="Candidate pool size (scaled up to M=20)")
    parser.add_argument("--k", type=int, default=3, help="Target rank k (fixed at k=3)")
    parser.add_argument("--z", type=float, default=1.0, help="Zipf skew z (fixed at z=1.0)")
    parser.add_argument("--c", type=int, default=1, help="Cluster parameter c (fixed at c=1)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance budget alpha (fixed at alpha=0.05)")
    parser.add_argument("--cal-seeds", type=int, default=100, help="Number of calibration seeds (default: 100)")
    parser.add_argument("--eval-seeds", type=int, default=25, help="Number of evaluation seeds (default: 25)")
    parser.add_argument("--queries", nargs="+", default=["q3", "q4"], help="Target queries (q3, q4)")
    parser.add_argument("--dataset-dir", type=str, default="data/scale_1_0_datasets", help="Directory for ephemeral datasets")
    parser.add_argument("--output-dir", type=str, default="results/scale_1_0", help="Output directory for scale results")
    parser.add_argument("--engine-bin-dir", type=str, default=str(REPO_ROOT / "mode_u_engine" / "target" / "release"))
    args = parser.parse_args()

    t_start = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    codec = SchemaCodec(out_dir / "schema_lookup.json")
    profiler = SweepProfiler("tpch_scale_comparison_sweep")

    print("\n" + "=" * 70)
    print("TPC-H Scale Comparison Runner (Part d: User 2 Requirements)")
    print(f"  Queries:       {args.queries}")
    print(f"  Dataset Scale: {args.scale} (100% scale = 50,000 stream tuples)")
    print(f"  M (Candidates):{args.M} (scaled up)")
    print(f"  k (Rank):      {args.k} (fixed)")
    print(f"  z (Skew):      {args.z} (fixed)")
    print(f"  c (Cluster):   {args.c} (fixed single-mode)")
    print(f"  alpha:         {args.alpha} (95% guarantee)")
    print(f"  Seeds:         {args.cal_seeds} cal / {args.eval_seeds} eval")
    print(f"  Output Dir:    {out_dir}")
    print("=" * 70 + "\n")

    combined_records = []
    cal_seeds_list = list(range(1000, 1000 + args.cal_seeds))
    eval_seeds_list = list(range(2000, 2000 + args.eval_seeds))

    for q in args.queries:
        print(f"\n>>> [{time.strftime('%H:%M:%S')}] Starting evaluation for query {q.upper()} at scale {args.scale}...")
        log_path = run_relational_tpch_sweep(
            query=q,
            dataset_dir=dataset_dir,
            output_dir=out_dir,
            M_list=[args.M],
            z_list=[args.z],
            c_list=[args.c],
            k_list=[args.k],
            alphas=[args.alpha],
            cal_seeds=cal_seeds_list,
            eval_seeds=eval_seeds_list,
            engine_bin_dir=Path(args.engine_bin_dir),
            codec=codec,
            profiler=profiler,
            dataset_scale=args.scale,
        )
        if log_path.exists():
            combined_records.extend(read_jsonl_gz(log_path, codec))
        print(f"✓ Completed {q.upper()} at scale {args.scale} -> {log_path}")

    # Emit unified file if multiple queries were run
    unified_path = out_dir / "unified_sweep_results.jsonl.gz"
    write_jsonl_gz(combined_records, unified_path, codec)
    codec.save_lookup(out_dir / "schema_lookup.json")

    elapsed_sec = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"✓ All scale-up benchmarks complete in {elapsed_sec:.1f}s ({elapsed_sec/60:.2f} min).")
    print(f"  Total records generated: {len(combined_records)}")
    print(f"  Unified results saved:   {unified_path}")
    print("  The V3 plotting suite will automatically incorporate these results upon next render.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
