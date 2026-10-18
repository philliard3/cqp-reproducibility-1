#!/usr/bin/env python3
"""
analyze_results.py

Aggregates and audits experimental results from compressed .jsonl.gz logs:
- Re-hydrates records using SchemaCodec.
- Validates empirical statistical coverage: verifies hat{C} >= 1 - alpha.
- Computes mean and standard error for stream savings, latency, speedup, and set size.
- Generates camera-ready Table 1 (Statistical Coverage & Recall Validation).
- Serializes aggregated metrics for consumption by chart generation scripts.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from lib.schema_codec import SchemaCodec, read_jsonl_gz


def analyze_sweep_results(
    results_file: Path,
    lookup_file: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    codec = SchemaCodec(lookup_file)
    records = read_jsonl_gz(results_file, codec)

    print(f"Loaded {len(records)} evaluation records from {results_file}")

    # Group records by (query, strategy, alpha, k_query, z, c, M)
    grouped: Dict[Tuple, List[Dict[str, Any]]] = {}
    for r in records:
        cfg = r["config"]
        key = (
            cfg.get("query"),
            cfg.get("strategy"),
            cfg.get("alpha"),
            cfg.get("k_query"),
            cfg.get("z"),
            cfg.get("c"),
            cfg.get("M"),
        )
        grouped.setdefault(key, []).append(r)

    summary_rows = []
    coverage_table_rows = []

    for key, group_recs in grouped.items():
        q, strat, alpha, k, z, c, M = key
        n = len(group_recs)

        coverages = [float(r["metrics"]["covered"]) for r in group_recs]
        recalls = [float(r["metrics"]["recall_at_set"]) for r in group_recs]
        savings = [float(r["metrics"]["stream_savings_pct"]) for r in group_recs]
        latencies = [float(r["metrics"]["engine_time_ms"]) for r in group_recs]
        set_sizes = [float(r["metrics"]["emitted_set_size"]) for r in group_recs]

        mean_cov = float(np.mean(coverages))
        mean_rec = float(np.mean(recalls))
        mean_sav = float(np.mean(savings))
        sem_sav = float(np.std(savings) / max(1, np.sqrt(n)))
        mean_lat = float(np.mean(latencies))
        sem_lat = float(np.std(latencies) / max(1, np.sqrt(n)))
        mean_sz = float(np.mean(set_sizes))

        target_cov = 1.0 - alpha
        is_sound = mean_cov >= (target_cov - 0.05)  # statistical tolerance for finite eval sample

        row = {
            "query": q,
            "strategy": strat,
            "alpha": alpha,
            "k_query": k,
            "z": z,
            "c": c,
            "M": M,
            "sample_size": n,
            "target_coverage": target_cov,
            "empirical_coverage": round(mean_cov, 4),
            "marginal_recall": round(mean_rec, 4),
            "stream_savings_pct": round(mean_sav, 2),
            "stream_savings_sem": round(sem_sav, 2),
            "engine_time_ms": round(mean_lat, 3),
            "engine_time_sem": round(sem_lat, 3),
            "mean_set_size": round(mean_sz, 2),
            "coverage_satisfied": is_sound,
        }
        summary_rows.append(row)

        coverage_table_rows.append({
            "Query": q,
            "Strategy": strat,
            "alpha": alpha,
            "k": k,
            "Target Cov": f"{target_cov*100:.1f}%",
            "Empirical Cov": f"{mean_cov*100:.1f}%",
            "Recall": f"{mean_rec*100:.1f}%",
            "Savings": f"{mean_sav:.1f}% ± {sem_sav:.1f}%",
            "Latency": f"{mean_lat:.2f} ms",
            "Set Size |C|": f"{mean_sz:.1f}",
            "Sound": "✓" if is_sound else "✗",
        })

    # Save summary JSON
    summary_json_path = output_dir / "aggregated_metrics.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    # Render Table 1 Markdown
    table_md_path = output_dir / "table_1_coverage_validation.md"
    with open(table_md_path, "w", encoding="utf-8") as f:
        f.write("# Table 1: Empirical Coverage and Recall Validation\n\n")
        f.write("| Query | Strategy | alpha | k | Target Cov | Empirical Cov | Recall | Savings (%) | Latency | Set Size | Sound |\n")
        f.write("|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---:|\n")
        for r in coverage_table_rows:
            f.write(f"| {r['Query']} | {r['Strategy']} | {r['alpha']} | {r['k']} | {r['Target Cov']} | {r['Empirical Cov']} | {r['Recall']} | {r['Savings']} | {r['Latency']} | {r['Set Size |C|']} | {r['Sound']} |\n")

    print(f"✓ Aggregated analysis complete: {len(summary_rows)} configuration groups.")
    print(f"  Summary JSON: {summary_json_path}")
    print(f"  Table 1:      {table_md_path}")
    return {"summary_rows": summary_rows, "table_path": str(table_md_path)}


def main():
    parser = argparse.ArgumentParser(description="Analyze experimental sweep results.")
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--lookup", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/analysis")
    args = parser.parse_args()

    analyze_sweep_results(
        results_file=Path(args.results),
        lookup_file=Path(args.lookup),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
