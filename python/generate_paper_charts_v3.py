#!/usr/bin/env python3
"""
generate_paper_charts_v3.py

Canonical V3 Publication and Presentation Chart Generator for Conformal In-DB Analytics.
Implements all User 2 recommendations and fixes:
- Dual Export Profiles: 'paper' (compact, clean, no title) and 'presentation' (with titles).
- Output Directory: figures_v3/
- Typography: Times New Roman serif, 13pt axis labels, 11pt tick labels.
- Vector PDF with TrueType font embedding (pdf.fonttype = 42).
- Strategy ordering strictly: MCDB-1000 -> Baseline -> CQP (Ours).
- MCDB-100 is strictly suppressed from all figures.
- Bottom shared legends across composite figures.
- Robust strategy mapping per query:
    * Q1a/Q1b: Baseline = Naive_Baseline, CQP = Conformal_InOperator_Stop
    * Q2a: Baseline = Naive, MCDB-1000 = MCDB-1k, CQP = Conformal QN1 (MutEx, In-Operator Stop)
    * Q2b: Baseline = Naive, MCDB-1000 = MCDB-1k, CQP = Conformal QN1 (Indep, In-Operator Stop)
    * Q3/Q4: Baseline = Naive, MCDB-1000 = MCDB_Sum_1000, CQP = Conformal_InOperator_Stop (Indep)
- Part a: 6-query representative strip (k=3, alpha=0.05, z in {0.5, 1.0, 2.0}), shared Y variant,
          and multiple normalized variants:
            1. Normalized Speedup (T_base / T, higher is better)
            2. Normalized Execution Time (T / T_base, lower is faster)
            3. Normalized Time Saved % ((T_base - T)/T_base * 100%, higher is better)
          plus Parity Correlation Plot (Time Saved % vs Stream Saved %).
- Part b: Top-k scaling (Q2a & Q3, z in {1.0, 2.0}, c=1):
          - Quad bars with Baseline alongside CQP
          - Quad bars with CQP only + Baseline dashed reference
          - Quad lines showing decay across k
          - Followup scatter plots (Savings vs k, Savings vs Delta W).
- Part c: Conformal set calibration dynamics (alpha in {0.01, 0.05, 0.10}, distinct hatches ///, //, /)
          and multi-panel tau_alpha threshold progression across all 6 queries.
- Part d: Database scale comparison (scale 0.5 vs scale 1.0 for Q3/Q4, k=3, z=1.0).
- Part e: Reasoning strategy ablations (Q2a MutEx vs Indep; Q3 In-Op Stop vs External Pull).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

# Set headless Matplotlib backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from lib.schema_codec import SchemaCodec, read_jsonl_gz
from lib.distribution_references import get_distribution_weights, compute_gap_metrics

# Canonical V3 Styling Palette
PALETTE = {
    "baseline": "#7f7f7f",            # Medium Slate Gray
    "mcdb_1000": "#d62728",           # Crimson / Vermillion
    "cqp": "#1f77b4",                 # Standard Blue (Ours)
    "mutex": "#2ca02c",               # Forest Green (Closed-World)
    "indep": "#1f77b4",               # Blue (Open-World)
    "in_op_stop": "#1f77b4",          # Blue
    "external_pull": "#ff7f0e",       # Orange
}

STRATEGY_STYLES = {
    "MCDB-1000": {
        "color": PALETTE["mcdb_1000"],
        "hatch": "xx",
        "edgecolor": "black",
        "linewidth": 0.8,
        "label": "MCDB-1000",
    },
    "Baseline": {
        "color": PALETTE["baseline"],
        "hatch": "",
        "edgecolor": "black",
        "linewidth": 0.8,
        "label": "Baseline",
    },
    "CQP": {
        "color": PALETTE["cqp"],
        "hatch": "//",
        "edgecolor": "black",
        "linewidth": 0.8,
        "label": "CQP (Ours)",
    },
    "MutEx": {
        "color": PALETTE["mutex"],
        "hatch": "..",
        "edgecolor": "black",
        "linewidth": 0.8,
        "label": "MutEx (Closed-World)",
    },
    "Indep": {
        "color": PALETTE["indep"],
        "hatch": "--",
        "edgecolor": "black",
        "linewidth": 0.8,
        "label": "Indep (Open-World)",
    },
    "In-Op Stop": {
        "color": PALETTE["in_op_stop"],
        "hatch": "//",
        "edgecolor": "black",
        "linewidth": 0.8,
        "label": "In-Operator Stop",
    },
    "External Pull": {
        "color": PALETTE["external_pull"],
        "hatch": "--",
        "edgecolor": "black",
        "linewidth": 0.8,
        "label": "External Pull",
    },
}

# Canonical Capabilities Taxonomy for Ablations (User 2 Specifications)
CAPABILITY_ORDER = [
    "Baseline",
    "External Pull + Indep",
    "External Pull + MutEx",
    "In-Operator Pull + Indep",
    "In-Operator Pull + MutEx",
]

CAPABILITY_STYLES = {
    "Baseline": {
        "color": PALETTE["baseline"],
        "hatch": "",
        "edgecolor": "black",
        "linewidth": 0.8,
        "label": "Baseline",
    },
    "External Pull + Indep": {
        "color": "#ff7f0e",
        "hatch": "--",
        "edgecolor": "black",
        "linewidth": 0.8,
        "label": "External Pull + Indep",
    },
    "External Pull + MutEx": {
        "color": "#d95f02",
        "hatch": "..",
        "edgecolor": "black",
        "linewidth": 0.8,
        "label": "External Pull + MutEx",
    },
    "In-Operator Pull + Indep": {
        "color": "#1f77b4",
        "hatch": "//",
        "edgecolor": "black",
        "linewidth": 0.8,
        "label": "In-Operator Pull + Indep",
    },
    "In-Operator Pull + MutEx": {
        "color": "#2ca02c",
        "hatch": "..",
        "edgecolor": "black",
        "linewidth": 0.8,
        "label": "In-Operator Pull + MutEx",
    },
}


def map_capability_strategy(query: str, raw_strat: str) -> Optional[str]:
    """Maps raw benchmark strategy string to unified capability taxonomy."""
    s = raw_strat.lower()
    if "mcdb" in s:
        return None
    if "naive" in s and "conformal" not in s:
        return "Baseline"
    if "naive_conformal_end" in s:
        return "Baseline"

    is_in_op = ("inoperator" in s or "in-operator" in s or "in_op" in s)
    is_external = ("external" in s or "qn1" in s) and not is_in_op

    is_mutex = "mutex" in s
    is_indep = "indep" in s

    if is_external and is_indep:
        return "External Pull + Indep"
    if is_external and is_mutex:
        return "External Pull + MutEx"
    if is_in_op and is_indep:
        return "In-Operator Pull + Indep"
    if is_in_op and is_mutex:
        return "In-Operator Pull + MutEx"

    if query in ["q3", "q4"]:
        if is_in_op:
            return "In-Operator Pull + Indep"
        if is_external:
            return "External Pull + Indep"

    return None

QUERY_INFO = {
    "q1a": {"name": "Q1a", "modality": "vision", "label": "Q1a: COCO (Flat)", "short": "Q1a (COCO)"},
    "q1b": {"name": "Q1b", "modality": "text", "label": "Q1b: Jigsaw (Flat)", "short": "Q1b (Jigsaw)"},
    "q2a": {"name": "Q2a", "modality": "vision", "label": "Q2a: COCO (Nested)", "short": "Q2a (COCO)"},
    "q2b": {"name": "Q2b", "modality": "text", "label": "Q2b: Jigsaw (Nested)", "short": "Q2b (Jigsaw)"},
    "q3":  {"name": "Q3",  "modality": "relational", "label": "Q3: TPC-H (3-Table)", "short": "Q3 (TPC-H)"},
    "q4":  {"name": "Q4",  "modality": "relational", "label": "Q4: TPC-H (4-Table)", "short": "Q4 (TPC-H)"},
}

CANONICAL_M = {"q1a": 10, "q1b": 6, "q2a": 10, "q2b": 6, "q3": 10, "q4": 10}

# Calibrated Non-Conformity Thresholds for TPC-H Queries at k=3, z=1.0, c=1 (from calib_summary)
TPCH_CALIB_THRESHOLDS = {
    "q3": {0.01: 273.952, 0.05: 278.082, 0.10: 279.410},
    "q4": {0.01: 58.344, 0.05: 58.774, 0.10: 59.536},
}


def format_sigfigs(val: float, sigfigs: int = 3, max_decimals: int = 3) -> str:
    """Formats float up to specified significant figures, capped at max_decimals."""
    if val == 0:
        return "0"
    import math
    decimals = sigfigs - int(math.floor(math.log10(abs(val)))) - 1
    decimals = min(decimals, max_decimals)
    if decimals <= 0:
        return f"{round(val, decimals):.0f}"
    else:
        formatted = f"{val:.{decimals}f}"
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted


def setup_v3_style(profile: str = "paper"):
    """Configures matplotlib rcParams for V3 publication standards with increased font sizes for legibility."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "STIXGeneral"],
        "font.size": 15,
        "axes.titlesize": 17 if profile == "paper" else 19,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 18,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "lines.linewidth": 2.2,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def canonicalize_record(query: str, raw_strat: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Returns (primary_strategy, ablation_strategy).
    Suppresses MCDB-100 completely.
    """
    s = raw_strat.lower()
    # Suppress MCDB-100
    if "mcdb" in s and "100" in s and "1000" not in s and "1k" not in s:
        return None

    # MCDB-1000
    if "mcdb" in s and ("1000" in s or "1k" in s):
        return ("MCDB-1000", None)

    # Baseline (Naive)
    if "naive" in s and "conformal" not in s:
        return ("Baseline", None)

    # Primary Conformal CQP and Ablations per query
    if query in ["q1a", "q1b"]:
        if "inoperator_stop" in s or "in-operator stop" in s or "conformal_inoperator" in s or "conformal qn1" in s:
            if "indep" in s and query == "q1a":
                return (None, "In-Op Indep")
            return ("CQP", "In-Op Stop")
        if "qn1" in s or "external" in s:
            if "mutex" in s:
                return (None, "External MutEx")
            return (None, "External Indep")
    elif query == "q2a":
        if "conformal qn1" in s and "mutex" in s:
            return ("CQP", "MutEx")
        if "conformal qn1" in s and "indep" in s:
            return (None, "In-Op Indep")
        if "qn1 (mutex)" in s or ("external" in s and "mutex" in s):
            return (None, "MutEx")
        if "qn1 (indep)" in s or ("external" in s and "indep" in s):
            return (None, "Indep")
    elif query == "q2b":
        if "conformal qn1" in s and "indep" in s:
            return ("CQP", "Indep")
        if "qn1" in s or "external" in s:
            return (None, "External Pull")
    elif query in ["q3", "q4"]:
        if "conformal_inoperator_stop" in s:
            return ("CQP", "In-Op Stop")
        if "external_pull" in s:
            return (None, "External Pull")

    return None


def load_and_preprocess_results(results_dir: Path, supplementary_dirs: Optional[List[Path]] = None) -> pd.DataFrame:
    """Loads unified_sweep_results.jsonl.gz and supplementary runs, flattening into a DataFrame."""
    codec = SchemaCodec()
    lookup_file = results_dir / "schema_lookup.json"
    if lookup_file.exists():
        codec.load_lookup(lookup_file)

    primary_file = results_dir / "unified_sweep_results.jsonl.gz"
    all_files = [primary_file] if primary_file.exists() else []

    if supplementary_dirs:
        for supp in supplementary_dirs:
            if supp.exists():
                all_files.extend(list(supp.glob("*.jsonl.gz")))

    if not all_files:
        raise FileNotFoundError(f"No results found in {results_dir}")

    rows = []
    for f in all_files:
        records = read_jsonl_gz(f, codec)
        for rec in records:
            cfg = rec.get("config", {})
            metrics = rec.get("metrics", {})
            query = cfg.get("query", "").lower()
            strat_raw = cfg.get("strategy", "")

            classification = canonicalize_record(query, strat_raw)
            if not classification:
                continue
            strat_primary, strat_ablation = classification

            q_info = QUERY_INFO.get(query, {"name": query.upper(), "modality": "unknown"})
            t_ing = int(metrics.get("tuples_ingested", 0))
            t_tot = int(metrics.get("tuples_total", t_ing))
            stream_sav = float(metrics.get("stream_savings_pct", metrics.get("stream_saved_pct", 0.0)))
            if stream_sav == 0.0 and t_tot > 0 and t_ing < t_tot:
                stream_sav = (1.0 - t_ing / t_tot) * 100.0

            tau_th = float(metrics.get("tau_threshold", 0.0))
            alpha_val = float(cfg.get("alpha", 0.05))

            # Inject TPC-H calibration thresholds if not recorded in metrics
            if query in TPCH_CALIB_THRESHOLDS and tau_th == 0.0:
                tau_th = TPCH_CALIB_THRESHOLDS[query].get(round(alpha_val, 2), 0.0)

            rows.append({
                "query": query,
                "query_name": q_info["name"],
                "modality": q_info["modality"],
                "strategy_raw": strat_raw,
                "strategy": strat_primary if strat_primary else "Ablation_Only",
                "ablation_strat": strat_ablation,
                "M": int(cfg.get("M", 10)),
                "z": float(cfg.get("z", 1.0)),
                "c": int(cfg.get("c", 1)),
                "k": int(cfg.get("k_query", cfg.get("k", 1))),
                "alpha": alpha_val,
                "seed": int(cfg.get("seed", 0)),
                "dataset_scale": float(cfg.get("dataset_scale", 0.5)),
                "engine_time_ms": float(metrics.get("engine_time_ms", 0.0)),
                "tuples_ingested": t_ing,
                "tuples_total": t_tot,
                "stream_savings_pct": stream_sav,
                "tau_threshold": tau_th,
            })

    df = pd.DataFrame(rows)
    # Filter to single-mode cluster c=1 everywhere
    df = df[df["c"] == 1]
    return df


def get_baseline_map(df: pd.DataFrame) -> Dict[Tuple[str, float], float]:
    """Computes Baseline runtime map per (query, z) for representative scale (dataset_scale=0.5, canonical M)."""
    base_rows = df[(df["strategy"] == "Baseline") & (df["dataset_scale"] == 0.5)]
    filtered = []
    for q, m_val in CANONICAL_M.items():
        filtered.append(base_rows[(base_rows["query"] == q) & (base_rows["M"] == m_val)])
    if filtered:
        base_rows = pd.concat(filtered)
    return base_rows.groupby(["query", "z"])["engine_time_ms"].mean().to_dict()


def get_mcdb_map(df: pd.DataFrame) -> Dict[Tuple[str, float], float]:
    """Computes MCDB-1000 runtime map per (query, z) for representative scale (dataset_scale=0.5, canonical M)."""
    mcdb_rows = df[(df["strategy"] == "MCDB-1000") & (df["dataset_scale"] == 0.5)]
    filtered = []
    for q, m_val in CANONICAL_M.items():
        filtered.append(mcdb_rows[(mcdb_rows["query"] == q) & (mcdb_rows["M"] == m_val)])
    if filtered:
        mcdb_rows = pd.concat(filtered)
    return mcdb_rows.groupby(["query", "z"])["engine_time_ms"].mean().to_dict()


def save_figure(fig, out_path_base: Path, profile: str):
    """Saves figure in both vector PDF (300 DPI) and raster PNG (200 DPI)."""
    out_path_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path_base.with_suffix(".pdf")
    png_path = out_path_base.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# PART A: Representative Performance Strip (k=3, z in {0.5, 1.0, 2.0}, alpha=0.05)
# =========================================================================

def plot_part_a_representative(df: pd.DataFrame, out_dir: Path, profile: str = "paper"):
    """Generates Part a figures: 6-panel strip, shared Y-axis, normalized speedup/time/saved, and parity correlation."""
    setup_v3_style(profile)
    # Filter to canonical M per query at representative scale 0.5
    queries = ["q1a", "q1b", "q2a", "q2b", "q3", "q4"]
    slice_parts = []
    for q in queries:
        m_val = CANONICAL_M[q]
        q_slice = df[(df["query"] == q) & (df["M"] == m_val) & (df["k"] == 3) & (df["alpha"] == 0.05) & (df["dataset_scale"] == 0.5)]
        slice_parts.append(q_slice)
    slice_df = pd.concat(slice_parts) if slice_parts else df[(df["k"] == 3) & (df["alpha"] == 0.05) & (df["dataset_scale"] == 0.5)]

    baseline_map = get_baseline_map(df)
    mcdb_map = get_mcdb_map(df)
    z_vals = [0.5, 1.0, 2.0]
    canonical_strats = ["MCDB-1000", "Baseline", "CQP"]

    # Canonical legend handles strictly ordered: MCDB-1000 -> Baseline -> CQP
    import matplotlib.patches as mpatches
    ordered_handles = [
        mpatches.Patch(
            facecolor=STRATEGY_STYLES[s]["color"],
            hatch=STRATEGY_STYLES[s]["hatch"],
            edgecolor=STRATEGY_STYLES[s]["edgecolor"],
            linewidth=STRATEGY_STYLES[s]["linewidth"],
        )
        for s in canonical_strats
    ]
    ordered_labels = [STRATEGY_STYLES[s]["label"] for s in canonical_strats]

    # --- Fig 1a: Representative Latency Strip (Full including MCDB-1000) ---
    fig, axes = plt.subplots(1, 6, figsize=(18, 2.9), sharey=False)
    plt.subplots_adjust(bottom=0.32, wspace=0.34, top=0.88)

    for idx, q in enumerate(queries):
        ax = axes[idx]
        q_df = slice_df[slice_df["query"] == q]
        q_name = QUERY_INFO[q]["name"]

        grouped = q_df.groupby(["z", "strategy"])["engine_time_ms"].agg(["mean", "sem"]).reset_index()
        x = np.arange(len(z_vals))
        bar_w = 0.19

        for s_idx, s in enumerate(canonical_strats):
            s_data = grouped[grouped["strategy"] == s]
            s_style = STRATEGY_STYLES[s]
            means = []
            sems = []
            for z in z_vals:
                row = s_data[s_data["z"] == z]
                if not row.empty:
                    means.append(row["mean"].values[0])
                    sems.append(row["sem"].values[0] if not pd.isna(row["sem"].values[0]) else 0.0)
                elif s == "Baseline" and (q, z) in baseline_map:
                    means.append(baseline_map[(q, z)])
                    sems.append(0.0)
                elif s == "MCDB-1000" and (q, z) in mcdb_map:
                    means.append(mcdb_map[(q, z)])
                    sems.append(0.0)
                else:
                    means.append(0.0)
                    sems.append(0.0)

            offset = (s_idx - 1) * bar_w
            ax.bar(
                x + offset, means, yerr=sems, width=bar_w,
                color=s_style["color"], hatch=s_style["hatch"],
                edgecolor=s_style["edgecolor"], linewidth=s_style["linewidth"],
                capsize=2.5,
            )

        ax.set_title(q_name if profile == "presentation" else f"({chr(97+idx)}) {q_name}", fontsize=15, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"z={z}" for z in z_vals], fontsize=13)
        if idx == 0:
            ax.set_ylabel("Execution Time (ms)", fontsize=15)

    fig.legend(
        ordered_handles, ordered_labels,
        loc="lower center", ncol=3, frameon=True, fontsize=14, bbox_to_anchor=(0.5, 0.02)
    )
    save_figure(fig, out_dir / "part_a" / "fig1a_representative_latency_strip", profile)

    # --- Fig 1a Focused: Baseline vs CQP Only (Zoomed into CQP Skew Dynamics) ---
    fig_f, axes_f = plt.subplots(1, 6, figsize=(18, 2.9), sharey=False)
    plt.subplots_adjust(bottom=0.32, wspace=0.34, top=0.88)
    focused_strats = ["Baseline", "CQP"]
    focused_handles = [
        mpatches.Patch(
            facecolor=STRATEGY_STYLES[s]["color"],
            hatch=STRATEGY_STYLES[s]["hatch"],
            edgecolor=STRATEGY_STYLES[s]["edgecolor"],
            linewidth=STRATEGY_STYLES[s]["linewidth"],
        )
        for s in focused_strats
    ]
    focused_labels = [STRATEGY_STYLES[s]["label"] for s in focused_strats]

    for idx, q in enumerate(queries):
        ax = axes_f[idx]
        q_df = slice_df[slice_df["query"] == q]
        q_name = QUERY_INFO[q]["name"]

        grouped = q_df.groupby(["z", "strategy"])["engine_time_ms"].agg(["mean", "sem"]).reset_index()
        x = np.arange(len(z_vals))
        bar_w_f = 0.28

        for s_idx, s in enumerate(focused_strats):
            s_data = grouped[grouped["strategy"] == s]
            s_style = STRATEGY_STYLES[s]
            means = []
            sems = []
            for z in z_vals:
                row = s_data[s_data["z"] == z]
                if not row.empty:
                    means.append(row["mean"].values[0])
                    sems.append(row["sem"].values[0] if not pd.isna(row["sem"].values[0]) else 0.0)
                elif s == "Baseline" and (q, z) in baseline_map:
                    means.append(baseline_map[(q, z)])
                    sems.append(0.0)
                else:
                    means.append(0.0)
                    sems.append(0.0)

            offset = (s_idx - 0.5) * bar_w_f
            ax.bar(
                x + offset, means, yerr=sems, width=bar_w_f,
                color=s_style["color"], hatch=s_style["hatch"],
                edgecolor=s_style["edgecolor"], linewidth=s_style["linewidth"],
                capsize=2.5,
            )

        ax.set_title(q_name if profile == "presentation" else f"({chr(97+idx)}) {q_name}", fontsize=15, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"z={z}" for z in z_vals], fontsize=13)
        if idx == 0:
            ax.set_ylabel("Execution Time (ms)", fontsize=15)

    fig_f.legend(
        focused_handles, focused_labels,
        loc="lower center", ncol=2, frameon=True, fontsize=14, bbox_to_anchor=(0.5, 0.02)
    )
    save_figure(fig_f, out_dir / "part_a" / "fig1a_baseline_vs_cqp_strip", profile)

    # --- Fig 1b: Representative Shared Y-Axis ---
    fig, axes = plt.subplots(1, 6, figsize=(18, 2.9), sharey=True)
    plt.subplots_adjust(bottom=0.32, wspace=0.16, top=0.88)
    for idx, q in enumerate(queries):
        ax = axes[idx]
        q_df = slice_df[slice_df["query"] == q]
        q_name = QUERY_INFO[q]["name"]
        grouped = q_df.groupby(["z", "strategy"])["engine_time_ms"].agg(["mean", "sem"]).reset_index()
        x = np.arange(len(z_vals))
        bar_w = 0.19

        for s_idx, s in enumerate(canonical_strats):
            s_data = grouped[grouped["strategy"] == s]
            s_style = STRATEGY_STYLES[s]
            means = []
            sems = []
            for z in z_vals:
                row = s_data[s_data["z"] == z]
                if not row.empty:
                    means.append(row["mean"].values[0])
                    sems.append(row["sem"].values[0] if not pd.isna(row["sem"].values[0]) else 0.0)
                elif s == "Baseline" and (q, z) in baseline_map:
                    means.append(baseline_map[(q, z)])
                    sems.append(0.0)
                elif s == "MCDB-1000" and (q, z) in mcdb_map:
                    means.append(mcdb_map[(q, z)])
                    sems.append(0.0)
                else:
                    means.append(0.0)
                    sems.append(0.0)

            offset = (s_idx - 1) * bar_w
            ax.bar(
                x + offset, means, yerr=sems, width=bar_w,
                color=s_style["color"], hatch=s_style["hatch"],
                edgecolor=s_style["edgecolor"], linewidth=s_style["linewidth"],
                capsize=2.5,
            )

        ax.set_title(q_name if profile == "presentation" else f"({chr(97+idx)}) {q_name}", fontsize=15, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"z={z}" for z in z_vals], fontsize=13)
        if idx == 0:
            ax.set_ylabel("Execution Time (ms)", fontsize=15)

    fig.legend(
        ordered_handles, ordered_labels,
        loc="lower center", ncol=3, frameon=True, fontsize=14, bbox_to_anchor=(0.5, 0.02)
    )
    save_figure(fig, out_dir / "part_a" / "fig1b_representative_shared_axis", profile)

    # --- Fig 1c Version 1: Normalized Speedup Ratio (T_base / T, higher is better) ---
    fig, axes = plt.subplots(1, 6, figsize=(18, 2.9), sharey=False)
    plt.subplots_adjust(bottom=0.32, wspace=0.34, top=0.88)
    for idx, q in enumerate(queries):
        ax = axes[idx]
        q_df = slice_df[slice_df["query"] == q]
        q_name = QUERY_INFO[q]["name"]
        grouped = q_df.groupby(["z", "strategy"])["engine_time_ms"].mean().reset_index()
        x = np.arange(len(z_vals))
        bar_w = 0.19

        for s_idx, s in enumerate(canonical_strats):
            s_data = grouped[grouped["strategy"] == s]
            s_style = STRATEGY_STYLES[s]
            speedups = []
            for z_i, z in enumerate(z_vals):
                t_base = baseline_map.get((q, z), 1.0)
                row = s_data[s_data["z"] == z]
                if not row.empty:
                    t_strat = row["engine_time_ms"].values[0]
                elif s == "Baseline":
                    t_strat = t_base
                elif s == "MCDB-1000" and (q, z) in mcdb_map:
                    t_strat = mcdb_map[(q, z)]
                else:
                    t_strat = 0.0
                sp = (t_base / t_strat) if t_strat > 0 else 0.0
                speedups.append(sp)

            offset = (s_idx - 1) * bar_w
            bars = ax.bar(
                x + offset, speedups, width=bar_w,
                color=s_style["color"], hatch=s_style["hatch"],
                edgecolor=s_style["edgecolor"], linewidth=s_style["linewidth"],
            )

            # Annotate raw milliseconds on Baseline bars
            if s == "Baseline":
                for z_i, z in enumerate(z_vals):
                    raw_ms = baseline_map.get((q, z), 0.0)
                    ax.text(
                        x[z_i] + offset, 1.05, f"{raw_ms:.0f}ms",
                        ha="center", va="bottom", fontsize=11.5, color="#333333"
                    )

        ax.axhline(1.0, color="#666666", linestyle=":", linewidth=1.0)
        ax.set_title(q_name if profile == "presentation" else f"({chr(97+idx)}) {q_name}", fontsize=15, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"z={z}" for z in z_vals], fontsize=13)
        if idx == 0:
            ax.set_ylabel("Speedup Ratio (T_base / T)", fontsize=15)

    fig.legend(
        ordered_handles, ordered_labels,
        loc="lower center", ncol=3, frameon=True, fontsize=14, bbox_to_anchor=(0.5, 0.02)
    )
    save_figure(fig, out_dir / "part_a" / "fig1c_normalized_speedup", profile)

    # --- Fig 1c Version 2: Normalized Execution Time (T / T_base, Baseline = 1.0) ---
    fig, axes = plt.subplots(1, 6, figsize=(18, 2.9), sharey=True)
    plt.subplots_adjust(bottom=0.32, wspace=0.16, top=0.88)
    for idx, q in enumerate(queries):
        ax = axes[idx]
        q_df = slice_df[slice_df["query"] == q]
        q_name = QUERY_INFO[q]["name"]
        grouped = q_df.groupby(["z", "strategy"])["engine_time_ms"].mean().reset_index()
        x = np.arange(len(z_vals))
        bar_w = 0.19

        for s_idx, s in enumerate(canonical_strats):
            s_data = grouped[grouped["strategy"] == s]
            s_style = STRATEGY_STYLES[s]
            norm_times = []
            for z_i, z in enumerate(z_vals):
                t_base = baseline_map.get((q, z), 1.0)
                row = s_data[s_data["z"] == z]
                if not row.empty:
                    t_strat = row["engine_time_ms"].values[0]
                elif s == "Baseline":
                    t_strat = t_base
                elif s == "MCDB-1000" and (q, z) in mcdb_map:
                    t_strat = mcdb_map[(q, z)]
                else:
                    t_strat = 0.0
                norm_t = (t_strat / t_base) if t_base > 0 else 0.0
                norm_times.append(norm_t)

            offset = (s_idx - 1) * bar_w
            bars = ax.bar(
                x + offset, norm_times, width=bar_w,
                color=s_style["color"], hatch=s_style["hatch"],
                edgecolor=s_style["edgecolor"], linewidth=s_style["linewidth"],
            )

            # Annotate raw milliseconds on Baseline bars
            if s == "Baseline":
                for z_i, z in enumerate(z_vals):
                    raw_ms = baseline_map.get((q, z), 0.0)
                    ax.text(
                        x[z_i] + offset, 1.05, f"{raw_ms:.0f}ms",
                        ha="center", va="bottom", fontsize=11.5, color="#333333"
                    )

        ax.axhline(1.0, color="#666666", linestyle=":", linewidth=1.0)
        ax.set_title(q_name if profile == "presentation" else f"({chr(97+idx)}) {q_name}", fontsize=15, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"z={z}" for z in z_vals], fontsize=13)
        if idx == 0:
            ax.set_ylabel("Normalized Time (T / T_base)", fontsize=15)

    fig.legend(
        ordered_handles, ordered_labels,
        loc="lower center", ncol=3, frameon=True, fontsize=14, bbox_to_anchor=(0.5, 0.02)
    )
    save_figure(fig, out_dir / "part_a" / "fig1c_normalized_execution_time", profile)

    # --- Fig 1c Version 3: Normalized Time Saved % ((T_base - T)/T_base * 100%, higher is better) ---
    fig, axes = plt.subplots(1, 6, figsize=(18, 2.9), sharey=True)
    plt.subplots_adjust(bottom=0.32, wspace=0.16, top=0.88)
    for idx, q in enumerate(queries):
        ax = axes[idx]
        q_df = slice_df[slice_df["query"] == q]
        q_name = QUERY_INFO[q]["name"]
        grouped = q_df.groupby(["z", "strategy"])["engine_time_ms"].mean().reset_index()
        x = np.arange(len(z_vals))
        bar_w = 0.19

        for s_idx, s in enumerate(canonical_strats):
            s_data = grouped[grouped["strategy"] == s]
            s_style = STRATEGY_STYLES[s]
            saved_pcts = []
            for z_i, z in enumerate(z_vals):
                t_base = baseline_map.get((q, z), 1.0)
                row = s_data[s_data["z"] == z]
                if not row.empty:
                    t_strat = row["engine_time_ms"].values[0]
                elif s == "Baseline":
                    t_strat = t_base
                elif s == "MCDB-1000" and (q, z) in mcdb_map:
                    t_strat = mcdb_map[(q, z)]
                else:
                    t_strat = 0.0
                ts = ((t_base - t_strat) / t_base * 100.0) if t_base > 0 and t_strat > 0 else 0.0
                saved_pcts.append(ts)

            offset = (s_idx - 1) * bar_w
            bars = ax.bar(
                x + offset, saved_pcts, width=bar_w,
                color=s_style["color"], hatch=s_style["hatch"],
                edgecolor=s_style["edgecolor"], linewidth=s_style["linewidth"],
            )

            # Annotate raw milliseconds on Baseline bars
            if s == "Baseline":
                for z_i, z in enumerate(z_vals):
                    raw_ms = baseline_map.get((q, z), 0.0)
                    ax.text(
                        x[z_i] + offset, 5.0, f"{raw_ms:.0f}ms",
                        ha="center", va="bottom", fontsize=11.5, color="#333333"
                    )

        ax.axhline(0.0, color="#666666", linestyle="-", linewidth=0.8)
        ax.set_title(q_name if profile == "presentation" else f"({chr(97+idx)}) {q_name}", fontsize=15, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"z={z}" for z in z_vals], fontsize=13)
        if idx == 0:
            ax.set_ylabel("Execution Time Saved (%)", fontsize=15)

    fig.legend(
        ordered_handles, ordered_labels,
        loc="lower center", ncol=3, frameon=True, fontsize=14, bbox_to_anchor=(0.5, 0.02)
    )
    save_figure(fig, out_dir / "part_a" / "fig1c_normalized_time_saved", profile)

    # --- Fig 1d: Parity Correlation (Time Saved % vs Stream Saved %) ---
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    cqp_df = slice_df[slice_df["strategy"] == "CQP"]

    points_x, points_y, colors = [], [], []
    modality_colors = {"vision": "#1B9E77", "text": "#D95F02", "relational": "#7570B3"}

    for _, row in cqp_df.iterrows():
        q = row["query"]
        z = row["z"]
        t_base = baseline_map.get((q, z), 0.0)
        t_cqp = row["engine_time_ms"]
        time_saved = ((t_base - t_cqp) / t_base * 100.0) if t_base > 0 else 0.0
        stream_saved = row["stream_savings_pct"]
        points_x.append(stream_saved)
        points_y.append(time_saved)
        colors.append(modality_colors.get(row["modality"], "#333333"))

    ax.scatter(points_x, points_y, c=colors, alpha=0.8, edgecolors="k", s=70, zorder=3)
    ax.plot([0, 100], [0, 100], "k--", label="Ideal Parity (y=x)", linewidth=1.5, zorder=2)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Stream Ingestion Saved (%)", fontsize=15)
    ax.set_ylabel("Execution Time Saved (%)", fontsize=15)
    if profile == "presentation":
        ax.set_title("Representative Efficiency Parity (k=3, alpha=0.05)", fontsize=16)
    ax.legend(loc="upper left", frameon=True, fontsize=14)
    save_figure(fig, out_dir / "part_a" / "fig1d_representative_correlation_parity", profile)


# =========================================================================
# PART B: Top-k Selection & Scaling Suite (Q2a & Q3, z in {1.0, 2.0}, c=1)
# =========================================================================

def plot_part_b_topk(df: pd.DataFrame, out_dir: Path, profile: str = "paper"):
    """Generates Part b figures: Quad bar grids (with Baseline and CQP-only), Line decay grid, and scatter plots."""
    setup_v3_style(profile)
    b_df = df[df["query"].isin(["q2a", "q3"]) & (df["M"] == 10) & df["z"].isin([1.0, 2.0]) & (df["alpha"] == 0.05) & (df["dataset_scale"] == 0.5)]
    baseline_map = get_baseline_map(df)
    queries = ["q2a", "q3"]
    z_vals = [1.0, 2.0]

    # --- Fig 2a: Quad Bar Grid with Baseline Alongside CQP ---
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.8), sharey=False)
    plt.subplots_adjust(hspace=0.38, wspace=0.28, bottom=0.18)

    legend_handles = {}
    for r_idx, q in enumerate(queries):
        for c_idx, z in enumerate(z_vals):
            ax = axes[r_idx, c_idx]
            sub = b_df[(b_df["query"] == q) & (b_df["z"] == z)]
            k_vals = sorted(sub[sub["strategy"] == "CQP"]["k"].unique())
            if not k_vals:
                k_vals = [1, 2, 3] if q == "q2a" else [1, 3, 5]

            bar_w = 0.26
            x = np.arange(len(k_vals))

            for s_idx, s in enumerate(["Baseline", "CQP"]):
                s_style = STRATEGY_STYLES[s]
                offset = (s_idx - 0.5) * bar_w
                if s == "Baseline":
                    b_val = baseline_map.get((q, z), 0.0)
                    means = [b_val] * len(k_vals)
                    sems = [0.0] * len(k_vals)
                else:
                    cqp_sub = sub[sub["strategy"] == "CQP"].groupby("k")["engine_time_ms"].agg(["mean", "sem"]).reindex(k_vals).reset_index()
                    means = cqp_sub["mean"].tolist()
                    sems = [s if not pd.isna(s) else 0.0 for s in cqp_sub["sem"].tolist()]

                bars = ax.bar(
                    x + offset, means, yerr=sems, width=bar_w,
                    color=s_style["color"], hatch=s_style["hatch"],
                    edgecolor=s_style["edgecolor"], linewidth=s_style["linewidth"],
                    capsize=2.5, label=s_style["label"]
                )
                if s_style["label"] not in legend_handles:
                    legend_handles[s_style["label"]] = bars[0]

            ax.set_xticks(x)
            ax.set_xticklabels([f"k={k}" for k in k_vals], fontsize=13)
            q_label = QUERY_INFO[q]["short"]
            ax.set_title(f"{q_label} (z={z})", fontsize=15)
            if c_idx == 0:
                ax.set_ylabel("Execution Time (ms)", fontsize=14)
            if r_idx == 1:
                ax.set_xlabel("Target Rank (k)", fontsize=14)

    fig.legend(
        list(legend_handles.values()), list(legend_handles.keys()),
        loc="lower center", ncol=2, frameon=True, fontsize=14, bbox_to_anchor=(0.5, 0.02)
    )
    save_figure(fig, out_dir / "part_b" / "fig2a_topk_quad_bars_with_baseline", profile)

    # --- Fig 2b: Quad Bar Grid with CQP Only + Baseline Reference Line ---
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.8), sharey=False)
    plt.subplots_adjust(hspace=0.38, wspace=0.28, bottom=0.15)

    for r_idx, q in enumerate(queries):
        for c_idx, z in enumerate(z_vals):
            ax = axes[r_idx, c_idx]
            sub = b_df[(b_df["query"] == q) & (b_df["z"] == z)]
            k_vals = sorted(sub[sub["strategy"] == "CQP"]["k"].unique())
            if not k_vals:
                k_vals = [1, 2, 3] if q == "q2a" else [1, 3, 5]

            base_mean = baseline_map.get((q, z), 0.0)
            cqp_data = sub[sub["strategy"] == "CQP"].groupby("k")["engine_time_ms"].agg(["mean", "sem"]).reindex(k_vals).reset_index()

            bar_w = 0.32
            x = np.arange(len(k_vals))
            s_style = STRATEGY_STYLES["CQP"]
            bars = ax.bar(
                x, cqp_data["mean"], yerr=cqp_data["sem"], width=bar_w,
                color=s_style["color"], hatch=s_style["hatch"],
                edgecolor=s_style["edgecolor"], linewidth=s_style["linewidth"],
                capsize=3.0, label="CQP (Ours)"
            )
            # Baseline dashed reference line
            ax.axhline(base_mean, color="#d62728", linestyle="--", linewidth=1.5, label=f"Baseline ({base_mean:.1f} ms)")

            ax.set_xticks(x)
            ax.set_xticklabels([f"k={k}" for k in k_vals], fontsize=13)
            q_label = QUERY_INFO[q]["short"]
            ax.set_title(f"{q_label} (z={z})", fontsize=15)
            if c_idx == 0:
                ax.set_ylabel("Execution Time (ms)", fontsize=14)
            if r_idx == 1:
                ax.set_xlabel("Target Rank (k)", fontsize=14)
            ax.legend(loc="upper left", fontsize=12)

    save_figure(fig, out_dir / "part_b" / "fig2b_topk_quad_bars_cqp_only", profile)

    # =========================================================================
    # Extended Q3 Top-k Suite (User 2 Specifications: M=20, Scale 0.5, k in {1,3,5,7,10,12,15})
    # =========================================================================
    q3_m20_s05 = df[(df["query"] == "q3") & (df["M"] == 20) & (df["dataset_scale"] == 0.5) & (df["alpha"] == 0.05)]
    if not q3_m20_s05.empty:
        q3_m20 = q3_m20_s05
        m20_scale_val = 0.5
    else:
        q3_m20 = df[(df["query"] == "q3") & (df["M"] == 20) & (df["alpha"] == 0.05)]
        m20_scale_val = 1.0 if not q3_m20.empty else 0.5

    if not q3_m20.empty:
        m20_z_vals = sorted(q3_m20["z"].unique())
        m20_k_vals = sorted(q3_m20[q3_m20["strategy"] == "CQP"]["k"].unique())
        scale_str = f"Scale {m20_scale_val:.1f}"

        # --- Fig 2c: Q3 M=20 Top-k Scaling Bar Chart (Baseline vs CQP) ---
        n_panels = len(m20_z_vals)
        fig_w = 6.0 if n_panels == 1 else 11.0
        fig, axes_m20 = plt.subplots(1, n_panels, figsize=(fig_w, 3.8), sharey=True)
        ax_list = [axes_m20] if n_panels == 1 else axes_m20
        plt.subplots_adjust(bottom=0.26, wspace=0.25)

        m20_legend_handles = {}
        for z_i, z_val in enumerate(m20_z_vals):
            ax = ax_list[z_i]
            z_sub = q3_m20[q3_m20["z"] == z_val]
            x_k = np.arange(len(m20_k_vals))
            bar_w_m20 = 0.28

            base_m20_sub = z_sub[z_sub["strategy"] == "Baseline"]
            default_base = baseline_map.get(("q3", z_val), 25.5 if m20_scale_val == 0.5 else 69.3)
            b_mean = base_m20_sub["engine_time_ms"].mean() if not base_m20_sub.empty else default_base

            for s_idx, s in enumerate(["Baseline", "CQP"]):
                s_style = STRATEGY_STYLES[s]
                offset = (s_idx - 0.5) * bar_w_m20
                if s == "Baseline":
                    means = [b_mean] * len(m20_k_vals)
                    sems = [0.0] * len(m20_k_vals)
                else:
                    cqp_k = z_sub[z_sub["strategy"] == "CQP"].groupby("k")["engine_time_ms"].agg(["mean", "sem"]).reindex(m20_k_vals).reset_index()
                    means = cqp_k["mean"].tolist()
                    sems = [s if not pd.isna(s) else 0.0 for s in cqp_k["sem"].tolist()]

                bars = ax.bar(
                    x_k + offset, means, yerr=sems, width=bar_w_m20,
                    color=s_style["color"], hatch=s_style["hatch"],
                    edgecolor=s_style["edgecolor"], linewidth=s_style["linewidth"],
                    capsize=2.5, label=s_style["label"]
                )
                if s_style["label"] not in m20_legend_handles:
                    m20_legend_handles[s_style["label"]] = bars[0]

            ax.set_xticks(x_k)
            ax.set_xticklabels([f"k={k}" for k in m20_k_vals], fontsize=11.5)
            ax.set_title(f"Q3 (M=20, z={z_val})", fontsize=15)
            if z_i == 0:
                ax.set_ylabel("Execution Time (ms)", fontsize=15)
            ax.set_xlabel("Target Rank (k)", fontsize=14)

        fig.legend(
            list(m20_legend_handles.values()), list(m20_legend_handles.keys()),
            loc="lower center", ncol=2, frameon=True, fontsize=14, bbox_to_anchor=(0.5, 0.02)
        )
        save_figure(fig, out_dir / "part_b" / "fig2c_q3_m20_topk_bars", profile)

        # --- Fig 2d: Q3 M=20 Correlation / Parity Scatter Plot ---
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        cqp_m20 = q3_m20[q3_m20["strategy"] == "CQP"]
        base_series = q3_m20[q3_m20["strategy"] == "Baseline"]["engine_time_ms"]
        base_m20 = base_series.mean() if not base_series.empty else baseline_map.get(("q3", 1.0), 25.5 if m20_scale_val == 0.5 else 69.29)

        pts_x = cqp_m20["stream_savings_pct"].tolist()
        pts_y = [((base_m20 - t) / base_m20 * 100.0) for t in cqp_m20["engine_time_ms"]]
        pts_k = cqp_m20["k"].tolist()

        scatter = ax.scatter(pts_x, pts_y, c=pts_k, cmap="viridis", alpha=0.85, edgecolors="k", s=80, zorder=3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Target Rank (k)", fontsize=13)

        ax.plot([0, 100], [0, 100], "k--", label="Ideal Parity (y=x)", linewidth=1.5, zorder=2)
        ax.set_xlim(0, max(30, max(pts_x, default=25) * 1.25))
        ax.set_ylim(0, max(35, max(pts_y, default=30) * 1.25))
        ax.set_xlabel("Stream Ingestion Saved (%)", fontsize=15)
        ax.set_ylabel("Execution Time Saved (%)", fontsize=15)
        ax.set_title(f"Q3 (M=20, {scale_str}) Efficiency Parity vs. k", fontsize=16)
        ax.legend(loc="upper left", frameon=True, fontsize=13)
        save_figure(fig, out_dir / "part_b" / "fig2d_q3_m20_correlation_parity", profile)

        # --- Fig 2e: Q3 M=20 Line Decay (k vs Stream Savings & k vs Time) ---
        fig, (ax_s, ax_t) = plt.subplots(1, 2, figsize=(11.0, 4.0))
        plt.subplots_adjust(bottom=0.25, top=0.88, wspace=0.28)

        # Style mapping for z=1.0 and z=2.0 (uniform colors and distinct dot shapes)
        z_styles = {
            1.0: {"color": "#1f77b4", "marker": "o", "label": "z = 1.0"},
            2.0: {"color": "#ff7f0e", "marker": "s", "label": "z = 2.0"},
        }
        target_z_vals = [zv for zv in [1.0, 2.0] if zv in m20_z_vals]

        for z_val in target_z_vals:
            zs = z_styles[z_val]
            z_sub = q3_m20[(q3_m20["z"] == z_val) & (q3_m20["strategy"] == "CQP")]
            grouped_k = z_sub.groupby("k")[["stream_savings_pct", "engine_time_ms"]].agg(["mean", "sem"]).reindex(m20_k_vals)

            # Panel 1: k vs Stream Savings
            ax_s.errorbar(
                m20_k_vals, grouped_k["stream_savings_pct"]["mean"],
                yerr=grouped_k["stream_savings_pct"]["sem"],
                marker=zs["marker"], markersize=7, capsize=3, linewidth=2.0,
                color=zs["color"], label=zs["label"]
            )
            # Panel 2: k vs Execution Time
            ax_t.errorbar(
                m20_k_vals, grouped_k["engine_time_ms"]["mean"],
                yerr=grouped_k["engine_time_ms"]["sem"],
                marker=zs["marker"], markersize=7, capsize=3, linewidth=2.0,
                color=zs["color"], label=zs["label"]
            )

        # Add baseline reference line on Panel 2 using uniform palette Medium Slate Gray (#7f7f7f)
        ax_t.axhline(base_m20, color=PALETTE["baseline"], linestyle="--", linewidth=1.6)
        ax_t.text(
            m20_k_vals[-1], base_m20 + 0.35, f"Baseline ({base_m20:.1f}ms)",
            color=PALETTE["baseline"], fontsize=12, fontweight="bold",
            ha="right", va="bottom"
        )

        ax_s.set_xticks(m20_k_vals)
        ax_s.set_xticklabels([f"{k}" for k in m20_k_vals], fontsize=13)
        ax_s.set_xlabel("Target Rank (k)", fontsize=15)
        ax_s.set_ylabel("Stream Ingestion Saved (%)", fontsize=15)
        ax_s.set_title("(a) Q3 (M=20)" if profile == "paper" else "Q3 (M=20)", fontsize=15)
        ax_s.grid(True, linestyle="--", alpha=0.5)

        ax_t.set_xticks(m20_k_vals)
        ax_t.set_xticklabels([f"{k}" for k in m20_k_vals], fontsize=13)
        ax_t.set_xlabel("Target Rank (k)", fontsize=15)
        ax_t.set_ylabel("Execution Time (ms)", fontsize=15)
        ax_t.set_title("(b) Q3 (M=20)" if profile == "paper" else "Q3 (M=20)", fontsize=15)
        ax_t.set_ylim(23.0, max(34.0, base_m20 * 1.08))
        ax_t.grid(True, linestyle="--", alpha=0.5)

        # Unified bottom legend shared across both subplots (matching other figures)
        handles, labels = ax_s.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower center", ncol=2, frameon=True, fontsize=13,
            bbox_to_anchor=(0.5, 0.02)
        )

        save_figure(fig, out_dir / "part_b" / "fig2e_q3_m20_decay_lines", profile)

    # --- Fig 2f: Quad Line Decay Grid ---
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.8), sharey=False)
    plt.subplots_adjust(hspace=0.38, wspace=0.28, bottom=0.15)

    for r_idx, q in enumerate(queries):
        for c_idx, z in enumerate(z_vals):
            ax = axes[r_idx, c_idx]
            sub = b_df[(b_df["query"] == q) & (b_df["z"] == z)]
            k_vals = sorted(sub[sub["strategy"] == "CQP"]["k"].unique())
            if not k_vals:
                k_vals = [1, 2, 3] if q == "q2a" else [1, 3, 5]

            base_val = baseline_map.get((q, z), 0.0)
            ax.plot(k_vals, [base_val]*len(k_vals), marker="s", markersize=6, color=PALETTE["baseline"], linestyle="--", label="Baseline", linewidth=1.8)

            cqp_sub = sub[sub["strategy"] == "CQP"].groupby("k")["engine_time_ms"].agg(["mean", "sem"]).reindex(k_vals).reset_index()
            ax.errorbar(
                cqp_sub["k"], cqp_sub["mean"], yerr=cqp_sub["sem"],
                marker="o", markersize=6, color=PALETTE["cqp"], capsize=3,
                label="CQP", linewidth=1.8
            )

            q_label = QUERY_INFO[q]["short"]
            ax.set_title(f"{q_label} (z={z})", fontsize=15)
            ax.set_xticks(k_vals)
            ax.set_xticklabels([f"k={k}" for k in k_vals], fontsize=13)
            if c_idx == 0:
                ax.set_ylabel("Execution Time (ms)", fontsize=14)
            if r_idx == 1:
                ax.set_xlabel("Target Rank (k)", fontsize=14)
            ax.legend(loc="center right", fontsize=12)

    save_figure(fig, out_dir / "part_b" / "fig2f_topk_quad_line_decay", profile)

    # --- Fig 2g: Bivariate Scatter Plots ---
    cqp_b = b_df[b_df["strategy"] == "CQP"].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    plt.subplots_adjust(wspace=0.28, bottom=0.18)

    for q in queries:
        q_data = cqp_b[cqp_b["query"] == q]
        ax1.scatter(q_data["k"], q_data["stream_savings_pct"], label=QUERY_INFO[q]["short"], alpha=0.8, s=60)
    ax1.set_xlabel("Target Rank (k)", fontsize=15)
    ax1.set_ylabel("Stream Savings (%)", fontsize=15)
    ax1.set_title("Stream Savings vs. Rank (k)", fontsize=15)
    ax1.legend(loc="upper right", fontsize=13)

    delta_w_list = []
    savings_list = []
    q_colors = []
    for _, row in cqp_b.iterrows():
        M = row["M"]
        z = row["z"]
        k = row["k"]
        weights = get_distribution_weights("zipf", M, z, c=1)
        gaps = compute_gap_metrics(weights, k)
        delta_w_list.append(gaps["delta_w"])
        savings_list.append(row["stream_savings_pct"])
        q_colors.append("#7570B3" if row["query"] == "q2a" else "#1B9E77")

    ax2.scatter(delta_w_list, savings_list, c=q_colors, alpha=0.8, s=60)
    ax2.set_xlabel("Weight Gap ΔW (W_k - W_{k+1})", fontsize=15)
    ax2.set_ylabel("Stream Savings (%)", fontsize=15)
    ax2.set_title("Stream Savings vs. Margin ΔW", fontsize=15)

    save_figure(fig, out_dir / "part_b" / "fig2g_followup_savings_and_delta_w_scatter", profile)

    # --- Fig 2h: Q3 M=20 Savings and Delta W Scatter (2-Panel, Grouped Means) ---
    q3_m20_cqp = df[(df["query"] == "q3") & (df["M"] == 20) & (df["dataset_scale"] == 0.5) & (df["strategy"] == "CQP")].copy()
    if not q3_m20_cqp.empty:
        delta_w_m20 = []
        ratio_w_m20 = []
        for _, row in q3_m20_cqp.iterrows():
            w = get_distribution_weights("zipf", 20, row["z"], c=1)
            g = compute_gap_metrics(w, int(row["k"]))
            delta_w_m20.append(g["delta_w"])
            ratio_w_m20.append(g["ratio_w"])
        q3_m20_cqp["delta_w"] = delta_w_m20
        q3_m20_cqp["ratio_w"] = ratio_w_m20

        z_styles = {
            1.0: {"color": "#1f77b4", "marker": "o", "label": "z = 1.0"},
            2.0: {"color": "#ff7f0e", "marker": "s", "label": "z = 2.0"},
        }

        # 1. Linear Delta W (Grouped Means)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
        plt.subplots_adjust(bottom=0.25, top=0.88, wspace=0.28)
        for z in [1.0, 2.0]:
            if z not in q3_m20_cqp["z"].values:
                continue
            sub = q3_m20_cqp[q3_m20_cqp["z"] == z]
            zs = z_styles[z]
            grouped = sub.groupby("k")[["stream_savings_pct", "delta_w"]].agg(["mean", "sem"]).reset_index()
            ax1.errorbar(
                grouped[("k", "")], grouped[("stream_savings_pct", "mean")],
                yerr=grouped[("stream_savings_pct", "sem")],
                marker=zs["marker"], markersize=7, capsize=3.5, linewidth=2.0,
                color=zs["color"], label=zs["label"]
            )
            ax2.errorbar(
                grouped[("delta_w", "mean")], grouped[("stream_savings_pct", "mean")],
                yerr=grouped[("stream_savings_pct", "sem")],
                marker=zs["marker"], markersize=7, capsize=3.5, linewidth=2.0,
                color=zs["color"], label=zs["label"]
            )
        ax1.set_xlabel("Target Rank (k)", fontsize=14)
        ax1.set_ylabel("Stream Ingestion Saved (%)", fontsize=14)
        ax1.set_title("(a) Q3 (M=20): Savings vs. Rank k", fontsize=15)
        ax1.set_xticks(sorted(q3_m20_cqp["k"].unique()))
        ax1.set_xticklabels([str(k) for k in sorted(q3_m20_cqp["k"].unique())], fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.set_xlabel(r"Weight Gap $\Delta W = W_k - W_{k+1}$", fontsize=14)
        ax2.set_ylabel("Stream Ingestion Saved (%)", fontsize=14)
        ax2.set_title(r"(b) Q3 (M=20): Savings vs. Margin $\Delta W$", fontsize=15)
        ax2.grid(True, linestyle="--", alpha=0.5)

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, fontsize=13, bbox_to_anchor=(0.5, 0.02))
        save_figure(fig, out_dir / "part_b" / "fig2g_q3_m20_savings_and_delta_w_scatter", profile)

        # 2. Linear Delta W Correlation (Grouped Means)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
        plt.subplots_adjust(bottom=0.20, top=0.88, wspace=0.28)
        for idx, (z, ax) in enumerate([(1.0, ax1), (2.0, ax2)]):
            if z not in q3_m20_cqp["z"].values:
                continue
            sub = q3_m20_cqp[q3_m20_cqp["z"] == z]
            zs = z_styles[z]
            grouped = sub.groupby("k")[["delta_w", "stream_savings_pct"]].agg(["mean", "sem"]).reset_index()
            x = grouped[("delta_w", "mean")].values
            y = grouped[("stream_savings_pct", "mean")].values
            yerr = grouped[("stream_savings_pct", "sem")].values

            ax.errorbar(
                x, y, yerr=yerr, fmt=zs["marker"], markersize=8, capsize=4,
                color=zs["color"], markeredgecolor="black", markeredgewidth=1.0,
                linewidth=1.8, label=f"Mean per Rank k (z={z})"
            )
            slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color="#d62728", linestyle="--", linewidth=2.0, label=f"Fit (r = {r_value:.2f})")

            rho_val, _ = stats.spearmanr(x, y)
            p_pow = int(np.floor(np.log10(max(p_value, 1e-100))))
            p_display = f"$p < 10^{{{p_pow}}}$" if p_value < 0.001 else f"$p = {p_value:.3f}$"
            stats_text = (
                f"Pearson $r = {r_value:.3f}$ ({p_display})\n"
                f"Spearman $\\rho = {rho_val:.3f}$\n"
                f"$R^2 = {r_value**2:.3f}$\n"
                f"Mean per $k$ (N=12)"
            )
            ax.text(
                0.05, 0.93, stats_text, transform=ax.transAxes,
                fontsize=11.5, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.92)
            )
            ax.set_xlabel(r"Weight Gap $\Delta W = W_k - W_{k+1}$", fontsize=14)
            if idx == 0:
                ax.set_ylabel("Stream Ingestion Saved (%)", fontsize=14)
            ax.set_title(f"Q3 (M=20, z={z}) Correlation", fontsize=15)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="lower right", fontsize=11, frameon=True)
        save_figure(fig, out_dir / "part_b" / "fig2g_q3_m20_delta_w_correlation", profile)

        # 3. Log-Scale Delta W (Grouped Means) - 2-Panel
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
        plt.subplots_adjust(bottom=0.25, top=0.88, wspace=0.28)
        for z in [1.0, 2.0]:
            if z not in q3_m20_cqp["z"].values:
                continue
            sub = q3_m20_cqp[q3_m20_cqp["z"] == z]
            zs = z_styles[z]
            grouped = sub.groupby("k")[["stream_savings_pct", "delta_w"]].agg(["mean", "sem"]).reset_index()
            ax1.errorbar(
                grouped[("k", "")], grouped[("stream_savings_pct", "mean")],
                yerr=grouped[("stream_savings_pct", "sem")],
                marker=zs["marker"], markersize=7, capsize=3.5, linewidth=2.0,
                color=zs["color"], label=zs["label"]
            )
            ax2.errorbar(
                grouped[("delta_w", "mean")], grouped[("stream_savings_pct", "mean")],
                yerr=grouped[("stream_savings_pct", "sem")],
                marker=zs["marker"], markersize=7, capsize=3.5, linewidth=2.0,
                color=zs["color"], label=zs["label"]
            )
        ax1.set_xlabel("Target Rank (k)", fontsize=14)
        ax1.set_ylabel("Stream Ingestion Saved (%)", fontsize=14)
        ax1.set_title("(a) Q3 (M=20): Savings vs. Rank k", fontsize=15)
        ax1.set_xticks(sorted(q3_m20_cqp["k"].unique()))
        ax1.set_xticklabels([str(k) for k in sorted(q3_m20_cqp["k"].unique())], fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.set_xscale("log")
        ax2.set_xlabel(r"Weight Gap $\Delta W = W_k - W_{k+1}$ (Log Scale)", fontsize=14)
        ax2.set_ylabel("Stream Ingestion Saved (%)", fontsize=14)
        ax2.set_title(r"(b) Q3 (M=20): Savings vs. $\Delta W$ (Log Scale)", fontsize=15)
        ax2.grid(True, which="both", linestyle="--", alpha=0.5)
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, fontsize=13, bbox_to_anchor=(0.5, 0.02))
        save_figure(fig, out_dir / "part_b" / "fig2g_q3_m20_savings_and_delta_w_log_grouped", profile)

        # 4. Log-Scale Delta W Correlation (Grouped Means)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
        plt.subplots_adjust(bottom=0.20, top=0.88, wspace=0.28)
        for idx, (z, ax) in enumerate([(1.0, ax1), (2.0, ax2)]):
            if z not in q3_m20_cqp["z"].values:
                continue
            sub = q3_m20_cqp[(q3_m20_cqp["z"] == z) & (q3_m20_cqp["k"] < 20)]
            zs = z_styles[z]
            grouped = sub.groupby("k")[["delta_w", "stream_savings_pct"]].agg(["mean", "sem"]).reset_index()
            x = grouped[("delta_w", "mean")].values
            y = grouped[("stream_savings_pct", "mean")].values
            yerr = grouped[("stream_savings_pct", "sem")].values

            ax.errorbar(
                x, y, yerr=yerr, fmt=zs["marker"], markersize=8, capsize=4,
                color=zs["color"], markeredgecolor="black", markeredgewidth=1.0,
                linewidth=1.8, label=f"Mean per Rank k (z={z})"
            )
            log_x = np.log10(x)
            slope, intercept, r_val, p_val, _ = stats.linregress(log_x, y)
            x_grid = np.geomspace(x.min(), x.max(), 100)
            ax.plot(x_grid, slope * np.log10(x_grid) + intercept, color="#d62728", linestyle="--", linewidth=2.0, label=f"Log-Linear Fit (r = {r_val:.2f})")

            rho_val, _ = stats.spearmanr(x, y)
            p_pow = int(np.floor(np.log10(max(p_val, 1e-100))))
            p_display = f"$p < 10^{{{p_pow}}}$" if p_val < 0.001 else f"$p = {p_val:.3f}$"
            stats_text = (
                f"Pearson $r = {r_val:.3f}$ ({p_display})\n"
                f"Spearman $\\rho = {rho_val:.3f}$\n"
                f"$R^2 = {r_val**2:.3f}$\n"
                f"Slope: ${slope:+.2f}\\%$ / decade"
            )
            ax.text(
                0.05, 0.93, stats_text, transform=ax.transAxes,
                fontsize=11.5, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.92)
            )
            ax.set_xscale("log")
            ax.set_xlabel(r"Weight Gap $\Delta W = W_k - W_{k+1}$ (Log Scale)", fontsize=14)
            if idx == 0:
                ax.set_ylabel("Stream Ingestion Saved (%)", fontsize=14)
            ax.set_title(fr"Q3 (M=20, z={z}) Log-$\Delta W$ Correlation", fontsize=15)
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            ax.legend(loc="lower right", fontsize=11, frameon=True)
        save_figure(fig, out_dir / "part_b" / "fig2g_q3_m20_delta_w_log_correlation_grouped", profile)

        # 5. Relative Ratio W_k / W_{k+1} in Normal Scale (Grouped Means) - 2-Panel
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
        plt.subplots_adjust(bottom=0.25, top=0.88, wspace=0.28)
        for z in [1.0, 2.0]:
            if z not in q3_m20_cqp["z"].values:
                continue
            sub = q3_m20_cqp[(q3_m20_cqp["z"] == z) & (q3_m20_cqp["k"] < 20)]
            zs = z_styles[z]
            grouped = sub.groupby("k")[["stream_savings_pct", "ratio_w"]].agg(["mean", "sem"]).reset_index()
            ax1.errorbar(
                grouped[("k", "")], grouped[("stream_savings_pct", "mean")],
                yerr=grouped[("stream_savings_pct", "sem")],
                marker=zs["marker"], markersize=7, capsize=3.5, linewidth=2.0,
                color=zs["color"], label=zs["label"]
            )
            ax2.errorbar(
                grouped[("ratio_w", "mean")], grouped[("stream_savings_pct", "mean")],
                yerr=grouped[("stream_savings_pct", "sem")],
                marker=zs["marker"], markersize=7, capsize=3.5, linewidth=2.0,
                color=zs["color"], label=zs["label"]
            )
        ax1.set_xlabel("Target Rank (k)", fontsize=14)
        ax1.set_ylabel("Stream Ingestion Saved (%)", fontsize=14)
        ax1.set_title("(a) Q3 (M=20): Savings vs. Rank k", fontsize=15)
        ax1.set_xticks(sorted(q3_m20_cqp[q3_m20_cqp["k"] < 20]["k"].unique()))
        ax1.set_xticklabels([str(k) for k in sorted(q3_m20_cqp[q3_m20_cqp["k"] < 20]["k"].unique())], fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.set_xlabel(r"Weight Ratio $R_W = W_k / W_{k+1}$ (Normal Scale)", fontsize=14)
        ax2.set_ylabel("Stream Ingestion Saved (%)", fontsize=14)
        ax2.set_title(r"(b) Q3 (M=20): Savings vs. Ratio $W_k/W_{k+1}$", fontsize=15)
        ax2.grid(True, linestyle="--", alpha=0.5)
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, fontsize=13, bbox_to_anchor=(0.5, 0.02))
        save_figure(fig, out_dir / "part_b" / "fig2g_q3_m20_savings_and_ratio_w_grouped", profile)

        # 6. Relative Ratio W_k / W_{k+1} Correlation (Grouped Means)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
        plt.subplots_adjust(bottom=0.20, top=0.88, wspace=0.28)
        for idx, (z, ax) in enumerate([(1.0, ax1), (2.0, ax2)]):
            if z not in q3_m20_cqp["z"].values:
                continue
            sub = q3_m20_cqp[(q3_m20_cqp["z"] == z) & (q3_m20_cqp["k"] < 20)]
            zs = z_styles[z]
            grouped = sub.groupby("k")[["ratio_w", "stream_savings_pct"]].agg(["mean", "sem"]).reset_index()
            x = grouped[("ratio_w", "mean")].values
            y = grouped[("stream_savings_pct", "mean")].values
            yerr = grouped[("stream_savings_pct", "sem")].values

            ax.errorbar(
                x, y, yerr=yerr, fmt=zs["marker"], markersize=8, capsize=4,
                color=zs["color"], markeredgecolor="black", markeredgewidth=1.0,
                linewidth=1.8, label=f"Mean per Rank k (z={z})"
            )
            slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color="#d62728", linestyle="--", linewidth=2.0, label=f"Linear Fit (r = {r_val:.2f})")

            rho_val, _ = stats.spearmanr(x, y)
            p_pow = int(np.floor(np.log10(max(p_val, 1e-100))))
            p_display = f"$p < 10^{{{p_pow}}}$" if p_val < 0.001 else f"$p = {p_val:.3f}$"
            stats_text = (
                f"Pearson $r = {r_val:.3f}$ ({p_display})\n"
                f"Spearman $\\rho = {rho_val:.3f}$\n"
                f"$R^2 = {r_val**2:.3f}$\n"
                f"Slope: ${slope:+.2f}\\%$ / unit ratio"
            )
            ax.text(
                0.05, 0.93, stats_text, transform=ax.transAxes,
                fontsize=11.5, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.92)
            )
            ax.set_xlabel(r"Weight Ratio $R_W = W_k / W_{k+1}$ (Normal Scale)", fontsize=14)
            if idx == 0:
                ax.set_ylabel("Stream Ingestion Saved (%)", fontsize=14)
            ax.set_title(f"Q3 (M=20, z={z}) Ratio $W_k/W_{{k+1}}$ Correlation", fontsize=15)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="lower right", fontsize=11, frameon=True)
        save_figure(fig, out_dir / "part_b" / "fig2g_q3_m20_ratio_w_correlation_grouped", profile)

    # --- Fig 2j: Frozen Controlled Parameter Slices Grid ---
    slice_configs = [
        {"query": "q3", "M": 20, "z": 1.0, "title": "Q3 (M=20, z=1.0)", "color": "#1f77b4", "marker": "o"},
        {"query": "q3", "M": 20, "z": 2.0, "title": "Q3 (M=20, z=2.0)", "color": "#ff7f0e", "marker": "s"},
        {"query": "q3", "M": 10, "z": 1.0, "title": "Q3 (M=10, z=1.0)", "color": "#2ca02c", "marker": "^"},
        {"query": "q3", "M": 10, "z": 2.0, "title": "Q3 (M=10, z=2.0)", "color": "#9467bd", "marker": "v"},
    ]

    # Grid 1: Linear Delta W (Grouped Means)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    plt.subplots_adjust(hspace=0.32, wspace=0.25, bottom=0.10, top=0.92)
    for idx, cfg in enumerate(slice_configs):
        r_idx, c_idx = idx // 2, idx % 2
        ax = axes[r_idx, c_idx]
        sub = df[(df["query"] == cfg["query"]) & (df["M"] == cfg["M"]) & (df["z"] == cfg["z"]) & 
                 (df["dataset_scale"] == 0.5) & (df["strategy"] == "CQP")].copy()
        if sub.empty:
            continue
        dws = []
        for _, row in sub.iterrows():
            w = get_distribution_weights("zipf", cfg["M"], cfg["z"], c=1)
            g = compute_gap_metrics(w, int(row["k"]))
            dws.append(g["delta_w"])
        sub["delta_w"] = dws

        grouped = sub.groupby("k")[["delta_w", "stream_savings_pct"]].agg(["mean", "sem"]).reset_index()
        x = grouped[("delta_w", "mean")].values
        y = grouped[("stream_savings_pct", "mean")].values
        yerr = grouped[("stream_savings_pct", "sem")].values

        ax.errorbar(x, y, yerr=yerr, fmt=cfg["marker"], markersize=7.5, capsize=3.5,
                    color=cfg["color"], markeredgecolor="black", markeredgewidth=1.0, linewidth=1.8)

        slope, intercept, r_val, p_val, _ = stats.linregress(x, y) if len(x) > 2 else (0.0, 0.0, 1.0, 0.0, 0.0)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="#d62728", linestyle="--", linewidth=1.8)

        p_pow = int(np.floor(np.log10(max(p_val, 1e-100))))
        p_str = f"$p < 10^{{{p_pow}}}$" if p_val < 0.001 else f"$p = {p_val:.3f}$"
        stats_text = f"Mean per $k$: $r = {r_val:.3f}$ ({p_str})\n$R^2 = {r_val**2:.3f}$"
        ax.text(0.05, 0.92, stats_text, transform=ax.transAxes, fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92))
        ax.set_title(cfg["title"], fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)
        if c_idx == 0:
            ax.set_ylabel("Stream Savings (%)", fontsize=13)
        if r_idx == 1:
            ax.set_xlabel(r"Weight Gap $\Delta W = W_k - W_{k+1}$", fontsize=13)
    save_figure(fig, out_dir / "part_b" / "fig2g_frozen_slices_grid", profile)

    # Grid 2: Log-Scale Delta W (Grouped Means)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    plt.subplots_adjust(hspace=0.32, wspace=0.25, bottom=0.10, top=0.92)
    for idx, cfg in enumerate(slice_configs):
        r_idx, c_idx = idx // 2, idx % 2
        ax = axes[r_idx, c_idx]
        sub = df[(df["query"] == cfg["query"]) & (df["M"] == cfg["M"]) & (df["z"] == cfg["z"]) & 
                 (df["dataset_scale"] == 0.5) & (df["strategy"] == "CQP") & (df["k"] < cfg["M"])].copy()
        if sub.empty:
            continue
        dws = []
        for _, row in sub.iterrows():
            w = get_distribution_weights("zipf", cfg["M"], cfg["z"], c=1)
            g = compute_gap_metrics(w, int(row["k"]))
            dws.append(g["delta_w"])
        sub["delta_w"] = dws

        grouped = sub.groupby("k")[["delta_w", "stream_savings_pct"]].agg(["mean", "sem"]).reset_index()
        x = grouped[("delta_w", "mean")].values
        y = grouped[("stream_savings_pct", "mean")].values
        yerr = grouped[("stream_savings_pct", "sem")].values

        ax.errorbar(x, y, yerr=yerr, fmt=cfg["marker"], markersize=7.5, capsize=3.5,
                    color=cfg["color"], markeredgecolor="black", markeredgewidth=1.0, linewidth=1.8)

        log_x = np.log10(x)
        slope, intercept, r_val, p_val, _ = stats.linregress(log_x, y) if len(x) > 2 else (0.0, 0.0, 1.0, 0.0, 0.0)
        x_grid = np.geomspace(x.min(), x.max(), 100)
        ax.plot(x_grid, slope * np.log10(x_grid) + intercept, color="#d62728", linestyle="--", linewidth=1.8)

        p_pow = int(np.floor(np.log10(max(p_val, 1e-100))))
        p_str = f"$p < 10^{{{p_pow}}}$" if p_val < 0.001 else f"$p = {p_val:.3f}$"
        stats_text = f"Mean per $k$: $r = {r_val:.3f}$ ({p_str})\n$R^2 = {r_val**2:.3f}$"
        ax.text(0.05, 0.92, stats_text, transform=ax.transAxes, fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92))
        ax.set_xscale("log")
        ax.set_title(cfg["title"], fontsize=14)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        if c_idx == 0:
            ax.set_ylabel("Stream Savings (%)", fontsize=13)
        if r_idx == 1:
            ax.set_xlabel(r"Weight Gap $\Delta W = W_k - W_{k+1}$ (Log Scale)", fontsize=13)
    save_figure(fig, out_dir / "part_b" / "fig2g_frozen_slices_grid_delta_w_log_grouped", profile)

    # Grid 3: Normal-Scale Ratio W_k / W_{k+1} (Grouped Means)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    plt.subplots_adjust(hspace=0.32, wspace=0.25, bottom=0.10, top=0.92)
    for idx, cfg in enumerate(slice_configs):
        r_idx, c_idx = idx // 2, idx % 2
        ax = axes[r_idx, c_idx]
        sub = df[(df["query"] == cfg["query"]) & (df["M"] == cfg["M"]) & (df["z"] == cfg["z"]) & 
                 (df["dataset_scale"] == 0.5) & (df["strategy"] == "CQP") & (df["k"] < cfg["M"])].copy()
        if sub.empty:
            continue
        ratios_list = []
        for _, row in sub.iterrows():
            w = get_distribution_weights("zipf", cfg["M"], cfg["z"], c=1)
            g = compute_gap_metrics(w, int(row["k"]))
            ratios_list.append(g["ratio_w"])
        sub["ratio_w"] = ratios_list

        grouped = sub.groupby("k")[["ratio_w", "stream_savings_pct"]].agg(["mean", "sem"]).reset_index()
        x = grouped[("ratio_w", "mean")].values
        y = grouped[("stream_savings_pct", "mean")].values
        yerr = grouped[("stream_savings_pct", "sem")].values

        ax.errorbar(x, y, yerr=yerr, fmt=cfg["marker"], markersize=7.5, capsize=3.5,
                    color=cfg["color"], markeredgecolor="black", markeredgewidth=1.0, linewidth=1.8)

        slope, intercept, r_val, p_val, _ = stats.linregress(x, y) if len(x) > 2 else (0.0, 0.0, 1.0, 0.0, 0.0)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="#d62728", linestyle="--", linewidth=1.8)

        p_pow = int(np.floor(np.log10(max(p_val, 1e-100))))
        p_str = f"$p < 10^{{{p_pow}}}$" if p_val < 0.001 else f"$p = {p_val:.3f}$"
        stats_text = f"Mean per $k$: $r = {r_val:.3f}$ ({p_str})\n$R^2 = {r_val**2:.3f}$"
        ax.text(0.05, 0.92, stats_text, transform=ax.transAxes, fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92))
        ax.set_title(cfg["title"], fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)
        if c_idx == 0:
            ax.set_ylabel("Stream Savings (%)", fontsize=13)
        if r_idx == 1:
            ax.set_xlabel(r"Weight Ratio $R_W = W_k / W_{k+1}$ (Normal Scale)", fontsize=13)
    save_figure(fig, out_dir / "part_b" / "fig2g_frozen_slices_grid_ratio_w_grouped", profile)


# =========================================================================
# PART C: Conformal Set Calibration Dynamics Suite (alpha in {0.01, 0.05, 0.10})
# =========================================================================

def plot_part_c_calibration(df: pd.DataFrame, out_dir: Path, profile: str = "paper"):
    """Generates Part c figures: Alpha sensitivity runtime bars, tau_alpha bar progression, and new +/-25% line plots."""
    setup_v3_style(profile)
    queries = ["q1a", "q1b", "q2a", "q2b", "q3", "q4"]
    c_parts = []
    for q in queries:
        m_val = CANONICAL_M[q]
        c_parts.append(df[(df["query"] == q) & (df["M"] == m_val) & (df["k"] == 3) & (df["z"] == 1.0) & (df["strategy"] == "CQP") & (df["dataset_scale"] == 0.5)])
    c_df = pd.concat(c_parts) if c_parts else df[(df["k"] == 3) & (df["z"] == 1.0) & (df["strategy"] == "CQP") & (df["dataset_scale"] == 0.5)]
    alphas = [0.01, 0.05, 0.10]
    alpha_hatches = {0.01: "///", 0.05: "//", 0.10: "/"}
    alpha_labels = {0.01: "α = 0.01 (99% coverage)", 0.05: "α = 0.05 (95% coverage)", 0.10: "α = 0.10 (90% coverage)"}

    # --- Fig 3a: Alpha Sensitivity Runtime Bars across all 6 queries ---
    fig, ax = plt.subplots(figsize=(11, 3.2))
    plt.subplots_adjust(bottom=0.26)

    bar_w = 0.20
    x = np.arange(len(queries))

    for a_idx, a in enumerate(alphas):
        a_data = c_df[c_df["alpha"] == a].groupby("query")["engine_time_ms"].agg(["mean", "sem"]).reindex(queries).reset_index()
        offset = (a_idx - 1) * bar_w
        ax.bar(
            x + offset, a_data["mean"], yerr=a_data["sem"], width=bar_w,
            color=PALETTE["cqp"], hatch=alpha_hatches[a],
            edgecolor="black", linewidth=0.8, capsize=3,
            label=alpha_labels[a]
        )

    ax.set_xticks(x)
    ax.set_xticklabels([QUERY_INFO[q]["short"] for q in queries], fontsize=13)
    ax.set_ylabel("CQP Execution Time (ms)", fontsize=14)
    if profile == "presentation":
        ax.set_title("Conformal Calibration Sensitivity Across Significance Budget α", fontsize=16)
    ax.legend(loc="upper left", frameon=True, fontsize=13)
    save_figure(fig, out_dir / "part_c" / "fig3a_alpha_runtime_bars", profile)

    # --- Fig 3b: Calibrated Tau Threshold Progression (6 Subplots to handle different units) ---
    fig, axes = plt.subplots(1, 6, figsize=(18, 2.9), sharey=False)
    plt.subplots_adjust(bottom=0.32, wspace=0.35, top=0.88)

    legend_handles = {}
    for idx, q in enumerate(queries):
        ax = axes[idx]
        q_name = QUERY_INFO[q]["name"]
        x_bars = np.arange(len(alphas))
        bar_w = 0.42

        tau_vals = []
        for a in alphas:
            row = c_df[(c_df["query"] == q) & (c_df["alpha"] == a)]
            tau_vals.append(float(row["tau_threshold"].mean()) if not row.empty else 0.0)

        for a_idx, a in enumerate(alphas):
            b = ax.bar(
                x_bars[a_idx], tau_vals[a_idx], width=bar_w,
                color="#009E73", hatch=alpha_hatches[a],
                edgecolor="black", linewidth=0.8, label=alpha_labels[a]
            )
            if a not in legend_handles:
                legend_handles[a] = b

        ax.set_title(q_name if profile == "presentation" else f"({chr(97+idx)}) {q_name}", fontsize=15, pad=6)
        ax.set_xticks(x_bars)
        ax.set_xticklabels([f"α={a:.2f}" for a in alphas], fontsize=13)
        ax.set_xlabel("Significance (α)", fontsize=14)
        if idx == 0:
            ax.set_ylabel("Threshold ($\\tau_\\alpha$)", fontsize=15)

    fig.legend(
        [legend_handles[a] for a in alphas], [alpha_labels[a] for a in alphas],
        loc="lower center", ncol=3, frameon=True, fontsize=14, bbox_to_anchor=(0.5, 0.02)
    )
    save_figure(fig, out_dir / "part_c" / "fig3b_alpha_tau_thresholds", profile)

    # --- Fig 3c: Calibrated Tau Line Plots Relative Percentage Scale (+/- 10%, lines every 2.5%) ---
    def render_fig3c(show_on_point_labels: bool, filename_suffix: str = ""):
        fig, axes = plt.subplots(1, 6, figsize=(18, 2.9), sharey=True)
        plt.subplots_adjust(bottom=0.28, wspace=0.18, top=0.88)
        for idx, q in enumerate(queries):
            ax = axes[idx]
            q_name = QUERY_INFO[q]["name"]

            taus = {}
            for a in alphas:
                row = c_df[(c_df["query"] == q) & (c_df["alpha"] == a)]
                taus[a] = float(row["tau_threshold"].mean()) if not row.empty else 0.0

            tau_ref = taus.get(0.05, 1.0)
            pct_devs = [((taus[a] - tau_ref) / tau_ref * 100.0) if tau_ref > 0 else 0.0 for a in alphas]

            ax.plot(alphas, pct_devs, marker="o", markersize=7, color="#0072B2", linewidth=2.0, zorder=4)
            ax.axhline(0.0, color="#333333", linestyle="-", linewidth=1.2, zorder=3)

            # Grid lines every 2.5% between -10% and +10%
            grid_ticks = np.arange(-10.0, 10.1, 2.5)
            ax.set_yticks(grid_ticks)
            ax.set_yticklabels([f"{t:+.1f}%" if t % 1 != 0 else (f"{int(t):+d}%" if t != 0 else "0%") for t in grid_ticks], fontsize=11.5)
            ax.set_ylim(-10, 10)

            # Point value annotations (if enabled)
            if show_on_point_labels:
                for a_val, dev in zip(alphas, pct_devs):
                    ax.annotate(
                        f"{dev:+.1f}%", (a_val, dev),
                        textcoords="offset points", xytext=(0, 7 if dev >= 0 else -14),
                        ha="center", fontsize=11.5, fontweight="bold", color="#0072B2"
                    )

            # Reference value text at bottom (preserved in both versions)
            ax.text(
                0.5, 0.08, f"Ref $\\tau_{{0.05}}={format_sigfigs(tau_ref, 3, max_decimals=3)}$",
                transform=ax.transAxes, ha="center", fontsize=11.5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f0f0", edgecolor="#cccccc", alpha=0.9)
            )

            ax.set_title(q_name if profile == "presentation" else f"({chr(97+idx)}) {q_name}", fontsize=15, pad=6)
            ax.set_xticks(alphas)
            ax.set_xticklabels([f"α={a:.2f}" for a in alphas], fontsize=13)
            ax.set_xlabel("Significance (α)", fontsize=14)
            if idx == 0:
                ax.set_ylabel("Relative Deviation from $\\tau_{0.05}$ (%)", fontsize=15)

        save_figure(fig, out_dir / "part_c" / f"fig3c_tau_relative_percent_lines{filename_suffix}", profile)

    # Render with on-point labels and without (clean lines)
    render_fig3c(show_on_point_labels=True, filename_suffix="")
    render_fig3c(show_on_point_labels=False, filename_suffix="_nolabels")

    # --- Fig 3d: Calibrated Tau Line Plots True Values up to 3 Significant Figures / 3 Decimals (+/- 10%, lines every 2.5%) ---
    def render_fig3d(show_on_point_labels: bool, filename_suffix: str = ""):
        fig, axes = plt.subplots(1, 6, figsize=(18, 2.9), sharey=False)
        plt.subplots_adjust(bottom=0.28, wspace=0.34, top=0.88)
        for idx, q in enumerate(queries):
            ax = axes[idx]
            q_name = QUERY_INFO[q]["name"]

            taus = {}
            for a in alphas:
                row = c_df[(c_df["query"] == q) & (c_df["alpha"] == a)]
                taus[a] = float(row["tau_threshold"].mean()) if not row.empty else 0.0

            tau_ref = taus.get(0.05, 1.0)
            tau_vals = [taus[a] for a in alphas]

            ax.plot(alphas, tau_vals, marker="s", markersize=7, color="#D55E00", linewidth=2.0, zorder=4)
            ax.axhline(tau_ref, color="#333333", linestyle="-", linewidth=1.2, zorder=3)

            # Bounds at +/- 10% from tau_ref
            y_min = 0.90 * tau_ref
            y_max = 1.10 * tau_ref
            ax.set_ylim(y_min, y_max)

            # Grid lines and ticks at every 2.5%
            grid_ticks = [tau_ref * (1.0 + 0.025 * i) for i in range(-4, 5)]
            ax.set_yticks(grid_ticks)
            ax.set_yticklabels([format_sigfigs(t, 3, max_decimals=3) for t in grid_ticks], fontsize=11.5)

            # Point value annotations formatted to 3 sigfigs with max 3 decimals (if enabled)
            if show_on_point_labels:
                for a_val, t_val in zip(alphas, tau_vals):
                    ax.annotate(
                        format_sigfigs(t_val, 3, max_decimals=3), (a_val, t_val),
                        textcoords="offset points", xytext=(0, 7 if t_val >= tau_ref else -14),
                        ha="center", fontsize=11.5, fontweight="bold", color="#D55E00"
                    )

            ax.set_title(q_name if profile == "presentation" else f"({chr(97+idx)}) {q_name}", fontsize=15, pad=6)
            ax.set_xticks(alphas)
            ax.set_xticklabels([f"α={a:.2f}" for a in alphas], fontsize=13)
            ax.set_xlabel("Significance (α)", fontsize=14)
            if idx == 0:
                ax.set_ylabel("Threshold $\\tau_\\alpha$ (3 Sig Figs / 3 Dec)", fontsize=15)

        save_figure(fig, out_dir / "part_c" / f"fig3d_tau_absolute_sigfig_lines{filename_suffix}", profile)

    # Render with on-point labels and without (clean lines)
    render_fig3d(show_on_point_labels=True, filename_suffix="")
    render_fig3d(show_on_point_labels=False, filename_suffix="_nolabels")


# =========================================================================
# PART D: Database Size Scaling Suite (Q3 & Q4, scale 0.5 vs scale 1.0)
# =========================================================================

def plot_part_d_scale(df: pd.DataFrame, out_dir: Path, profile: str = "paper"):
    """Generates Part d figures: Database size scaling linearity for Q3 and Q4."""
    setup_v3_style(profile)
    # Explicitly filter canonical configurations: scale 0.5 uses M=10, scale 1.0 uses M=20
    d_df = df[
        df["query"].isin(["q3", "q4"]) &
        (df["k"] == 3) & (df["z"] == 1.0) & (df["alpha"] == 0.05) &
        (((df["dataset_scale"] == 0.5) & (df["M"] == 10)) | ((df["dataset_scale"] == 1.0) & (df["M"] == 20)))
    ]
    baseline_map = get_baseline_map(df)
    scales = sorted(d_df["dataset_scale"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    plt.subplots_adjust(bottom=0.25, wspace=0.30)

    legend_handles = {}
    for idx, q in enumerate(["q3", "q4"]):
        ax = axes[idx]
        sub = d_df[d_df["query"] == q]
        x = np.arange(len(scales))
        bar_w = 0.28

        for s_idx, s in enumerate(["Baseline", "CQP"]):
            s_style = STRATEGY_STYLES[s]
            offset = (s_idx - 0.5) * bar_w
            means = []
            sems = []
            for sc in scales:
                sc_sub = sub[(sub["dataset_scale"] == sc) & (sub["strategy"] == s)]
                if not sc_sub.empty:
                    means.append(sc_sub["engine_time_ms"].mean())
                    sems.append(sc_sub["engine_time_ms"].sem() if not pd.isna(sc_sub["engine_time_ms"].sem()) else 0.0)
                elif s == "Baseline" and (q, 1.0) in baseline_map:
                    means.append(baseline_map[(q, 1.0)])
                    sems.append(0.0)
                else:
                    means.append(0.0)
                    sems.append(0.0)

            bars = ax.bar(
                x + offset, means, yerr=sems, width=bar_w,
                color=s_style["color"], hatch=s_style["hatch"],
                edgecolor=s_style["edgecolor"], linewidth=s_style["linewidth"],
                capsize=3, label=s_style["label"]
            )
            if s_style["label"] not in legend_handles and len(bars) > 0:
                legend_handles[s_style["label"]] = bars[0]

        ax.set_xticks(x)
        ax.set_xticklabels([f"Scale {s:.1f} ({int(s*50)}k tuples)" for s in scales], fontsize=13)
        ax.set_title(QUERY_INFO[q]["label"], fontsize=15)
        if idx == 0:
            ax.set_ylabel("Execution Time (ms)", fontsize=15)

    fig.legend(
        list(legend_handles.values()), list(legend_handles.keys()),
        loc="lower center", ncol=2, frameon=True, fontsize=14, bbox_to_anchor=(0.5, 0.02)
    )
    save_figure(fig, out_dir / "part_d" / "fig4_database_scale_linearity", profile)


# =========================================================================
# PART E: Reasoning Strategy Ablation Suite (Q2a & Q3)
# =========================================================================

def plot_part_e_ablations(df: pd.DataFrame, out_dir: Path, profile: str = "paper"):
    """
    Generates Part e figures:
    1. fig5a_strategy_capabilities_ablation: Unified grouped bar chart grouped by query (Q2a, Q3, Q4)
       with capability-based naming and squeezed contiguous bars (no gaps).
    2. fig5b_strategy_capabilities_strip: 3-panel strip with skinnier subplots.
    3. fig5_strategy_ablation_dual_panel: Legacy dual-panel z-sweep ablation comparison.
    """
    setup_v3_style(profile)

    # Filter for User 2 canonical ablation point: scale=0.5, M=10, k=3, alpha=0.05, z=1.0
    ab_df = df[
        (df["dataset_scale"] == 0.5)
        & (df["M"] == 10)
        & (df["k"] == 3)
        & (abs(df["alpha"] - 0.05) < 1e-4)
        & (abs(df["z"] - 1.0) < 1e-4)
        & df["query"].isin(["q2a", "q3", "q4"])
    ].copy()

    ab_df["capability"] = ab_df.apply(lambda r: map_capability_strategy(r["query"], r["strategy_raw"]), axis=1)
    ab_df = ab_df[ab_df["capability"].notnull()]

    q_list = ["q2a", "q3", "q4"]
    q_labels = [
        "Q2a: COCO\n(Nested)",
        "Q3: TPC-H\n(3-Table)",
        "Q4: TPC-H\n(4-Table)",
    ]
    bar_w = 0.18

    # Common Legend Handles in strict CAPABILITY_ORDER
    handles = [
        mpatches.Patch(
            facecolor=CAPABILITY_STYLES[s]["color"],
            hatch=CAPABILITY_STYLES[s]["hatch"],
            edgecolor=CAPABILITY_STYLES[s]["edgecolor"],
            linewidth=CAPABILITY_STYLES[s]["linewidth"],
            label=CAPABILITY_STYLES[s]["label"],
        )
        for s in CAPABILITY_ORDER
    ]

    # --- 1. fig5a: Single Unified Grouped Bar Chart (Grouped by Query) ---
    figsize_a = (8.6, 5.0) if profile == "paper" else (9.6, 5.5)
    fig_a, ax_a = plt.subplots(figsize=figsize_a)
    plt.subplots_adjust(bottom=0.32, top=0.91, left=0.11, right=0.96)

    x_centers = [0.0, 1.35, 2.70]
    bar_w = 0.19

    for q_idx, q in enumerate(q_list):
        q_sub = ab_df[ab_df["query"] == q]
        means = q_sub.groupby("capability")["engine_time_ms"].mean()
        sems = q_sub.groupby("capability")["engine_time_ms"].sem()

        # Extract only present strategies in canonical order
        present = [s for s in CAPABILITY_ORDER if s in means.index]
        N = len(present)

        for i, s in enumerate(present):
            offset = -(N - 1) * bar_w / 2.0 + i * bar_w
            x_pos = x_centers[q_idx] + offset
            val = means[s]
            err = sems.get(s, 0.0)
            st = CAPABILITY_STYLES[s]

            ax_a.bar(
                x_pos, val, yerr=err, width=bar_w,
                color=st["color"], hatch=st["hatch"],
                edgecolor=st["edgecolor"], linewidth=1.1,
                capsize=3.5,
            )

    ax_a.set_xticks(x_centers)
    ax_a.set_xticklabels(q_labels, fontsize=15, fontweight="semibold")
    ax_a.tick_params(axis="y", labelsize=14)
    ax_a.set_ylabel("Execution Time (ms)", fontsize=16, fontweight="bold", labelpad=8)
    ax_a.set_ylim(0, 32)
    ax_a.set_title("Strategy Capability Ablation (M=10, k=3, z=1.0)", fontsize=17, fontweight="bold", pad=12)
    ax_a.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax_a.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.21),
        ncol=3, frameon=True, fontsize=13.0, handlelength=1.8, handleheight=1.1,
        columnspacing=1.1, borderpad=0.6
    )

    save_figure(fig_a, out_dir / "part_e" / "fig5a_strategy_capabilities_ablation", profile)
    save_figure(fig_a, out_dir / "part_e" / "fig5_strategy_capabilities_ablation", profile)

    # --- 2. fig5b: 3-Panel Strip (Skinnier Subplots) ---
    figsize_b = (10.8, 4.0) if profile == "paper" else (12.0, 4.4)
    fig_b, axes_b = plt.subplots(1, 3, figsize=figsize_b, sharey=True)
    plt.subplots_adjust(bottom=0.28, wspace=0.18, top=0.88, left=0.09, right=0.97)

    sub_titles = [
        "(a) Q2a: COCO (Nested)" if profile == "paper" else "Q2a: COCO (Nested)",
        "(b) Q3: TPC-H (3-Table)" if profile == "paper" else "Q3: TPC-H (3-Table)",
        "(c) Q4: TPC-H (4-Table)" if profile == "paper" else "Q4: TPC-H (4-Table)",
    ]

    for q_idx, (q, q_title, ax) in enumerate(zip(q_list, sub_titles, axes_b)):
        q_sub = ab_df[ab_df["query"] == q]
        means = q_sub.groupby("capability")["engine_time_ms"].mean()
        sems = q_sub.groupby("capability")["engine_time_ms"].sem()
        present = [s for s in CAPABILITY_ORDER if s in means.index]
        N = len(present)

        for i, s in enumerate(present):
            offset = -(N - 1) * bar_w / 2.0 + i * bar_w
            val = means[s]
            err = sems.get(s, 0.0)
            st = CAPABILITY_STYLES[s]

            ax.bar(
                offset, val, yerr=err, width=bar_w,
                color=st["color"], hatch=st["hatch"],
                edgecolor=st["edgecolor"], linewidth=1.1,
                capsize=3.5,
            )

        ax.set_xticks([])
        ax.set_title(q_title, fontsize=14, fontweight="semibold")
        ax.tick_params(axis="y", labelsize=13)
        if q_idx == 0:
            ax.set_ylabel("Execution Time (ms)", fontsize=15, fontweight="bold")
        ax.set_ylim(0, 32)
        ax.set_xlim(-(N + 1) * bar_w / 2.0, (N + 1) * bar_w / 2.0)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    fig_b.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=5, frameon=True, fontsize=12.0, handlelength=1.6, handleheight=1.0,
        columnspacing=1.0
    )

    save_figure(fig_b, out_dir / "part_e" / "fig5b_strategy_capabilities_strip", profile)

    # --- 3. Legacy Dual-Panel Comparison (MutEx vs Indep; In-Op vs External Pull) ---
    fig_leg, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0), sharey=False)
    plt.subplots_adjust(bottom=0.25, wspace=0.30)

    z_vals = [1.0, 2.0]
    bar_w_leg = 0.28
    x_leg = np.arange(len(z_vals))

    # Panel 1: Q2a MutEx vs Indep under Volcano Pull (canonical M=10, scale 0.5)
    # Disambiguate: filter strictly to External Pull QN1 so In-Operator Stop (3ms) is not pooled into External Pull (16ms)
    q2a_df = df[(df["query"] == "q2a") & (df["M"] == 10) & (df["k"] == 3) & df["z"].isin(z_vals) & (df["alpha"] == 0.05) & (df["dataset_scale"] == 0.5)]
    q2a_pull_df = q2a_df[q2a_df["strategy_raw"].str.lower().str.contains("qn1 \\(") | q2a_df["strategy_raw"].str.lower().str.contains("external_pull")]
    mutex_data = q2a_pull_df[q2a_pull_df["ablation_strat"] == "MutEx"].groupby("z")["engine_time_ms"].mean().reindex(z_vals)
    indep_data = q2a_pull_df[q2a_pull_df["ablation_strat"] == "Indep"].groupby("z")["engine_time_ms"].mean().reindex(z_vals)

    b1 = ax1.bar(x_leg - bar_w_leg/2, mutex_data, width=bar_w_leg, color=PALETTE["mutex"], hatch="..", edgecolor="k", linewidth=0.8, label="MutEx (Closed-World)")
    b2 = ax1.bar(x_leg + bar_w_leg/2, indep_data, width=bar_w_leg, color=PALETTE["indep"], hatch="--", edgecolor="k", linewidth=0.8, label="Indep (Open-World)")
    ax1.set_xticks(x_leg)
    ax1.set_xticklabels([f"z={z}" for z in z_vals], fontsize=13)
    ax1.set_ylabel("Execution Time (ms)", fontsize=15)
    ax1.set_title("(a) Q2a: COCO Semantic Assumption (Pull)", fontsize=14, fontweight="semibold")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, fontsize=12)

    # Panel 2: Q3 In-Op Stop vs External Pull (canonical M=10, scale 0.5)
    q3_df = df[(df["query"] == "q3") & (df["M"] == 10) & (df["k"] == 3) & df["z"].isin(z_vals) & (df["alpha"] == 0.05) & (df["dataset_scale"] == 0.5)]
    in_op_data = q3_df[q3_df["ablation_strat"] == "In-Op Stop"].groupby("z")["engine_time_ms"].mean().reindex(z_vals)
    pull_data = q3_df[q3_df["ablation_strat"] == "External Pull"].groupby("z")["engine_time_ms"].mean().reindex(z_vals)

    b3 = ax2.bar(x_leg - bar_w_leg/2, in_op_data, width=bar_w_leg, color=PALETTE["in_op_stop"], hatch="//", edgecolor="k", linewidth=0.8, label="In-Operator")
    b4 = ax2.bar(x_leg + bar_w_leg/2, pull_data, width=bar_w_leg, color=PALETTE["external_pull"], hatch="--", edgecolor="k", linewidth=0.8, label="External Pull")
    ax2.set_xticks(x_leg)
    ax2.set_xticklabels([f"z={z}" for z in z_vals], fontsize=13)
    ax2.set_ylabel("Execution Time (ms)", fontsize=15)
    ax2.set_title("(b) Q3: Relational Execution Model", fontsize=14, fontweight="semibold")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, fontsize=12)

    save_figure(fig_leg, out_dir / "part_e" / "fig5_strategy_ablation_dual_panel", profile)


# =========================================================================
# MAIN EXECUTION
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate V3 Publication & Presentation Charts.")
    parser.add_argument("--results-dir", type=str, default="prod_sweep_results_oct13_3pm/paper_focused_quick_results_oct13_3pm/results/paper_focused_quick")
    parser.add_argument("--figures-dir", type=str, default="prod_sweep_results_oct13_3pm/paper_focused_quick_results_oct13_3pm/results/paper_focused_quick/figures_v3")
    parser.add_argument("--supplementary-dirs", nargs="*", default=[
        "prod_sweep_results_oct13_3pm/paper_focused_quick_results_oct13_3pm/results/scale_1_0",
        "prod_sweep_results_oct13_3pm/paper_focused_quick_results_oct13_3pm/results/mcdb_flat",
    ])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    supp_dirs = [Path(d) for d in args.supplementary_dirs]

    print("=================================================================")
    print("V3 Canonical Paper Charts Generator (User 2 Recommendations)")
    print(f"Results Source:     {results_dir}")
    print(f"Figures Output Dir: {figures_dir}")
    print("=================================================================")

    df = load_and_preprocess_results(results_dir, supp_dirs)
    print(f"✓ Loaded {len(df)} total filtered run records (c=1 single-mode cluster).")

    for profile in ["paper", "presentation"]:
        print(f"\n>>> Generating figures for profile: '{profile}'...")
        prof_dir = figures_dir / profile

        plot_part_a_representative(df, prof_dir, profile)
        print("  ✓ Part a (Representative Performance Suite) complete.")

        plot_part_b_topk(df, prof_dir, profile)
        print("  ✓ Part b (Top-k Selection & Scaling Suite) complete.")

        plot_part_c_calibration(df, prof_dir, profile)
        print("  ✓ Part c (Conformal Set Calibration Dynamics) complete.")

        plot_part_d_scale(df, prof_dir, profile)
        print("  ✓ Part d (Database Size Scaling Suite) complete.")

        plot_part_e_ablations(df, prof_dir, profile)
        print("  ✓ Part e (Reasoning Strategy Ablations) complete.")

    print("\n=================================================================")
    print(f"✓ All V3 paper and presentation figures generated successfully!")
    print(f"  Artifact directory: {figures_dir}")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
