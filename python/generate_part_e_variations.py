#!/usr/bin/env python3
"""
Comprehensive Part E Ablation Figure Generator
Produces all requested variations across:
  - Three slot-alignment modes:
      1. closed_gap (compact): Packs only available strategies side-by-side.
      2. gap: Preserves fixed 5-slot alignment across all queries, leaving open empty gaps for inapplicable strategies.
      3. na_bar: Preserves fixed 5-slot alignment, drawing a flat ground bar at 0 with a bold red "N/A" label over each inapplicable slot.
  - Optional CQP (Full) right-most summary bar:
      Copies the height of the best performing strategy for that query group in white with black border and legend label "CQP (Full)".

Figures Generated:
1. fig5a: Single grouped bar charts (k=1, k=3, subset, all queries)
2. fig5b: Multi-panel strip charts (k=1, k=3, subset, all queries)
3. fig5c: Grouped bars for k=1 next to k=3 for a query (subset, all queries)
4. fig5d: Dual-panel subplots (k=1 left, k=3 right, shared Y-axis) (subset, all queries)
5. fig5e: Subplots per query comparing k=1 vs k=3 (subset, all queries)
6. fig5f: Stacked vertical (k=1 top, k=3 bottom) (subset, all queries)
7. fig5g: 2-row grid of query subplots (Top k=1, Bottom k=3) (subset, all queries)
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Add repo to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "production_sweep_package"))

from lib.schema_codec import SchemaCodec, read_jsonl_gz

PALETTE = {
    "baseline": "#7f7f7f",
    "external_indep": "#ff7f0e",
    "external_mutex": "#d95f02",
    "inop_indep": "#1f77b4",
    "inop_mutex": "#2ca02c",
}

CAPABILITY_ORDER = [
    "Baseline",
    "External Pull + Indep",
    "External Pull + MutEx",
    "In-Operator + Indep",
    "In-Operator + MutEx",
]

CAPABILITY_STYLES = {
    "Baseline": {
        "color": PALETTE["baseline"],
        "hatch": "",
        "edgecolor": "black",
        "linewidth": 0.9,
        "label": "Baseline",
    },
    "External Pull + Indep": {
        "color": PALETTE["external_indep"],
        "hatch": "--",
        "edgecolor": "black",
        "linewidth": 0.9,
        "label": "External Pull + Indep",
    },
    "External Pull + MutEx": {
        "color": PALETTE["external_mutex"],
        "hatch": "..",
        "edgecolor": "black",
        "linewidth": 0.9,
        "label": "External Pull + MutEx",
    },
    "In-Operator + Indep": {
        "color": PALETTE["inop_indep"],
        "hatch": "//",
        "edgecolor": "black",
        "linewidth": 0.9,
        "label": "In-Operator + Indep",
    },
    "In-Operator + MutEx": {
        "color": PALETTE["inop_mutex"],
        "hatch": "\\\\",
        "edgecolor": "black",
        "linewidth": 0.9,
        "label": "In-Operator + MutEx",
    },
}

QUERY_INFO = {
    "q1a": {"name": "Q1a", "label": "Q1a: COCO\n(Flat)", "short": "Q1a (COCO)"},
    "q1b": {"name": "Q1b", "label": "Q1b: Jigsaw\n(Flat)", "short": "Q1b (Jigsaw)"},
    "q2a": {"name": "Q2a", "label": "Q2a: COCO\n(Nested)", "short": "Q2a (COCO)"},
    "q2b": {"name": "Q2b", "label": "Q2b: Jigsaw\n(Nested)", "short": "Q2b (Jigsaw)"},
    "q3":  {"name": "Q3",  "label": "Q3: TPC-H\n(3-Table)", "short": "Q3 (TPC-H)"},
    "q4":  {"name": "Q4",  "label": "Q4: TPC-H\n(4-Table)", "short": "Q4 (TPC-H)"},
}

SUBSET_QUERIES = ["q2a", "q3", "q4"]
ALL_QUERIES = ["q1a", "q1b", "q2a", "q2b", "q3", "q4"]


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "STIXGeneral"],
        "font.size": 17,
        "axes.titlesize": 19,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14.5,
        "figure.titlesize": 21,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "lines.linewidth": 2.0,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def map_capability(query: str, raw_strat: str) -> str:
    s = raw_strat.lower()
    if "mcdb" in s:
        return None
    if "naive" in s and "conformal" not in s:
        return "Baseline"
    if "naive_conformal_end" in s:
        return "Baseline"

    is_in_op = ("inoperator" in s or "in-operator" in s or "in_op" in s)
    is_external = ("external" in s or "qn1" in s or "relative" in s) and not is_in_op

    is_mutex = "mutex" in s
    is_indep = "indep" in s

    if query == "q1a":
        if is_in_op:
            return "In-Operator + MutEx"
        if is_external:
            return "External Pull + MutEx"
    if query == "q1b":
        if is_in_op:
            return "In-Operator + Indep"
        if is_external:
            return "External Pull + Indep"

    if is_external and is_indep:
        return "External Pull + Indep"
    if is_external and is_mutex:
        return "External Pull + MutEx"
    if is_in_op and is_indep:
        return "In-Operator + Indep"
    if is_in_op and is_mutex:
        return "In-Operator + MutEx"

    if query in ["q3", "q4"]:
        if is_in_op:
            return "In-Operator + Indep"
        if is_external:
            return "External Pull + Indep"
    return None


def load_dataset(results_dir: Path) -> pd.DataFrame:
    codec = SchemaCodec(results_dir / "schema_lookup.json")
    records = read_jsonl_gz(results_dir / "unified_sweep_results.jsonl.gz", codec)
    rows = []
    for r in records:
        c = r.get("config", {})
        m = r.get("metrics", {})
        rows.append({
            "query": c.get("query"),
            "strategy_raw": c.get("strategy"),
            "k": c.get("k_query"),
            "z": c.get("z"),
            "alpha": c.get("alpha"),
            "dataset_scale": c.get("dataset_scale", 0.5),
            "M": c.get("M", 10),
            "engine_time_ms": m.get("engine_time_ms"),
            "stream_savings_pct": m.get("stream_savings_pct"),
            "covered": m.get("covered"),
        })
    df = pd.DataFrame(rows)
    df["capability"] = df.apply(lambda r: map_capability(r["query"], r["strategy_raw"]), axis=1)
    return df


def get_baseline_map(df: pd.DataFrame) -> dict:
    """Fallback baseline map for queries where baseline was only evaluated once."""
    base_df = df[df["capability"] == "Baseline"]
    return base_df.groupby(["query", "z"])["engine_time_ms"].mean().to_dict()


def save_fig(fig, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {out_base.name}.png / .pdf")


def get_canonical_handles(include_cqp_full: bool = False):
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
    if include_cqp_full:
        handles.append(
            mpatches.Patch(
                facecolor="white",
                edgecolor="black",
                linewidth=1.2,
                label="CQP (Optimal)",
            )
        )
    return handles


def draw_capability_bars(
    ax,
    center_x: float,
    means: dict,
    sems: dict,
    bar_w: float,
    alignment_mode: str = "closed_gap",
    alpha_val: float = 1.0,
    na_fontsize: float = 10.5,
    rotation: int = None,
    include_cqp_full: bool = False,
):
    """
    Renders capability bars under one of three layout modes:
      - 'closed_gap': compact, centers only present capabilities (+ optional right-most CQP Full).
      - 'gap': fixed slots based on CAPABILITY_ORDER (+ optional CQP Full), leaving open empty gaps.
      - 'na_bar': fixed slots, drawing a flat ground bar at y=0 and red bold 'N/A' label.
    """
    if rotation is None:
        rotation = 90 if bar_w < 0.10 else 0

    # Determine best conformal strategy for CQP (Full)
    conformal_means = {s: means[s] for s in means if s != "Baseline"}
    if conformal_means:
        best_strat = min(conformal_means, key=conformal_means.get)
        best_val = means[best_strat]
        best_err = sems.get(best_strat, 0.0)
    else:
        best_val = None
        best_err = 0.0

    if alignment_mode == "closed_gap":
        present = [s for s in CAPABILITY_ORDER if s in means]
        N = len(present) + (1 if (include_cqp_full and best_val is not None) else 0)
        for i, s in enumerate(present):
            offset = -(N - 1) * bar_w / 2.0 + i * bar_w
            x_pos = center_x + offset
            val = means[s]
            err = sems.get(s, 0.0)
            st = CAPABILITY_STYLES[s]
            ax.bar(
                x_pos, val, yerr=err, width=bar_w,
                color=st["color"], hatch=st["hatch"],
                edgecolor=st["edgecolor"], linewidth=1.1,
                alpha=alpha_val, capsize=3.0,
            )
        if include_cqp_full and best_val is not None:
            offset = -(N - 1) * bar_w / 2.0 + len(present) * bar_w
            x_pos = center_x + offset
            ax.bar(
                x_pos, best_val, yerr=best_err, width=bar_w,
                color="white", edgecolor="black", linewidth=1.2,
                alpha=alpha_val, capsize=3.0,
            )
    else:
        # Fixed 5 capability slots (+ optional 6th slot for CQP Full)
        total_slots = len(CAPABILITY_ORDER) + (1 if include_cqp_full else 0)
        N = total_slots
        for i, s in enumerate(CAPABILITY_ORDER):
            offset = -(N - 1) * bar_w / 2.0 + i * bar_w
            x_pos = center_x + offset
            if s in means:
                val = means[s]
                err = sems.get(s, 0.0)
                st = CAPABILITY_STYLES[s]
                ax.bar(
                    x_pos, val, yerr=err, width=bar_w,
                    color=st["color"], hatch=st["hatch"],
                    edgecolor=st["edgecolor"], linewidth=1.1,
                    alpha=alpha_val, capsize=3.0,
                )
            elif alignment_mode == "na_bar":
                # Flat grounding bar at 0 with subtle dashed border
                ax.bar(
                    x_pos, 0.35, width=bar_w,
                    color="#f5f5f5", edgecolor="#cccccc",
                    linestyle=":", linewidth=0.8, alpha=0.7
                )
                # Soft neutral N/A label centered horizontally over the slot
                ax.text(
                    x_pos, 0.8, "N/A",
                    ha="center", va="bottom",
                    fontsize=na_fontsize, fontweight="bold",
                    color="#555555", rotation=rotation
                )
        if include_cqp_full:
            offset = -(N - 1) * bar_w / 2.0 + len(CAPABILITY_ORDER) * bar_w
            x_pos = center_x + offset
            if best_val is not None:
                ax.bar(
                    x_pos, best_val, yerr=best_err, width=bar_w,
                    color="white", edgecolor="black", linewidth=1.2,
                    alpha=alpha_val, capsize=3.0,
                )
            elif alignment_mode == "na_bar":
                ax.bar(
                    x_pos, 0.35, width=bar_w,
                    color="#f5f5f5", edgecolor="#cccccc",
                    linestyle=":", linewidth=0.8, alpha=0.7
                )
                ax.text(
                    x_pos, 0.8, "N/A",
                    ha="center", va="bottom",
                    fontsize=na_fontsize, fontweight="bold",
                    color="#d62728", rotation=rotation
                )


# -------------------------------------------------------------------------
# 1. FIG 5A: Single Grouped Bar Chart (Fixed k)
# -------------------------------------------------------------------------
def plot_fig5a(
    df: pd.DataFrame,
    out_dir: Path,
    q_list: list,
    k_val: int,
    suffix: str,
    alignment_mode: str = "closed_gap",
    include_cqp_full: bool = False,
):
    setup_style()
    sub_df = df[
        (df["dataset_scale"] == 0.5)
        & ((abs(df["alpha"] - 0.05) < 1e-4) | (df["alpha"] == 0.0))
        & (abs(df["z"] - 1.0) < 1e-4)
        & df["query"].isin(q_list)
        & (df["capability"].notnull())
    ].copy()

    b_map = get_baseline_map(df)
    bar_w = 0.11
    if include_cqp_full:
        spacing = 0.95 if len(q_list) <= 3 else 0.88
        figsize = (8.6, 5.8) if len(q_list) <= 3 else (13.8, 6.2)
    else:
        spacing = 0.85 if len(q_list) <= 3 else 0.80
        figsize = (8.2, 5.8) if len(q_list) <= 3 else (13.2, 6.2)

    x_centers = [i * spacing for i in range(len(q_list))]
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.34, top=0.91, left=0.10, right=0.97)

    for q_idx, q in enumerate(q_list):
        q_sub = sub_df[(sub_df["query"] == q) & ((sub_df["k"] == k_val) | (sub_df["capability"] == "Baseline"))]
        means = q_sub.groupby("capability")["engine_time_ms"].mean().to_dict()
        sems = q_sub.groupby("capability")["engine_time_ms"].sem().to_dict()

        if "Baseline" not in means and (q, 1.0) in b_map:
            means["Baseline"] = b_map[(q, 1.0)]
            sems["Baseline"] = 0.0

        draw_capability_bars(
            ax, x_centers[q_idx], means, sems, bar_w,
            alignment_mode=alignment_mode, na_fontsize=11.5,
            include_cqp_full=include_cqp_full
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels([QUERY_INFO[q]["label"] for q in q_list], fontsize=17.5, fontweight="bold")
    ax.tick_params(axis="y", labelsize=16.0)
    ax.set_ylabel("Execution Time (ms)", fontsize=18.5, fontweight="bold", labelpad=8)
    ax.set_ylim(0, 32)
    q_scope = "Focused Queries" if len(q_list) <= 3 else "All Workloads"
    mode_desc = "" if alignment_mode == "closed_gap" else (" [Gaps]" if alignment_mode == "gap" else " [N/A Slots]")
    ax.set_title(f"Strategy Capability Ablation ({q_scope}, k={k_val}, z=1.0, α=0.05){mode_desc}", fontsize=18.5, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    handles = get_canonical_handles(include_cqp_full=include_cqp_full)
    leg_ncol = 3 if len(q_list) <= 3 else min(6, len(handles))
    leg_y = -0.26 if len(q_list) <= 3 else -0.22
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, leg_y),
        ncol=leg_ncol, frameon=True, fontsize=14.5, handlelength=1.6, handleheight=1.0,
        columnspacing=0.8, borderpad=0.5
    )

    mode_tag = "" if alignment_mode == "closed_gap" else f"_{alignment_mode}"
    full_tag = "_cqp_full" if include_cqp_full else ""
    save_fig(fig, out_dir / f"fig5a_strategy_capabilities_ablation_{suffix}{mode_tag}{full_tag}")


# -------------------------------------------------------------------------
# 2. FIG 5B: Multi-Panel Strip (Subplot per Query, Fixed k)
# -------------------------------------------------------------------------
def plot_fig5b(
    df: pd.DataFrame,
    out_dir: Path,
    q_list: list,
    k_val: int,
    suffix: str,
    alignment_mode: str = "closed_gap",
    include_cqp_full: bool = False,
):
    setup_style()
    sub_df = df[
        (df["dataset_scale"] == 0.5)
        & ((abs(df["alpha"] - 0.05) < 1e-4) | (df["alpha"] == 0.0))
        & (abs(df["z"] - 1.0) < 1e-4)
        & df["query"].isin(q_list)
        & (df["capability"].notnull())
    ].copy()

    b_map = get_baseline_map(df)
    bar_w = 0.12
    n_plots = len(q_list)
    figsize = (10.5, 4.8) if n_plots <= 3 else (16.5, 5.2)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize, sharey=True)
    if n_plots == 1:
        axes = [axes]
    plt.subplots_adjust(bottom=0.32, wspace=0.10, top=0.88, left=0.09, right=0.98)

    for q_idx, (q, ax) in enumerate(zip(q_list, axes)):
        q_sub = sub_df[(sub_df["query"] == q) & ((sub_df["k"] == k_val) | (sub_df["capability"] == "Baseline"))]
        means = q_sub.groupby("capability")["engine_time_ms"].mean().to_dict()
        sems = q_sub.groupby("capability")["engine_time_ms"].sem().to_dict()

        if "Baseline" not in means and (q, 1.0) in b_map:
            means["Baseline"] = b_map[(q, 1.0)]
            sems["Baseline"] = 0.0

        draw_capability_bars(
            ax, 0.0, means, sems, bar_w,
            alignment_mode=alignment_mode, na_fontsize=12.0,
            include_cqp_full=include_cqp_full
        )

        ax.set_xticks([])
        q_letter = chr(97 + q_idx)
        ax.set_title(f"({q_letter}) {QUERY_INFO[q]['short']}", fontsize=17.0, fontweight="bold")
        ax.tick_params(axis="y", labelsize=15.0)
        if q_idx == 0:
            ax.set_ylabel("Execution Time (ms)", fontsize=18.0, fontweight="bold")
        ax.set_ylim(0, 32)

        present = [s for s in CAPABILITY_ORDER if s in means]
        extra = 1 if include_cqp_full else 0
        N_xlim = (len(present) + extra) if alignment_mode == "closed_gap" else (5 + extra)
        ax.set_xlim(-(N_xlim + 1) * bar_w / 2.0, (N_xlim + 1) * bar_w / 2.0)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    handles = get_canonical_handles(include_cqp_full=include_cqp_full)
    leg_ncol = 3 if n_plots <= 3 else min(6, len(handles))
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=leg_ncol, frameon=True, fontsize=14.5, handlelength=1.6, handleheight=1.0,
        columnspacing=0.8
    )
    mode_tag = "" if alignment_mode == "closed_gap" else f"_{alignment_mode}"
    full_tag = "_cqp_full" if include_cqp_full else ""
    save_fig(fig, out_dir / f"fig5b_strategy_capabilities_strip_{suffix}{mode_tag}{full_tag}")


# -------------------------------------------------------------------------
# 3. FIG 5C: Grouped Bars for k=1 Next to k=3 for Each Query
# -------------------------------------------------------------------------
def plot_fig5c(
    df: pd.DataFrame,
    out_dir: Path,
    q_list: list,
    suffix: str,
    alignment_mode: str = "closed_gap",
    include_cqp_full: bool = False,
):
    setup_style()
    sub_df = df[
        (df["dataset_scale"] == 0.5)
        & ((abs(df["alpha"] - 0.05) < 1e-4) | (df["alpha"] == 0.0))
        & (abs(df["z"] - 1.0) < 1e-4)
        & df["query"].isin(q_list)
        & (df["capability"].notnull())
    ].copy()

    b_map = get_baseline_map(df)
    bar_w = 0.072
    k_gap = 0.035
    if include_cqp_full:
        query_spacing = 1.25 if len(q_list) <= 3 else 1.16
        figsize = (9.6, 7.0) if len(q_list) <= 3 else (16.2, 7.5)
    else:
        query_spacing = 1.15 if len(q_list) <= 3 else 1.08
        figsize = (9.2, 7.0) if len(q_list) <= 3 else (15.6, 7.5)

    x_centers = [i * query_spacing for i in range(len(q_list))]
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.36, top=0.90, left=0.09, right=0.98)

    k_label_positions = []
    k_labels = []

    for q_idx, q in enumerate(q_list):
        for k_idx, k_val in enumerate([1, 3]):
            q_sub = sub_df[(sub_df["query"] == q) & ((sub_df["k"] == k_val) | (sub_df["capability"] == "Baseline"))]
            means = q_sub.groupby("capability")["engine_time_ms"].mean().to_dict()
            sems = q_sub.groupby("capability")["engine_time_ms"].sem().to_dict()

            if "Baseline" not in means and (q, 1.0) in b_map:
                means["Baseline"] = b_map[(q, 1.0)]
                sems["Baseline"] = 0.0

            extra = 1 if include_cqp_full else 0
            if alignment_mode == "closed_gap":
                present = [s for s in CAPABILITY_ORDER if s in means]
                N = len(present) + extra
            else:
                N = 5 + extra
            cluster_w = N * bar_w

            # Center for this k cluster
            k_cluster_center = x_centers[q_idx] + (-0.5 * (cluster_w + k_gap) if k_idx == 0 else 0.5 * (cluster_w + k_gap))
            k_label_positions.append(k_cluster_center)
            k_labels.append(f"k={k_val}")

            alpha_val = 0.78 if k_idx == 0 else 1.0
            draw_capability_bars(
                ax, k_cluster_center, means, sems, bar_w,
                alignment_mode=alignment_mode, alpha_val=alpha_val, na_fontsize=9.0,
                include_cqp_full=include_cqp_full
            )

    # Primary X-ticks: Query labels centered with extra pad
    ax.set_xticks(x_centers)
    ax.set_xticklabels([QUERY_INFO[q]["label"] for q in q_list], fontsize=16.5, fontweight="bold")
    ax.tick_params(axis="x", pad=28)

    # Secondary text annotations for k=1 vs k=3 directly under each bar group
    for k_pos, k_lbl in zip(k_label_positions, k_labels):
        ax.text(k_pos, -1.8, k_lbl, ha="center", va="top", fontsize=14.0, fontweight="bold", color="#222222")

    ax.tick_params(axis="y", labelsize=15)
    ax.set_ylabel("Execution Time (ms)", fontsize=18.0, fontweight="bold", labelpad=8)
    ax.set_ylim(0, 32)
    q_scope = "Focused Queries" if len(q_list) <= 3 else "All Workloads"
    mode_desc = "" if alignment_mode == "closed_gap" else (" [Gaps]" if alignment_mode == "gap" else " [N/A Slots]")
    ax.set_title(f"Ablation Comparison Across Ranks (k=1 vs k=3, {q_scope}, z=1.0, α=0.05){mode_desc}", fontsize=18.0, fontweight="bold", pad=14)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Legend with k notation
    handles = get_canonical_handles(include_cqp_full=include_cqp_full)
    k_note = mpatches.Patch(facecolor="white", edgecolor="none", label="[Lighter = k=1, Solid = k=3]")
    handles.append(k_note)

    leg_ncol = 3 if len(q_list) <= 3 else min(7, len(handles))
    leg_y = -0.27 if len(q_list) <= 3 else -0.25
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, leg_y),
        ncol=leg_ncol, frameon=True, fontsize=13.0, handlelength=1.6, handleheight=1.0,
        columnspacing=0.8, borderpad=0.5
    )

    mode_tag = "" if alignment_mode == "closed_gap" else f"_{alignment_mode}"
    full_tag = "_cqp_full" if include_cqp_full else ""
    save_fig(fig, out_dir / f"fig5c_k_comparison_grouped_bars_{suffix}{mode_tag}{full_tag}")


# -------------------------------------------------------------------------
# 4. FIG 5D: Dual-Panel Subplots (k=1 on Left, k=3 on Right, Shared Y-axis)
# -------------------------------------------------------------------------
def plot_fig5d(
    df: pd.DataFrame,
    out_dir: Path,
    q_list: list,
    suffix: str,
    alignment_mode: str = "closed_gap",
    include_cqp_full: bool = False,
):
    setup_style()
    sub_df = df[
        (df["dataset_scale"] == 0.5)
        & ((abs(df["alpha"] - 0.05) < 1e-4) | (df["alpha"] == 0.0))
        & (abs(df["z"] - 1.0) < 1e-4)
        & df["query"].isin(q_list)
        & (df["capability"].notnull())
    ].copy()

    b_map = get_baseline_map(df)
    bar_w = 0.11
    if include_cqp_full:
        spacing = 0.95 if len(q_list) <= 3 else 0.88
        figsize = (12.8, 5.6) if len(q_list) <= 3 else (17.5, 6.0)
    else:
        spacing = 0.85 if len(q_list) <= 3 else 0.80
        figsize = (12.2, 5.6) if len(q_list) <= 3 else (16.8, 6.0)

    x_centers = [i * spacing for i in range(len(q_list))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    plt.subplots_adjust(bottom=0.32, wspace=0.10, top=0.89, left=0.08, right=0.98)

    for ax, k_val, panel_title in zip([ax1, ax2], [1, 3], ["(a) k = 1", "(b) k = 3"]):
        for q_idx, q in enumerate(q_list):
            q_sub = sub_df[(sub_df["query"] == q) & ((sub_df["k"] == k_val) | (sub_df["capability"] == "Baseline"))]
            means = q_sub.groupby("capability")["engine_time_ms"].mean().to_dict()
            sems = q_sub.groupby("capability")["engine_time_ms"].sem().to_dict()

            if "Baseline" not in means and (q, 1.0) in b_map:
                means["Baseline"] = b_map[(q, 1.0)]
                sems["Baseline"] = 0.0

            rot_val = 90 if len(q_list) > 3 else 0
            draw_capability_bars(
                ax, x_centers[q_idx], means, sems, bar_w,
                alignment_mode=alignment_mode, na_fontsize=10.0, rotation=rot_val,
                include_cqp_full=include_cqp_full
            )

        ax.set_xticks(x_centers)
        ax.set_xticklabels([QUERY_INFO[q]["label"] for q in q_list], fontsize=15.0, fontweight="bold")
        ax.set_title(panel_title, fontsize=16.5, fontweight="bold", pad=10)
        ax.tick_params(axis="y", labelsize=14.5)
        ax.set_ylim(0, 32)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    ax1.set_ylabel("Execution Time (ms)", fontsize=17.5, fontweight="bold", labelpad=8)

    handles = get_canonical_handles(include_cqp_full=include_cqp_full)
    leg_ncol = 3 if len(q_list) <= 3 else min(6, len(handles))
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=leg_ncol, frameon=True, fontsize=13.5, handlelength=1.6, handleheight=1.0,
        columnspacing=0.8
    )
    mode_tag = "" if alignment_mode == "closed_gap" else f"_{alignment_mode}"
    full_tag = "_cqp_full" if include_cqp_full else ""
    save_fig(fig, out_dir / f"fig5d_k_dual_panel_side_by_side_{suffix}{mode_tag}{full_tag}")


# -------------------------------------------------------------------------
# 5. FIG 5E: Subplots per Query Strip comparing k=1 vs k=3
# -------------------------------------------------------------------------
def plot_fig5e(
    df: pd.DataFrame,
    out_dir: Path,
    q_list: list,
    suffix: str,
    alignment_mode: str = "closed_gap",
    include_cqp_full: bool = False,
):
    setup_style()
    sub_df = df[
        (df["dataset_scale"] == 0.5)
        & ((abs(df["alpha"] - 0.05) < 1e-4) | (df["alpha"] == 0.0))
        & (abs(df["z"] - 1.0) < 1e-4)
        & df["query"].isin(q_list)
        & (df["capability"].notnull())
    ].copy()

    b_map = get_baseline_map(df)
    bar_w = 0.11
    n_plots = len(q_list)
    figsize = (10.5, 5.0) if n_plots <= 3 else (16.8, 5.4)

    fig, axes = plt.subplots(1, n_plots, figsize=figsize, sharey=True)
    if n_plots == 1:
        axes = [axes]
    plt.subplots_adjust(bottom=0.32, wspace=0.10, top=0.88, left=0.08, right=0.98)

    for q_idx, (q, ax) in enumerate(zip(q_list, axes)):
        for k_idx, k_val in enumerate([1, 3]):
            q_sub = sub_df[(sub_df["query"] == q) & ((sub_df["k"] == k_val) | (sub_df["capability"] == "Baseline"))]
            means = q_sub.groupby("capability")["engine_time_ms"].mean().to_dict()
            sems = q_sub.groupby("capability")["engine_time_ms"].sem().to_dict()

            if "Baseline" not in means and (q, 1.0) in b_map:
                means["Baseline"] = b_map[(q, 1.0)]
                sems["Baseline"] = 0.0

            cluster_offset = -0.44 if k_idx == 0 else 0.44
            alpha_val = 0.78 if k_idx == 0 else 1.0
            draw_capability_bars(
                ax, cluster_offset, means, sems, bar_w,
                alignment_mode=alignment_mode, alpha_val=alpha_val, na_fontsize=9.0,
                include_cqp_full=include_cqp_full
            )

        ax.set_xticks([-0.44, 0.44])
        ax.set_xticklabels(["k = 1", "k = 3"], fontsize=14.0, fontweight="bold")
        q_letter = chr(97 + q_idx)
        ax.set_title(f"({q_letter}) {QUERY_INFO[q]['short']}", fontsize=16.0, fontweight="bold")
        ax.tick_params(axis="y", labelsize=14.0)
        if q_idx == 0:
            ax.set_ylabel("Execution Time (ms)", fontsize=17.0, fontweight="bold")
        ax.set_ylim(0, 32)
        ax.set_xlim(-0.90, 0.90)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    handles = get_canonical_handles(include_cqp_full=include_cqp_full)
    leg_ncol = 3 if n_plots <= 3 else min(6, len(handles))
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=leg_ncol, frameon=True, fontsize=14.0, handlelength=1.6, handleheight=1.0,
        columnspacing=0.8
    )
    mode_tag = "" if alignment_mode == "closed_gap" else f"_{alignment_mode}"
    full_tag = "_cqp_full" if include_cqp_full else ""
    save_fig(fig, out_dir / f"fig5e_k_comparison_query_strip_{suffix}{mode_tag}{full_tag}")


# -------------------------------------------------------------------------
# 6. FIG 5F: Stacked Vertical (k=1 Top Row, k=3 Bottom Row)
# -------------------------------------------------------------------------
def plot_fig5f(
    df: pd.DataFrame,
    out_dir: Path,
    q_list: list,
    suffix: str,
    alignment_mode: str = "closed_gap",
    include_cqp_full: bool = False,
):
    setup_style()
    sub_df = df[
        (df["dataset_scale"] == 0.5)
        & ((abs(df["alpha"] - 0.05) < 1e-4) | (df["alpha"] == 0.0))
        & (abs(df["z"] - 1.0) < 1e-4)
        & df["query"].isin(q_list)
        & (df["capability"].notnull())
    ].copy()

    b_map = get_baseline_map(df)
    bar_w = 0.11
    if include_cqp_full:
        spacing = 0.95 if len(q_list) <= 3 else 0.88
        figsize = (8.8, 8.8) if len(q_list) <= 3 else (14.0, 9.2)
    else:
        spacing = 0.85 if len(q_list) <= 3 else 0.80
        figsize = (8.4, 8.8) if len(q_list) <= 3 else (13.2, 9.2)

    x_centers = [i * spacing for i in range(len(q_list))]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True)
    plt.subplots_adjust(bottom=0.22, hspace=0.25, top=0.92, left=0.10, right=0.96)

    for ax, k_val, panel_title in zip([ax1, ax2], [1, 3], ["(a) k = 1", "(b) k = 3"]):
        for q_idx, q in enumerate(q_list):
            q_sub = sub_df[(sub_df["query"] == q) & ((sub_df["k"] == k_val) | (sub_df["capability"] == "Baseline"))]
            means = q_sub.groupby("capability")["engine_time_ms"].mean().to_dict()
            sems = q_sub.groupby("capability")["engine_time_ms"].sem().to_dict()

            if "Baseline" not in means and (q, 1.0) in b_map:
                means["Baseline"] = b_map[(q, 1.0)]
                sems["Baseline"] = 0.0

            draw_capability_bars(
                ax, x_centers[q_idx], means, sems, bar_w,
                alignment_mode=alignment_mode, na_fontsize=10.5,
                include_cqp_full=include_cqp_full
            )

        ax.set_title(panel_title, fontsize=16.0, fontweight="bold", pad=8)
        ax.tick_params(axis="y", labelsize=14.5)
        ax.set_ylabel("Execution Time (ms)", fontsize=16.0, fontweight="bold", labelpad=8)
        ax.set_ylim(0, 32)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    ax2.set_xticks(x_centers)
    ax2.set_xticklabels([QUERY_INFO[q]["label"] for q in q_list], fontsize=16.0, fontweight="bold")

    handles = get_canonical_handles(include_cqp_full=include_cqp_full)
    leg_ncol = 3 if len(q_list) <= 3 else min(6, len(handles))
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.03),
        ncol=leg_ncol, frameon=True, fontsize=13.5, handlelength=1.6, handleheight=1.0,
        columnspacing=0.8
    )
    mode_tag = "" if alignment_mode == "closed_gap" else f"_{alignment_mode}"
    full_tag = "_cqp_full" if include_cqp_full else ""
    save_fig(fig, out_dir / f"fig5f_k_stacked_vertical_{suffix}{mode_tag}{full_tag}")


# -------------------------------------------------------------------------
# 7. FIG 5G: 2-Row Grid (Top Row k=1, Bottom Row k=3 per Query Column)
# -------------------------------------------------------------------------
def plot_fig5g(
    df: pd.DataFrame,
    out_dir: Path,
    q_list: list,
    suffix: str,
    alignment_mode: str = "closed_gap",
    include_cqp_full: bool = False,
):
    setup_style()
    sub_df = df[
        (df["dataset_scale"] == 0.5)
        & ((abs(df["alpha"] - 0.05) < 1e-4) | (df["alpha"] == 0.0))
        & (abs(df["z"] - 1.0) < 1e-4)
        & df["query"].isin(q_list)
        & (df["capability"].notnull())
    ].copy()

    b_map = get_baseline_map(df)
    bar_w = 0.12
    n_cols = len(q_list)
    figsize = (10.8, 8.2) if n_cols <= 3 else (17.0, 8.5)

    fig, axes = plt.subplots(2, n_cols, figsize=figsize, sharey=True)
    plt.subplots_adjust(bottom=0.20, hspace=0.26, wspace=0.10, top=0.91, left=0.08, right=0.98)

    for row_idx, k_val in enumerate([1, 3]):
        for col_idx, q in enumerate(q_list):
            ax = axes[row_idx, col_idx] if n_cols > 1 else axes[row_idx]
            q_sub = sub_df[(sub_df["query"] == q) & ((sub_df["k"] == k_val) | (sub_df["capability"] == "Baseline"))]
            means = q_sub.groupby("capability")["engine_time_ms"].mean().to_dict()
            sems = q_sub.groupby("capability")["engine_time_ms"].sem().to_dict()

            if "Baseline" not in means and (q, 1.0) in b_map:
                means["Baseline"] = b_map[(q, 1.0)]
                sems["Baseline"] = 0.0

            draw_capability_bars(
                ax, 0.0, means, sems, bar_w,
                alignment_mode=alignment_mode, na_fontsize=11.0,
                include_cqp_full=include_cqp_full
            )

            ax.set_xticks([])
            if row_idx == 0:
                ax.set_title(f"{QUERY_INFO[q]['short']}\n(k=1)", fontsize=15.0, fontweight="bold")
            else:
                ax.set_title(f"{QUERY_INFO[q]['short']}\n(k=3)", fontsize=15.0, fontweight="bold")

            if col_idx == 0:
                ax.set_ylabel("Execution Time (ms)", fontsize=16.0, fontweight="bold")
            ax.tick_params(axis="y", labelsize=14.0)
            ax.set_ylim(0, 32)

            present = [s for s in CAPABILITY_ORDER if s in means]
            extra = 1 if include_cqp_full else 0
            N_xlim = (len(present) + extra) if alignment_mode == "closed_gap" else (5 + extra)
            ax.set_xlim(-(N_xlim + 1) * bar_w / 2.0, (N_xlim + 1) * bar_w / 2.0)
            ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    handles = get_canonical_handles(include_cqp_full=include_cqp_full)
    leg_ncol = 3 if n_cols <= 3 else min(6, len(handles))
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=leg_ncol, frameon=True, fontsize=14.0, handlelength=1.6, handleheight=1.0,
        columnspacing=0.8
    )
    mode_tag = "" if alignment_mode == "closed_gap" else f"_{alignment_mode}"
    full_tag = "_cqp_full" if include_cqp_full else ""
    save_fig(fig, out_dir / f"fig5g_k_stacked_grid_{suffix}{mode_tag}{full_tag}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Render Comprehensive Part E Ablation Figure Variations")
    parser.add_argument("--results-dir", type=str, default="v3_results_oct17_9pm/v3_results_oct17_9pm/results/canonical_v3",
                        help="Path to results directory containing unified_sweep_results.jsonl.gz")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Path to output figures directory. If omitted, saves under results_dir/figures_v3")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} not found.")
        sys.exit(1)

    print("Loading unified canonical results...")
    df = load_dataset(results_dir)
    print(f"Loaded {len(df)} records.")

    if args.output_dir:
        base_out = Path(args.output_dir)
        out_dirs = [
            base_out / "paper" / "part_e",
            base_out / "presentation" / "part_e",
        ]
    else:
        out_dirs = [
            results_dir / "figures_v3" / "paper" / "part_e",
            results_dir / "figures_v3" / "presentation" / "part_e",
        ]

    modes = ["closed_gap", "gap", "na_bar"]
    cqp_full_options = [False, True]

    for out_dir in out_dirs:
        print(f"\n=======================================================")
        print(f"Rendering Part E Figures into: {out_dir}")
        print(f"=======================================================")

        for cqp_full in cqp_full_options:
            cqp_str = "with CQP (Full) bar" if cqp_full else "standard ablations only"
            print(f"\n>>> Suite: {cqp_str} <<<")

            for mode in modes:
                print(f"  --- Generating Mode: {mode} ---")
                # 1. FIG 5A: Single Grouped Bar Chart (k=1, k=3 for subset and all_queries)
                plot_fig5a(df, out_dir, SUBSET_QUERIES, k_val=3, suffix="subset_k3", alignment_mode=mode, include_cqp_full=cqp_full)
                plot_fig5a(df, out_dir, SUBSET_QUERIES, k_val=1, suffix="subset_k1", alignment_mode=mode, include_cqp_full=cqp_full)
                plot_fig5a(df, out_dir, ALL_QUERIES, k_val=3, suffix="all_queries_k3", alignment_mode=mode, include_cqp_full=cqp_full)
                plot_fig5a(df, out_dir, ALL_QUERIES, k_val=1, suffix="all_queries_k1", alignment_mode=mode, include_cqp_full=cqp_full)

                # 2. FIG 5B: Multi-Panel Strip (k=1, k=3 for subset and all_queries)
                plot_fig5b(df, out_dir, SUBSET_QUERIES, k_val=3, suffix="subset_k3", alignment_mode=mode, include_cqp_full=cqp_full)
                plot_fig5b(df, out_dir, SUBSET_QUERIES, k_val=1, suffix="subset_k1", alignment_mode=mode, include_cqp_full=cqp_full)
                plot_fig5b(df, out_dir, ALL_QUERIES, k_val=3, suffix="all_queries_k3", alignment_mode=mode, include_cqp_full=cqp_full)
                plot_fig5b(df, out_dir, ALL_QUERIES, k_val=1, suffix="all_queries_k1", alignment_mode=mode, include_cqp_full=cqp_full)

                # 3. FIG 5C: Grouped bars for k=1 next to k=3 for a query
                plot_fig5c(df, out_dir, SUBSET_QUERIES, suffix="subset", alignment_mode=mode, include_cqp_full=cqp_full)
                plot_fig5c(df, out_dir, ALL_QUERIES, suffix="all_queries", alignment_mode=mode, include_cqp_full=cqp_full)

                # 4. FIG 5D: Dual-panel subplots (k=1 on left, k=3 on right)
                plot_fig5d(df, out_dir, SUBSET_QUERIES, suffix="subset", alignment_mode=mode, include_cqp_full=cqp_full)
                plot_fig5d(df, out_dir, ALL_QUERIES, suffix="all_queries", alignment_mode=mode, include_cqp_full=cqp_full)

                # 5. FIG 5E: Subplots per query comparing k=1 vs k=3
                plot_fig5e(df, out_dir, SUBSET_QUERIES, suffix="subset", alignment_mode=mode, include_cqp_full=cqp_full)
                plot_fig5e(df, out_dir, ALL_QUERIES, suffix="all_queries", alignment_mode=mode, include_cqp_full=cqp_full)

                # 6. FIG 5F: Stacked vertical (k=1 top, k=3 bottom)
                plot_fig5f(df, out_dir, SUBSET_QUERIES, suffix="subset", alignment_mode=mode, include_cqp_full=cqp_full)
                plot_fig5f(df, out_dir, ALL_QUERIES, suffix="all_queries", alignment_mode=mode, include_cqp_full=cqp_full)

                # 7. FIG 5G: 2-row grid of query subplots (Top k=1, Bottom k=3)
                plot_fig5g(df, out_dir, SUBSET_QUERIES, suffix="subset", alignment_mode=mode, include_cqp_full=cqp_full)
                plot_fig5g(df, out_dir, ALL_QUERIES, suffix="all_queries", alignment_mode=mode, include_cqp_full=cqp_full)

    print("\n✓ All Part E figure variations (both standard and CQP (Full) versions, across closed_gap, gap, and na_bar modes) generated successfully in both paper/ and presentation/ directories!")


if __name__ == "__main__":
    main()
