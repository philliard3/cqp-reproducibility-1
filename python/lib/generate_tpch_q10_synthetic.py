#!/usr/bin/env python3
"""
generate_tpch_q10_synthetic.py

Synthetic Data Generator for Probabilistic TPC-H Q10 (Returned Item Reporting Query).
Simulates a 4-table relational join pipeline:
  Customer(c) ⋈ Orders(o) ⋈ Lineitem(l) ⋈ Nation(n)
under customer return adjudication and refund approval uncertainty.

Standard TPC-H Q10 Semantics:
- Predicates:
    o_orderdate >= date '1993-10-01' AND o_orderdate < date '1994-01-01'
    l_returnflag = 'R'
- Joins:
    c_custkey = o_custkey, l_orderkey = o_orderkey, c_nationkey = n_nationkey
- Group By:
    c_custkey (Customer candidate space |C| ~ 1,000)
- Order By:
    lost_revenue DESC LIMIT 20

Uncertainty Model:
- Return refund approval probability: p_return(l) in [0.10, 0.95]
- Lost revenue: filtered_revenue = l_extendedprice * (1 - l_discount)
- Expected lost revenue: expected_revenue = p_return * filtered_revenue
- Proxy score for stream ordering: S(l) = p_return * l_extendedprice
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


GLOBAL_NATIONS = [
    "ALGERIA", "ARGENTINA", "BRAZIL", "CANADA", "EGYPT",
    "ETHIOPIA", "FRANCE", "GERMANY", "INDIA", "INDONESIA",
    "IRAN", "IRAQ", "JAPAN", "JORDAN", "KENYA",
    "MOROCCO", "MOZAMBIQUE", "PERU", "CHINA", "ROMANIA",
    "SAUDI ARABIA", "VIETNAM", "RUSSIA", "UNITED KINGDOM", "UNITED STATES"
]


def sample_zipf(n_items: int, n_categories: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Sample category indices 0..n_categories-1 from a Zipf distribution."""
    if alpha <= 0.001:
        return rng.integers(0, n_categories, size=n_items)
    ranks = np.arange(1, n_categories + 1, dtype=np.float64)
    weights = 1.0 / (ranks ** alpha)
    probs = weights / np.sum(weights)
    return rng.choice(n_categories, size=n_items, p=probs)


def generate_tpch_q10_dataset(
    n_customers: int = 1000,
    n_orders: Optional[int] = None,
    target_stream_tuples: Optional[int] = None,
    zipf_skew: float = 1.0,
    n_mc_worlds: int = 1000,
    seed: int = 42,
    stream_sort: str = "proxy",
    include_mc_bits: bool = True,
) -> Dict[str, Any]:
    """
    Generates a full synthetic probabilistic TPC-H Q10 dataset.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    n_groups = n_customers

    if target_stream_tuples is not None:
        # Predicate selectivity modeling:
        # o_orderdate in Q4 (days 275..365 of 365) ~ 0.25
        # l_returnflag == 'R' ~ 0.25 of line items
        # Average line items per order ~ 4
        # Expected qualifying tuples per order ~ 0.25 * 0.25 * 4 = 0.25
        expected_selectivity_per_order = 0.25
        n_orders = int(np.round(target_stream_tuples / expected_selectivity_per_order))
    else:
        if n_orders is None:
            n_orders = 30000

    # 1. Generate CUSTOMERS with nation keys and account balances
    cust_nations = rng.integers(0, len(GLOBAL_NATIONS), size=n_customers)
    cust_acctbal = rng.uniform(-999.0, 9999.0, size=n_customers)

    # 2. Generate ORDERS
    # Map orders to customers using Zipf skew to model power-law commercial order frequencies
    order_custkeys = sample_zipf(n_orders, n_customers, zipf_skew, rng)
    # Order dates across 365 days; Q4 is days [275, 365)
    order_dates = rng.integers(0, 365, size=n_orders)

    # Lineitems per order: Poisson(4) clamped [1, 7]
    n_items_per_order = np.clip(rng.poisson(4, size=n_orders), 1, 7)

    stream_tuples = []
    ground_truth_world_revenues = np.zeros((n_groups, n_mc_worlds), dtype=np.float64)
    ground_truth_expected_revenues = np.zeros(n_groups, dtype=np.float64)
    ground_truth_proxy_scores = np.zeros(n_groups, dtype=np.float64)
    v_max = 50000.0  # Max revenue domain bound for normalization

    qualifying_orders = np.flatnonzero(order_dates >= 275)

    for o_idx in qualifying_orders:
        o_date = int(order_dates[o_idx])
        c_id = int(order_custkeys[o_idx])
        c_nat = int(cust_nations[c_id])
        nat_name = GLOBAL_NATIONS[c_nat]

        n_lines = int(n_items_per_order[o_idx])
        line_prices = rng.uniform(100.0, 5000.0, size=n_lines)
        line_discounts = rng.uniform(0.00, 0.10, size=n_lines)
        
        # In TPC-H, l_returnflag is 'R', 'A', or 'N'; 'R' corresponds to return event
        # Probability of return request ~ 0.25
        is_return_req = rng.uniform(0.0, 1.0, size=n_lines) < 0.25
        
        # Return refund likelihood p_return drawn from Beta(2, 3) shifted to [0.10, 0.95]
        return_probs = 0.10 + rng.beta(2.0, 3.0, size=n_lines) * (0.95 - 0.10)

        for l_idx in range(n_lines):
            if not is_return_req[l_idx]:
                continue

            p_ret = float(return_probs[l_idx])
            price = float(line_prices[l_idx])
            disc = float(line_discounts[l_idx])
            filtered_rev = float(price * (1.0 - disc))
            expected_rev = float(p_ret * filtered_rev)
            proxy_score = float(p_ret * price)

            ground_truth_expected_revenues[c_id] += expected_rev
            ground_truth_proxy_scores[c_id] += proxy_score

            # MCDB realizations across W worlds
            if include_mc_bits:
                return_world_mask = (rng.uniform(0.0, 1.0, size=n_mc_worlds) < p_ret)
                ground_truth_world_revenues[c_id, return_world_mask] += filtered_rev

                pad_len = (64 - (n_mc_worlds % 64)) % 64
                padded = np.pad(return_world_mask, (0, pad_len), 'constant') if pad_len > 0 else return_world_mask
                bytes_data = np.packbits(padded, bitorder='little').tobytes()
                mc_words = np.frombuffer(bytes_data, dtype=np.uint64).tolist()
                packed_words = bytes_data.hex()
                active_worlds = int(np.sum(return_world_mask))
            else:
                mc_words = None
                packed_words = None
                active_worlds = 0

            norm_score = min(1.0, max(0.0, proxy_score / v_max))

            stream_tuples.append({
                "order_id": int(o_idx),
                "line_id": int(l_idx),
                "cust_id": c_id,
                "group_key": c_id,
                "group_label": f"Customer_{c_id:04d}",
                "nation_name": nat_name,
                "acct_bal": round(float(cust_acctbal[c_id]), 2),
                "order_date": o_date,
                "return_flag": "R",
                "prob": round(p_ret, 6),
                "filtered_revenue": round(filtered_rev, 4),
                "expected_revenue": round(expected_rev, 4),
                "normalized_score": round(norm_score, 8),
                "proxy_score": round(proxy_score, 4),
                "unit_score": round(p_ret, 6),
                "exact_score": round(expected_rev, 4),
                "mc_bits": mc_words,
                "mcdb_bits_hex": packed_words,
                "mcdb_active_worlds": active_worlds,
            })

    # Sort stream by chosen score space descending
    if stream_sort == "exact":
        sorted_stream = sorted(stream_tuples, key=lambda t: t["exact_score"], reverse=True)
    elif stream_sort == "unit":
        sorted_stream = sorted(stream_tuples, key=lambda t: t["unit_score"], reverse=True)
    else:
        # Default to proxy score space
        sorted_stream = sorted(stream_tuples, key=lambda t: t["proxy_score"], reverse=True)

    sorted_gt_groups = np.argsort(-ground_truth_expected_revenues).tolist()
    sorted_proxy_groups = np.argsort(-ground_truth_proxy_scores).tolist()

    # MCDB world plurality winners:
    world_winning_groups = np.argmax(ground_truth_world_revenues, axis=0)
    mcdb_group_win_counts = np.bincount(world_winning_groups, minlength=n_groups)
    mcdb_sorted_groups = np.argsort(-mcdb_group_win_counts).tolist()

    return {
        "metadata": {
            "query": "TPC-H Q10",
            "n_customers": n_customers,
            "n_orders": n_orders,
            "n_groups": n_groups,
            "qualifying_tuples": len(stream_tuples),
            "zipf_skew": zipf_skew,
            "n_mc_worlds": n_mc_worlds,
            "v_max": v_max,
            "seed": seed,
            "stream_sort": stream_sort,
            "k_standard": 20,
        },
        "ground_truth": {
            "expected_revenue_by_group": [round(float(v), 4) for v in ground_truth_expected_revenues],
            "proxy_score_by_group": [round(float(v), 4) for v in ground_truth_proxy_scores],
            "top_groups_expected": sorted_gt_groups,
            "top_groups_proxy": sorted_proxy_groups,
            "mcdb_win_counts": [int(c) for c in mcdb_group_win_counts],
            "top_groups_mcdb": mcdb_sorted_groups,
        },
        "stream": sorted_stream,
    }


def _generate_and_save_q10_dataset(task_args: Tuple[float, int, int, int, str, int, bool, str]) -> Tuple[str, float, bool]:
    import time
    skew, seed, target_stream_tuples, n_customers, stream_sort, n_mc_worlds, include_mc_bits, out_dir = task_args
    out_file = os.path.join(out_dir, f"tpch_q10_z{skew}_seed{seed}_{stream_sort}.json")
    if os.path.exists(out_file):
        return out_file, 0.0, True

    t0 = time.time()
    data = generate_tpch_q10_dataset(
        n_customers=n_customers,
        target_stream_tuples=target_stream_tuples,
        zipf_skew=skew,
        n_mc_worlds=n_mc_worlds,
        seed=seed,
        stream_sort=stream_sort,
        include_mc_bits=include_mc_bits,
    )
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    elapsed = time.time() - t0
    return out_file, elapsed, False


def main():
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="Generate synthetic TPC-H Q10 probabilistic benchmark data.")
    parser.add_argument("--out_dir", "--output-dir", "--out-dir", "--output_dir", type=str, default="experiment_datasets/tpch_q10_synthetic", dest="out_dir")
    parser.add_argument("--out_file", "--out-file", type=str, default=None, dest="out_file")
    parser.add_argument("--n_customers", "--n-customers", type=int, default=1000, dest="n_customers")
    parser.add_argument("--target_stream_tuples", "--target-stream-tuples", type=int, default=None, dest="target_stream_tuples")
    parser.add_argument("--zipf", "--zipf-skew", "--zipf_skew", type=float, default=1.0, dest="zipf")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--stream_sort", "--score-space", "--stream-sort", "--score_space", type=str, default="proxy", choices=["proxy", "exact", "unit"], dest="stream_sort")
    parser.add_argument("--n_mc_worlds", "--n-mc-worlds", type=int, default=1000, dest="n_mc_worlds")
    parser.add_argument("--num_seeds", "--num-seeds", type=int, default=None, dest="num_seeds")
    parser.add_argument("--start_seed", "--start-seed", type=int, default=1000, dest="start_seed")
    parser.add_argument("--skews", type=str, default=None, help="Comma-separated list of skews e.g. '1.0,2.0'")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--no_mc_bits", "--no-mc-bits", action="store_true", default=False, dest="no_mc_bits")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.num_seeds is not None:
        skews = [float(z.strip()) for z in args.skews.split(",")] if args.skews else [args.zipf]
        tasks = []
        for z in skews:
            for s in range(args.start_seed, args.start_seed + args.num_seeds):
                tasks.append((z, s, args.target_stream_tuples, args.n_customers, args.stream_sort, args.n_mc_worlds, not args.no_mc_bits, args.out_dir))

        total_tasks = len(tasks)
        print(f"==========================================================================")
        print(f"  PARALLEL TPC-H Q10 DATASET GENERATION ({args.workers} Workers)")
        print(f"  Total Tasks: {total_tasks} ({len(skews)} Skews x {args.num_seeds} Seeds) | Target Tuples: {args.target_stream_tuples}")
        print(f"==========================================================================")

        t_start = time.time()
        completed = 0
        if args.workers <= 1:
            for task in tasks:
                out_path, elapsed, cached = _generate_and_save_q10_dataset(task)
                completed += 1
                status = "cached" if cached else f"{elapsed:.2f}s"
                print(f"  [{completed:>3}/{total_tasks}] Seed {task[1]} (z={task[0]}) -> {status}")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(_generate_and_save_q10_dataset, task): task for task in tasks}
                for fut in as_completed(futures):
                    out_path, elapsed, cached = fut.result()
                    completed += 1
                    status = "cached" if cached else f"{elapsed:.2f}s"
                    task = futures[fut]
                    if completed % 5 == 0 or completed == total_tasks or completed == 1:
                        total_elapsed = time.time() - t_start
                        rate = completed / max(0.001, total_elapsed)
                        print(f"  [{completed:>3}/{total_tasks}] Seed {task[1]} (z={task[0]}) -> {status} (rate: {rate:.1f} ds/s)")

        total_elapsed = time.time() - t_start
        print(f"✓ Completed {total_tasks} datasets in {total_elapsed:.2f}s ({total_tasks/max(0.001, total_elapsed):.1f} ds/s) into {args.out_dir}")
        return

    data = generate_tpch_q10_dataset(
        n_customers=args.n_customers,
        target_stream_tuples=args.target_stream_tuples,
        zipf_skew=args.zipf,
        n_mc_worlds=args.n_mc_worlds,
        seed=args.seed,
        stream_sort=args.stream_sort,
        include_mc_bits=not args.no_mc_bits,
    )

    if args.out_file:
        out_file = args.out_file
    else:
        suffix = f"_{args.stream_sort}" if args.stream_sort != "proxy" else ""
        out_file = os.path.join(args.out_dir, f"tpch_q10_z{int(args.zipf*10):02d}_seed{args.seed}{suffix}.json")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"[TPC-H Q10] Generated {data['metadata']['qualifying_tuples']} qualifying return tuples across {data['metadata']['n_groups']} customers.")
    print(f"Top 5 customers by expected lost revenue: {data['ground_truth']['top_groups_expected'][:5]}")
    print(f"Saved to: {out_file}")


if __name__ == "__main__":
    main()
