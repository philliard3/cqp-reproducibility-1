"""
schema_codec.py

Compact serialization and schema management for experimental runs:
- Interns strings (query names, strategies, candidate group keys) to stable integer IDs.
- Companion lookup dictionary management (schema_lookup.json).
- Supports streaming compressed JSONLines (.jsonl.gz) to satisfy storage constraints.
- Re-hydrates compact records back to human-readable structures on demand.
- Supports candidate score bounds [lo, hi] and MCDB winning world sums.
"""

import gzip
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple


class SchemaCodec:
    """
    Maintains bidirectional mappings between human-readable string tokens
    and stable integer IDs.
    """
    def __init__(self, lookup_path: Optional[Union[str, Path]] = None):
        self.lookup_path = Path(lookup_path) if lookup_path else None
        self.str_to_id: Dict[str, Dict[str, int]] = {
            "queries": {},
            "strategies": {},
            "entities": {},
        }
        self.id_to_str: Dict[str, Dict[int, str]] = {
            "queries": {},
            "strategies": {},
            "entities": {},
        }
        if self.lookup_path and self.lookup_path.exists():
            self.load_lookup(self.lookup_path)
        else:
            self._seed_default_vocabularies()

    def _seed_default_vocabularies(self):
        """Seed known canonical identifiers."""
        canonical_queries = ["q1a", "q1b", "q2a", "q2b", "q3", "q4"]
        for q in canonical_queries:
            self.get_or_create_id("queries", q)

        canonical_strategies = [
            "Naive_Conformal_End",
            "MCDB_Count_1000",
            "MCDB_Sum_1000",
            "Conformal_InOperator_Stop",
            "Conformal_InOperator_Stop (Indep)",
            "Conformal_InOperator_Stop (MutEx)",
            "QN1_External_Pull",
            "QN1_External_Pull (Indep)",
            "QN1_External_Pull (MutEx)",
        ]
        for s in canonical_strategies:
            self.get_or_create_id("strategies", s)

    def get_or_create_id(self, domain: str, value: str) -> int:
        if domain not in self.str_to_id:
            self.str_to_id[domain] = {}
            self.id_to_str[domain] = {}
        if value in self.str_to_id[domain]:
            return self.str_to_id[domain][value]
        new_id = len(self.str_to_id[domain])
        self.str_to_id[domain][value] = new_id
        self.id_to_str[domain][new_id] = value
        return new_id

    def get_str(self, domain: str, token_id: int) -> str:
        return self.id_to_str.get(domain, {}).get(token_id, f"unknown_{token_id}")

    def save_lookup(self, path: Optional[Union[str, Path]] = None):
        p = Path(path) if path else self.lookup_path
        if not p:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            domain: {k: v for k, v in table.items()}
            for domain, table in self.str_to_id.items()
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_lookup(self, path: Union[str, Path]):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for domain, table in data.items():
            self.str_to_id[domain] = {k: int(v) for k, v in table.items()}
            self.id_to_str[domain] = {int(v): k for k, v in table.items()}

    def encode_record(self, raw_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress record fields using integer mappings.
        """
        cfg = raw_record.get("config", {})
        metrics = raw_record.get("metrics", {})

        q_id = self.get_or_create_id("queries", str(cfg.get("query", "")))
        s_id = self.get_or_create_id("strategies", str(cfg.get("strategy", "")))

        # Intern group names in sets/bounds
        emitted_set = [
            self.get_or_create_id("entities", str(item))
            for item in metrics.get("emitted_set", [])
        ]
        gt_top_k = [
            self.get_or_create_id("entities", str(item))
            for item in metrics.get("ground_truth_top_k", [])
        ]

        # Score bounds: {group: [lo, hi]} -> {group_id: [round(lo, 5), round(hi, 5)]}
        compact_bounds = {}
        for grp, (lo, hi) in metrics.get("score_bounds", {}).items():
            gid = self.get_or_create_id("entities", str(grp))
            compact_bounds[gid] = [round(float(lo), 5), round(float(hi), 5)]

        # MCDB world wins: {group: wins} -> {group_id: wins}
        compact_mcdb = {}
        for grp, wins in metrics.get("mcdb_world_wins", {}).items():
            gid = self.get_or_create_id("entities", str(grp))
            compact_mcdb[gid] = int(wins)

        compact = {
            "c": {
                "q": q_id,
                "s": s_id,
                "m": cfg.get("M", 10),
                "z": cfg.get("z", 1.0),
                "c": cfg.get("c", 1),
                "k": cfg.get("k_query", 1),
                "a": cfg.get("alpha", 0.05),
                "sd": cfg.get("seed", 0),
                "sc": cfg.get("dataset_scale", 1.0),
            },
            "m": {
                "in": metrics.get("tuples_ingested", 0),
                "tot": metrics.get("tuples_total", 0),
                "sav": round(float(metrics.get("stream_savings_pct", 0.0)), 3),
                "eng_ms": round(float(metrics.get("engine_time_ms", 0.0)), 4),
                "e2e_ms": round(float(metrics.get("e2e_time_ms", 0.0)), 4),
                "cov": bool(metrics.get("covered", False)),
                "rec": round(float(metrics.get("recall_at_set", 0.0)), 4),
                "tau": round(float(metrics.get("tau_threshold", 0.0)), 5),
                "sz": metrics.get("emitted_set_size", len(emitted_set)),
                "rel_sz": round(float(metrics.get("relative_set_size", 1.0)), 3),
                "emit": emitted_set,
                "gt": gt_top_k,
            }
        }
        if compact_bounds:
            compact["m"]["bnd"] = compact_bounds
        if compact_mcdb:
            compact["m"]["mcdb"] = compact_mcdb
        return compact

    def decode_record(self, compact_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Re-hydrate compact record into standard human-readable JSON dictionary.
        """
        c = compact_record.get("c", {})
        m = compact_record.get("m", {})

        q_name = self.get_str("queries", c.get("q", 0))
        s_name = self.get_str("strategies", c.get("s", 0))

        emitted_set = [self.get_str("entities", gid) for gid in m.get("emit", [])]
        gt_top_k = [self.get_str("entities", gid) for gid in m.get("gt", [])]

        score_bounds = {}
        for gid_str, (lo, hi) in m.get("bnd", {}).items():
            gid = int(gid_str)
            score_bounds[self.get_str("entities", gid)] = [lo, hi]

        mcdb_world_wins = {}
        for gid_str, wins in m.get("mcdb", {}).items():
            gid = int(gid_str)
            mcdb_world_wins[self.get_str("entities", gid)] = wins

        return {
            "config": {
                "query": q_name,
                "strategy": s_name,
                "M": c.get("m"),
                "z": c.get("z"),
                "c": c.get("c"),
                "k_query": c.get("k"),
                "alpha": c.get("a"),
                "seed": c.get("sd"),
                "dataset_scale": c.get("sc", 1.0),
            },
            "metrics": {
                "tuples_ingested": m.get("in"),
                "tuples_total": m.get("tot"),
                "stream_savings_pct": m.get("sav"),
                "engine_time_ms": m.get("eng_ms"),
                "e2e_time_ms": m.get("e2e_ms"),
                "emitted_set": emitted_set,
                "emitted_set_size": m.get("sz", len(emitted_set)),
                "ground_truth_top_k": gt_top_k,
                "covered": m.get("cov"),
                "recall_at_set": m.get("rec"),
                "tau_threshold": m.get("tau"),
                "relative_set_size": m.get("rel_sz"),
                "score_bounds": score_bounds,
                "mcdb_world_wins": mcdb_world_wins,
            }
        }


def write_jsonl_gz(records: List[Dict[str, Any]], output_path: Union[str, Path], codec: Optional[SchemaCodec] = None):
    """
    Write records to a gzip-compressed jsonlines file, optionally encoding with SchemaCodec.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as f:
        for rec in records:
            to_write = codec.encode_record(rec) if codec else rec
            f.write(json.dumps(to_write) + "\n")


def read_jsonl_gz(input_path: Union[str, Path], codec: Optional[SchemaCodec] = None) -> List[Dict[str, Any]]:
    """
    Read records from a gzip-compressed or plain jsonlines file, optionally decoding with SchemaCodec.
    """
    p = Path(input_path)
    records = []
    open_fn = gzip.open if p.suffix == ".gz" else open
    with open_fn(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if codec and "c" in rec and "m" in rec:
                records.append(codec.decode_record(rec))
            else:
                records.append(rec)
    return records
