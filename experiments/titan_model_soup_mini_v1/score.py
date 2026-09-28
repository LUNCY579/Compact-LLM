"""Paired, provenance-bound reports and explicit parent_id leakage checks.

Consumes externally scored predictions; does not certify the external collector/scorer.
Raw exact McNemar p-values are descriptive, not multiplicity-adjusted selection gates.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import sys
from soup import EXPERIMENT, read_json, require, sha256

MATCHED = ("experiment", "suite_id", "split", "dataset_sha256", "scorer_sha256",
           "runtime_sha256", "decoding_sha256", "tokenizer_sha256")
HASHES = tuple(k for k in MATCHED if k.endswith("sha256")) + ("checkpoint_sha256",)


def load_run(path: Path):
    obj = read_json(path)
    meta = obj.get("metadata", {})
    require(meta.get("experiment") == EXPERIMENT, "wrong experiment")
    require(meta.get("split") in ("CAL", "CHECK", "REPORT_ONLY"), "unknown instrument role")
    require(isinstance(meta.get("suite_id"), str) and bool(meta["suite_id"]), "missing suite ID")
    for key in HASHES:
        require(isinstance(meta.get(key), str) and bool(re.fullmatch(r"[0-9a-f]{64}", meta[key])), f"unbound {key}")
    rows = obj.get("rows")
    require(isinstance(rows, list) and bool(rows), "empty or absent rows")
    indexed = {}
    for r in rows:
        require(isinstance(r, dict), "bad row")
        for key in ("id", "parent_id", "domain"):
            require(isinstance(r.get(key), str) and bool(r[key].strip()), f"missing {key}")
        require(r["id"] not in indexed, "duplicate row ID")
        require(type(r.get("correct")) is bool, "correct must be a boolean, not a string/number")
        indexed[r["id"]] = r
    return meta, indexed


def exact_mcnemar(gains: int, losses: int) -> float:
    n = gains + losses
    if not n:
        return 1.0
    # Compute in log space, avoiding overflow at large n. Stable sum of positive terms.
    k = min(gains, losses)
    logs = [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) - n * math.log(2)
            for j in range(k + 1)]
    largest = max(logs)
    return min(1.0, 2 * math.exp(largest) * math.fsum(math.exp(x - largest) for x in logs))


def summary(pairs):
    n = len(pairs)
    source = sum(a for a, b in pairs)
    retained = sum(a and b for a, b in pairs)
    gains = sum(not a and b for a, b in pairs)
    losses = sum(a and not b for a, b in pairs)
    return {"n": n, "reference_correct": source, "candidate_correct": retained + gains,
            "source_correct_retained": retained, "source_correct_lost": losses,
            "newly_correct": gains, "net_correct": gains - losses,
            "retention_fraction": retained / source if source else None,
            "raw_exact_mcnemar_two_sided_p": exact_mcnemar(gains, losses)}


def paired_report(reference: Path, candidate: Path):
    ma, a = load_run(reference)
    mb, b = load_run(candidate)
    require(all(ma[k] == mb[k] for k in MATCHED), "unmatched dataset/scorer/runtime/decoding/tokenizer/role")
    require(set(a) == set(b), "row sets differ; no intersections, truncation, or dropped failures")
    groups = defaultdict(list)
    parent_rows = defaultdict(list)
    for key in sorted(a):
        require((a[key]["parent_id"], a[key]["domain"]) == (b[key]["parent_id"], b[key]["domain"]), "row identity mismatch")
        pair = (a[key]["correct"], b[key]["correct"])
        groups[a[key]["domain"]].append(pair)
        parent_rows[(a[key]["domain"], a[key]["parent_id"])].append(pair)
    # Repeated views are dependent. Parent-all-correct is an additional descriptive endpoint,
    # not a relabelling of a preregistered canonical-view score.
    parent_groups = defaultdict(list)
    for (domain, parent), pairs in parent_rows.items():
        parent_groups[domain].append((all(x for x, y in pairs), all(y for x, y in pairs)))
    independent = len(parent_rows) == len(a)
    result = {"experiment": EXPERIMENT, "metadata": ma,
              "reference_checkpoint_sha256": ma["checkpoint_sha256"],
              "candidate_checkpoint_sha256": mb["checkpoint_sha256"],
              "reference_run_sha256": sha256(reference), "candidate_run_sha256": sha256(candidate),
              "domains": {k: summary(v) for k, v in sorted(groups.items())},
              "parent_all_views_correct_descriptive": {k: summary(v) for k, v in sorted(parent_groups.items())},
              "row_level_p_values_assume_independent_parents": independent,
              "quality_improvement_established": False, "release_permitted": False,
              "status": "DESCRIPTIVE_PAIRED_REPORT_NOT_SELECTION_OR_CONFIRMATION",
              "external_scoring_verified": False}
    if not independent:
        for stats in result["domains"].values():
            stats["raw_exact_mcnemar_two_sided_p"] = None
    return result


def parent_ids(path: Path) -> set[str]:
    ids, parents = set(), set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            require(isinstance(r, dict), "invalid dataset row")
            require(isinstance(r.get("parent_id"), str) and bool(r["parent_id"].strip()), "explicit parent_id required; never infer from id")
            require(isinstance(r.get("id"), str) and bool(r["id"].strip()), "explicit row id required")
            require(r["id"] not in ids, "duplicate dataset row id")
            ids.add(r["id"])
            parents.add(r["parent_id"])
    require(bool(parents), "empty dataset")
    return parents


def audit_splits(cal: Path, check: Path, train: list[Path]):
    require(bool(train), "supply training-parent ledgers for every fine-tuned ancestor")
    cp, hp = parent_ids(cal), parent_ids(check)
    tp = set().union(*(parent_ids(x) for x in train))
    overlaps = {"cal_check": len(cp & hp), "train_cal": len(tp & cp), "train_check": len(tp & hp)}
    require(not any(overlaps.values()), "parent contamination: " + json.dumps(overlaps))
    return {"status": "SUPPLIED_LEDGER_DISJOINTNESS_ONLY", "overlaps": overlaps,
            "files": {str(p): sha256(p) for p in [cal, check, *train]},
            "ancestor_ledger_completeness_verified": False, "native_execution_authorized": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    q = s.add_parser("compare")
    q.add_argument("--reference", required=True, type=Path)
    q.add_argument("--candidate", required=True, type=Path)
    q = s.add_parser("audit-splits")
    q.add_argument("--cal", required=True, type=Path)
    q.add_argument("--check", required=True, type=Path)
    q.add_argument("--train", required=True, nargs="+", type=Path)
    a = p.parse_args()
    try:
        result = paired_report(a.reference, a.candidate) if a.command == "compare" else audit_splits(a.cal, a.check, a.train)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "REFUSED", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
