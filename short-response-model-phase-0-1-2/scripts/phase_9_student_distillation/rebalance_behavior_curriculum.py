from __future__ import annotations

"""Build a deterministic Phase 9 behavior-mixture training view.

This script never edits the source curriculum. It creates a training-only sampled
view with explicit class proportions so the grounding decision boundary is not
swamped by repeated supported-fact rows. Sampling is deterministic and may use
replacement; the manifest records every source ID and realized proportion.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_jsonl, sha256, write_json, write_jsonl
from scripts.phase_9_student_distillation.phase9_metrics import decision_class

DEFAULT_TARGETS = {
    "supported": 0.45,
    "unsupported": 0.25,
    "conflict": 0.10,
    "injection": 0.10,
    "procedure": 0.05,
    "social": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebalance a Phase 9 curriculum by behavior class.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--target-size", type=int, default=0, help="0 keeps the source row count.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ratios-json",
        default=json.dumps(DEFAULT_TARGETS, sort_keys=True),
        help="JSON object of behavior-class proportions.",
    )
    return parser.parse_args()


def behavior_bucket(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    category = str(row.get("category") or "").casefold()
    variant = str(metadata.get("curriculum_variant") or "").casefold()
    mode = str(row.get("mode") or "").casefold()
    decision = str(row.get("decision") or "").casefold()
    if "injection" in category or "injection" in variant or row.get("adversarial_evidence"):
        return "injection"
    if mode == "procedure" or decision == "procedure" or "procedure" in category:
        return "procedure"
    klass = decision_class(row)
    if klass == "social":
        return "social"
    if klass == "conflict":
        return "conflict"
    if klass == "unsupported":
        return "unsupported"
    return "supported"


def normalize_ratios(raw: dict[str, Any], available: set[str]) -> dict[str, float]:
    values = {name: max(0.0, float(raw.get(name, 0.0))) for name in DEFAULT_TARGETS}
    active = {name: value for name, value in values.items() if name in available and value > 0.0}
    if not active:
        raise ValueError("None of the requested behavior classes are present in the input")
    total = sum(active.values())
    return {name: value / total for name, value in active.items()}


def integer_quotas(total: int, ratios: dict[str, float]) -> dict[str, int]:
    exact = {name: total * value for name, value in ratios.items()}
    quotas = {name: int(value) for name, value in exact.items()}
    remainder = total - sum(quotas.values())
    order = sorted(ratios, key=lambda name: (exact[name] - quotas[name], ratios[name], name), reverse=True)
    for name in order[:remainder]:
        quotas[name] += 1
    return quotas


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    if not rows:
        raise ValueError("Input curriculum is empty")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[behavior_bucket(row)].append(row)
    ratios = normalize_ratios(json.loads(args.ratios_json), set(grouped))
    target_size = args.target_size if args.target_size > 0 else len(rows)
    if target_size < len(ratios):
        raise ValueError("target-size must be at least the number of available behavior classes")
    quotas = integer_quotas(target_size, ratios)
    rng = random.Random(args.seed)
    sampled: list[dict[str, Any]] = []
    source_use_count: Counter[str] = Counter()
    for bucket in sorted(quotas):
        pool = list(grouped[bucket])
        rng.shuffle(pool)
        for index in range(quotas[bucket]):
            source = dict(pool[index % len(pool)])
            source_id = str(source.get("id") or f"{bucket}_{index}")
            source_use_count[source_id] += 1
            copy = dict(source)
            copy["id"] = f"{source_id}__balanced_{bucket}_{source_use_count[source_id]:04d}"
            copy["source_record_id"] = str(source.get("source_record_id") or source_id)
            copy["metadata"] = {
                **(source.get("metadata") or {}),
                "behavior_bucket": bucket,
                "balance_source_id": source_id,
                "balance_seed": args.seed,
            }
            sampled.append(copy)
    rng.shuffle(sampled)
    write_jsonl(args.output, sampled)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    realized = Counter(behavior_bucket(row) for row in sampled)
    manifest = {
        "phase": "9_student_distillation",
        "schema_version": "1.3",
        "pipeline_revision": "phase9_grounded_recovery_v1_7_2",
        "source": str(args.input),
        "source_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "seed": args.seed,
        "source_row_count": len(rows),
        "output_row_count": len(sampled),
        "source_class_counts": dict(Counter(behavior_bucket(row) for row in rows)),
        "requested_ratios": json.loads(args.ratios_json),
        "effective_ratios": ratios,
        "realized_counts": dict(realized),
        "realized_ratios": {name: count / len(sampled) for name, count in realized.items()},
        "sampling_policy": "deterministic stratified sampling with replacement; source curriculum remains unchanged",
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"rows": len(sampled), "manifest": str(manifest_path), "counts": dict(realized)}, indent=2))


if __name__ == "__main__":
    main()
