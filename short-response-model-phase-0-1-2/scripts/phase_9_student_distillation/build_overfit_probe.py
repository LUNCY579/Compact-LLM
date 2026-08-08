from __future__ import annotations

"""Build a tiny deliberate Phase 9 overfit probe.

The benchmark contains unique rows; the training view repeats those exact rows
with unique IDs. This is a wiring test, not held-out quality evidence.
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
from scripts.phase_9_student_distillation.phase9_common import build_grounded_messages
from scripts.phase_9_student_distillation.rebalance_behavior_curriculum import behavior_bucket

DEFAULT_QUOTAS = {
    "supported": 6,
    "unsupported": 4,
    "conflict": 2,
    "injection": 2,
    "procedure": 2,
    "social": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced tiny Phase 9 overfit probe.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--benchmark-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--rows", type=int, default=18)
    parser.add_argument("--repeat", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _answer(row: dict[str, Any]) -> str:
    if row.get("reference"):
        return str(row["reference"])
    if row.get("answer"):
        return str(row["answer"])
    messages = row.get("messages") or []
    return str(messages[-1].get("content") or "") if messages else ""


def materialize_injection(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    if behavior_bucket(result) != "injection" or not result.get("adversarial_evidence"):
        return result
    evidence = [dict(item) for item in result.get("adversarial_evidence") or []]
    answer = _answer(result)
    messages = build_grounded_messages(
        question=str(result.get("question") or ""),
        evidence=evidence,
        procedures=[dict(item) for item in result.get("procedures") or []],
        mode=str(result.get("mode") or "grounded_factual"),
        system_prompt=result.get("system_prompt"),
    )
    messages.append({"role": "assistant", "content": answer})
    result["evidence"] = evidence
    result["messages"] = messages
    result["metadata"] = {**(result.get("metadata") or {}), "overfit_uses_adversarial_evidence": True}
    return result


def choose_rows(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for raw in rows:
        source_key = str(raw.get("source_record_id") or raw.get("id") or "")
        # Prefer unique underlying examples so the benchmark is not made of mere repeats.
        key = f"{behavior_bucket(raw)}:{source_key}"
        if key in seen:
            continue
        seen.add(key)
        grouped[behavior_bucket(raw)].append(materialize_injection(raw))
    missing_buckets = [name for name in DEFAULT_QUOTAS if not grouped.get(name)]
    if missing_buckets:
        raise ValueError(
            "The deliberate overfit probe requires all six behavior buckets; "
            f"missing: {missing_buckets}"
        )
    rng = random.Random(seed)
    for pool in grouped.values():
        rng.shuffle(pool)
    quotas = dict(DEFAULT_QUOTAS)
    scale = count / sum(quotas.values())
    requested = {name: max(0, round(value * scale)) for name, value in quotas.items()}
    while sum(requested.values()) < count:
        requested[max(requested, key=lambda name: DEFAULT_QUOTAS[name] - requested[name])] += 1
    while sum(requested.values()) > count:
        candidates = [name for name, value in requested.items() if value > 0]
        requested[max(candidates, key=lambda name: requested[name] - DEFAULT_QUOTAS[name])] -= 1
    chosen: list[dict[str, Any]] = []
    for bucket in DEFAULT_QUOTAS:
        take = min(requested.get(bucket, 0), len(grouped.get(bucket, [])))
        chosen.extend(grouped.get(bucket, [])[:take])
    if len(chosen) < count:
        leftovers = [row for bucket in DEFAULT_QUOTAS for row in grouped.get(bucket, [])[requested.get(bucket, 0) :]]
        rng.shuffle(leftovers)
        chosen.extend(leftovers[: count - len(chosen)])
    if len(chosen) < min(12, count):
        raise ValueError(f"Only {len(chosen)} unique rows available; at least {min(12, count)} are required")
    rng.shuffle(chosen)
    return chosen[:count]


def main() -> None:
    args = parse_args()
    if not 12 <= args.rows <= 24:
        raise ValueError("--rows must be between 12 and 24 for the deliberate overfit probe")
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    rows = read_jsonl(args.input)
    benchmark = choose_rows(rows, args.rows, args.seed)
    training: list[dict[str, Any]] = []
    for repeat_index in range(args.repeat):
        for row in benchmark:
            copy = dict(row)
            original_id = str(row.get("id"))
            copy["id"] = f"{original_id}__overfit_repeat_{repeat_index:03d}"
            copy["source_record_id"] = str(row.get("source_record_id") or original_id)
            copy["metadata"] = {
                **(row.get("metadata") or {}),
                "overfit_probe": True,
                "overfit_repeat_index": repeat_index,
            }
            training.append(copy)
    random.Random(args.seed).shuffle(training)
    write_jsonl(args.benchmark_output, benchmark)
    write_jsonl(args.train_output, training)
    manifest_path = args.manifest or args.train_output.with_suffix(".manifest.json")
    manifest = {
        "phase": "9_student_distillation",
        "schema_version": "1.4",
        "pipeline_revision": "phase9_large_corrective_curriculum_v1_7_5",
        "purpose": "tiny_deliberate_overfit_wiring_test_not_held_out_quality_evidence",
        "source": str(args.input),
        "source_sha256": sha256(args.input),
        "benchmark": str(args.benchmark_output),
        "benchmark_sha256": sha256(args.benchmark_output),
        "train": str(args.train_output),
        "train_sha256": sha256(args.train_output),
        "unique_rows": len(benchmark),
        "repeat": args.repeat,
        "training_rows": len(training),
        "behavior_counts": dict(Counter(behavior_bucket(row) for row in benchmark)),
        "seed": args.seed,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
