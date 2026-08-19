from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    BASE_SEED,
    DATASET_NAME,
    DATASET_VERSION,
    SCHEMA_VERSION,
    full_response,
    infer_split,
    json_dumps,
    normalize_problem,
    repair_adjacent_problem_ids,
    sha256_text,
    stable_key,
    write_jsonl,
)
from generators_math import MATH_GENERATORS  # noqa: E402
from generators_reasoning import REASONING_GENERATORS  # noqa: E402

ALL_GENERATORS = {**MATH_GENERATORS, **REASONING_GENERATORS}
DEFAULT_PER_CATEGORY = 1600


def build_full_record(
    *,
    problem_id: str,
    split: str,
    domain: str,
    category: str,
    generated: Any,
    generation_seed: int,
) -> dict[str, Any]:
    response = full_response(generated.steps)
    source_hash = sha256_text(generated.problem + "\n" + response)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "variant": "full_context",
        "problem_id": problem_id,
        "split": split,
        "domain": domain,
        "category": category,
        "difficulty": generated.difficulty,
        "problem": generated.problem,
        "answer": generated.answer,
        "answer_type": generated.answer_type,
        "full_response": response,
        "reasoning_steps": [step.target_step for step in generated.steps],
        "compact_state_trace": [step.state_before for step in generated.steps]
        + [generated.steps[-1].state_after],
        "step_count": len(generated.steps),
        "messages": [
            {"role": "user", "content": generated.problem},
            {"role": "assistant", "content": response},
        ],
        "generator": generated.generator,
        "tags": list(generated.tags),
        "generation_seed": generation_seed,
        "source_pair_sha256": source_hash,
        "provenance": {
            "type": "procedurally_generated_synthetic",
            "external_dataset_text_copied": False,
            "construction": "A canonical complete trajectory was generated first; compact-state transitions were derived from that same trajectory.",
        },
    }


def build_transition_records(full: dict[str, Any], generated: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    total_steps = len(generated.steps)
    for i, step in enumerate(generated.steps):
        transition_id = f"{full['problem_id']}::step_{i + 1:02d}"
        user_content = (
            "Original problem:\n"
            f"{full['problem']}\n\n"
            "Current compact state:\n"
            f"{step.state_before}\n\n"
            "Give exactly the next correct reasoning step. Do not replay the omitted reasoning history."
        )
        stateful_target = json_dumps({
            "next_reasoning_step": step.target_step,
            "updated_compact_state": step.state_after,
            "done": step.done,
            "final_answer": full["answer"] if step.done else None,
        })
        result.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_name": DATASET_NAME,
                "dataset_version": DATASET_VERSION,
                "variant": "split_transition",
                "transition_id": transition_id,
                "problem_id": full["problem_id"],
                "split": full["split"],
                "domain": full["domain"],
                "category": full["category"],
                "difficulty": full["difficulty"],
                "step_index": i + 1,
                "total_steps": total_steps,
                "problem": full["problem"],
                "current_state": step.state_before,
                "target_step": step.target_step,
                "next_state": step.state_after,
                "done": step.done,
                "final_answer": full["answer"] if step.done else None,
                "answer_type": full["answer_type"],
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": step.target_step},
                ],
                "stateful_rollout_messages": [
                    {"role": "user", "content": user_content + "\nReturn a JSON object with next_reasoning_step, updated_compact_state, done, and final_answer."},
                    {"role": "assistant", "content": stateful_target},
                ],
                "stateful_rollout_target": stateful_target,
                "generator": full["generator"],
                "tags": full["tags"],
                "source_pair_sha256": full["source_pair_sha256"],
                "sequence_free_training": True,
                "state_construction": "derived_from_complete_canonical_trajectory",
            }
        )
    return result


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_words(text: str) -> int:
    return len(text.split())


def generate(root: Path, per_category: int, seed: int) -> dict[str, Any]:
    if per_category < 20 or per_category % 20 != 0:
        raise ValueError("--per-category must be at least 20 and divisible by 20 for exact 90/5/5 splits")

    full_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    transitions_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    seen_prompts: set[str] = set()
    manifest_rows: list[dict[str, Any]] = []
    sample_candidates: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]]]]] = defaultdict(list)

    categories = list(ALL_GENERATORS.items())
    print(f"Generating {len(categories)} categories × {per_category} problems = {len(categories) * per_category:,} problems")

    for category_index, (category, (domain, generator)) in enumerate(categories):
        category_counts = Counter()
        for local_index in range(per_category):
            retry = 0
            while True:
                generation_seed = seed + category_index * 10_000_000 + local_index * 1000 + retry
                rng = random.Random(generation_seed)
                generated = generator(rng, local_index)
                normalized = normalize_problem(generated.problem)
                if normalized not in seen_prompts:
                    seen_prompts.add(normalized)
                    break
                retry += 1
                if retry > 1000:
                    raise RuntimeError(f"Unable to create a unique prompt for {category} index {local_index}")

            split = infer_split(local_index)
            problem_id = f"rineya_si_v1_{category}_{local_index:04d}"
            full = build_full_record(
                problem_id=problem_id,
                split=split,
                domain=domain,
                category=category,
                generated=generated,
                generation_seed=generation_seed,
            )
            transitions = build_transition_records(full, generated)
            full_by_split[split].append(full)
            transitions_by_split[split].extend(transitions)
            category_counts[split] += 1
            manifest_rows.append(
                {
                    "problem_id": problem_id,
                    "split": split,
                    "domain": domain,
                    "category": category,
                    "difficulty": generated.difficulty,
                    "step_count": len(transitions),
                    "source_pair_sha256": full["source_pair_sha256"],
                    "problem_sha256": sha256_text(full["problem"]),
                    "full_response_sha256": sha256_text(full["full_response"]),
                }
            )
            if len(sample_candidates[category]) < 2 and split in {"train", "test"}:
                sample_candidates[category].append((full, transitions))

        expected = {"train": per_category * 18 // 20, "dev": per_category // 20, "test": per_category // 20}
        if dict(category_counts) != expected:
            raise AssertionError(f"Unexpected split counts for {category}: {category_counts}, expected {expected}")
        print(f"  {category:26s} train={category_counts['train']:4d} dev={category_counts['dev']:3d} test={category_counts['test']:3d}")

    # Deterministically shuffle all three splits. The transition order intentionally destroys trajectory chronology.
    for split in ("train", "dev", "test"):
        full_by_split[split].sort(key=lambda row: stable_key(row["problem_id"], f"full|{seed}|{split}"))
        transitions_by_split[split].sort(key=lambda row: stable_key(row["transition_id"], f"transition|{seed}|{split}"))
        repair_adjacent_problem_ids(transitions_by_split[split])

    output_files: list[Path] = []
    for split in ("train", "dev", "test"):
        full_path = root / "data" / "full_context" / f"{split}.jsonl"
        split_path = root / "data" / "split_transitions" / f"{split}.jsonl"
        write_jsonl(full_path, full_by_split[split])
        write_jsonl(split_path, transitions_by_split[split])
        output_files.extend([full_path, split_path])

    manifest_path = root / "data" / "manifests" / "problem_manifest.jsonl"
    manifest_rows.sort(key=lambda row: row["problem_id"])
    write_jsonl(manifest_path, manifest_rows)
    output_files.append(manifest_path)

    # Write category distribution CSV.
    category_csv = root / "data" / "manifests" / "category_distribution.csv"
    category_csv.parent.mkdir(parents=True, exist_ok=True)
    with category_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "domain", "train_problems", "dev_problems", "test_problems", "total_problems", "train_transitions", "dev_transitions", "test_transitions"])
        for category, (domain, _) in categories:
            row_full = {s: sum(1 for r in full_by_split[s] if r["category"] == category) for s in ("train", "dev", "test")}
            row_trans = {s: sum(1 for r in transitions_by_split[s] if r["category"] == category) for s in ("train", "dev", "test")}
            writer.writerow([category, domain, row_full["train"], row_full["dev"], row_full["test"], sum(row_full.values()), row_trans["train"], row_trans["dev"], row_trans["test"]])
    output_files.append(category_csv)

    # Paired samples: two examples per category, with every transition preserved beside its source record.
    paired_sample_rows: list[dict[str, Any]] = []
    for category, pairs in sample_candidates.items():
        for full, transitions in pairs:
            paired_sample_rows.append(
                {
                    "problem_id": full["problem_id"],
                    "category": category,
                    "split": full["split"],
                    "full_context_record": full,
                    "split_transition_records": transitions,
                }
            )
    paired_sample_rows.sort(key=lambda row: row["problem_id"])
    paired_path = root / "samples" / "paired_examples.jsonl"
    write_jsonl(paired_path, paired_sample_rows)
    output_files.append(paired_path)

    # Human-readable examples from one representative item in each domain.
    md_path = root / "samples" / "PAIRED_EXAMPLES.md"
    seen_domains: set[str] = set()
    md_lines = ["# Paired Dataset Examples", "", "Each example shows one canonical full-context record and the unordered transition records derived from it.", ""]
    for row in paired_sample_rows:
        full = row["full_context_record"]
        if full["domain"] in seen_domains:
            continue
        seen_domains.add(full["domain"])
        md_lines.extend([
            f"## {full['domain']} — {full['category']}",
            "",
            "### Original problem",
            "",
            full["problem"],
            "",
            "### Full-context target",
            "",
            "```text",
            full["full_response"],
            "```",
            "",
            "### Derived compact-state transitions",
            "",
        ])
        for transition in row["split_transition_records"]:
            md_lines.extend([
                f"**Step {transition['step_index']} / {transition['total_steps']}**",
                "",
                f"Current state: `{transition['current_state']}`",
                "",
                f"Target: `{transition['target_step']}`",
                "",
            ])
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    output_files.append(md_path)

    # Aggregate statistics.
    all_full = [r for split in full_by_split.values() for r in split]
    all_trans = [r for split in transitions_by_split.values() for r in split]
    step_counts = [r["step_count"] for r in all_full]
    full_word_counts = [count_words(r["full_response"]) for r in all_full]
    transition_word_counts = [count_words(r["target_step"]) for r in all_trans]

    stats = {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "generation_seed": seed,
        "category_count": len(categories),
        "problems_per_category": per_category,
        "total_unique_problems": len(all_full),
        "total_transition_records": len(all_trans),
        "problem_splits": {s: len(full_by_split[s]) for s in ("train", "dev", "test")},
        "transition_splits": {s: len(transitions_by_split[s]) for s in ("train", "dev", "test")},
        "step_count": {
            "min": min(step_counts),
            "max": max(step_counts),
            "mean": round(statistics.mean(step_counts), 4),
            "median": statistics.median(step_counts),
        },
        "full_response_word_count": {
            "mean": round(statistics.mean(full_word_counts), 4),
            "median": statistics.median(full_word_counts),
            "min": min(full_word_counts),
            "max": max(full_word_counts),
            "total": sum(full_word_counts),
        },
        "transition_target_word_count": {
            "mean": round(statistics.mean(transition_word_counts), 4),
            "median": statistics.median(transition_word_counts),
            "min": min(transition_word_counts),
            "max": max(transition_word_counts),
            "total": sum(transition_word_counts),
        },
        "sequence_policy": {
            "full_context": "One problem and one uninterrupted complete reasoning target per row.",
            "split_transitions": "Every transition is independently sufficient and globally shuffled; chronological adjacency is intentionally removed.",
        },
        "exact_problem_alignment": True,
        "cross_split_problem_overlap": False,
    }
    stats_path = root / "data" / "manifests" / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    output_files.append(stats_path)

    # File hashes are written last; do not hash the hash file itself.
    hashes_path = root / "data" / "manifests" / "sha256sums.txt"
    hash_lines = []
    for path in sorted(output_files):
        relative = path.relative_to(root).as_posix()
        hash_lines.append(f"{sha256_file(path)}  {relative}")
    hashes_path.write_text("\n".join(hash_lines) + "\n", encoding="utf-8")

    print("\nGeneration complete")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the paired full-context and sequence-free split-transition datasets.")
    parser.add_argument("--root", type=Path, default=SCRIPT_DIR.parent, help="Dataset repository root")
    parser.add_argument("--per-category", type=int, default=DEFAULT_PER_CATEGORY)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.root.resolve(), args.per_category, args.seed)
