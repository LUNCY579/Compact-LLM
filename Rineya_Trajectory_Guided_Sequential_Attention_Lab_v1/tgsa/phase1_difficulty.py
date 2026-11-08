"""Outcome-blind task-difficulty and shortcut-baseline diagnostics.

Pointer mappings are random permutations, so fixed points and short cycles let a
nominal multi-hop item resolve to the start node or need fewer distinct
transitions than its hop label claims. Modular state sequences can likewise
cancel. Such items are valid, but they dilute the iterative mechanism signal and
reduce power (``PHASE1_LAUNCH_AUDIT.md`` finding P1-5).

Held-out ranges also *overlap* training ranges for some registered hop and step
values, so the result is held-out generated-task performance rather than uniform
out-of-range extrapolation (finding P1-6). This module measures and freezes both
facts before any confirmatory outcome exists. Nothing here reads a model or a
prediction file, so it can be run and published pre-launch.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence, cast

from .phase1_data import (
    Phase1Example,
    SyntheticVocabulary,
    _family_spec,
    load_examples,
    sha256_file,
    validate_dataset_manifest,
)

DIFFICULTY_SCHEMA_VERSION = "tgsa.phase1-task-difficulty.v1"


def _rate(values: Sequence[bool]) -> float:
    return (sum(1 for value in values if value) / len(values)) if values else 0.0


def _family_difficulty(examples: Sequence[Phase1Example]) -> dict[str, object]:
    if not examples:
        return {"examples": 0}
    metadata = [cast(Mapping[str, object], example.metadata) for example in examples]
    summary: dict[str, object] = {
        "examples": len(examples),
        "answer_equals_start_rate": _rate([bool(m.get("answer_equals_start", False)) for m in metadata]),
        "trivial_rate": _rate([bool(m.get("trivial", False)) for m in metadata]),
        "token_length": {
            "min": min(int(m["token_length"]) for m in metadata),
            "max": max(int(m["token_length"]) for m in metadata),
        },
    }
    family = examples[0].family
    if family in {"pointer_chase", "direct_lookup"}:
        requested = [int(m["hops"]) for m in metadata]
        effective = [int(m.get("effective_hops", m["hops"])) for m in metadata]
        summary.update(
            {
                "requested_hops": {"min": min(requested), "max": max(requested)},
                "effective_hops": {
                    "min": min(effective),
                    "max": max(effective),
                    "mean": sum(effective) / len(effective),
                },
                "short_circuit_rate": _rate(
                    [effective[index] < requested[index] for index in range(len(requested))]
                ),
                "fixed_point_start_rate": _rate(
                    [bool(m.get("start_is_fixed_point", False)) for m in metadata]
                ),
                "minimal_cycle_distance": {
                    "min": min(int(m.get("start_cycle_length", 1)) for m in metadata),
                    "mean": sum(int(m.get("start_cycle_length", 1)) for m in metadata) / len(metadata),
                },
                "requested_hop_histogram": dict(sorted(Counter(requested).items())),
            }
        )
    elif family == "state_update":
        requested = [int(m["step_count"]) for m in metadata]
        effective = [int(m.get("effective_steps", m["step_count"])) for m in metadata]
        summary.update(
            {
                "requested_steps": {"min": min(requested), "max": max(requested)},
                "effective_steps": {
                    "min": min(effective),
                    "max": max(effective),
                    "mean": sum(effective) / len(effective),
                },
                "cancellation_rate": _rate([bool(m.get("cancels_to_start", False)) for m in metadata]),
                "early_return_rate": _rate(
                    [effective[index] < requested[index] for index in range(len(requested))]
                ),
                "requested_step_histogram": dict(sorted(Counter(requested).items())),
            }
        )
    return summary


def shortcut_baselines(
    train: Sequence[Phase1Example],
    test: Sequence[Phase1Example],
    *,
    vocab: SyntheticVocabulary,
) -> dict[str, object]:
    """Registered non-learned baselines the mechanism must beat to be meaningful.

    * ``copy_start`` answers with the queried start value. It is exactly the
      degenerate-item rate, so it is the floor a real iterative solver must clear.
    * ``copy_last_value`` answers with the final value token in the prompt.
    * ``train_majority`` answers with the most frequent training answer token.
    """

    majority = Counter(example.answer_token for example in train).most_common(1)
    majority_token = majority[0][0] if majority else -1
    results: dict[str, object] = {}
    for family in sorted({example.family for example in test}):
        selection = [example for example in test if example.family == family]
        copy_start = 0
        copy_last = 0
        majority_hits = 0
        for example in selection:
            inputs = example.tokens[: example.answer_input_index + 1]
            value_tokens = [
                token
                for token in inputs
                if vocab.value_base <= token < vocab.op_base
            ]
            if family in {"pointer_chase", "direct_lookup"}:
                # The queried start is a KEY token; its value-space counterpart
                # is the answer a naive copier would emit.
                keys = [token for token in inputs if vocab.key_base <= token < vocab.value_base]
                if keys:
                    start_value = vocab.value_base + (keys[-1] - vocab.key_base)
                    copy_start += int(start_value == example.answer_token)
            else:
                if value_tokens:
                    copy_start += int(value_tokens[0] == example.answer_token)
            if value_tokens:
                copy_last += int(value_tokens[-1] == example.answer_token)
            majority_hits += int(majority_token == example.answer_token)
        count = len(selection)
        results[family] = {
            "examples": count,
            "copy_start_accuracy": copy_start / count,
            "copy_last_value_accuracy": copy_last / count,
            "train_majority_accuracy": majority_hits / count,
        }
    return results


def _range_overlap(
    dataset_config: Mapping[str, object],
    *,
    family: str,
    low_key: str,
    high_key: str,
) -> dict[str, object]:
    train = _family_spec(dataset_config, family, split="train")
    test = _family_spec(dataset_config, family, split="test")
    train_low, train_high = int(train[low_key]), int(train[high_key])
    test_low, test_high = int(test[low_key]), int(test[high_key])
    overlap_low = max(train_low, test_low)
    overlap_high = min(train_high, test_high)
    overlapping = list(range(overlap_low, overlap_high + 1)) if overlap_low <= overlap_high else []
    return {
        "train_range": [train_low, train_high],
        "test_range": [test_low, test_high],
        "overlapping_values": overlapping,
        "fully_out_of_range": not overlapping,
    }


def difficulty_report(
    *,
    dataset_dir: Path,
    dataset_config: Mapping[str, object],
) -> dict[str, object]:
    """Freeze the pre-launch difficulty and shortcut-baseline picture."""

    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = validate_dataset_manifest(manifest_path)
    splits = cast(Mapping[str, Mapping[str, object]], manifest["splits"])
    by_split = {
        split: load_examples(dataset_dir / str(splits[split]["path"]))
        for split in ("train", "dev", "test")
    }
    vocabulary = cast(Mapping[str, object], manifest["vocabulary"])
    vocab = SyntheticVocabulary(
        domain_size=int(vocabulary["domain_size"]),
        max_hops=int(vocabulary["max_hops"]),
        max_abs_step=int(vocabulary["max_abs_step"]),
    )

    difficulty: dict[str, object] = {}
    for split, examples in by_split.items():
        per_family: dict[str, object] = {}
        for family in sorted({example.family for example in examples}):
            per_family[family] = _family_difficulty(
                [example for example in examples if example.family == family]
            )
        difficulty[split] = per_family

    # Finding P1-6: describe held-out ranges honestly.
    overlap = {
        "pointer_chase_hops": _range_overlap(
            dataset_config, family="pointer_chase", low_key="min_hops", high_key="max_hops"
        ),
        "state_update_steps": _range_overlap(
            dataset_config, family="state_update", low_key="min_steps", high_key="max_steps"
        ),
        "direct_lookup_nodes": _range_overlap(
            dataset_config, family="direct_lookup", low_key="min_nodes", high_key="max_nodes"
        ),
    }
    any_overlap = any(not entry["fully_out_of_range"] for entry in overlap.values())

    return {
        "schema_version": DIFFICULTY_SCHEMA_VERSION,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_config_sha256": manifest["dataset_config_sha256"],
        "difficulty_by_split": difficulty,
        "shortcut_baselines_test": shortcut_baselines(
            by_split["train"], by_split["test"], vocab=vocab
        ),
        "held_out_range_overlap": overlap,
        "claim_scope": (
            "held-out generated-task performance on instance-held-out examples; "
            + (
                "registered hop/step/node ranges partially overlap training, so this is NOT "
                "uniform out-of-range extrapolation"
                if any_overlap
                else "registered ranges are disjoint from training"
            )
        ),
        "uniform_out_of_range_extrapolation": not any_overlap,
    }


def render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# TGSA Phase 1 task-difficulty and shortcut-baseline report",
        "",
        "Outcome-blind. Computed from the materialized dataset only; no model, ",
        "checkpoint, or prediction file is read.",
        "",
        f"**Dataset manifest SHA-256:** `{report['dataset_manifest_sha256']}`",
        "",
        f"**Uniform out-of-range extrapolation:** `{report['uniform_out_of_range_extrapolation']}`",
        "",
        "## Held-out range overlap",
        "",
        "| Dimension | Train | Test | Overlapping values |",
        "|---|---|---|---|",
    ]
    for name, entry in cast(Mapping[str, Mapping[str, object]], report["held_out_range_overlap"]).items():
        lines.append(
            f"| `{name}` | {entry['train_range']} | {entry['test_range']} | "
            f"{entry['overlapping_values'] or 'none'} |"
        )
    lines.extend(["", "## Test-split shortcut baselines", "", "| Family | copy start | copy last value | train majority |", "|---|---:|---:|---:|"])
    for family, entry in cast(Mapping[str, Mapping[str, object]], report["shortcut_baselines_test"]).items():
        lines.append(
            f"| `{family}` | {float(entry['copy_start_accuracy']):.4f} | "
            f"{float(entry['copy_last_value_accuracy']):.4f} | "
            f"{float(entry['train_majority_accuracy']):.4f} |"
        )
    lines.extend(["", "## Test-split degeneracy rates", "", "| Family | trivial | answer == start |", "|---|---:|---:|"])
    test = cast(Mapping[str, Mapping[str, object]], cast(Mapping[str, object], report["difficulty_by_split"])["test"])
    for family, entry in test.items():
        lines.append(
            f"| `{family}` | {float(entry['trivial_rate']):.4f} | "
            f"{float(entry['answer_equals_start_rate']):.4f} |"
        )
    lines.append("")
    return "\n".join(lines)
