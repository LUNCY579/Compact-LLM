from __future__ import annotations

"""Build a deterministic concise-reasoning augmentation for Phase 2C.

The Phase 2C training files intentionally retain the approved Phase 1 dataset
and add rule-generated examples whose answers are mechanically verifiable.  No
sealed examples are read, and train/dev entities are disjoint by construction.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, read_jsonl, write_json, write_jsonl
from scripts.phase_2c_targeted_recovery.phase2c_common import relative_to_root

TRAIN_NAMES = [
    "Ava", "Ben", "Cora", "Dylan", "Esme", "Felix", "Gia", "Hugo",
    "Iris", "Jalen", "Kira", "Liam", "Mina", "Nolan", "Opal", "Pavel",
]
DEV_NAMES = ["Rina", "Soren", "Talia", "Uma", "Vera", "Wes", "Xena", "Yuri"]
TRAIN_OBJECTS = [
    "amber key", "blue card", "cedar box", "delta badge", "emerald tag",
    "frost token", "gold pass", "harbor ticket", "ivory seal", "jade coin",
    "kite permit", "linen note", "maple stamp", "navy chip", "opal marker",
    "pearl pin",
]
DEV_OBJECTS = [
    "quartz key", "ruby card", "silver box", "teal badge", "umber tag",
    "violet token", "white pass", "yellow ticket",
]


def indefinite_article(phrase: str) -> str:
    return "an" if phrase[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def pluralize(phrase: str) -> str:
    words = phrase.split()
    if not words:
        return phrase
    noun = words[-1]
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        noun += "es"
    elif len(noun) > 1 and noun.endswith("y") and noun[-2].lower() not in "aeiou":
        noun = noun[:-1] + "ies"
    else:
        noun += "s"
    return " ".join([*words[:-1], noun])


def record(
    identifier: str,
    prompt: str,
    answer: str,
    category: str,
    *,
    reasoning_steps: int = 1,
    target_max: int = 12,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "system_policy_version": "v1",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "reply_class": category,
            "length_mode": "SHORT" if target_max > 6 else "TINY",
            "target_token_min": 1,
            "target_token_max": target_max,
            "requires_external_fact": False,
            "safety_tier": "normal",
            "tone": ["direct", "concise"],
            "reasoning_steps": reasoning_steps,
            "source_provenance": "deterministic_rule_template_phase2c_v1",
        },
    }


def implication_rows(prefix: str, names: list[str], objects: list[str]) -> list[dict[str, Any]]:
    rooms = ["studio", "archive", "garage", "kitchen", "lab", "office", "garden", "storage room"]
    rows: list[dict[str, Any]] = []
    for index, (name, obj) in enumerate(zip(names, objects)):
        room = rooms[index % len(rooms)]
        article = indefinite_article(obj)
        prompt = f"Every {obj} opens the {room}. {name} has {article} {obj}. What can it open?"
        rows.append(record(f"{prefix}_implication_{index:02d}", prompt, f"The {room}.", "concise_logic_implication", reasoning_steps=2))
    return rows


def ordering_rows(prefix: str, names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(0, len(names) - 2):
        a, b, c = names[index : index + 3]
        prompt = f"{a} arrived before {b}, and {b} arrived before {c}. Who arrived first?"
        rows.append(record(f"{prefix}_ordering_{index:02d}", prompt, f"{a} arrived first.", "concise_ordering_logic", reasoning_steps=2))
    return rows


def comparison_rows(prefix: str, names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(0, len(names) - 2):
        a, b, c = names[index : index + 3]
        prompt = f"{a} is taller than {b}, and {b} is taller than {c}. Who is shortest?"
        rows.append(record(f"{prefix}_comparison_{index:02d}", prompt, f"{c} is shortest.", "concise_comparison_logic", reasoning_steps=2))
    return rows


def arithmetic_rows(prefix: str, count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        start = rng.randint(3, 24)
        added = rng.randint(2, 12)
        removed = rng.randint(1, min(added + 2, start + added - 1))
        result = start + added - removed
        prompt = f"A tray had {start} clips, gained {added}, then lost {removed}. How many remain?"
        rows.append(record(f"{prefix}_arithmetic_{index:02d}", prompt, f"{result} clips remain.", "concise_arithmetic", reasoning_steps=2, target_max=7))
    return rows


def contradiction_rows(prefix: str, names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        item = ["report", "package", "message", "reservation", "invoice", "ticket", "draft", "backup"][index % 8]
        prompt = f"The note says {name} submitted the {item}. Another line says {name} never submitted it. Can both claims be true as written?"
        rows.append(record(f"{prefix}_contradiction_{index:02d}", prompt, "No, they contradict each other.", "concise_contradiction_detection", reasoning_steps=1, target_max=8))
    return rows


def set_rows(prefix: str, names: list[str], objects: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (name, obj) in enumerate(zip(names, objects)):
        group = ["priority items", "fragile items", "approved items", "returnable items"][index % 4]
        article = indefinite_article(obj)
        prompt = (
            f"All {pluralize(obj)} are {group}. {name}'s item is {article} {obj}. "
            f"Is it one of the {group}?"
        )
        rows.append(record(f"{prefix}_set_{index:02d}", prompt, "Yes.", "concise_set_membership", reasoning_steps=2, target_max=4))
    return rows


def temporal_rows(prefix: str, names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for index, name in enumerate(names):
        first = days[index % 4]
        second = days[(index % 4) + 1]
        prompt = f"{name} finished the outline on {first} and the draft on {second}. Which was finished later?"
        rows.append(record(f"{prefix}_temporal_{index:02d}", prompt, "The draft was finished later.", "concise_temporal_reasoning", reasoning_steps=1, target_max=8))
    return rows


def clarification_rows(prefix: str, names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        prompt = f"{name} said, 'Can you change it?' No object was mentioned. What should you ask?"
        rows.append(record(f"{prefix}_clarification_{index:02d}", prompt, "What should I change?", "concise_clarification_logic", reasoning_steps=1, target_max=7))
    return rows






def deterministic_shuffle(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled

def interleave_rows(
    base_rows: list[dict[str, Any]],
    added_rows: list[dict[str, Any]],
    *,
    base_per_added: int,
) -> list[dict[str, Any]]:
    """Place specialized records throughout the prefix used by limited evaluations."""

    if base_per_added < 1:
        raise ValueError("base_per_added must be at least 1")
    output: list[dict[str, Any]] = []
    base_index = 0
    added_index = 0
    while base_index < len(base_rows) or added_index < len(added_rows):
        for _ in range(base_per_added):
            if base_index < len(base_rows):
                output.append(base_rows[base_index])
                base_index += 1
        if added_index < len(added_rows):
            output.append(added_rows[added_index])
            added_index += 1
    return output

def generated_rows(prefix: str, *, train: bool) -> list[dict[str, Any]]:
    names = TRAIN_NAMES if train else DEV_NAMES
    objects = TRAIN_OBJECTS if train else DEV_OBJECTS
    seed = 20260730 if train else 20260731
    rows: list[dict[str, Any]] = []
    rows.extend(implication_rows(prefix, names, objects))
    rows.extend(ordering_rows(prefix, names))
    rows.extend(comparison_rows(prefix, names))
    rows.extend(arithmetic_rows(prefix, len(names), seed))
    rows.extend(contradiction_rows(prefix, names))
    rows.extend(set_rows(prefix, names, objects))
    rows.extend(temporal_rows(prefix, names))
    rows.extend(clarification_rows(prefix, names))
    return rows




def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def prompt_target_key(row: dict[str, Any]) -> tuple[str, str]:
    messages = row.get("messages") or []
    prompt = "\n".join(str(message.get("content", "")) for message in messages[:-1]).strip().lower()
    target = str(messages[-1].get("content", "")).strip().lower() if messages else ""
    return prompt, target


def validate(rows: Iterable[dict[str, Any]], label: str) -> None:
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for row in rows:
        identifier = str(row.get("id", ""))
        if not identifier or identifier in seen_ids:
            raise ValueError(f"{label}: missing or duplicate id {identifier!r}")
        seen_ids.add(identifier)
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2 or messages[-1].get("role") != "assistant":
            raise ValueError(f"{label}:{identifier}: invalid SFT messages")
        pair = prompt_target_key(row)
        if pair in seen_pairs:
            raise ValueError(f"{label}:{identifier}: duplicate prompt/target pair")
        seen_pairs.add(pair)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the deterministic Phase 2C training/development mixture.")
    parser.add_argument("--base-train", type=Path, default=ROOT / "data/train/train.jsonl")
    parser.add_argument("--base-dev", type=Path, default=ROOT / "data/dev/dev.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/phase_2c_targeted_recovery")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.output_dir / "train.jsonl"
    dev_path = args.output_dir / "dev.jsonl"
    manifest_path = args.output_dir / "dataset_manifest.json"
    if not args.force and train_path.exists() and dev_path.exists() and manifest_path.exists():
        manifest = read_json(manifest_path)
        expected_hashes = {
            "base_train_sha256": sha256_file(args.base_train),
            "base_dev_sha256": sha256_file(args.base_dev),
            "train_output_sha256": sha256_file(train_path),
            "dev_output_sha256": sha256_file(dev_path),
        }
        mismatches = [
            key for key, value in expected_hashes.items()
            if manifest.get(key) != value
        ]
        if mismatches:
            raise RuntimeError(
                "Existing Phase 2C dataset does not match its source/output hashes "
                f"({', '.join(mismatches)}). Review the changes, then regenerate with --force."
            )
        train_rows = read_jsonl(train_path)
        dev_rows = read_jsonl(dev_path)
        validate(train_rows, "train")
        validate(dev_rows, "dev")
        if {prompt_target_key(row) for row in train_rows} & {prompt_target_key(row) for row in dev_rows}:
            raise RuntimeError("Existing Phase 2C train/dev files contain exact prompt-target overlap.")
        print(manifest_path)
        return

    base_train = read_jsonl(args.base_train)
    base_dev = read_jsonl(args.base_dev)
    added_train = deterministic_shuffle(generated_rows("p2c_train", train=True), 20260730)
    added_dev = deterministic_shuffle(generated_rows("p2c_dev", train=False), 20260731)
    # Training is shuffled by Trainer, but deterministic interleaving makes the
    # file itself inspectable. Development uses a denser specialized mixture so
    # the default 64-case ranking and 180-case candidate evaluation limits both
    # contain concise-reasoning and original conversational controls.
    train_rows = interleave_rows(base_train, added_train, base_per_added=8)
    dev_rows = interleave_rows(base_dev, added_dev, base_per_added=2)
    validate(train_rows, "train")
    validate(dev_rows, "dev")

    train_pairs = {prompt_target_key(row) for row in train_rows}
    dev_pairs = {prompt_target_key(row) for row in dev_rows}
    overlap = train_pairs & dev_pairs
    if overlap:
        raise ValueError(f"Phase 2C train/dev exact prompt-target overlap: {len(overlap)}")

    write_jsonl(train_path, train_rows)
    write_jsonl(dev_path, dev_rows)
    categories = Counter((row.get("metadata") or {}).get("reply_class", "unknown") for row in [*added_train, *added_dev])
    write_json(
        manifest_path,
        {
            "manifest_version": 1,
            "sealed_data_used": False,
            "base_train": relative_to_root(args.base_train, ROOT),
            "base_dev": relative_to_root(args.base_dev, ROOT),
            "train_output": relative_to_root(train_path, ROOT),
            "dev_output": relative_to_root(dev_path, ROOT),
            "base_train_sha256": sha256_file(args.base_train),
            "base_dev_sha256": sha256_file(args.base_dev),
            "train_output_sha256": sha256_file(train_path),
            "dev_output_sha256": sha256_file(dev_path),
            "base_train_count": len(base_train),
            "base_dev_count": len(base_dev),
            "added_train_count": len(added_train),
            "added_dev_count": len(added_dev),
            "final_train_count": len(train_rows),
            "final_dev_count": len(dev_rows),
            "exact_prompt_target_overlap": 0,
            "added_category_counts": dict(sorted(categories.items())),
            "generation_method": "deterministic rule templates with mechanically verified answers",
            "ordering": {
                "train": "8 base records then 1 added record while available",
                "dev": "2 base records then 1 added record while available",
                "reason": "limited-prefix calibration/evaluation must include both original and Phase 2C categories",
                "added_rows_shuffled_deterministically": True,
            },
            "evidence_boundary": (
                "These records strengthen concise local reasoning and do not claim to add broad "
                "world knowledge or general reasoning beyond the base model."
            ),
        },
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
