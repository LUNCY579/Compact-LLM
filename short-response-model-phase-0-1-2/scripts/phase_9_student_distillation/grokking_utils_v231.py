from __future__ import annotations

"""Shared helpers for the v2.3.1 Grokking Dynamics Observatory."""

import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ABSTAIN = "I don't know based on the available information."


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def norm(text: str) -> str:
    text = (text or "").strip().casefold()
    text = re.sub(r'^[\s"\'`]+|[\s"\'`]+$', "", text)
    text = re.sub(r"[.!?]+$", "", text)
    return re.sub(r"\s+", " ", text)


def is_abstain(text: str) -> bool:
    value = norm(text)
    return value == norm(ABSTAIN) or value.startswith("i don't know") or value.startswith("i do not know")


def exact_correct(answer: str, row: dict[str, Any]) -> bool:
    gold = [str(row.get("approved_answer") or row.get("answer") or "")]
    gold.extend(str(value) for value in (row.get("answer_aliases") or []))
    value = norm(answer)
    return any(value == norm(target) for target in gold if norm(target))


def apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]], *, generation: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": generation}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def rate(values: Iterable[bool]) -> float | None:
    values = list(values)
    return sum(bool(value) for value in values) / len(values) if values else None


def balanced_sample(
    rows: list[dict[str, Any]],
    n: int,
    *,
    seed: int,
    group_fields: tuple[str, ...] = ("reasoning_family",),
) -> list[dict[str, Any]]:
    """Round-robin sample across semantic groups without inventing weights."""
    if n <= 0 or not rows:
        return []
    if n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(field) or "") for field in group_fields)
        groups[key].append(row)
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    rng.shuffle(keys)
    out: list[dict[str, Any]] = []
    cursor = 0
    while len(out) < n and keys:
        key = keys[cursor % len(keys)]
        bucket = groups[key]
        if bucket:
            out.append(bucket.pop())
        else:
            keys.remove(key)
            cursor -= 1
        cursor += 1
    return out


def checkpoint_step(path: Path) -> int | None:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else None


def discover_adapters(training_root: Path) -> list[dict[str, Any]]:
    """Return step-zero placeholder, Trainer checkpoints, and final adapter metadata."""
    out: list[dict[str, Any]] = [{"step": 0, "adapter": None, "kind": "anchor"}]
    checkpoint_root = training_root / "checkpoints"
    if checkpoint_root.exists():
        for child in checkpoint_root.iterdir():
            if not child.is_dir():
                continue
            step = checkpoint_step(child)
            if step is None:
                continue
            # PEFT Trainer checkpoints should contain adapter_config.json.  If a
            # transformers version nests the adapter one level deeper, accept it.
            adapter = child
            if not (adapter / "adapter_config.json").exists():
                candidates = [p.parent for p in child.rglob("adapter_config.json")]
                if candidates:
                    adapter = candidates[0]
            if (adapter / "adapter_config.json").exists():
                out.append({"step": step, "adapter": str(adapter), "kind": "checkpoint"})
    snapshot_root = training_root / "dynamics_snapshots"
    if snapshot_root.exists():
        for child in snapshot_root.iterdir():
            if not child.is_dir():
                continue
            match = re.search(r"step_(\d+)$", child.name)
            if match and (child / "adapter_config.json").exists():
                step = int(match.group(1))
                out = [item for item in out if int(item["step"]) != step]
                out.append({"step": step, "adapter": str(child), "kind": "dynamics_snapshot"})
    manifest = training_root / "training_manifest.json"
    final = training_root / "adapter"
    final_step = None
    if manifest.exists():
        try:
            final_step = int(read_json(manifest).get("optimizer_steps") or 0)
        except (TypeError, ValueError):
            final_step = None
    if final.exists() and (final / "adapter_config.json").exists():
        if not final_step:
            final_step = max([int(item["step"]) for item in out if int(item["step"]) > 0] or [0])
        if not any(int(item["step"]) == final_step for item in out):
            out.append({"step": final_step, "adapter": str(final), "kind": "final"})
        else:
            # Prefer the explicitly published final adapter for the final step.
            for item in out:
                if int(item["step"]) == final_step:
                    item["adapter"] = str(final)
                    item["kind"] = "final"
    return sorted(out, key=lambda item: int(item["step"]))


def parse_number_list(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values]


def fraction_tag(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def nearest_steps(available: list[int], requested: Iterable[float | int], max_step: int) -> list[int]:
    chosen: set[int] = set()
    for value in requested:
        target = int(round(float(value) * max_step)) if isinstance(value, float) and 0.0 <= value <= 1.0 else int(value)
        if available:
            chosen.add(min(available, key=lambda step: abs(step - target)))
    return sorted(chosen)


def progress_step(points: list[tuple[int, float]], fraction: float) -> int | None:
    if not points:
        return None
    points = sorted(points)
    baseline = points[0][1]
    best = max(value for _, value in points)
    gain = best - baseline
    if gain <= 1e-12:
        return None
    target = baseline + float(fraction) * gain
    for step, value in points:
        if value >= target:
            return int(step)
    return None


def finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None
