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


def critical_dose_at_step(
    dose_rows: list[dict[str, Any]],
    *,
    progress_fraction: float = 0.5,
    absolute_floor: float = 0.01,
    accuracy_key: str = "ood_answerable_correctness",
) -> dict[str, Any]:
    """Lowest dose whose OOD accuracy reaches progress_fraction of that step's max-dose accuracy."""
    if not dose_rows:
        return {"critical_dose": None, "max_ood_accuracy": None, "target_accuracy": None}
    ordered = sorted(dose_rows, key=lambda row: float(row["dose"]))
    accuracies = [float(row.get(accuracy_key) or 0.0) for row in ordered]
    max_acc = max(accuracies) if accuracies else 0.0
    if max_acc < float(absolute_floor):
        return {
            "critical_dose": None,
            "max_ood_accuracy": max_acc,
            "target_accuracy": None,
            "reason": "below_absolute_floor",
        }
    target = float(progress_fraction) * max_acc
    for row, acc in zip(ordered, accuracies):
        if acc >= target:
            return {
                "critical_dose": float(row["dose"]),
                "max_ood_accuracy": max_acc,
                "target_accuracy": target,
            }
    return {
        "critical_dose": float(ordered[-1]["dose"]),
        "max_ood_accuracy": max_acc,
        "target_accuracy": target,
    }


def critical_dose_trajectory(
    surface_rows: list[dict[str, Any]],
    *,
    progress_fraction: float = 0.5,
    absolute_floor: float = 0.01,
    leftward_shift_minimum: float = 0.15,
    fixed_high_minimum: float = 0.75,
) -> dict[str, Any]:
    """Reduce a step×dose surface to critical_dose(t) and leftward-migration flags."""
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in surface_rows:
        by_step.setdefault(int(row["step"]), []).append(row)
    series: list[dict[str, Any]] = []
    for step in sorted(by_step):
        info = critical_dose_at_step(
            by_step[step],
            progress_fraction=progress_fraction,
            absolute_floor=absolute_floor,
        )
        series.append({"step": step, **info})
    defined = [row for row in series if row.get("critical_dose") is not None]
    start = float(defined[0]["critical_dose"]) if defined else None
    end = float(defined[-1]["critical_dose"]) if defined else None
    leftward_shift = (start - end) if start is not None and end is not None else None
    monotonic_leftward = False
    if len(defined) >= 2:
        doses = [float(row["critical_dose"]) for row in defined]
        monotonic_leftward = all(right <= left + 1e-12 for left, right in zip(doses, doses[1:]))
    migrates_left = bool(
        leftward_shift is not None
        and leftward_shift >= float(leftward_shift_minimum)
        and (monotonic_leftward or leftward_shift >= float(leftward_shift_minimum))
    )
    fixed_high = bool(
        end is not None
        and end >= float(fixed_high_minimum)
        and not migrates_left
    )
    return {
        "critical_dose_by_step": series,
        "critical_dose_start": start,
        "critical_dose_end": end,
        "critical_dose_leftward_shift": leftward_shift,
        "critical_dose_monotonic_leftward": monotonic_leftward,
        "critical_dose_migrates_left": migrates_left,
        "fixed_high_critical_dose": fixed_high,
    }


def _linear_r2(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_yy = sum((y - mean_y) ** 2 for y in ys)
    if ss_xx <= 1e-12 or ss_yy <= 1e-12:
        return None
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return float((ss_xy * ss_xy) / (ss_xx * ss_yy))


def temporal_readout_summary(
    rows: list[dict[str, Any]],
    *,
    accuracy_jump_fraction_minimum: float = 0.5,
    margin_linear_r2_minimum: float = 0.7,
    nll_linear_r2_minimum: float = 0.7,
) -> dict[str, Any]:
    """Detect sharp discrete OOD accuracy jumps accompanied by smooth logit trajectories."""
    rows = sorted(rows, key=lambda row: int(row["step"]))
    if len(rows) < 3:
        return {
            "smooth_logits_with_sharp_accuracy": False,
            "reason": "insufficient_points",
        }
    steps = [float(row["step"]) for row in rows]
    accuracies = [float(row.get("ood_answerable_correctness") or 0.0) for row in rows]
    gain = max(accuracies) - min(accuracies)
    jumps = [right - left for left, right in zip(accuracies, accuracies[1:])]
    largest_jump = max(jumps) if jumps else 0.0
    jump_fraction = (largest_jump / gain) if gain > 1e-12 else None
    sharp_accuracy = bool(jump_fraction is not None and jump_fraction >= float(accuracy_jump_fraction_minimum))

    margin_xs: list[float] = []
    margin_ys: list[float] = []
    for step, row in zip(steps, rows):
        value = finite(row.get("ood_first_target_margin_mean"))
        if value is None:
            value = finite(row.get("ood_target_token_margin_mean"))
        if value is not None:
            margin_xs.append(step)
            margin_ys.append(value)
    margin_r2 = _linear_r2(margin_xs, margin_ys)
    margin_improving = False
    if len(margin_ys) >= 2:
        margin_improving = margin_ys[-1] >= margin_ys[0]
    smooth_margin = bool(
        margin_r2 is not None
        and margin_r2 >= float(margin_linear_r2_minimum)
        and margin_improving
    )

    nll_xs: list[float] = []
    nll_ys: list[float] = []
    for step, row in zip(steps, rows):
        value = finite(row.get("ood_target_token_nll_mean"))
        if value is None:
            value = finite(row.get("ood_first_target_nll_mean"))
        if value is not None:
            nll_xs.append(step)
            nll_ys.append(value)
    nll_r2 = _linear_r2(nll_xs, nll_ys)
    nll_improving = False
    if len(nll_ys) >= 2:
        nll_improving = nll_ys[-1] <= nll_ys[0]
    smooth_nll = bool(
        nll_r2 is not None
        and nll_r2 >= float(nll_linear_r2_minimum)
        and nll_improving
    )

    return {
        "ood_accuracy_total_gain": gain,
        "largest_ood_accuracy_adjacent_jump": largest_jump,
        "largest_ood_accuracy_jump_fraction_of_gain": jump_fraction,
        "sharp_ood_accuracy_jump": sharp_accuracy,
        "ood_margin_linear_r2": margin_r2,
        "ood_margin_improving": margin_improving,
        "smooth_ood_margin_trajectory": smooth_margin,
        "ood_nll_linear_r2": nll_r2,
        "ood_nll_improving": nll_improving,
        "smooth_ood_nll_trajectory": smooth_nll,
        "smooth_logits_with_sharp_accuracy": bool(sharp_accuracy and (smooth_margin or smooth_nll)),
    }
