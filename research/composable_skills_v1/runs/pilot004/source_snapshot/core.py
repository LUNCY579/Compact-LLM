"""Pure data, scoring and accounting. No model calls or historical mutations."""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rate_at(peak: float, completed: int, total: int) -> float:
    """RRP-015 token-position WSD: square-root decay, floor 0.1*peak."""
    if not math.isfinite(peak) or peak <= 0 or not 0 < completed <= total:
        raise ValueError("invalid learning rate position")
    progress = completed / total
    if progress < 0.2:
        return peak * progress / 0.2
    if progress <= 0.8:
        return peak
    phase = min(1.0, (progress - 0.8) / 0.2)
    return peak * (0.1 + 0.9 * (1.0 - math.sqrt(phase)))


def exact(text: str, reference: str, family: str, terminated: bool = True) -> bool:
    if not terminated:
        return False
    if family in {"ledger_replay", "prerequisite_plan", "register_machine"}:
        try:
            # Serializing avoids Python's True == 1 and float == int shortcuts.
            return json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"), allow_nan=False) == reference
        except (ValueError, TypeError):
            return False
    return text.strip() == reference


def paired(source: dict[str, bool], candidate: dict[str, bool]) -> dict:
    if not source or source.keys() != candidate.keys():
        raise ValueError("paired endpoints require identical, nonempty ID sets")
    if any(type(v) is not bool for v in [*source.values(), *candidate.values()]):
        raise ValueError("correctness must be boolean")
    lost = sorted(i for i in source if source[i] and not candidate[i])
    gained = sorted(i for i in source if not source[i] and candidate[i])
    retained = sum(source[i] and candidate[i] for i in source)
    return {"n": len(source), "source_correct": sum(source.values()),
            "candidate_correct": sum(candidate.values()), "retained": retained,
            "lost": len(lost), "gained": len(gained), "lost_ids": lost, "gained_ids": gained,
            "conditional_forgetting": len(lost) / sum(source.values()) if any(source.values()) else None,
            "net": len(gained) - len(lost)}


def bank(seed: int, worlds: int, split: str) -> list[dict]:
    """Each binding world has answerable/missing and identity-swap controls.

    Train and evaluation use disjoint rendering templates. IDs are never in prompts.
    This is visible synthetic development data, NOT protected confirmation.
    """
    if split not in {"anchor", "dev"} or worlds < 1:
        raise ValueError("invalid bank request")
    rng = random.Random(seed)
    names = ["Nora", "Theo", "Mira", "Omar", "Lena", "Ivan", "Rosa", "Hugo", "Ada", "Noah", "Iris", "Leon"]
    roles = ["keeper", "courier", "clerk", "editor", "painter", "baker", "singer", "driver"]
    rows = []
    for w in range(worlds):
        people = rng.sample(names, 4)
        selected = rng.sample(roles, 5)
        target = w % 4
        world_id = f"CS1-{split}-{seed}-{w:04}"
        for swap in (False, True):
            occupants = people.copy()
            if swap:
                j = (target + 1) % 4
                occupants[target], occupants[j] = occupants[j], occupants[target]
            for missing in (False, True):
                assignments = list(zip(selected[:4], occupants))
                if missing:
                    assignments[target] = (selected[4], occupants[target])
                if split == "anchor":
                    facts = [f"{person} is the {role}." for role, person in assignments]
                    question = f"Who is the {selected[target]}?"
                    template = "roster_is"
                else:
                    facts = [f"The {role} position is held by {person}." for role, person in assignments]
                    question = f"Name the person holding the {selected[target]} position."
                    template = "position_held"
                # Position changes independently of answerability, keeping paired rows aligned.
                shift = (w // 4) % 4
                facts = facts[shift:] + facts[:shift]
                filler = ""
                if w % 2:
                    filler = ("The building has a blue door. The hall contains a wooden bench. " * 20)
                system = "Answer using only the roster. Return just the person's name. If the roster does not identify that person, return UNKNOWN and nothing else."
                rows.append({"id": f"{world_id}-b{int(swap)}-u{int(missing)}", "world_id": world_id,
                             "split": split, "family": "binding", "template": template,
                             "answerable": not missing, "long": bool(w % 2),
                             "reference": "UNKNOWN" if missing else occupants[target],
                             "prompt": [{"role": "system", "content": system},
                                        {"role": "user", "content": filler + " ".join(facts) + "\n" + question}],
                             "semantic_key": json.dumps([sorted(assignments), selected[target]])})
        a, b = rng.randrange(12, 90), rng.randrange(2, 10)
        op = "+" if w % 2 else "-"
        answer = a + b if op == "+" else a - b
        q = f"Calculate {a} {op} {b}." if split == "anchor" else f"What is the result of {a} {op} {b}?"
        rows.append({"id": f"{world_id}-arithmetic", "world_id": world_id + "-arithmetic",
                     "split": split, "family": "arithmetic", "template": split + "_arithmetic",
                     "reference": str(answer), "answerable": True,
                     "semantic_key": f"arithmetic:{a}{op}{b}",
                     "prompt": [{"role": "system", "content": "Return only the integer answer."}, {"role": "user", "content": q}]})
        hour, minute, delay = rng.randrange(8, 18), rng.randrange(0, 60), rng.randrange(5, 90)
        end = hour * 60 + minute + delay
        answer = f"{end // 60:02}:{end % 60:02}"
        q = (f"A clock shows {hour:02}:{minute:02}. What time is it {delay} minutes later?"
             if split == "anchor" else f"The start time is {hour:02}:{minute:02}; {delay} minutes pass. Give the end time.")
        rows.append({"id": f"{world_id}-time", "world_id": world_id + "-time",
                     "split": split, "family": "time", "template": split + "_time",
                     "reference": answer, "answerable": True,
                     "semantic_key": f"time:{hour}:{minute}:{delay}",
                     "prompt": [{"role": "system", "content": "Use 24-hour HH:MM format only."}, {"role": "user", "content": q}]})
    return rows


def validate_banks(anchor: list[dict], dev: list[dict]) -> dict:
    for name, rows in (("anchor", anchor), ("dev", dev)):
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"duplicate IDs in {name}")
        if any(r["split"] != name for r in rows):
            raise ValueError("split mismatch")
    for key in ("world_id", "semantic_key", "template"):
        overlap = {r[key] for r in anchor} & {r[key] for r in dev}
        if overlap:
            raise ValueError(f"cross-split {key} overlap: {sorted(overlap)[:3]}")
    return {"anchor_rows": len(anchor), "dev_rows": len(dev),
            "semantic_and_world_overlap": 0, "template_overlap": 0,
            "anchor_families": dict(Counter(r["family"] for r in anchor)),
            "dev_families": dict(Counter(r["family"] for r in dev)),
            "role": "VISIBLE_SYNTHETIC_DEVELOPMENT", "independently_sealed": False,
            "limitations": ["Only three narrow synthetic old-skill families.",
                            "Long contexts use repetitive irrelevant filler; not broad long-context coverage.",
                            "Template split does not establish generalization outside this generator."]}
