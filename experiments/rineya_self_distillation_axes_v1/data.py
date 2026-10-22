"""Explicit TRAIN/DEV admission, independent parents, fixed privileged context.

No benchmark downloads, sealed-test reads, generated-code execution or data relabeling.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
import re

FIELDS = {"id", "parent_id", "task", "split", "purpose", "source", "admitted_for",
          "prompt", "reference", "answer", "scorer"}
TEACHER_TEMPLATE = (
    "{prompt}\n\nReference response for this training example:\n{reference}\n\n"
    "Use the reference to solve the original task. Return the requested answer."
)


def last_box(text):
    matches = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return matches[-1].strip() if matches else None


def score(text: str, answer: str, scorer: str) -> bool:
    if not isinstance(text, str) or not isinstance(answer, str):
        raise ValueError("Text and canonical answer must be strings")
    if scorer not in {"exact", "json_exact", "boxed_int", "boxed_base9", "boxed_pair"}:
        raise ValueError("Unsupported scorer")
    if scorer == "exact":
        return text.strip() == answer.strip()
    if scorer == "json_exact":
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("Duplicate JSON key")
                result[key] = value
            return result
        def invalid_constant(value):
            raise ValueError("Nonstandard JSON constant")
        def canonical(value):
            parsed = json.loads(value, object_pairs_hook=pairs, parse_constant=invalid_constant)
            return json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)
        try:
            return canonical(text) == canonical(answer)
        except (ValueError, TypeError):
            return False
    box = last_box(text)
    if box is None:
        return False
    if scorer in {"boxed_int", "boxed_base9"}:
        grammar = r"[+-]?[0-8]+" if scorer == "boxed_base9" else r"[+-]?[0-9]+"
        if not re.fullmatch(grammar, box) or not re.fullmatch(grammar, answer):
            return False
        base = 9 if scorer == "boxed_base9" else 10
        return int(box, base) == int(answer, base)
    if scorer == "boxed_pair":
        grammar = r"\(\s*([+-]?[0-9]+)\s*,\s*([+-]?[0-9]+)\s*\)"
        actual, expected = re.fullmatch(grammar, box), re.fullmatch(grammar, answer)
        return bool(actual and expected and tuple(map(int, actual.groups())) == tuple(map(int, expected.groups())))
    raise ValueError("Unsupported scorer")


def validate_rows(rows, task):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 512:
        raise ValueError("Micro-pilot expects 1--512 admitted rows")
    ids, parents, prompts = set(), {}, {}
    for r in rows:
        if not isinstance(r, dict) or set(r) != FIELDS or any(not isinstance(v, str) or not v for v in r.values()):
            raise ValueError("Rows need exactly the documented nonempty string fields")
        if r["split"] not in {"train", "dev"} or r["admitted_for"] != r["split"]:
            raise ValueError("Only admitted TRAIN/DEV; sealed or confirmation inputs are forbidden")
        if r["purpose"] not in {"acquisition", "retention"} or (r["split"] == "train" and r["purpose"] != "acquisition"):
            raise ValueError("Retention is evaluation-only in this first factorial")
        if r["id"] in ids:
            raise ValueError("Duplicate row id")
        ids.add(r["id"])
        # Paired ordinary/contradictory tasks may share x only WITHIN a split.
        for table, key in ((parents, r["parent_id"]), (prompts, " ".join(r["prompt"].split()))):
            if key in table and table[key] != r["split"]:
                raise ValueError("TRAIN/DEV parent or normalized-prompt leakage")
            table[key] = r["split"]
        if not score(r["reference"], r["answer"], r["scorer"]):
            raise ValueError("Reference fails its declared scorer")
    selected = [r for r in rows if r["task"] == task and r["purpose"] == "acquisition"]
    if not any(r["split"] == "train" for r in selected) or not any(r["split"] == "dev" for r in selected):
        raise ValueError("Selected adaptation task needs TRAIN and independent DEV")
    retention = [r for r in rows if r["purpose"] == "retention"]
    if not retention:
        raise ValueError("An independent retention cohort is required")
    # Identical prompts with incompatible targets cannot be retained and acquired together.
    if {r["prompt"] for r in selected} & {r["prompt"] for r in retention}:
        raise ValueError("Retention cannot reuse the same acquisition prompt")
    return selected + retention


def load_rows(path: Path, task: str):
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Dataset exceeds 4 MiB micro-pilot budget")
    return validate_rows([json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()], task)


def messages(row, privileged=False):
    # Exactly the original student prompt; no answer, hidden convention or reference appended.
    text = TEACHER_TEMPLATE.format(prompt=row["prompt"], reference=row["reference"]) if privileged else row["prompt"]
    return [{"role": "user", "content": text}]


def base9(number):
    sign = "-" if number < 0 else ""
    number = abs(number)
    digits = ""
    while number:
        number, rem = divmod(number, 9)
        digits = str(rem) + digits
    return sign + (digits or "0")


def synthetic_rows(seed=17, parents_per_split=12):
    """Fresh small diagnostic, NOT the paper's original corpora or broad benchmarks.

    Ordinary and clockwise-rotated spatial tasks have identical prompts/parents.
    Select one task per run. Hidden contradictory rules are diagnostic-only.
    """
    if not 2 <= parents_per_split <= 60:
        raise ValueError("Choose 2--60 parents per split")
    rng, rows, seen = random.Random(seed), [], set()
    directions = {"north": (0, 1), "east": (1, 0), "south": (0, -1), "west": (-1, 0)}
    for split in ("train", "dev"):
        count = 0
        while count < parents_per_split:
            anchor = (rng.randrange(-9, 10), rng.randrange(-9, 10))
            steps = [(rng.choice(list(directions)), rng.randrange(1, 5)) for _ in range(rng.randrange(1, 4))]
            dx = sum(directions[d][0] * n for d, n in steps)
            dy = sum(directions[d][1] * n for d, n in steps)
            signature = json.dumps([anchor, steps])
            if (dx == 0 and dy == 0) or signature in seen:
                continue
            seen.add(signature)
            pid = hashlib.sha256(signature.encode()).hexdigest()
            prompt = f"A marker starts at {anchor}. " + " Then ".join(f"move {n} cells {d}." for d, n in steps)
            prompt += " Give its final coordinates as a boxed integer pair."
            for task, delta in (("spatial_ordinary", (dx, dy)), ("spatial_rotated", (dy, -dx))):
                answer = str((anchor[0] + delta[0], anchor[1] + delta[1]))
                rows.append({"id": f"{task}-{pid}", "parent_id": pid, "task": task,
                             "split": split, "purpose": "acquisition", "source": "DSD-001 fresh synthetic diagnostic",
                             "admitted_for": split, "prompt": prompt, "reference": f"\\boxed{{{answer}}}",
                             "answer": answer, "scorer": "boxed_pair"})
            count += 1
    for i in range(parents_per_split):
        a, b = 1000 + 13 * i, 43 + 7 * i
        answer = str(a + b)
        rows.append({"id": f"retention-{i}", "parent_id": f"fresh-arithmetic-{i}", "task": "fixture_arithmetic",
                     "split": "dev", "purpose": "retention", "source": "DSD-001 narrow retention diagnostic",
                     "admitted_for": "dev", "prompt": f"Compute {a} + {b}. Give a boxed integer.",
                     "reference": f"\\boxed{{{answer}}}", "answer": answer, "scorer": "boxed_int"})
    return rows
