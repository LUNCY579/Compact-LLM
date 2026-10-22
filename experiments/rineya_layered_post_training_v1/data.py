"""Local JSONL only. Gold never enters the generation API; confirmation is separate."""
from __future__ import annotations

import json
import re
import random
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import canonical, digest, require


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


@dataclass(frozen=True)
class Row:
    id: str
    group: str
    split: str
    suite: str
    prompt: str
    target: str
    source: str
    revision: str
    rejected: str | None = None
    verifier: str = "exact"
    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()

    def __post_init__(self):
        for key in ("id", "group", "suite", "prompt", "target", "source", "revision"):
            require(isinstance(getattr(self, key), str) and bool(getattr(self, key).strip()), f"missing {key}")
        require(self.split in {"train", "select", "confirm"}, "unknown split")
        require(self.verifier in {"exact", "json_equal", "constraints"}, "unregistered verifier")
        require(self.rejected is None or (isinstance(self.rejected, str) and normalize(self.rejected) != normalize(self.target)), "invalid preference pair")
        require(all(isinstance(x, str) and x for x in (*self.must_contain, *self.must_not_contain)), "invalid textual constraints")
        require(self.verifier != "constraints" or bool(self.must_contain or self.must_not_contain), "empty constraints verifier")
        require(score(self, self.target) == 1., "teacher target fails declared verifier")


def score(row: Row, text: str, terminated: bool = True) -> float:
    if not terminated:
        return 0.
    if row.verifier == "exact":
        return float(normalize(text) == normalize(row.target))
    if row.verifier == "json_equal":
        try:
            return float(canonical(json.loads(text)) == canonical(json.loads(row.target)))
        except (ValueError, TypeError):
            return 0.
    return float(all(x in text for x in row.must_contain) and not any(x in text for x in row.must_not_contain))


def validate(rows):
    require(bool(rows), "empty data")
    ids, groups, prompts = set(), {}, {}
    for row in rows:
        require(row.id not in ids, f"duplicate row id: {row.id}")
        ids.add(row.id)
        for table, key in ((groups, row.group), (prompts, digest(normalize(row.prompt)))):
            require(key not in table or table[key] == row.split, "cross-split group/prompt leakage")
            table[key] = row.split
    return rows


def load_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            for key in ("must_contain", "must_not_contain"):
                if key in value:
                    value[key] = tuple(value[key])
            rows.append(Row(**value))
    return validate(rows)


def fingerprint(rows):
    return digest([asdict(x) for x in sorted(rows, key=lambda r: r.id)])


def partition(rows, split):
    return [r for r in rows if r.split == split]


def fixtures():
    """Generated microtasks, NEVER a Rineya benchmark or generalism certification."""
    rows = []
    for suite in ("acquisition", "retention"):
        for a in range(5):
            for b in range(5):
                n = a * 5 + b
                split = ("train", "select", "confirm")[min(n // 8, 2)]
                target = (a + b) % 5 if suite == "acquisition" else a
                op = "sum" if suite == "acquisition" else "keep"
                key = f"{suite}-{a}-{b}"
                rows.append(Row(key, key, split, suite, f"{op} {a} {b}", str(target), "synthetic-mod5", "v1", str((target + 1) % 5)))
    return validate(rows)


def balanced_order(rows, seed):
    """Round-robin suites so a short replay stage does not see only the first suite."""
    buckets = {}
    for row in rows:
        buckets.setdefault(row.suite, []).append(row)
    for suite, values in buckets.items():
        values.sort(key=lambda r: r.id)
        random.Random(int(digest([seed, suite])[:12], 16)).shuffle(values)
    ordered = []
    for index in range(max((len(v) for v in buckets.values()), default=0)):
        for suite in sorted(buckets):
            if index < len(buckets[suite]):
                ordered.append(buckets[suite][index])
    return ordered
