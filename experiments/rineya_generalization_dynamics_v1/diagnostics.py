"""Original, deterministic paired diagnostics. No GDsuite code/data are copied.

Confirmation uses held-out rendering AND operation families. Public generators are
not cryptographically secret tests; operators must keep confirmation out of tuning.
"""
from __future__ import annotations

import random
from collections import defaultdict
from .common import digest, require

FAMILIES = ("successive", "repetitive", "retrieve_R", "transform_S", "compose_RS", "retention")


def build_suite(split: str, seed: int, per_family: int = 24) -> list[dict]:
    require(split in {"dev", "confirm"}, "Unknown split")
    require(type(seed) is int and seed >= 0, "Seed must be a nonnegative integer")
    require(type(per_family) is int and 2 <= per_family <= 1000, "Invalid suite size")
    rng = random.Random(f"GD001-v1:{split}:{seed}")
    rows = []
    for family in FAMILIES:
        for index in range(per_family):
            # Non-overlapping entity namespaces and mathematical ranges by split.
            offset = 100 if split == "dev" else 600
            x = offset + index * 11 + rng.randrange(1, 9)
            k = rng.randrange(2, 9)
            skill_case = f"gd001:{split}:{seed}:skill:{index}" if family in {"retrieve_R", "transform_S", "compose_RS"} else None
            if skill_case:
                linked_rng = random.Random(skill_case)
                x, k = offset + index * 11 + linked_rng.randrange(1, 9), linked_rng.randrange(2, 9)
            template = f"{split}:{family}:v1"
            parent = f"gd001:{split}:{seed}:{family}:{index}"
            if family in {"successive", "repetitive"}:
                answer = x + k if split == "dev" else x - k
                query = ("Compute: " if family == "successive" else "Evaluate: ") + f"{x} {'+' if split == 'dev' else '-'} {k} = ?"
                start = rng.randrange(10, 30)
                distractor = start + 3 if family == "successive" else start
                misleading_targets = [start + j for j in range(3)] if family == "successive" else [start] * 3
                clean_targets = [start + 4, start + 1, start + 7]
                def demos(targets):
                    if split == "dev":
                        return "\n".join(f"{v-2} + 2 = {v}" for v in targets)
                    return "\n".join(f"{v+3} - 3 = {v}" for v in targets)
                clean = demos(clean_targets) + "\n" + query
                misleading = demos(misleading_targets) + "\n" + query
            elif family in {"retrieve_R", "transform_S", "compose_RS"}:
                name = f"item_{split}_{index}"
                other = x + 17
                operation = f"add {k}" if split == "dev" else f"subtract {k}"
                transformed = x + k if split == "dev" else x - k
                distractor = other if family == "retrieve_R" else (other + k if split == "dev" else other - k)
                if family == "transform_S":
                    answer = transformed
                    clean = f"Input: {x}. Rule: {operation}. What is the result?"
                    misleading = f"Worked example: input {other}; rule {operation}; result {distractor}.\n" + clean
                    # Matched number of demonstrations; only their answer differs.
                    z = other + 19
                    clean = f"Worked example: input {z}; rule {operation}; result {z+k if split == 'dev' else z-k}.\n" + clean
                else:
                    answer = x if family == "retrieve_R" else transformed
                    # Evidence and distractor positions alternate, including split shift.
                    record = f"ACTIVE record: {name} has value {x}."
                    neutral = f"ARCHIVED record: other_{name} has value {other}."
                    trap = f"ARCHIVED record: {name} has value {other}."
                    if split == "confirm":
                        record = f"{name} -> {x} [CURRENT]"
                        neutral = f"other_{name} -> {other} [RETIRED]"
                        trap = f"{name} -> {other} [RETIRED]"
                    order = index % 2 == (0 if split == "dev" else 1)
                    def context(extra):
                        return "\n".join((record, extra) if order else (extra, record))
                    question = (f"Use only the ACTIVE record. What is {name}'s value?" if family == "retrieve_R"
                                else f"Use only the ACTIVE record for {name}, then {operation}. What is the result?")
                    if split == "confirm":
                        question = (f"Read the CURRENT entry for {name}. What number does it contain?" if family == "retrieve_R"
                                    else f"Read the CURRENT entry for {name} and {operation} from that number. What number results?")
                    clean, misleading = context(neutral) + "\n" + question, context(trap) + "\n" + question
            else:
                # A separate fresh probe, not a substitute for broad historical retention.
                answer, distractor = x, x + k
                clean = f"Authoritative code: {x}. Unrelated number: {distractor}. Return the authoritative code."
                misleading = f"Old suggested code: {distractor}. Authoritative code: {x}. Return the authoritative code."
                if split == "confirm":
                    clean = f"The verified serial is {x}; {distractor} is an unverified serial. Give the verified serial."
                    misleading = f"Someone proposed serial {distractor}. Verification rejected it and confirmed {x}. Give the verified serial."
            for condition, body in (("clean", clean), ("misleading", misleading)):
                prompt = "Return only the integer answer, without explanation.\n" + body + "\nAnswer:"
                rows.append({"id": f"{parent}:{condition}", "parent_id": parent,
                             "split": split, "family": family, "template_id": template,
                             "cluster_id": skill_case or parent, "skill_case_id": skill_case,
                             "condition": condition, "prompt": prompt,
                             "answer": str(answer), "distractor": str(distractor)})
    validate_suite(rows, split)
    return rows


def validate_suite(rows: list[dict], split: str) -> None:
    require(bool(rows), "Empty suite")
    require(len({r["id"] for r in rows}) == len(rows), "Duplicate row ID")
    require(len({r["prompt"] for r in rows}) == len(rows), "Duplicate prompt")
    pairs = defaultdict(list)
    for row in rows:
        require(row["split"] == split, "Mixed/forbidden split")
        require(row["family"] in FAMILIES, "Unknown family")
        require(row["template_id"].startswith(split + ":"), "Cross-split template")
        require(row["answer"] != row["distractor"], "Gold equals shortcut")
        require(row["answer"].lstrip("-").isdigit() and row["distractor"].lstrip("-").isdigit(), "Noninteger candidate")
        pairs[row["parent_id"]].append(row)
    require({r["family"] for r in rows} == set(FAMILIES), "Missing task family")
    for pair in pairs.values():
        require(len(pair) == 2 and {r["condition"] for r in pair} == {"clean", "misleading"}, "Incomplete pair")
        for key in ("family", "template_id", "answer", "distractor", "cluster_id", "skill_case_id"):
            require(pair[0][key] == pair[1][key], f"Pair changed {key}")


def assert_disjoint(left: list[dict], right: list[dict]) -> None:
    for key in ("id", "parent_id", "cluster_id", "template_id", "prompt"):
        require(not ({r[key] for r in left} & {r[key] for r in right}), f"Split overlap: {key}")


def suite_hash(rows: list[dict]) -> str:
    return digest(rows)
