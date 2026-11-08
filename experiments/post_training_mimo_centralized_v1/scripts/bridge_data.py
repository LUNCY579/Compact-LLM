"""Original deterministic E1 bridge data; preparation is not training admission.

Allocate canonical parents before rendering or solving. This module deliberately
has no API for opening CHECK or FINAL: their exports contain identity hashes only.
Run with a frozen FAST-lane registration; no model, tokenizer or network is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

VERSION = "rineya-e1-bridge-v1"
VERIFIER_VERSION = "rineya-e1-independent-verifier-v1"
FAMILIES = (
    "bounded_integer_arithmetic", "symbolic_variable_updates",
    "finite_set_queries", "constrained_text_transformations",
)
ROLES = ("TRAIN", "TRAIN_SELECTION", "DEV", "CHECK", "FINAL")
OPEN_ROLES = frozenset(ROLES[:3])
ROLE_BUCKETS = ((7000, "TRAIN"), (7800, "TRAIN_SELECTION"), (8600, "DEV"),
                (9400, "CHECK"), (10000, "FINAL"))
ROLE_DOMAIN = b"rineya-e1-parent-role-v1\0"
ANSWER_LIMIT = 100000
WORDS = ("amber", "birch", "cedar", "dawn", "ember", "fern", "glen", "harbor",
         "iris", "jade", "kelp", "linen", "moss", "north", "opal", "pine")


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _keys(value: Any, expected: set[str]) -> None:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"expected object with keys {sorted(expected)}")


def _integer(value: Any, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"expected integer in [{low}, {high}]")
    return value


def _normalize_tree(node: Any, sets: bool, depth: int = 0) -> Any:
    if depth > 4:
        raise ValueError("expression depth exceeds contract")
    if sets and type(node) is list:
        if len(node) > 16:
            raise ValueError("set too large")
        return sorted({_integer(x, -30, 30) for x in node})
    if not sets and type(node) is int:
        return _integer(node, -50, 50)
    _keys(node, {"op", "args"})
    allowed = {"union", "intersection", "difference"} if sets else {"add", "sub", "mul"}
    if node["op"] not in allowed or type(node["args"]) is not list or len(node["args"]) != 2:
        raise ValueError("invalid binary operation")
    args = [_normalize_tree(x, sets, depth + 1) for x in node["args"]]
    if node["op"] in {"add", "mul", "union", "intersection"}:
        args.sort(key=json_bytes)
    return {"op": node["op"], "args": args}


def canonicalize(spec: dict) -> dict:
    """Normalize the declared equivalences, not all mathematical equivalences."""
    if type(spec) is not dict or spec.get("family") not in FAMILIES:
        raise ValueError("unknown family")
    family = spec["family"]
    if family in FAMILIES[:1] + FAMILIES[2:3]:
        _keys(spec, {"family", "expression"})
        return {"family": family, "expression": _normalize_tree(spec["expression"], family == FAMILIES[2])}
    if family == FAMILIES[1]:
        _keys(spec, {"family", "registers", "program"})
        regs, program = spec["registers"], spec["program"]
        if type(regs) is not list or len(regs) != 3 or type(program) is not list or not 1 <= len(program) <= 6:
            raise ValueError("invalid register program")
        regs = [_integer(x, -9, 9) for x in regs]
        instructions = []
        for ins in program:
            _keys(ins, {"op", "dst", "src"})
            if ins["op"] not in {"add", "sub", "copy"}:
                raise ValueError("invalid register operation")
            instructions.append({"op": ins["op"], "dst": _integer(ins["dst"], 0, 2),
                                 "src": _integer(ins["src"], 0, 2)})
        return {"family": family, "registers": regs, "program": instructions}
    _keys(spec, {"family", "tokens", "program", "separator"})
    tokens, program = spec["tokens"], spec["program"]
    if (type(tokens) is not list or not 1 <= len(tokens) <= 12
            or any(type(x) is not str or not re.fullmatch("[a-z]{1,12}", x) for x in tokens)
            or type(program) is not list or not 1 <= len(program) <= 6
            or spec["separator"] not in {"|", ",", "/"}):
        raise ValueError("invalid text program")
    instructions = []
    for ins in program:
        if type(ins) is not dict:
            raise ValueError("invalid text instruction")
        op = ins.get("op")
        if op in {"reverse", "sort", "upper"}:
            _keys(ins, {"op"})
            instructions.append({"op": op})
        elif op in {"rotate", "take", "drop"}:
            _keys(ins, {"op", "n"})
            instructions.append({"op": op, "n": _integer(ins["n"], 0, 12)})
        else:
            raise ValueError("invalid text operation")
    return {"family": family, "tokens": list(tokens), "program": instructions, "separator": spec["separator"]}


def _shape(spec: dict) -> str:
    def tree(node):
        if type(node) is not dict:
            return "leaf"
        children = [tree(x) for x in node["args"]]
        if node["op"] in {"add", "mul", "union", "intersection"}:
            children.sort()
        return node["op"] + "(" + ",".join(children) + ")"
    if "expression" in spec:
        return tree(spec["expression"])
    return ">".join(ins["op"] for ins in spec["program"])


@dataclass(frozen=True)
class AllocatedParent:
    canonical_json: str
    canonical_parent_sha256: str
    parent_id: str
    split_sha256: str
    role: str
    structure_id: str
    generator_seed: int
    candidate_index: int


def allocate_parent(spec: dict, seed: int = 20260927, candidate_index: int = 0) -> AllocatedParent:
    """The sole allocator; called before any render/solve/score operation."""
    _integer(seed, 0, 2**63 - 1)
    _integer(candidate_index, 0, 2**63 - 1)
    raw = json_bytes(canonicalize(spec))
    identity, split = sha256(raw), sha256(ROLE_DOMAIN + raw)
    bucket = int(split[:16], 16) % 10000
    role = next(role for upper, role in ROLE_BUCKETS if bucket < upper)
    return AllocatedParent(raw.decode("utf-8"), identity, "bridge-" + identity, split, role,
                           _shape(json.loads(raw)), seed, candidate_index)


def _validated_spec(parent: AllocatedParent, require_open: bool = True) -> dict:
    spec = json.loads(parent.canonical_json)
    if allocate_parent(spec, parent.generator_seed, parent.candidate_index) != parent:
        raise ValueError("parent identity, allocation or canonical metadata mismatch")
    if require_open and parent.role not in OPEN_ROLES:
        raise PermissionError("CHECK and FINAL remain sealed; rendering/solving/verifying is forbidden")
    return spec


def reference_answer(parent: AllocatedParent) -> Any:
    """Reference execution. The verifier below uses a separate implementation."""
    spec = _validated_spec(parent)
    family = spec["family"]
    if family == FAMILIES[0]:
        def arithmetic(node):
            if type(node) is int:
                return node
            a, b = map(arithmetic, node["args"])
            value = {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b}[node["op"]]()
            return _integer(value, -ANSWER_LIMIT, ANSWER_LIMIT)
        return arithmetic(spec["expression"])
    if family == FAMILIES[1]:
        regs = list(spec["registers"])
        for ins in spec["program"]:
            a, b = regs[ins["dst"]], regs[ins["src"]]
            regs[ins["dst"]] = {"add": a + b, "sub": a - b, "copy": b}[ins["op"]]
        return regs
    if family == FAMILIES[2]:
        def finite(node):
            if type(node) is list:
                return set(node)
            a, b = map(finite, node["args"])
            return {"union": lambda: a | b, "intersection": lambda: a & b,
                    "difference": lambda: a - b}[node["op"]]()
        return sorted(finite(spec["expression"]))
    words = list(spec["tokens"])
    for ins in spec["program"]:
        op, n = ins["op"], ins.get("n", 0)
        if op == "reverse":
            words.reverse()
        elif op == "sort":
            words.sort()
        elif op == "upper":
            words = [word.upper() for word in words]
        elif op == "take":
            words = words[:n]
        elif op == "drop":
            words = words[n:]
        elif words:
            n %= len(words)
            words = words[n:] + words[:n]
    return spec["separator"].join(words)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def parse_answer(response: str) -> Any:
    if type(response) is not str or len(response) > 4096:
        raise ValueError("answer must be a bounded JSON string")
    parsed = json.loads(response, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    _keys(parsed, {"answer"})
    return parsed["answer"]


def verify_answer(parent: AllocatedParent, response: str) -> bool:
    """Strict JSON and independent semantics; corrupt host metadata raises.

    Arithmetic uses a stack machine, registers use immutable state transitions,
    sets use per-element membership predicates, and text uses indexed emission.
    No reference_answer call or cached reference target is trusted here.
    """
    spec = _validated_spec(parent)  # Outside parser exception handling by design.
    try:
        answer = parse_answer(response)
    except (ValueError, TypeError, RecursionError):
        return False
    family = spec["family"]
    if family == FAMILIES[0]:
        if type(answer) is not int:
            return False
        todo, stack = [(spec["expression"], False)], []
        while todo:
            node, visited = todo.pop()
            if type(node) is int:
                stack.append(node)
            elif not visited:
                todo.extend([(node, True), (node["args"][1], False), (node["args"][0], False)])
            else:
                right, left = stack.pop(), stack.pop()
                if node["op"] == "add":
                    result = left + right
                elif node["op"] == "sub":
                    result = left - right
                else:
                    result = left * right
                _integer(result, -ANSWER_LIMIT, ANSWER_LIMIT)
                stack.append(result)
        return answer == stack[0]
    if family == FAMILIES[1]:
        if type(answer) is not list or len(answer) != 3 or any(type(x) is not int for x in answer):
            return False
        state = tuple(spec["registers"])
        for ins in spec["program"]:
            src, dst, op = ins["src"], ins["dst"], ins["op"]
            value = state[src]
            if op == "add":
                value = state[dst] + value
            elif op == "sub":
                value = state[dst] - value
            state = tuple(value if i == dst else state[i] for i in range(3))
        return tuple(answer) == state
    if family == FAMILIES[2]:
        if (type(answer) is not list or any(type(x) is not int for x in answer)
                or answer != sorted(set(answer))):
            return False
        def contains(node, item):
            if type(node) is list:
                return any(item == x for x in node)
            left = contains(node["args"][0], item)
            right = contains(node["args"][1], item)
            if node["op"] == "union":
                return left or right
            if node["op"] == "intersection":
                return left and right
            return left and not right
        expected = [x for x in range(-30, 31) if contains(spec["expression"], x)]
        return answer == expected
    if type(answer) is not str:
        return False
    state = tuple(spec["tokens"])
    for ins in spec["program"]:
        op, n = ins["op"], ins.get("n", 0)
        if op == "reverse":
            state = tuple(state[len(state) - 1 - i] for i in range(len(state)))
        elif op == "sort":
            state = tuple(sorted(state))
        elif op == "upper":
            state = tuple("".join(chr(ord(c) - 32) if "a" <= c <= "z" else c for c in w) for w in state)
        elif op == "take":
            state = tuple(w for i, w in enumerate(state) if i < n)
        elif op == "drop":
            state = tuple(w for i, w in enumerate(state) if i >= n)
        elif state:
            state = tuple(state[(i + n) % len(state)] for i in range(len(state)))
    expected = ""
    for i, word in enumerate(state):
        expected += (spec["separator"] if i else "") + word
    return answer == expected


def render_prompt(parent: AllocatedParent, variant: int = 0) -> str:
    spec = _validated_spec(parent)
    _integer(variant, 0, 1)
    prefix = ("Solve the following task.", "Follow these instructions exactly.")[variant]
    family = spec["family"]
    if family == FAMILIES[0]:
        def expression(node):
            if type(node) is int:
                return str(node)
            a, b = map(expression, node["args"])
            return f"({a} {dict(add='+', sub='-', mul='*')[node['op']]} {b})"
        body = "Compute using integer arithmetic: " + expression(spec["expression"]) + ". Answer with an integer."
    elif family == FAMILIES[1]:
        names = ("A", "B", "C") if variant == 0 else ("X", "Y", "Z")
        body = "Registers: " + ", ".join(f"{name}={x}" for name, x in zip(names, spec["registers"])) + ".\n"
        body += "Apply updates in order, using current values at each step:\n"
        for i, ins in enumerate(spec["program"], 1):
            dst, src = names[ins["dst"]], names[ins["src"]]
            rhs = src if ins["op"] == "copy" else dst + (" + " if ins["op"] == "add" else " - ") + src
            body += f"{i}. {dst} = {rhs}\n"
        body += "Answer with the final integer list in order [" + ",".join(names) + "]."
    elif family == FAMILIES[2]:
        def expression(node):
            if type(node) is list:
                return "{" + ",".join(map(str, node)) + "}"
            a, b = map(expression, node["args"])
            return f"{node['op']}({a}, {b})"
        body = ("union includes elements in either set; intersection includes elements in both; "
                "difference(left,right) includes elements in left but not right.\nCompute: "
                + expression(spec["expression"]) + ". Answer with a sorted distinct integer list.")
    else:
        body = "Start with this token list: " + json.dumps(spec["tokens"]) + ".\nApply in order:\n"
        for i, ins in enumerate(spec["program"], 1):
            op, n = ins["op"], ins.get("n", 0)
            instruction = {"reverse": "Reverse token order", "sort": "Sort tokens in ASCII lexicographic order",
                           "upper": "Uppercase every ASCII letter", "rotate": f"Rotate tokens left by {n} positions, wrapping around (empty stays empty)",
                           "take": f"Keep only the first {n} tokens (or all if fewer)",
                           "drop": f"Remove the first {n} tokens (or all if fewer)"}[op]
            body += f"{i}. {instruction}.\n"
        body += "Join the remaining tokens with " + json.dumps(spec["separator"]) + " and no spaces. Answer with that exact string."
    return prefix + "\n" + body + '\nReturn one JSON object with exactly the key "answer". No rationale or extra text.'


def _op(name, a, b):
    return {"op": name, "args": [a, b]}


def generate_spec(family: str, seed: int, index: int, catalog: str = "familiar") -> dict:
    """Sample only the latent specification; no answer or surface text exists yet."""
    if family not in FAMILIES or catalog not in {"familiar", "check_composition", "final_composition"}:
        raise ValueError("invalid family/catalog")
    _integer(seed, 0, 2**63 - 1)
    _integer(index, 0, 2**63 - 1)
    rng = random.Random(int(sha256(json_bytes([VERSION, family, seed, index, catalog])), 16))
    choice = index % 4 if catalog == "familiar" else (4 if catalog == "check_composition" else 6) + index % 2
    if family == FAMILIES[0]:
        a, b, c, d = [rng.randint(-20, 20) for _ in range(4)]
        catalog_expr = [
            _op("add", _op("sub", a, b), c), _op("mul", a, _op("add", b, c)),
            _op("sub", _op("mul", a, b), c), _op("add", _op("mul", a, b), c),
            _op("sub", _op("add", a, b), _op("mul", c, d)),
            _op("mul", _op("sub", a, b), _op("add", c, d)),
            _op("sub", _op("mul", a, _op("sub", b, c)), d),
            _op("add", _op("mul", _op("add", a, b), c), d),
        ]
        return {"family": family, "expression": catalog_expr[choice]}
    if family == FAMILIES[1]:
        sequences = [("add", "copy"), ("sub", "add"), ("copy", "sub"), ("add", "sub"),
                     ("copy", "add", "sub"), ("sub", "copy", "add"),
                     ("add", "copy", "sub", "add"), ("copy", "sub", "add", "copy")]
        return {"family": family, "registers": [rng.randint(-9, 9) for _ in range(3)],
                "program": [{"op": op, "dst": rng.randrange(3), "src": rng.randrange(3)} for op in sequences[choice]]}
    if family == FAMILIES[2]:
        a, b, c, d = [rng.sample(range(-15, 16), rng.randint(3, 7)) for _ in range(4)]
        catalog_expr = [
            _op("union", a, b), _op("intersection", a, b), _op("difference", a, b),
            _op("union", _op("intersection", a, b), c),
            _op("difference", _op("union", a, b), c),
            _op("intersection", _op("union", a, b), c),
            _op("difference", _op("union", a, b), _op("intersection", c, d)),
            _op("union", _op("difference", a, b), _op("intersection", c, d)),
        ]
        return {"family": family, "expression": catalog_expr[choice]}
    sequences = [("reverse", "take"), ("rotate", "upper"), ("drop", "reverse"), ("sort", "take"),
                 ("rotate", "reverse", "take"), ("sort", "drop", "upper"),
                 ("drop", "rotate", "reverse", "upper"), ("reverse", "sort", "drop", "rotate")]
    return {"family": family, "tokens": [rng.choice(WORDS) for _ in range(rng.randint(5, 9))],
            "program": [{"op": op, **({"n": rng.randint(1, 5)} if op in {"rotate", "take", "drop"} else {})}
                        for op in sequences[choice]], "separator": rng.choice(("|", ",", "/"))}


def wrong_answer(answer: Any) -> Any:
    """One deterministic semantic mutation per generated positive control."""
    if type(answer) is int:
        return answer + 1
    if type(answer) is list:
        return [answer[0] + 1000, *answer[1:]] if answer else [1000]
    return answer + "!"


def make_row(parent: AllocatedParent, source_revision: str, variant: int = 0) -> dict:
    spec = _validated_spec(parent)
    if not re.fullmatch("[0-9a-f]{40}", source_revision):
        raise ValueError("source revision must be a full lowercase Git SHA")
    prompt = render_prompt(parent, variant)
    answer = reference_answer(parent)
    target = json_bytes({"answer": answer}).decode("utf-8")
    if not verify_answer(parent, target) or verify_answer(parent, json_bytes({"answer": wrong_answer(answer)}).decode("utf-8")):
        raise RuntimeError("reference/verifier positive or mutation-control failure")
    return {
        "schema_version": 1, "row_id": parent.parent_id + f"-v{variant}", "parent_id": parent.parent_id,
        "source_id": "original-rineya-e1-bridge", "source_revision": source_revision,
        "source_revision_verification": "CALLER_SUPPLIED_GIT_SHA; exact generator bytes bound by receipt",
        "family": spec["family"], "structure_id": parent.structure_id, "role": parent.role,
        "variant_id": variant, "canonical_parent_sha256": parent.canonical_parent_sha256,
        "split_sha256": parent.split_sha256, "spec": spec, "prompt": prompt,
        "prompt_sha256": sha256(prompt.encode("utf-8")),
        "normalized_prompt_sha256": sha256(" ".join(prompt.split()).encode("utf-8")),
        "prompt_normalization": "collapse Unicode whitespace only",
        "tokenized_prompt_sha256": None, "tokenizer_identity": None, "tokenizer_sha256": None,
        "chat_template_sha256": None, "tokenization_status": "UNMEASURED",
        "prompt_tokens": None, "target_tokens": None, "total_nonpadding_tokens": None,
        "prompt_characters": len(prompt), "target": target, "target_sha256": sha256(target.encode("utf-8")),
        "verifier_version": VERIFIER_VERSION, "verifier_result": True, "mutation_control_rejected": True,
        "generator_seed": parent.generator_seed, "candidate_index": parent.candidate_index,
        "generation_attempts": 1, "attempt_number": 1, "policy_or_teacher_identity": "DETERMINISTIC_REFERENCE_NO_MODEL",
        "policy_version": None, "behavior_logprobs": None,
        "environment_identity": VERSION, "environment_revision": source_revision,
        "tool_actions": [], "tool_outputs": [], "reward": 1, "termination_reason": "verified_exact_answer",
        "cost": {"model_calls": 0, "teacher_calls": 0, "model_generation_tokens": 0,
                 "verifier_calls": 2, "training_tokens": None, "tokenizer_tokens_status": "UNMEASURED"},
        "training_role_eligible": parent.role == "TRAIN", "training_eligibility": False,
        "training_eligibility_reasons": ["TOKENIZER_CONTEXT_FIT_UNMEASURED", "HISTORICAL_EXCLUSION_UNMEASURED",
                                         "NATIVE_AND_COMPLETE_E1_DATA_GATES_NOT_QUALIFIED"],
        "license_provenance": {"origin": "original deterministic task and prompt generated locally for the user project",
                               "external_dataset_or_teacher": False, "redistribution_license": "UNRESOLVED_NO_NEW_LICENSE_GRANT",
                               "code_notice_policy": "retain repository source notices"},
    }


def validate_registration(registration: dict) -> dict[str, int]:
    if registration.get("status") != "FROZEN_FAST_LANE_DATA_ONLY":
        raise ValueError("registration must be FROZEN_FAST_LANE_DATA_ONLY")
    authorization = registration.get("authorization", {})
    if (authorization.get("dataset_artifact_generation") is not True
            or authorization.get("check_rendering") is not False
            or authorization.get("final_rendering") is not False):
        raise ValueError("registration must authorize data preparation and keep CHECK/FINAL closed")
    if registration.get("families") != list(FAMILIES):
        raise ValueError("registration family catalog mismatch")
    expected_buckets = {role: [0 if i == 0 else ROLE_BUCKETS[i - 1][0], high]
                        for i, (high, role) in enumerate(ROLE_BUCKETS)}
    declared_buckets = registration.get("parent_policy", {}).get("buckets")
    if declared_buckets is not None and declared_buckets != expected_buckets:
        raise ValueError("registered role buckets differ from implementation")
    _integer(registration.get("selection_seed"), 0, 2**63 - 1)
    counts = {}
    for role in ROLES:
        count = registration.get("counts", {}).get(role, {}).get("acquisition_parents")
        count = 0 if role == "FINAL" and count is None else count
        counts[role] = _integer(count, 0, 100000)
        if count % (8 if role == "CHECK" else 4):
            raise ValueError("counts must balance families and CHECK structure strata")
        declared_per_family = registration["counts"][role].get("per_family")
        if declared_per_family is not None and declared_per_family != count // 4:
            raise ValueError("per_family count mismatch")
    if sum(counts.values()) == 0:
        raise ValueError("empty registration")
    return counts


def prepare_parents(counts: dict[str, int], seed: int) -> tuple[list[AllocatedParent], dict]:
    """Select exact quotas by fixed candidate order, never by answer/model quality."""
    if set(counts) != set(ROLES):
        raise ValueError("all five roles required")
    for role, count in counts.items():
        _integer(count, 0, 100000)
        if count % (8 if role == "CHECK" else 4):
            raise ValueError("unbalanced requested counts")
    selected, seen, attempts, duplicate_candidates = [], set(), 0, 0
    quotas = {}
    for family in FAMILIES:
        for role in ROLES:
            if role == "CHECK":
                quotas[(family, role, "familiar")] = counts[role] // 8
                quotas[(family, role, "check_composition")] = counts[role] // 8
            else:
                quotas[(family, role, "final_composition" if role == "FINAL" else "familiar")] = counts[role] // 4
    for family in FAMILIES:
        for catalog in ("familiar", "check_composition", "final_composition"):
            keys = [key for key in quotas if key[0] == family and key[2] == catalog]
            wanted = sum(quotas[key] for key in keys)
            index, ceiling = 0, max(10000, wanted * 1000)
            while any(quotas[key] for key in keys):
                if index >= ceiling:
                    raise RuntimeError("deterministic candidate ceiling exhausted; do not silently change seed")
                parent = allocate_parent(generate_spec(family, seed, index, catalog), seed, index)
                index += 1
                attempts += 1
                key = (family, parent.role, catalog)
                if quotas.get(key, 0) <= 0:
                    continue
                if parent.canonical_parent_sha256 in seen:
                    duplicate_candidates += 1
                    continue
                seen.add(parent.canonical_parent_sha256)
                selected.append(parent)
                quotas[key] -= 1
    return selected, {"candidate_specifications": attempts, "duplicate_candidates_skipped": duplicate_candidates,
                      "selection_basis": "fixed generator order, role and structure quotas; no solver/model scoring"}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("xb") as handle:
        for row in rows:
            handle.write(json_bytes(row) + b"\n")


def build_dataset(registration_path: Path, output: Path, source_revision: str, variants: int = 1) -> dict:
    """Write once to a new directory after every row passes semantic controls."""
    _integer(variants, 1, 2)
    if not re.fullmatch("[0-9a-f]{40}", source_revision):
        raise ValueError("source revision must be a full lowercase Git SHA")
    started = time.perf_counter()
    registration_raw = registration_path.read_bytes()
    registration = json.loads(registration_raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    counts = validate_registration(registration)
    if output.exists():
        raise FileExistsError("export is write-once; output directory already exists")
    parents, selection = prepare_parents(counts, registration["selection_seed"])
    rows = {role: [] for role in OPEN_ROLES}
    reservations, normalized_seen, prompt_seen = [], {}, {}
    for parent in parents:
        if parent.role not in OPEN_ROLES:
            # Intentionally no latent specification, structure tree, prompt, target or verifier call.
            reservations.append({"parent_id": parent.parent_id, "role": parent.role,
                                 "canonical_parent_sha256": parent.canonical_parent_sha256,
                                 "split_sha256": parent.split_sha256,
                                 "status": "HASH_ONLY_UNOPENED"})
            continue
        for variant in range(variants):
            row = make_row(parent, source_revision, variant)
            for key, inventory in (("normalized_prompt_sha256", normalized_seen), ("prompt_sha256", prompt_seen)):
                previous = inventory.setdefault(row[key], parent.parent_id)
                if previous != parent.parent_id:
                    raise RuntimeError("prompt collision across parents; no silent substitution permitted")
            rows[parent.role].append(row)
    output.mkdir(parents=True, exist_ok=False)
    (output / "REGISTRATION.json").write_bytes(registration_raw)
    for role in ROLES[:3]:
        _write_jsonl(output / f"{role}.jsonl", rows[role])
    _write_jsonl(output / "SEALED_RESERVATIONS.jsonl", reservations)
    files = {}
    for name in ("REGISTRATION.json", "TRAIN.jsonl", "TRAIN_SELECTION.jsonl", "DEV.jsonl", "SEALED_RESERVATIONS.jsonl"):
        raw = (output / name).read_bytes()
        files[name] = {"sha256": sha256(raw), "bytes": len(raw)}
    per_family = Counter((json.loads(p.canonical_json)["family"], p.role) for p in parents)
    receipt = {
        "schema_version": 1, "status": "MEASURED_DETERMINISTIC_DATA_MECHANICS_ONLY",
        "generator_version": VERSION, "verifier_version": VERIFIER_VERSION,
        "generator_file_sha256": sha256(Path(__file__).read_bytes()), "source_revision": source_revision,
        "source_revision_verification": "CALLER_SUPPLIED; generator_file_sha256 binds actual executed implementation",
        "registration_sha256": sha256(registration_raw), "selection_seed": registration["selection_seed"],
        "runtime": {"python_version": sys.version, "platform": sys.platform,
                    "dependencies": "Python standard library only", "determinism_scope": "same code, registration and Python runtime"},
        "role_allocator": {"domain_separator_hex": ROLE_DOMAIN.hex(),
                           "hash": "SHA256(domain_separator + canonical_UTF8_JSON); first 64 bits modulo 10000",
                           "exclusive_upper_buckets": [[high, role] for high, role in ROLE_BUCKETS]},
        "created_utc": datetime.now(timezone.utc).isoformat(), "files": files,
        "independent_unit": "canonical_latent_parent", "parent_counts": counts,
        "family_parent_counts": {role: {family: per_family[(family, role)] for family in FAMILIES} for role in ROLES},
        "open_row_counts": {role: len(rows[role]) for role in ROLES[:3]}, "variants_per_open_parent": variants,
        "sealed_exports": "hash identities only; no prompt/target/spec/verifier payload",
        "final_count_status": "UNRESOLVED_UNOPENED" if registration["counts"]["FINAL"]["acquisition_parents"] is None else "RESERVED_UNOPENED",
        "selection": selection,
        "controls": {"positive_reference_answers_verified": sum(map(len, rows.values())),
                     "wrong_answer_mutations_rejected": sum(map(len, rows.values())),
                     "canonical_parent_duplicates": 0, "cross_parent_prompt_collisions": 0},
        "normalization_scope": ["commutative arithmetic operands", "finite-set sorting/deduplication and commutative operands",
                                "symbolic register names use indices", "text tokens and ordered program retained"],
        "not_proven": ["general semantic equivalence exclusion", "historical/pretraining exclusion", "native learning efficacy"],
        "cost": {"wall_seconds": time.perf_counter() - started, "model_calls": 0, "teacher_calls": 0,
                 "gpu_seconds": 0, "model_generation_tokens": 0, "actual_training_tokens": 0,
                 "dataset_token_count": None, "dataset_token_count_status": "UNMEASURED"},
        "tokenizer_context_qualified": False, "tokenized_prompt_hashes_measured": False,
        "historical_exclusion_qualified": False, "retention_dataset_qualified": False,
        "native_qualified": False, "e1_launch_ready": False,
    }
    (output / "DATA_RECEIPT.json").write_bytes(json_bytes(receipt) + b"\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--variants", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    receipt = build_dataset(args.registration, args.output, args.source_revision, args.variants)
    receipt_path = args.output / "DATA_RECEIPT.json"
    print(json.dumps({"receipt": str(receipt_path.resolve()), "receipt_sha256": sha256(receipt_path.read_bytes()),
                      "parent_counts": receipt["parent_counts"], "e1_launch_ready": False}, indent=2))


if __name__ == "__main__":
    main()
