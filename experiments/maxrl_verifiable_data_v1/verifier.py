"""Bounded semantic binary rewards. Never imports the generator or oracle."""
from __future__ import annotations
import ast
from collections import Counter
from fractions import Fraction
import heapq
import json
from typing import Any
from .specs import canonical, ints, require, validate_spec

MAX_BYTES = 4096

def _pairs(pairs: list[tuple]) -> dict:
    result = {}
    for k, v in pairs:
        if k in result: raise ValueError("duplicate JSON key")
        result[k] = v
    return result

def _constant(_: str) -> None:
    raise ValueError("nonfinite JSON")

def parse_answer(text: str) -> Any:
    if type(text) is not str or len(text.encode("utf-8")) > MAX_BYTES:
        raise ValueError("response length/type")
    obj = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    require(type(obj) is dict and set(obj) == {"answer"}, "answer envelope")
    return obj["answer"]

def expression_value(text: str, numbers: list[int]) -> Fraction:
    require(type(text) is str and 0 < len(text) <= 512, "bad expression length")
    root = ast.parse(text, mode="eval")
    require(sum(1 for _ in ast.walk(root)) <= 50, "expression node budget")
    leaves = []
    def walk(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant) and type(node.value) is int and 1 <= node.value <= 100:
            leaves.append(node.value)
            return Fraction(node.value)
        require(isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div), "forbidden expression syntax")
        a, b = walk(node.left), walk(node.right)
        if isinstance(node.op, ast.Add): v = a + b
        elif isinstance(node.op, ast.Sub): v = a - b
        elif isinstance(node.op, ast.Mult): v = a * b
        else: v = a / b
        require(abs(v.numerator) <= 10**9 and v.denominator <= 10**9, "expression arithmetic budget")
        return v
    value = walk(root.body)
    require(Counter(leaves) == Counter(numbers), "wrong number multiset")
    return value

def shortest_cost(s: dict) -> int | None:
    edges = [[] for _ in range(s["n"])]
    for u, v, w in s["edges"]: edges[u].append((v, w))
    heap = [(0, s["start"])]
    done = {}
    while heap:
        d, u = heapq.heappop(heap)
        if u in done: continue
        done[u] = d
        if u == s["goal"]: return d
        for v, w in edges[u]:
            if v not in done: heapq.heappush(heap, (d+w, v))
    return None

def _accept(f: str, s: dict, a: Any) -> bool:
    if f == "countdown":
        return expression_value(a, s["numbers"]) == s["target"]
    if f == "subset_sum":
        return ints(a, 0, len(s["values"])-1, s["k"]) and len(set(a)) == len(a) and sum(s["values"][i] for i in a) == s["target"]
    if f == "linear_constraints":
        return ints(a, 0, s["upper"], s["n"]) and all(sum(c*x for c, x in zip(eq["coefficients"], a)) == eq["rhs"] for eq in s["equations"])
    if f == "cnf_assignment":
        return ints(a, 0, 1, s["n"]) and all(any(a[abs(lit)-1] == int(lit > 0) for lit in clause) for clause in s["clauses"])
    if f == "shortest_path":
        if not (ints(a, 0, s["n"]-1) and 2 <= len(a) <= s["n"] and len(set(a)) == len(a) and a[0] == s["start"] and a[-1] == s["goal"]): return False
        weights = {(u, v): w for u, v, w in s["edges"]}
        if any((u, v) not in weights for u, v in zip(a, a[1:])): return False
        return sum(weights[(u, v)] for u, v in zip(a, a[1:])) == shortest_cost(s)
    if f == "precedence_order":
        if not ints(a, 0, s["n"]-1, s["n"]) or len(set(a)) != s["n"]: return False
        pos = {v: i for i, v in enumerate(a)}
        return all(pos[u] < pos[v] for u, v in s["before"])
    if f == "finite_program":
        if not ints(a, 0, len(s["operations"])-1, s["length"]): return False
        for start, target in s["cases"]:
            state = start
            for op in a:
                mul, add = s["operations"][op]
                state = (mul * state + add) % s["modulus"]
            if state != target: return False
        return True
    if not ints(a, -1000, 1000, len(s["initial"])): return False
    state = s["initial"][:]
    for op, i, x in s["operations"]:
        if op == "add": state[i] += x
        elif op == "set": state[i] = x
        elif op == "copy": state[i] = state[x]
        else: state[i], state[x] = state[x], state[i]
    return a == state

def score(family: str, spec: dict, response: str) -> int:
    validate_spec(family, spec)  # fail on corrupt/truncated metadata, not a learner error
    try:
        return int(_accept(family, spec, parse_answer(response)))
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError):
        return 0

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None) -> float:
    """verl custom_reward_function entry point; ground_truth contains SPEC, not a trace.

    Configure the package import path on the training host; this adapter is data-level
    tested only. Never award partial credit for format, brevity or explanations.
    """
    require(type(ground_truth) is str and len(ground_truth) <= 16000, "invalid verifier payload")
    payload = json.loads(ground_truth, object_pairs_hook=_pairs, parse_constant=_constant)
    require(type(payload) is dict and set(payload) == {"version", "family", "spec"} and payload["version"] == "1.0.0", "verifier version/payload mismatch")
    require(data_source == "rineya_maxrl_v1/" + payload["family"], "data_source/validator mismatch")
    return float(score(payload["family"], payload["spec"], solution_str))
