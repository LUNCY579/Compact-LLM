"""Public problem contracts and deterministic prompt rendering (stdlib only)."""
from __future__ import annotations
import hashlib
import json
from typing import Any

VERSION = "1.0.0"
FAMILIES = ("countdown", "subset_sum", "linear_constraints", "cnf_assignment",
            "shortest_path", "precedence_order", "finite_program", "state_tracking")
SYSTEM = ('Solve the supplied self-contained problem. Return exactly one JSON object '
          'with the single key "answer". Do not include explanations, code fences, or extra keys. '
          'Any answer satisfying ALL stated rules is accepted; no particular reference answer is required.')
INTRO = (
    "Find a valid solution to this problem.",
    "Complete this exact, self-contained reasoning task.",
    "Use the rules and the specification below to produce one accepted answer.",
    "Solve the following instance; any solution meeting every constraint is valid.",
    "Construct a certificate that satisfies the complete specification below.",
    "Return one admissible result under all of the following requirements.",
)
RULES = {
    "countdown": 'answer is a string expression. Use every integer in numbers exactly once, including repetitions. Use only binary +, -, *, / and parentheses; no unary operators, powers, concatenation, or other syntax. Division is exact rational arithmetic and cannot divide by zero. The expression must equal target. Every intermediate numerator and denominator must have absolute value <= 1000000000.',
    "subset_sum": 'answer is an array of exactly k distinct zero-based indices into values. The selected values must sum to target. Index order does not matter.',
    "linear_constraints": 'answer is an integer array of length n, in variable-index order. Each integer must lie in the inclusive interval [0,upper]. For every equation, the dot product of its coefficients with answer must equal its rhs.',
    "cnf_assignment": 'answer is an array of n integers, each 0 or 1, in variable-index order. Each clause is a disjunction, and all clauses must be true. Literal j denotes variable j-1; literal -j denotes its negation. Variables are one-based in clauses only.',
    "shortest_path": 'answer is an array of distinct node indices describing a directed path from start to goal. Each [u,v,w] entry is a directed edge u to v with positive integer cost w. Return any globally minimum-total-cost path, not merely a feasible path.',
    "precedence_order": 'answer is a permutation of all task indices 0 through n-1. Every pair [u,v] in before requires u to appear strictly before v. Any valid topological ordering is accepted.',
    "finite_program": 'answer is an array of exactly length zero-based operation indices; repetition is allowed. An operation [a,b] maps state s to (a*s+b) modulo modulus. Apply the same sequence in order to each input in cases, obtaining its paired output. All cases must pass.',
    "state_tracking": 'answer is the final integer array of register values after applying all operations in listed order. ["add",i,v] adds v to register i. ["set",i,v] sets register i to v. ["swap",i,j] exchanges registers i and j. ["copy",i,j] copies the CURRENT value of register j into register i. Indices are zero-based.',
}
KEYS = {
    "countdown": {"numbers", "target"}, "subset_sum": {"values", "k", "target"},
    "linear_constraints": {"n", "upper", "equations"},
    "cnf_assignment": {"n", "clauses"},
    "shortest_path": {"n", "edges", "start", "goal"},
    "precedence_order": {"n", "before"},
    "finite_program": {"modulus", "operations", "length", "cases"},
    "state_tracking": {"initial", "operations"},
}

def canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)

def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def integer(x: Any, lo: int, hi: int) -> bool:
    return type(x) is int and lo <= x <= hi

def ints(xs: Any, lo: int, hi: int, n: int | None = None) -> bool:
    return type(xs) is list and (n is None or len(xs) == n) and all(integer(x, lo, hi) for x in xs)

def validate_spec(family: str, s: dict) -> None:
    """Corrupt problem metadata is an error, NEVER an ordinary zero reward."""
    require(family in FAMILIES and type(s) is dict and set(s) == KEYS[family], "invalid spec keys/family")
    if family == "countdown":
        require(ints(s["numbers"], 1, 100) and 3 <= len(s["numbers"]) <= 5 and integer(s["target"], -10000, 10000), "bad countdown")
    elif family == "subset_sum":
        require(ints(s["values"], -100, 100) and 5 <= len(s["values"]) <= 15 and integer(s["k"], 1, len(s["values"])) and integer(s["target"], -1500, 1500), "bad subset")
    elif family == "linear_constraints":
        require(integer(s["n"], 2, 6) and integer(s["upper"], 1, 5) and type(s["equations"]) is list and 1 <= len(s["equations"]) <= 5, "bad linear")
        for eq in s["equations"]:
            require(type(eq) is dict and set(eq) == {"coefficients", "rhs"} and ints(eq["coefficients"], -5, 5, s["n"]) and any(eq["coefficients"]) and integer(eq["rhs"], -150, 150), "bad equation")
    elif family == "cnf_assignment":
        require(integer(s["n"], 3, 11) and type(s["clauses"]) is list and 1 <= len(s["clauses"]) <= 30, "bad cnf")
        for clause in s["clauses"]:
            require(ints(clause, -s["n"], s["n"]) and 2 <= len(clause) <= 3 and 0 not in clause and len({abs(x) for x in clause}) == len(clause), "bad clause")
    elif family in ("shortest_path", "precedence_order"):
        require(integer(s["n"], 3, 11), "bad node count")
        edges = s["edges"] if family == "shortest_path" else s["before"]
        require(type(edges) is list and 1 <= len(edges) <= s["n"]**2, "bad edges")
        seen = set()
        for edge in edges:
            require(ints(edge, 0, 100, 3 if family == "shortest_path" else 2), "bad edge")
            u, v = edge[:2]
            require(u < s["n"] and v < s["n"] and u != v and (u, v) not in seen, "bad edge endpoints")
            if family == "shortest_path":
                require(1 <= edge[2] <= 100, "bad weight")
            seen.add((u, v))
        if family == "shortest_path":
            require(integer(s["start"], 0, s["n"]-1) and integer(s["goal"], 0, s["n"]-1) and s["start"] != s["goal"], "bad endpoints")
    elif family == "finite_program":
        require(integer(s["modulus"], 5, 37) and integer(s["length"], 2, 5), "bad program domain")
        require(type(s["operations"]) is list and 2 <= len(s["operations"]) <= 5, "bad operation bank")
        require(all(ints(op, 0, s["modulus"]-1, 2) for op in s["operations"]), "bad affine op")
        require(type(s["cases"]) is list and 2 <= len(s["cases"]) <= 4 and all(ints(c, 0, s["modulus"]-1, 2) for c in s["cases"]), "bad cases")
    else:
        require(ints(s["initial"], -50, 50) and 2 <= len(s["initial"]) <= 6 and type(s["operations"]) is list and 2 <= len(s["operations"]) <= 20, "bad ledger")
        for op in s["operations"]:
            require(type(op) is list and len(op) == 3 and op[0] in ("add", "set", "swap", "copy") and integer(op[1], 0, len(s["initial"])-1), "bad ledger op")
            require(integer(op[2], -30, 30) if op[0] in ("add", "set") else integer(op[2], 0, len(s["initial"])-1), "bad ledger argument")

def fingerprint(family: str, s: dict) -> str:
    """Canonical public-instance identity, independent of wording, seed and split."""
    s = json.loads(canonical(s))
    if family == "countdown": s["numbers"].sort()
    if family == "cnf_assignment": s["clauses"] = sorted(sorted(c) for c in s["clauses"])
    if family == "linear_constraints": s["equations"].sort(key=canonical)
    if family == "shortest_path": s["edges"].sort()
    if family == "precedence_order": s["before"].sort()
    if family == "finite_program": s["cases"].sort()
    return digest({"family": family, "spec": s})

def partition(fp: str) -> str:
    bucket = int(fp[:12], 16) % 1000
    return "warmstart" if bucket < 50 else "train" if bucket < 750 else "dev" if bucket < 850 else "test" if bucket < 950 else "challenge"

def render(family: str, s: dict, template: int) -> list[dict[str, str]]:
    validate_spec(family, s)
    require(integer(template, 0, 5), "bad template")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
        INTRO[template] + "\nTask: " + family + "\n" + RULES[family] + "\nSPEC\n" + canonical(s) + "\nEND_SPEC"}]

def parse_visible(prompt: list[dict]) -> tuple[str, dict]:
    require(type(prompt) is list and len(prompt) == 2 and prompt[0] == {"role": "system", "content": SYSTEM} and prompt[1].get("role") == "user", "bad prompt boundary")
    text = prompt[1]["content"]
    family = text.split("\nTask: ", 1)[1].split("\n", 1)[0]
    s = json.loads(text.split("\nSPEC\n", 1)[1].removesuffix("\nEND_SPEC"))
    require(any(prompt == render(family, s, t) for t in range(6)), "prompt/rules mismatch")
    return family, s
