"""Original seeded instance construction. Witnesses are discarded, not training labels."""
from __future__ import annotations
from fractions import Fraction
import random
from .specs import validate_spec

def make(f: str, difficulty: int, rng: random.Random) -> dict | None:
    d = difficulty
    if f == "countdown":
        n = (3, 4, 4, 5)[d]
        numbers = [rng.randint(1, (20, 16, 30, 20)[d]) for _ in range(n)]
        pool = [Fraction(x) for x in numbers]
        while len(pool) > 1:
            a = pool.pop(rng.randrange(len(pool)))
            b = pool.pop(rng.randrange(len(pool)))
            op = rng.choice("+-*" if d == 0 else "+-*/")
            if op == "/" and not b: return None
            value = a+b if op == "+" else a-b if op == "-" else a*b if op == "*" else a/b
            if abs(value.numerator) > 10**9 or value.denominator > 10**9: return None
            pool.append(value)
        if pool[0].denominator != 1 or not 2 <= abs(pool[0]) <= 2000: return None
        s = {"numbers": numbers, "target": int(pool[0])}
    elif f == "subset_sum":
        n, k = (6, 2) if d == 0 else (9, 3) if d == 1 else (12, 4) if d == 2 else (15, 5)
        values = [rng.randint(-8 if d > 1 else 1, 35) for _ in range(n)]
        chosen = rng.sample(range(n), k)
        s = {"values": values, "k": k, "target": sum(values[i] for i in chosen)}
    elif f == "linear_constraints":
        n = (3, 4, 5, 6)[d]
        upper = 2 if d == 0 else 3
        witness = [rng.randrange(upper+1) for _ in range(n)]
        if not any(witness): return None
        equations = []
        for _ in range((1, 2, 3, 4)[d]):
            coefficients = [rng.randint(-3, 4) for _ in range(n)]
            if not any(coefficients): return None
            equations.append({"coefficients": coefficients, "rhs": sum(c*x for c, x in zip(coefficients, witness))})
        s = {"n": n, "upper": upper, "equations": equations}
    elif f == "cnf_assignment":
        n, count = (5, 5) if d == 0 else (7, 9) if d == 1 else (9, 14) if d == 2 else (11, 20)
        witness = [rng.randrange(2) for _ in range(n)]
        clauses = set()
        tries = 0
        while len(clauses) < count and tries < 500:
            tries += 1
            variables = rng.sample(range(n), 2 if d == 0 else 3)
            lits = [(j+1)*rng.choice((-1, 1)) for j in variables]
            if not any(witness[abs(lit)-1] == int(lit > 0) for lit in lits):
                ix = rng.randrange(len(lits)); lits[ix] *= -1
            clauses.add(tuple(sorted(lits)))
        s = {"n": n, "clauses": [list(c) for c in sorted(clauses)]}
    elif f in ("shortest_path", "precedence_order"):
        n = (5, 6, 8, 10)[d]
        order = rng.sample(range(n), n)
        if f == "shortest_path":
            edge_map = {(order[i], order[i+1]): rng.randint(1, 9) for i in range(n-1)}
            for i in range(n):
                for j in range(i+2, n):
                    if rng.random() < 0.35: edge_map[(order[i], order[j])] = rng.randint(1, 15)
            edges = [[u, v, w] for (u, v), w in sorted(edge_map.items())]
            s = {"n": n, "edges": edges, "start": order[0], "goal": order[-1]}
        else:
            edges = [[order[i], order[j]] for i in range(n) for j in range(i+1, n) if rng.random() < (0.20, 0.28, 0.35, 0.42)[d]]
            if not edges: return None
            s = {"n": n, "before": sorted(edges)}
    elif f == "finite_program":
        modulus, length = (11, 2) if d == 0 else (17, 3) if d == 1 else (23, 4) if d == 2 else (31, 5)
        operations = rng.sample([(a, b) for a in range(1, 5) for b in range(modulus) if (a, b) != (1, 0)], 4)
        program = [rng.randrange(4) for _ in range(length)]
        cases = []
        for start in rng.sample(range(modulus), 2 if d < 2 else 3):
            state = start
            for ix in program:
                a, b = operations[ix]; state = (a*state+b) % modulus
            cases.append([start, state])
        s = {"modulus": modulus, "operations": [list(op) for op in operations], "length": length, "cases": cases}
    elif f == "state_tracking":
        n, count = (3, 4) if d == 0 else (4, 7) if d == 1 else (5, 11) if d == 2 else (6, 16)
        initial = [rng.randint(-15, 25) for _ in range(n)]
        ops = []
        for _ in range(count):
            op = rng.choice(("add", "set", "swap", "copy"))
            i = rng.randrange(n)
            if op in ("swap", "copy"):
                j = rng.choice([v for v in range(n) if v != i])
            else:
                j = rng.choice([v for v in range(-12, 13) if v != 0])
            ops.append([op, i, j])
        s = {"initial": initial, "operations": ops}
    else:
        raise ValueError("unknown family")
    validate_spec(f, s)
    return s
