"""Independent bounded reference search, using only public problem specifications.

Does not call the reward verifier or use the generator's private construction witness.
References are audit/warm-start material, NEVER stored in online policy prompts.
"""
from __future__ import annotations
from fractions import Fraction
from itertools import combinations, product
from .specs import canonical, validate_spec

def solutions(f: str, s: dict, limit: int = 3) -> list[str]:
    validate_spec(f, s)
    found: list = []
    if f == "countdown":
        # Bottom-up subset DP, independent from the verifier's AST interpreter.
        numbers = s["numbers"]
        dp: dict[int, dict[Fraction, str]] = {1 << i: {Fraction(v): str(v)} for i, v in enumerate(numbers)}
        full = (1 << len(numbers)) - 1
        for mask in range(1, full+1):
            if mask in dp: continue
            table = {}
            sub = (mask-1) & mask
            while sub:
                other = mask ^ sub
                if other and sub < other:
                    for a, ea in dp[sub].items():
                        for b, eb in dp[other].items():
                            pairs = [(a+b, f"({ea}+{eb})"), (a-b, f"({ea}-{eb})"), (b-a, f"({eb}-{ea})"), (a*b, f"({ea}*{eb})")]
                            if b: pairs.append((a/b, f"({ea}/{eb})"))
                            if a: pairs.append((b/a, f"({eb}/{ea})"))
                            for val, expr in pairs:
                                if abs(val.numerator) <= 10**9 and val.denominator <= 10**9:
                                    if mask == full and val == s["target"] and expr not in found:
                                        found.append(expr)
                                        if len(found) >= limit: return [canonical({"answer": x}) for x in found]
                                    table.setdefault(val, expr)
                sub = (sub-1) & mask
            dp[mask] = table
    elif f == "subset_sum":
        for ix in combinations(range(len(s["values"])), s["k"]):
            if sum(s["values"][j] for j in ix) == s["target"]:
                found.append(list(ix))
                if len(found) >= limit: break
    elif f == "linear_constraints":
        # Product enumeration of the finite assignment domain.
        for xs in product(range(s["upper"]+1), repeat=s["n"]):
            residuals = [e["rhs"] for e in s["equations"]]
            for j, x in enumerate(xs):
                for i, eq in enumerate(s["equations"]): residuals[i] -= eq["coefficients"][j]*x
            if not any(residuals):
                found.append(list(xs))
                if len(found) >= limit: break
    elif f == "cnf_assignment":
        # Bitset clause satisfaction rather than the reward verifier's literal loop.
        masks = []
        for clause in s["clauses"]:
            pos = sum(1 << (x-1) for x in clause if x > 0)
            neg = sum(1 << (-x-1) for x in clause if x < 0)
            masks.append((pos, neg))
        universe = (1 << s["n"]) - 1
        for value in range(1 << s["n"]):
            if all((value & p) or ((universe ^ value) & n) for p, n in masks):
                found.append([(value >> j) & 1 for j in range(s["n"])])
                if len(found) >= limit: break
    elif f == "shortest_path":
        # Exhaustive simple-path search; verifier uses Dijkstra.
        adj = [[] for _ in range(s["n"])]
        for u, v, w in s["edges"]: adj[u].append((v, w))
        best = float("inf")
        def dfs(path: list[int], cost: int) -> None:
            nonlocal best, found
            if cost > best: return
            if path[-1] == s["goal"]:
                if cost < best: best, found = cost, []
                if len(found) < limit: found.append(path)
                return
            for v, w in adj[path[-1]]:
                if v not in path: dfs(path+[v], cost+w)
        dfs([s["start"]], 0)
    elif f == "precedence_order":
        prereq = {i: set() for i in range(s["n"])}
        for u, v in s["before"]: prereq[v].add(u)
        def expand(prefix: list[int], remaining: set[int]) -> None:
            if len(found) >= limit: return
            if not remaining: found.append(prefix); return
            for v in sorted(remaining):
                if not (prereq[v] & remaining): expand(prefix+[v], remaining-{v})
        expand([], set(prereq))
    elif f == "finite_program":
        # Compose affine maps algebraically; verifier executes instructions per input.
        for seq in product(range(len(s["operations"])), repeat=s["length"]):
            a, b = 1, 0
            for ix in seq:
                c, d = s["operations"][ix]
                a, b = c*a % s["modulus"], (c*b+d) % s["modulus"]
            if all((a*x+b) % s["modulus"] == y for x, y in s["cases"]):
                found.append(list(seq))
                if len(found) >= limit: break
    else:
        # Symbolic register provenance, evaluated at end, rather than concrete replay.
        exprs = [(i, 0) for i in range(len(s["initial"]))]
        for op, i, v in s["operations"]:
            if op == "set": exprs[i] = (None, v)
            elif op == "add": exprs[i] = (exprs[i][0], exprs[i][1]+v)
            elif op == "copy": exprs[i] = exprs[v]
            else: exprs[i], exprs[v] = exprs[v], exprs[i]
        found = [[(s["initial"][i] if i is not None else 0)+b for i, b in exprs]]
    return [canonical({"answer": x}) for x in found[:limit]]
