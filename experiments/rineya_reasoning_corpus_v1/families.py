"""Task families for RTG-1, with two independent solvers each.

These mirror the four MRS scout families so that training can transfer to that fixed,
evaluation-only bank. Three deliberate departures from the scout's generator, each
motivated by a measured defect or a measured capacity limit:

1. ANSWERABILITY IS DECORRELATED FROM ITS SURFACE CUE. In the scout bank,
   `initial is None` <-> UNKNOWN is a perfect biconditional for register_machine and
   counterfactual_state (48/12, zero off-diagonal), and "terminal node has no records"
   <-> UNKNOWN is perfect for evidence_chain (24/6). A model can therefore score every
   abstention from one cue without doing the task. RTG-1 breaks both directions:
     * `SET k` makes a value determined even when the initial value is unrecorded;
     * an unrecorded event makes it undetermined even when the initial value is known;
     * a branching link makes an evidence chain undetermined with every record present;
     * records missing from a node off the path leave the answer determined.
   Only `prerequisite_schedule` was already sound in the scout, and its mechanism is kept.

2. COMPLEXITY RANGES EXTEND BEYOND THE EVAL RANGE. The `match` tier reproduces the
   scout's generative range exactly; `easy` and `hard` tiers sit below and above it. This
   is a curriculum choice, registered prospectively, not a change to the eval surface.

3. `prerequisite_schedule` ADMITS MORE JOBS. Only 480 distinct 4-job instances exist in
   the whole scout construction (verified by exhaustive enumeration: 256 + 128 + 64 + 32
   by `missing`), so a large corpus of unique 4-job structures is arithmetically
   impossible. Larger job counts teach the identical algorithm.

Instruction text is assembled so that a row using only the scout's own operation set
renders a byte-identical preamble to the scout's. Extra clauses appear only when the
operations they describe are actually present.

Every solver pair is cross-validated against the scout's own independent solvers on all
120 scout worlds (see tests/test_families.py).
"""
from __future__ import annotations
import itertools
import json

FAMILIES = ("register_machine", "counterfactual_state", "prerequisite_schedule", "evidence_chain")
ANSWERS = ("0", "1", "2", "3", "UNKNOWN")
NAME_ALPHABET = "abcdefghjkmnpqrstuvwxyz"   # same 23 letters the scout uses
NAME_PREFIX = "q"                            # scout uses "n"
NAME_BODY = 7                                # scout uses 6
ALL_VALUES = frozenset(range(4))


# --------------------------------------------------------------------------- names

def names(rng, count):
    """Namespace provably disjoint from the scout's `n` + 6 chars, by prefix AND length."""
    out = set()
    while len(out) < count:
        out.add(NAME_PREFIX + "".join(rng.choices(NAME_ALPHABET, k=NAME_BODY)))
    return sorted(out)


# ----------------------------------------------------------------- register semantics

def apply_op(values, op):
    """Advance a SET of possible values through one operation.

    add/xor are bijections on Z4, so they preserve uncertainty; `set` collapses it; an
    unrecorded event destroys it. That asymmetry is what decorrelates answerability from
    the "not recorded" surface marker.
    """
    kind, n = op
    if kind == "add":
        return {(x + n) % 4 for x in values}
    if kind == "xor":
        return {x ^ n for x in values}
    if kind == "set":
        return {n}
    if kind == "unrec":
        return set(ALL_VALUES)
    raise ValueError("unregistered operation: " + str(kind))


def _register_states(w):
    return set(ALL_VALUES) if w["initial"] is None else {w["initial"]}


def _register_ops(w):
    skip = w.get("skip")
    return [op for i, op in enumerate(w["ops"]) if i != skip]


def solve_register_forward(w):
    values = _register_states(w)
    for op in _register_ops(w):
        values = apply_op(values, op)
    return _one(values)


def solve_register_backward(w):
    """Independent algorithm: test each candidate result by inverting the operations.

    Rather than pushing a set forward, this pulls each of the four possible results back
    through the operation list and asks whether the initial state could have produced it.
    """
    ops = _register_ops(w)
    start = _register_states(w)
    outcomes = set()
    for result in range(4):
        preimage = {result}
        reachable = True
        for kind, n in reversed(ops):
            if kind == "add":
                preimage = {(v - n) % 4 for v in preimage}
            elif kind == "xor":
                preimage = {v ^ n for v in preimage}
            elif kind == "set":
                # Only results equal to n survive; anything before is unconstrained.
                if preimage != {n} and n not in preimage:
                    reachable = False
                    break
                preimage = set(ALL_VALUES)
            elif kind == "unrec":
                preimage = set(ALL_VALUES)
            else:
                raise ValueError("unregistered operation: " + str(kind))
        if reachable and preimage & start:
            outcomes.add(result)
    return _one(outcomes)


def render_register(w):
    names_ = w["names"]
    initial = "not recorded" if w["initial"] is None else str(w["initial"])
    kinds = {op[0] for op in w["ops"]}
    head = (f"Register {names_[0]} holds one of 0,1,2,3. Initial value: {initial}. "
            "Apply numbered events in order. ADD k adds k modulo 4. XOR k is bitwise XOR on two bits.")
    if "set" in kinds:
        head += " SET k replaces the value with k."
    if "unrec" in kinds:
        head += " An event recorded as NOT RECORDED has an unknown effect."
    lines = []
    for i, (kind, n) in enumerate(w["ops"]):
        lines.append(f"{i+1}. NOT RECORDED" if kind == "unrec" else f"{i+1}. {kind.upper()} {n}")
    text = head + "\n" + "\n".join(lines)
    if w.get("skip") is not None:
        text += (f"\nCounterfactual: event {w['skip'] + 1} is cancelled; "
                 "all other events stay in order.")
    return text + f"\nWhat is the final value of {names_[0]}?"


# -------------------------------------------------------------- prerequisite schedule

def _dep_alternatives(w):
    """All dependency tables consistent with the record, including an unrecorded row."""
    missing = w["missing"]
    if missing is None:
        return [[list(d) for d in w["deps"]]]
    out = []
    for bits in itertools.product((0, 1), repeat=missing):
        deps = [list(d) for d in w["deps"]]
        deps[missing] = [i for i, b in enumerate(bits) if b]
        out.append(deps)
    return out


def solve_schedule_forward(w):
    """Forward: relax earliest start times in list order (dependencies point backwards)."""
    outcomes = set()
    for deps in _dep_alternatives(w):
        starts = []
        for row in deps:
            starts.append(max((starts[j] + 1 for j in row), default=0))
        outcomes.add(starts[w["query"]])
    return _one(outcomes)


def solve_schedule_backward(w):
    """Independent algorithm: longest dependency path by recursive depth, memoised.

    Uses a different formulation from the forward relaxation: the earliest start of a job
    is the length of the longest chain of prerequisites beneath it.
    """
    outcomes = set()
    for deps in _dep_alternatives(w):
        depth = {}

        def longest(job):
            if job not in depth:
                depth[job] = max((longest(j) + 1 for j in deps[job]), default=0)
            return depth[job]

        outcomes.add(longest(w["query"]))
    return _one(outcomes)


def render_schedule(w):
    names_, n = w["names"], len(w["deps"])
    word = {3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven"}[n]
    text = (f"{word} unit-duration jobs may run in parallel. Times begin at 0. A job starts only after "
            "all prerequisites finish. All listed dependencies are complete except any explicitly "
            "unrecorded row. The only possible prerequisites of each job are jobs earlier in "
            "this list. Find the earliest possible start under the actual dependencies.\n")
    for i, ds in enumerate(w["deps"]):
        row = "not recorded" if i == w["missing"] else (", ".join(names_[j] for j in ds) or "none")
        text += f"Job {names_[i]}: prerequisites {row}.\n"
    return text + f"What is the earliest start time of {names_[w['query']]}?"


# -------------------------------------------------------------------- evidence chain

def _link_map(w):
    out = {}
    for a, b in w["links"]:
        out.setdefault(a, []).append(b)
    return out


def _terminals(w):
    """Every node with no outgoing link reachable from the query, following all branches."""
    links = _link_map(w)
    seen, stack, ends = set(), [w["query"]], set()
    while stack:
        node = stack.pop()
        if node in seen:
            raise ValueError("cyclic evidence chain")
        seen.add(node)
        if node in links:
            stack.extend(links[node])
        else:
            ends.add(node)
    return ends


def _node_value(w, node):
    found = [(t, v) for name, t, v in w["facts"] if name == node]
    return {max(found)[1]} if found else set(ALL_VALUES)


def solve_evidence_forward(w):
    values = set()
    for end in _terminals(w):
        values |= _node_value(w, end)
    return _one(values)


def solve_evidence_backward(w):
    """Independent algorithm: enumerate simple paths from the query and read each endpoint.

    Path enumeration rather than a reachability walk, and the latest record is selected by
    an explicit sort rather than by max() over tuples.
    """
    edges = {(a, b) for a, b in w["links"]}
    nodes = sorted({a for a, _ in edges} | {b for _, b in edges} | {w["query"]})
    sources = {a for a, _ in edges}
    endpoints = set()
    for length in range(1, len(nodes) + 1):
        for tail in itertools.permutations([n for n in nodes if n != w["query"]], length - 1):
            path = (w["query"],) + tail
            if all((a, b) in edges for a, b in zip(path, path[1:])) and path[-1] not in sources:
                endpoints.add(path[-1])
    values = set()
    for end in endpoints:
        rows = sorted((row for row in w["facts"] if row[0] == end), key=lambda f: (f[1], f[2]))
        values |= {rows[-1][2]} if rows else set(ALL_VALUES)
    return _one(values)


def render_evidence(w):
    names_ = w["names"]
    text = ("Follow directed links until a node has no outgoing link. Its value is the most recent "
            "dated value record for that node; older records are superseded. Values are 0,1,2,3. "
            "Links below are complete.\n")
    text += "\n".join(f"{names_[a]} -> {names_[b]}" for a, b in w["links"])
    text += "\n" + "\n".join(f"time {t}: {names_[i]} value {v}" for i, t, v in w["facts"])
    return text + f"\nWhat value is reached from {names_[w['query']]}?"


# ------------------------------------------------------------------------- dispatch

def _one(values):
    values = set(values)
    if not values:
        raise ValueError("solver produced no outcome")
    return str(next(iter(values))) if len(values) == 1 else "UNKNOWN"


_FORWARD = {"register_machine": solve_register_forward,
            "counterfactual_state": solve_register_forward,
            "prerequisite_schedule": solve_schedule_forward,
            "evidence_chain": solve_evidence_forward}
_BACKWARD = {"register_machine": solve_register_backward,
             "counterfactual_state": solve_register_backward,
             "prerequisite_schedule": solve_schedule_backward,
             "evidence_chain": solve_evidence_backward}
_RENDER = {"register_machine": render_register, "counterfactual_state": render_register,
           "prerequisite_schedule": render_schedule, "evidence_chain": render_evidence}


def solve_forward(w):
    return _FORWARD[w["family"]](w)


def solve_backward(w):
    return _BACKWARD[w["family"]](w)


def render(w):
    return _RENDER[w["family"]](w)


def solve(w):
    """The registered answer. Both independent solvers must agree or the row is rejected."""
    forward, backward = solve_forward(w), solve_backward(w)
    if forward != backward:
        raise ValueError("independent solvers disagree: " + json.dumps(w, sort_keys=True))
    return forward


# ---------------------------------------------------------------------- fingerprints

def _evidence_signature(w):
    """Isomorphism-invariant form for an evidence chain.

    Node integers are arbitrary labels, so sorting raw indices would let two identical
    structures under a relabelling receive different fingerprints -- which would silently
    admit duplicates across a train/holdout boundary. This computes a recursive subtree
    signature instead, making the form invariant to node numbering.
    """
    links = _link_map(w)
    facts = {}
    for node, t, v in w["facts"]:
        facts.setdefault(node, []).append((t, v))
    memo = {}

    def signature(node, ancestors=frozenset()):
        if node in ancestors:
            raise ValueError("cyclic evidence chain")
        if node not in memo:
            kids = sorted(signature(c, ancestors | {node}) for c in links.get(node, []))
            memo[node] = [sorted(facts.get(node, [])), kids]
        return memo[node]

    root = signature(w["query"])
    reachable, stack = set(), [w["query"]]
    while stack:
        node = stack.pop()
        reachable.add(node)
        stack.extend(c for c in links.get(node, []) if c not in reachable)
    others = sorted(signature(n) for n in
                    ({a for a, _ in w["links"]} | {b for _, b in w["links"]}
                     | {f[0] for f in w["facts"]}) - reachable)
    return {"root": root, "detached": others}


def fingerprint(w):
    """Canonical name-blind structure.

    Entity names carry no information about the answer, so two worlds differing only in
    names are the same structure. Fields that cannot affect the rendered instance are
    erased: an unrecorded dependency row hides that job's actual prerequisites. A
    cancelled event still appears in the listing, so it is retained.
    """
    family = w["family"]
    if family in ("register_machine", "counterfactual_state"):
        core = {"initial": w["initial"], "ops": [list(op) for op in w["ops"]], "skip": w.get("skip")}
    elif family == "prerequisite_schedule":
        core = {"deps": [("NOT_RECORDED" if i == w["missing"] else sorted(d))
                         for i, d in enumerate(w["deps"])],
                "query": w["query"], "missing": w["missing"], "jobs": len(w["deps"])}
    else:
        core = _evidence_signature(w)
    return json.dumps({"family": family, "core": core}, sort_keys=True, separators=(",", ":"))


def answerability_cue(w):
    """Whether the instance contains an information gap at all.

    This is a NECESSARY condition for an undetermined answer in every family: a complete
    record always determines the result. It is deliberately not sufficient, and the corpus
    targets P(UNKNOWN | gap) = 0.5 -- against 1.0 in the scout bank for three of its four
    families, where abstention was therefore obtainable from this cue alone.
    """
    family = w["family"]
    if family in ("register_machine", "counterfactual_state"):
        # A cancelled event cannot create uncertainty, so only effective ops count.
        return w["initial"] is None or any(op[0] == "unrec" for op in _register_ops(w))
    if family == "prerequisite_schedule":
        # An unrecorded row for a job that is not an ancestor of the queried one is a gap
        # that does not matter -- exactly the irrelevant-gap case.
        return w["missing"] is not None
    # Any node missing a record, or an ambiguous route. Counting records missing from nodes
    # OFF the route matters: those gaps are real but irrelevant, and including them is what
    # forces a relevance judgement instead of a marker lookup. The condition stays NECESSARY
    # for UNKNOWN, since an undetermined answer still requires either the route's endpoint
    # to lack a record or the route itself to branch.
    links = _link_map(w)
    mentioned = ({a for a, _ in w["links"]} | {b for _, b in w["links"]}
                 | {row[0] for row in w["facts"]} | {w["query"]})
    return (any(not any(row[0] == node for row in w["facts"]) for node in mentioned)
            or any(len(v) > 1 for v in links.values()))
