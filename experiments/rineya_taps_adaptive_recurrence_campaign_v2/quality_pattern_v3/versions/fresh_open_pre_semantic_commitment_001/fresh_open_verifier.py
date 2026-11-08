"""Visible-text verification for fresh OPEN tasks; pure stdlib, no bank import.

Algorithms deliberately differ from construction/forward reference execution:
incidence equations, backward Horn proofs, transitive closure, immutable
expression trees, backward event witnesses and pairwise evidence elimination.
The same author wrote both modules; this is algorithm separation, not an
independent author or independent source-review receipt.
"""
from __future__ import annotations

import re

INTEGER = r"-?[0-9]+"
NAME = r"[A-Z]{3}"
DOCUMENT_HEADER = "DOCUMENTS: use only A-approved records"
DOCUMENT_RULE = ". Equal approved values give that value; absent/conflicting values give UNKNOWN."
LATEST = "; latest approved date per entity"
HEADERS = {
    "JOURNALS: balance is opening plus signed entries.",
    "TRANSFERS: each move subtracts from source and adds to destination; balances may be negative.",
    "HORNS: infer a rule's right symbol only if every left symbol is known; otherwise NO.",
    "LINKS: YES requires a directed path of one or more links; otherwise NO.",
    "REGISTERS: execute numbered assignments in order with current values.",
    "STACKS: numbered commands; P pushes; ADD/MUL replace the top two by sum/product.",
    "MEMORY: numbered SET replaces; CLEAR removes; QUOTE changes nothing.",
    "MESSAGES: numbered OFFER sets pending; matching ACK replaces active and consumes pending; REVOKE clears active only; QUOTE inert.",
    "SNAPSHOTS: only named current revision and SIGNED rows count; agree or UNKNOWN for absent/conflicting values.",
    "ATTESTATIONS: authorized unrevoked receipts count; agree or UNKNOWN for absent/conflicting values.",
    DOCUMENT_HEADER + DOCUMENT_RULE, DOCUMENT_HEADER + LATEST + DOCUMENT_RULE,
}


def _query(line):
    match = re.fullmatch(r"(?:Question: (.+)\?|Query: (.+)\.) (Reply one integer(?: or UNKNOWN)?\.|Reply only YES or NO\.)", line)
    if match is None:
        raise ValueError("unknown complete query contract")
    question = match[1] or match[2]
    if not re.fullmatch(NAME + r"(?: to " + NAME + r")?", question):
        raise ValueError("query symbols invalid")
    return question.split(" to ")


def _items(line):
    if not line.endswith("."):
        raise ValueError("body must end in period")
    return line[:-1].split(";")


def _named_values(line):
    result = {}
    for item in line.split(","):
        m = re.fullmatch(r"(" + NAME + r")=(" + INTEGER + r")", item)
        if m is None or m[1] in result:
            raise ValueError("named initial values invalid/duplicate")
        result[m[1]] = int(m[2])
    return result


def _steps(line):
    result = []
    for item in _items(line):
        m = re.fullmatch(r"([1-9][0-9]*) (.+)", item)
        if m is None:
            raise ValueError("numbered step invalid")
        result.append((int(m[1]), m[2]))
    if len({i for i, _ in result}) != len(result):
        raise ValueError("duplicate step numbers")
    return sorted(result)


def _answer(value, status="ANSWER", kind="integer"):
    return {"answer": str(value), "source_status": status, "answer_kind": kind}


def _consensus(values):
    if not values:
        return _answer("UNKNOWN", "MISSING", "integer_or_unknown")
    low, high = min(values), max(values)
    return _answer(low, "ANSWER", "integer_or_unknown") if low == high else _answer("UNKNOWN", "CONFLICT", "integer_or_unknown")


def _expression(text, state):
    m = re.fullmatch(r"(" + NAME + r")([+*-])([0-9]+)", text)
    if m is None or m[1] not in state:
        raise ValueError("register RHS invalid")
    return (m[2], state[m[1]], ("n", int(m[3])))


def _evaluate(tree):
    op = tree[0]
    if op == "n":
        return tree[1]
    if op == "if":
        return _evaluate(tree[3] if _evaluate(tree[1]) > tree[2] else tree[4])
    left, right = _evaluate(tree[1]), _evaluate(tree[2])
    return left + right if op == "+" else left - right if op == "-" else left * right


def verify_prompt(prompt):
    """Reads only visible prompt, never a world, case, gold or family record."""
    if type(prompt) is not str:
        raise TypeError("visible prompt string required")
    lines = prompt.split("\n")
    if len(lines) < 3:
        raise ValueError("complete prompt missing")
    query = _query(lines[-1])
    header, body = lines[0], lines[1:-1]
    if header not in HEADERS:
        raise ValueError("unregistered complete header")
    kind = "choice" if header.startswith(("HORNS:", "LINKS:")) else "integer_or_unknown" if header.startswith(("MEMORY:", "MESSAGES:", "DOCUMENTS:", "SNAPSHOTS:", "ATTESTATIONS:")) else "integer"
    expected = "Reply only YES or NO." if kind == "choice" else "Reply one integer or UNKNOWN." if kind == "integer_or_unknown" else "Reply one integer."
    if not lines[-1].endswith(" " + expected) or len(query) != (2 if header.startswith("LINKS:") else 1):
        raise ValueError("answer/query contract differs from task")
    expected_body = 3 if header.startswith("ATTESTATIONS:") else 2 if header.startswith(("TRANSFERS:", "HORNS:", "REGISTERS:", "STACKS:", "SNAPSHOTS:")) else 1
    if len(body) != expected_body:
        raise ValueError("complete body line count differs")
    if header.startswith("JOURNALS:"):
        balances = {}
        for row in _items(body[0]):
            m = re.fullmatch(r"(" + NAME + r") open (" + INTEGER + r") entries ([+-][0-9]+(?:,[+-][0-9]+)*)", row)
            if m is None or m[1] in balances:
                raise ValueError("journal row invalid")
            terms = [int(m[2]), *map(int, m[3].split(","))]
            # Separate positive/negative totals, not the constructor closure.
            balances[m[1]] = sum(x for x in terms if x >= 0) - sum(-x for x in terms if x < 0)
        return _answer(balances[query[0]])
    if header.startswith("TRANSFERS:"):
        if len(body) != 2 or not body[0].startswith("Initial ") or not body[1].startswith("Moves "):
            raise ValueError("transfer body invalid")
        opening = _named_values(body[0][8:].rstrip("."))
        moves = []
        for row in _items(body[1][6:]):
            m = re.fullmatch(r"(" + NAME + r")->(" + NAME + r"):([0-9]+)", row)
            if m is None or m[1] not in opening or m[2] not in opening:
                raise ValueError("transfer endpoints invalid")
            moves.append((m[1], m[2], int(m[3])))
        target = query[0]
        incidence = [int(dest == target) - int(src == target) for src, dest, _ in moves]
        return _answer(opening[target] + sum(coefficient * amount for coefficient, (_, _, amount) in zip(incidence, moves)))
    if header.startswith("HORNS:"):
        if len(body) != 2 or not body[0].startswith("Facts ") or not body[1].startswith("Rules "):
            raise ValueError("Horn body invalid")
        facts = set(body[0][6:].rstrip(".").split(","))
        productions = {}
        for row in _items(body[1][6:]):
            m = re.fullmatch(r"(" + NAME + r"(?:&" + NAME + r")*)>(" + NAME + r")", row)
            if m is None:
                raise ValueError("Horn rule invalid")
            productions.setdefault(m[2], []).append(m[1].split("&"))
        def prove(name, trail):
            if name in facts:
                return True
            if name in trail:
                return False
            return any(all(prove(x, trail | {name}) for x in antecedents) for antecedents in productions.get(name, []))
        return _answer("YES" if prove(query[0], set()) else "NO", kind="choice")
    if header.startswith("LINKS:"):
        if len(query) != 2:
            raise ValueError("link query needs origin and target")
        edges = []
        for row in _items(body[0]):
            m = re.fullmatch(r"(" + NAME + r")>(" + NAME + r")", row)
            if m is None:
                raise ValueError("link invalid")
            edges.append((m[1], m[2]))
        names = sorted({n for edge in edges for n in edge} | set(query))
        reach = {(a, b): False for a in names for b in names}
        for edge in edges:
            reach[edge] = True
        for middle in names:
            for left in names:
                for right in names:
                    reach[left, right] = reach[left, right] or reach[left, middle] and reach[middle, right]
        return _answer("YES" if reach[tuple(query)] else "NO", kind="choice")
    if header.startswith("REGISTERS:"):
        if len(body) != 2 or not body[0].startswith("Initial ") or not body[1].startswith("Steps "):
            raise ValueError("register body invalid")
        state = {k: ("n", v) for k, v in _named_values(body[0][8:].rstrip(".")).items()}
        for _, instruction in _steps(body[1][6:]):
            branch = re.fullmatch(r"IF (" + NAME + r")>(" + INTEGER + r") THEN (" + NAME + r")=(.+) ELSE (" + NAME + r")=(.+)", instruction)
            if branch:
                if branch[3] != branch[5] or branch[1] not in state or branch[3] not in state:
                    raise ValueError("register branch targets invalid")
                state[branch[3]] = ("if", state[branch[1]], int(branch[2]), _expression(branch[4], state), _expression(branch[6], state))
            else:
                assignment = re.fullmatch(r"(" + NAME + r")=(.+)", instruction)
                if assignment is None or assignment[1] not in state:
                    raise ValueError("register assignment invalid")
                state[assignment[1]] = _expression(assignment[2], state)
        return _answer(_evaluate(state[query[0]]))
    if header.startswith("STACKS:"):
        if len(body) != 2 or not body[0].startswith("Initial ") or not body[1].startswith("Steps "):
            raise ValueError("stack body invalid")
        stacks = {}
        for item in body[0][8:].rstrip(".").split("|"):
            m = re.fullmatch(r"(" + NAME + r")=\[(" + INTEGER + r"(?:," + INTEGER + r")*)\]", item)
            if m is None or m[1] in stacks:
                raise ValueError("initial stack invalid")
            stacks[m[1]] = [("n", int(x)) for x in m[2].split(",")]
        for _, instruction in _steps(body[1][6:]):
            m = re.fullmatch(r"(" + NAME + r") (P (" + INTEGER + r")|ADD|MUL)", instruction)
            if m is None or m[1] not in stacks:
                raise ValueError("stack instruction invalid")
            stack = stacks[m[1]]
            if m[2].startswith("P "):
                stack.append(("n", int(m[3])))
            elif len(stack) < 2:
                raise ValueError("stack underflow")
            else:
                right, left = stack.pop(), stack.pop()
                stack.append(("+" if m[2] == "ADD" else "*", left, right))
        return _answer(_evaluate(stacks[query[0]][-1]))
    if header.startswith("MEMORY:") or header.startswith("MESSAGES:"):
        histories = {}
        for number, text in _steps(body[0]):
            m = re.fullmatch(r"(" + NAME + r") (SET|CLEAR|QUOTE|OFFER|ACK|REVOKE)(?: (" + INTEGER + r"))?", text)
            if m is None or (m[2] in ("SET", "QUOTE", "OFFER", "ACK")) != (m[3] is not None):
                raise ValueError("owner event invalid")
            histories.setdefault(m[1], []).append((number, m[2], None if m[3] is None else int(m[3])))
        history = histories[query[0]]
        if header.startswith("MEMORY:"):
            effective = [(action, value) for _, action, value in reversed(history) if action in ("SET", "CLEAR")]
            value = effective[0][1] if effective and effective[0][0] == "SET" else None
        else:
            cutoff = max((i for i, (_, action, _) in enumerate(history) if action == "REVOKE"), default=-1)
            value = None
            for i in range(len(history) - 1, cutoff, -1):
                _, action, code = history[i]
                if action != "ACK":
                    continue
                offer = max((j for j in range(i) if history[j][1] == "OFFER"), default=-1)
                if offer >= 0 and history[offer][2] == code and not any(a == "ACK" and v == code for _, a, v in history[offer + 1:i]):
                    value = code; break
        return _answer("UNKNOWN", "MISSING", "integer_or_unknown") if value is None else _answer(value, kind="integer_or_unknown")
    if header in (DOCUMENT_HEADER + DOCUMENT_RULE, DOCUMENT_HEADER + LATEST + DOCUMENT_RULE):
        dated = header == DOCUMENT_HEADER + LATEST + DOCUMENT_RULE
        records = []
        pattern = r"(" + NAME + r") ([AD]) " + (r"d([1-9][0-9]*) " if dated else "") + r"(" + INTEGER + r")"
        for row in _items(body[0]):
            m = re.fullmatch(pattern, row)
            if m is None:
                raise ValueError("approved record invalid")
            if m[1] == query[0] and m[2] == "A":
                records.append((int(m[3]) if dated else 0, int(m[4] if dated else m[3])))
        # A current record has no higher-date approved witness. This is distinct
        # from the bank reference's max-date filter.
        current = [value for date, value in records if not any(other > date for other, _ in records)]
        return _consensus(current)
    if header.startswith("SNAPSHOTS:"):
        if len(body) != 2 or not body[0].startswith("Current ") or not body[1].startswith("Rows "):
            raise ValueError("snapshot body invalid")
        current = {}
        for row in _items(body[0][8:]):
            m = re.fullmatch(r"(" + NAME + r")=v([1-9][0-9]*)", row)
            if m is None or m[1] in current:
                raise ValueError("current pointer invalid")
            current[m[1]] = int(m[2])
        values = []
        for row in _items(body[1][5:]):
            m = re.fullmatch(r"(" + NAME + r")/v([1-9][0-9]*) (SIGNED|UNSIGNED) (" + INTEGER + r")", row)
            if m is None:
                raise ValueError("snapshot row invalid")
            if m[1] == query[0] and m[3] == "SIGNED" and int(m[2]) == current[query[0]]:
                values.append(int(m[4]))
        return _consensus(values)
    if header.startswith("ATTESTATIONS:"):
        if len(body) != 3 or not body[0].startswith("Authorized ") or not body[1].startswith("Revoked ") or not body[2].startswith("Receipts "):
            raise ValueError("attestation body invalid")
        authorized = set(body[0][11:].rstrip(".").split(","))
        revoked = set(body[1][8:].rstrip(".").split(","))
        values = []
        seen = set()
        for row in _items(body[2][9:]):
            m = re.fullmatch(r"(R[1-9][0-9]*) (" + NAME + r") (" + NAME + r") (" + INTEGER + r")", row)
            if m is None or m[1] in seen:
                raise ValueError("receipt invalid/duplicate")
            seen.add(m[1])
            if m[2] == query[0] and m[3] in authorized and m[1] not in revoked:
                values.append(int(m[4]))
        return _consensus(values)
    raise ValueError("unknown fresh task grammar")
