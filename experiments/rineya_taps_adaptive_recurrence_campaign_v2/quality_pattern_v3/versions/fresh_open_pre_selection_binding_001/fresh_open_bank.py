"""Frozen-size fresh OPEN CPU apparatus; no old-world/model/tokenizer import.

Production generation is release-gated and returns data only. The actual bank
must not be generated before independent source review and selected-policy
freeze. Tests use a separate unmistakably fixture-only nonce. Both construction
and verification authorship is disclosed; distinct algorithms are not distinct
human authors. Root owns later prompt-only runtime export and native admission.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

STUDY = Path(__file__).resolve().parent
VERSION = "FRESH-OPEN-216-V1"
NONCE = "f0216a9d60b74c8e"  # Assigned once before any production case/outcome.
SEED = 17
FIXTURE_NONCE = "CPU-FIXTURE-ONLY-NOT-REPLICATION"
FAMILIES = ("signed_ledgers", "deduction_composition", "program_state", "dialogue_memory",
            "document_retrieval_reconciliation", "answerability_source_conflict")
MECHANISMS = {
    "signed_ledgers": ("mixed_sign_journal", "transfer_conservation"),
    "deduction_composition": ("horn_dag", "directed_reachability"),
    "program_state": ("register_reassignment_branch", "stack_expression_trace"),
    "dialogue_memory": ("declaration_clear_quote", "offer_ack_revoke"),
    "document_retrieval_reconciliation": ("approved_consensus", "named_current_snapshot"),
    "answerability_source_conflict": ("approved_bundle_exchange", "issuer_revocation"),
}
PARENTS_PER_MECHANISM = 18
PARENTS = 216
CASES = 432
GENERATION_CAP = 8
DOCUMENT_HEADER = "DOCUMENTS: use only A-approved records"
DOCUMENT_RULE = ". Equal approved values give that value; absent/conflicting values give UNKNOWN."
LATEST = "; latest approved date per entity"
HEADERS = {
    "journal": "JOURNALS: balance is opening plus signed entries.",
    "transfer": "TRANSFERS: each move subtracts from source and adds to destination; balances may be negative.",
    "horn": "HORNS: infer a rule's right symbol only if every left symbol is known; otherwise NO.",
    "links": "LINKS: YES requires a directed path of one or more links; otherwise NO.",
    "register": "REGISTERS: execute numbered assignments in order with current values.",
    "stack": "STACKS: numbered commands; P pushes; ADD/MUL replace the top two by sum/product.",
    "memory": "MEMORY: numbered SET replaces; CLEAR removes; QUOTE changes nothing.",
    "messages": "MESSAGES: numbered OFFER sets pending; matching ACK replaces active and consumes pending; REVOKE clears active only; QUOTE inert.",
    "snapshots": "SNAPSHOTS: only named current revision and SIGNED rows count; agree or UNKNOWN for absent/conflicting values.",
    "attestations": "ATTESTATIONS: authorized unrevoked receipts count; agree or UNKNOWN for absent/conflicting values.",
}
RESERVED = {"YES", "NOT", "AND", "SET", "ADD", "MUL", "SUB", "ACK", "GET", "PUT", "PAY", "TOP", "ROW", "NIL", "ALL"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def semantic_world_digest(world):
    """Commit facts and paired interventions, never bookkeeping or answers.

    This is a symbolic-fact byte commitment, not a graph-isomorphism or proof
    of independent authorship. Common primitives/templates remain disclosed.
    Query/owner lists are reconstructed from the actual fact fields; opaque
    indices or a different question cannot pad a duplicated world into a new
    independent parent.
    """
    excluded = {"family", "mechanism", "index", "hard", "query_role", "alternate_role",
        "pair_kind", "kind", "answers", "statuses", "positive_gold", "negative_status",
        "known_variant", "branch_truth", "names", "query", "alternate"}
    facts = {key: value for key, value in world.items() if key not in excluded}
    # The exchange pair's second factual state is a real intervention. Commit
    # both states independently of presentation/known_variant ordering.
    if world.get("type") == "documents" and world.get("pair_kind") == "EVIDENCE_TRANSFER":
        alternate = {name: list(records) for name, records in world["bundles"].items()}
        q, a = world["query"], world["alternate"]
        alternate[q], alternate[a] = alternate[a], alternate[q]
        facts["intervened_bundles"] = alternate
    return digest(facts)


def _rng(nonce, *parts):
    return random.Random(int.from_bytes(hashlib.sha256(json.dumps([VERSION, nonce, SEED, *parts], separators=(",", ":")).encode()).digest(), "big"))


def _names(rng, count):
    names = []
    while len(names) < count:
        name = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(3))
        if name not in RESERVED and name not in names:
            names.append(name)
    return names


def _put_roles(values, q, a, wanted, alternative):
    values = list(values)
    for role, value in ((q, wanted), (a, alternative)):
        where = values.index(value)
        values[role], values[where] = values[where], values[role]
    return values


def _numbers(rng, names, q, a, index):
    vals = [-rng.randint(31, 149), 0, rng.randint(31, 149), rng.randint(211, 399)]
    wanted = index // 2 % 3
    assigned = _put_roles(vals, q, a, vals[wanted], vals[(wanted + 1) % 3])
    return dict(zip(names, assigned))


def _status_roles(q, a, known, unknown):
    values = ["ANSWER", "ANSWER", "MISSING", "CONFLICT"]
    return _put_roles(values, q, a, "ANSWER" if known else unknown, unknown if known else "ANSWER")


def _assignment(target, source, offset):
    return f"{target}={source}{'+' if offset >= 0 else '-'}{abs(offset)}"


def _construct(nonce, family, mechanism, index):
    rng = _rng(nonce, family, mechanism, index, "semantic")
    names = _names(rng, 14 if family == "deduction_composition" else 7)
    owners = names[:4]
    q, a = (index + 2 * mechanism) % 4, (index + 2 * mechanism + 2) % 4
    hard = (index + index // 2 + mechanism) % 2 == 1
    known = (index // 2 + mechanism) % 2 == 0
    w = {"names": owners, "query": owners[q], "alternate": owners[a], "query_role": q,
         "alternate_role": a, "hard": hard, "family": family, "mechanism": mechanism,
         "index": index, "pair_kind": "NEUTRAL_VARIANT" if index % 2 == 0 else "QUERY_COUNTERFACTUAL"}
    if family == "signed_ledgers":
        final = _numbers(rng, owners, q, a, index)
        if mechanism == 0:
            journals = []
            for name in owners:
                opening = rng.randint(-20, 20)
                changes = [rng.randint(11, 35), -rng.randint(11, 35), rng.randint(-9, 9)]
                if hard:
                    x = rng.randint(2, 8); changes += [x, -x]
                changes.append(final[name] - opening - sum(changes))
                # Terms are independent of print order and mixed/cancelling.
                journals.append((name, opening, changes))
            w.update(type="journal", journals=journals)
        else:
            moves = [(owners[i], owners[(i + 1) % 4], rng.randint(4, 23)) for i in range(4)]
            if hard:
                moves += [(owners[0], owners[2], rng.randint(4, 15)), (owners[3], owners[1], rng.randint(4, 15))]
            opening = {name: final[name] - sum(amount for _, dest, amount in moves if dest == name)
                       + sum(amount for src, _, amount in moves if src == name) for name in owners}
            w.update(type="transfer", opening=opening, moves=moves)
        w.update(kind="integer", answers=final, statuses=dict.fromkeys(owners, "ANSWER"))
    elif family == "deduction_composition":
        positive, negative = owners[q] if known else owners[a], owners[a] if known else owners[q]
        if mechanism == 0:
            s, t, mid, extra = names[4:8]
            facts = [s, t]
            rules = [([s, t], mid), ([mid], positive), ([extra], negative)]
            if hard:
                more = names[8]
                rules[1] = ([mid], more); rules.append(([more, t], positive))
            # Construction proof: positive chain has all facts; negative needs
            # extra, which has no fact/production. The text verifier works back.
            answers = dict.fromkeys(owners, "NO"); answers[positive] = "YES"
            w.update(type="horn", facts=facts, rules=rules, answers=answers)
        else:
            origin, middle, decoy = names[4:7]
            edges = [(origin, middle), (middle, positive), (decoy, negative)]
            if hard:
                bridge = names[7]; edges[1] = (middle, bridge); edges.append((bridge, positive))
            # The unconnected decoy is deliberately a printed path to negative.
            w.update(type="links", origin=origin, edges=edges,
                     answers={name: "YES" if name == positive else "NO" for name in owners})
        w.update(kind="choice", statuses=dict.fromkeys(owners, "ANSWER"))
    elif family == "program_state":
        final = _numbers(rng, owners, q, a, index)
        if mechanism == 0:
            opening = {name: rng.randint(-13, 13) for name in owners}
            current = dict(opening); instructions = []
            for i, name in enumerate(owners):
                other = owners[(i + 1) % 4]
                instructions.append(_assignment(name, other, final[name] - current[other]))
                current[name] = final[name]
            if hard:
                condition = owners[(q + 1) % 4]
                truth = index % 2 == 0
                threshold = current[condition] - 3 if truth else current[condition] + 3
                source = owners[(q + 3) % 4]
                good = _assignment(owners[q], source, final[owners[q]] - current[source])
                bad = _assignment(owners[q], source, final[owners[q]] + 7 - current[source])
                instructions.append(f"IF {condition}>{threshold} THEN {good if truth else bad} ELSE {bad if truth else good}")
            w.update(type="register", opening=opening, instructions=instructions,
                     branch_truth=None if not hard else truth)
        else:
            initial, instructions = {}, []
            for name in owners:
                factor = rng.choice((2, 3))
                # Closed construction expression, independent of both execution
                # solvers. The visible stack evaluates (left+right)*factor.
                desired = final[name] * factor
                left = rng.randint(-20, 20); right = final[name] - left
                initial[name] = [left, right]
                instructions += [f"{name} ADD", f"{name} P {factor}", f"{name} MUL"]
                final[name] = desired
            if hard:
                shift = rng.randint(2, 8)
                instructions += [f"{owners[q]} P {shift}", f"{owners[q]} ADD"]
                # Make the added shift a cancelling construction term by
                # changing the initial right operand; final value remains fixed.
                factor = int(instructions[3 * q + 1].split()[-1])
                # A multiple of factor gives an exact integer cancellation.
                instructions[-2] = f"{owners[q]} P {shift * factor}"
                initial[owners[q]][1] -= shift
            w.update(type="stack", initial=initial, instructions=instructions)
        w.update(kind="integer", answers=final, statuses=dict.fromkeys(owners, "ANSWER"))
    elif family == "dialogue_memory":
        status = _status_roles(q, a, known, "MISSING")
        # Dialogue has two active and two absent owners; CONFLICT is not an
        # owner-state. Convert the distractor CONFLICT slot to MISSING.
        status = ["MISSING" if s == "CONFLICT" else s for s in status]
        codes = rng.sample(range(101, 9900), 12)
        histories = []
        for i, name in enumerate(owners):
            old, new, noise = codes[3 * i:3 * i + 3]
            if mechanism == 0:
                h = [("SET", old), ("CLEAR", None), ("QUOTE", noise), ("SET", new)] if status[i] == "ANSWER" else [("SET", old), ("QUOTE", new), ("CLEAR", None), ("QUOTE", noise)]
            else:
                h = [("OFFER", old), ("ACK", old), ("OFFER", new), ("ACK", new)] if status[i] == "ANSWER" else [("OFFER", old), ("ACK", old), ("OFFER", new), ("REVOKE", None)]
            histories.append((name, h))
        events = []
        for stage in range(4):
            order = list(range(4)); rng.shuffle(order)
            for i in order:
                name, h = histories[i]; action, value = h[stage]
                events.append((len(events) + 1, name, action, value))
        if mechanism == 1 and hard:
            events.append((len(events) + 1, owners[q], "QUOTE", codes[-1]))
        w.update(type="memory" if mechanism == 0 else "messages", events=events, kind="integer_or_unknown",
                 answers={name: str(codes[3 * i + 1]) if status[i] == "ANSWER" else "UNKNOWN" for i, name in enumerate(owners)},
                 statuses=dict(zip(owners, status)))
    elif family == "document_retrieval_reconciliation" or mechanism == 0:
        desired_unknown = "MISSING" if index % 2 == 0 else "CONFLICT"
        if family == "answerability_source_conflict":
            status = _status_roles(q, a, True, desired_unknown)
            w["pair_kind"] = "EVIDENCE_TRANSFER"
        else:
            status = _status_roles(q, a, known, desired_unknown)
        values = rng.sample(range(101, 900), 12)
        answers, statuses = {}, dict(zip(owners, status))
        if mechanism == 0:
            bundles = {}
            for i, name in enumerate(owners):
                v, other, noise = values[3 * i:3 * i + 3]
                flag = "D" if status[i] == "MISSING" else "A"
                bundles[name] = [(flag, 2, v), (flag, 2, other if status[i] == "CONFLICT" else v), ("D", 2, noise)]
                if hard:
                    bundles[name].append(("A" if status[i] != "MISSING" else "D", 1, noise))
                answers[name] = str(v) if status[i] == "ANSWER" else "UNKNOWN"
            w.update(type="documents", bundles=bundles, dated=hard)
        else:
            current, snapshots = {}, []
            for i, name in enumerate(owners):
                rev = rng.randint(2, 6); current[name] = rev
                v, other, noise = values[3 * i:3 * i + 3]
                flag = "UNSIGNED" if status[i] == "MISSING" else "SIGNED"
                snapshots += [(name, rev, flag, v), (name, rev, flag, other if status[i] == "CONFLICT" else v),
                              (name, rev - 1, "SIGNED", noise), (name, rev + 2, "UNSIGNED", other)]
                answers[name] = str(v) if status[i] == "ANSWER" else "UNKNOWN"
            w.update(type="snapshots", current=current, snapshots=snapshots)
        w.update(kind="integer_or_unknown", answers=answers, statuses=statuses)
        if family == "answerability_source_conflict":
            w.update(positive_gold=answers[owners[q]], negative_status=desired_unknown,
                     known_variant=(index + mechanism) % 2)
    else:
        # New revocation mechanism: receipt data is identical across variants;
        # only the revocation relation changes. Issuer authority is global.
        issuers = names[4:7]
        authorized = issuers[:2]
        value = rng.randint(101, 899); other = value + rng.randint(1, 20)
        target = owners[q]; dummy = owners[a]
        receipts = [("R1", target, authorized[0], value), ("R2", target, authorized[1], value),
                    ("R3", target, authorized[0], other), ("R4", dummy, authorized[0], value + 31),
                    ("R5", dummy, authorized[1], value + 31), ("R6", dummy, authorized[0], value + 31),
                    ("R7", owners[(q + 1) % 4], issuers[2], value + 42),
                    ("R8", owners[(q + 3) % 4], issuers[2], value + 57)]
        missing = index % 2 == 0
        positive_revoked = ["R3", "R5", "R6"]
        negative_revoked = ["R1", "R2", "R3"] if missing else ["R4", "R5", "R6"]
        shuffled_ids = [r[0] for r in receipts]; rng.shuffle(shuffled_ids)
        identifiers = dict(zip([r[0] for r in receipts], shuffled_ids))
        receipts = [(identifiers[rid], name, issuer, amount) for rid, name, issuer, amount in receipts]
        positive_revoked = [identifiers[rid] for rid in positive_revoked]
        negative_revoked = [identifiers[rid] for rid in negative_revoked]
        w.update(type="attestations", receipts=receipts, authorized=authorized,
                 positive_revoked=positive_revoked, negative_revoked=negative_revoked,
                 positive_gold=str(value), negative_status="MISSING" if missing else "CONFLICT",
                 known_variant=(index + mechanism) % 2, pair_kind="EVIDENCE_TRANSFER", kind="integer_or_unknown")
    return w


def _question(target, kind, style):
    reply = "Reply only YES or NO." if kind == "choice" else "Reply one integer or UNKNOWN." if kind == "integer_or_unknown" else "Reply one integer."
    return ("Question: " + target + "? " if style == 0 else "Query: " + target + ". ") + reply


def _render(world, nonce, variant):
    rng = _rng(nonce, world["family"], world["mechanism"], world["index"], "render", variant)
    type_ = world["type"]
    target = world["query"] if variant == 0 or world["pair_kind"] != "QUERY_COUNTERFACTUAL" else world["alternate"]
    style = (world["index"] // 2 + world["mechanism"]) % 2
    if world["pair_kind"] == "NEUTRAL_VARIANT":
        style = (style + variant) % 2
    body = []
    if type_ == "journal":
        records = []
        for name, opening, changes in world["journals"]:
            changes = list(changes); rng.shuffle(changes)
            records.append(f"{name} open {opening} entries " + ",".join(f"{v:+}" for v in changes))
        rng.shuffle(records); body = [";".join(records) + "."]
    elif type_ == "transfer":
        opening = list(world["opening"].items()); rng.shuffle(opening)
        moves = list(world["moves"]); rng.shuffle(moves)
        body = ["Initial " + ",".join(f"{k}={v}" for k, v in opening) + ".",
                "Moves " + ";".join(f"{a}->{b}:{v}" for a, b, v in moves) + "."]
    elif type_ == "horn":
        facts, rules = list(world["facts"]), list(world["rules"]); rng.shuffle(facts); rng.shuffle(rules)
        body = ["Facts " + ",".join(facts) + ".", "Rules " + ";".join("&".join(a) + ">" + b for a, b in rules) + "."]
    elif type_ == "links":
        edges = list(world["edges"]); rng.shuffle(edges)
        body = [";".join(a + ">" + b for a, b in edges) + "."]
        target = world["origin"] + " to " + target
    elif type_ in ("register", "stack"):
        if type_ == "register":
            items = list(world["opening"].items()); rng.shuffle(items)
            initial = ",".join(f"{k}={v}" for k, v in items)
        else:
            items = list(world["initial"].items()); rng.shuffle(items)
            initial = "|".join(k + "=[" + ",".join(map(str, v)) + "]" for k, v in items)
        steps = list(enumerate(world["instructions"], 1)); rng.shuffle(steps)
        body = ["Initial " + initial + ".", "Steps " + ";".join(f"{i} {text}" for i, text in steps) + "."]
    elif type_ in ("memory", "messages"):
        events = list(world["events"]); rng.shuffle(events)
        body = [";".join(f"{n} {owner} {action}" + (f" {value}" if value is not None else "") for n, owner, action, value in events) + "."]
    elif type_ == "documents":
        bundles = {k: list(v) for k, v in world["bundles"].items()}
        if world["pair_kind"] == "EVIDENCE_TRANSFER" and variant != world["known_variant"]:
            bundles[world["query"]], bundles[world["alternate"]] = bundles[world["alternate"]], bundles[world["query"]]
        records = [f"{name} {flag}" + (f" d{date}" if world["dated"] else "") + f" {value}"
                   for name, rows in bundles.items() for flag, date, value in rows]
        rng.shuffle(records); body = [";".join(records) + "."]
    elif type_ == "snapshots":
        currents = list(world["current"].items()); rows = list(world["snapshots"]); rng.shuffle(currents); rng.shuffle(rows)
        body = ["Current " + ";".join(f"{name}=v{rev}" for name, rev in currents) + ".",
                "Rows " + ";".join(f"{name}/v{rev} {flag} {value}" for name, rev, flag, value in rows) + "."]
    elif type_ == "attestations":
        rows = list(world["receipts"]); rng.shuffle(rows)
        revoked = list(world["positive_revoked"] if variant == world["known_variant"] else world["negative_revoked"]); rng.shuffle(revoked)
        issuers = list(world["authorized"]); rng.shuffle(issuers)
        body = ["Authorized " + ",".join(issuers) + ".", "Revoked " + ",".join(revoked) + ".",
                "Receipts " + ";".join(f"{rid} {name} {issuer} {value}" for rid, name, issuer, value in rows) + "."]
    if type_ == "documents":
        header = DOCUMENT_HEADER + (LATEST if world["dated"] else "") + DOCUMENT_RULE
    else:
        header = HEADERS[type_]
    prompt = "\n".join([header, *body, _question(target, world["kind"], style)])
    if world["pair_kind"] == "EVIDENCE_TRANSFER":
        gold, status = (world["positive_gold"], "ANSWER") if variant == world["known_variant"] else ("UNKNOWN", world["negative_status"])
    else:
        gold, status = str(world["answers"][target if type_ != "links" else target.split(" to ")[1]]), world["statuses"][target if type_ != "links" else target.split(" to ")[1]]
    return prompt, gold, status, style


def reference_solve(prompt):
    """Forward reference algorithms; input is a visible string, not case labels."""
    lines = prompt.split("\n"); header = lines[0]; body = lines[1:-1]
    match = re.fullmatch(r"(?:Question: (.+)\?|Query: (.+)\.) Reply (?:one integer(?: or UNKNOWN)?|only YES or NO)\.", lines[-1])
    if match is None:
        raise ValueError("reference query invalid")
    query = (match[1] or match[2]).split(" to ")
    def result(value, status="ANSWER", kind="integer"):
        return {"answer": str(value), "source_status": status, "answer_kind": kind}
    def consensus(values):
        unique = set(values)
        return result(next(iter(unique)), kind="integer_or_unknown") if len(unique) == 1 else result("UNKNOWN", "MISSING" if not unique else "CONFLICT", "integer_or_unknown")
    if header == HEADERS["journal"]:
        values = {}
        for row in body[0][:-1].split(";"):
            m = re.fullmatch(r"([A-Z]{3}) open (-?[0-9]+) entries (.+)", row)
            if m is None: raise ValueError("journal syntax")
            values[m[1]] = int(m[2]) + sum(int(v) for v in m[3].split(","))
        return result(values[query[0]])
    if header == HEADERS["transfer"]:
        values = {k: int(v) for k, v in (x.split("=") for x in body[0][8:-1].split(","))}
        for row in body[1][6:-1].split(";"):
            m = re.fullmatch(r"([A-Z]{3})->([A-Z]{3}):([0-9]+)", row)
            if m is None: raise ValueError("transfer syntax")
            values[m[1]] -= int(m[3]); values[m[2]] += int(m[3])
        return result(values[query[0]])
    if header == HEADERS["horn"]:
        known = set(body[0][6:-1].split(",")); rules = [(x.split(">")[0].split("&"), x.split(">")[1]) for x in body[1][6:-1].split(";")]
        for _ in range(len(rules) + 1):
            before = len(known)
            for left, right in rules:
                if all(n in known for n in left): known.add(right)
            if len(known) == before: break
        return result("YES" if query[0] in known else "NO", kind="choice")
    if header == HEADERS["links"]:
        edges = [tuple(x.split(">")) for x in body[0][:-1].split(";")]
        reached = {b for a, b in edges if a == query[0]}
        for _ in range(len(edges) + 1):
            after = reached | {b for a, b in edges if a in reached}
            if after == reached: break
            reached = after
        return result("YES" if query[1] in reached else "NO", kind="choice")
    if header in (HEADERS["register"], HEADERS["stack"]):
        instructions = sorted((int(x.split(" ", 1)[0]), x.split(" ", 1)[1]) for x in body[1][6:-1].split(";"))
        if header == HEADERS["register"]:
            values = {k: int(v) for k, v in (x.split("=") for x in body[0][8:-1].split(","))}
            def apply(text):
                m = re.fullmatch(r"([A-Z]{3})=([A-Z]{3})([+*-])([0-9]+)", text)
                if m is None: raise ValueError("register expression")
                x, c = values[m[2]], int(m[4]); values[m[1]] = x + c if m[3] == "+" else x - c if m[3] == "-" else x * c
            for _, text in instructions:
                m = re.fullmatch(r"IF ([A-Z]{3})>(-?[0-9]+) THEN (.+) ELSE (.+)", text)
                apply(m[3] if values[m[1]] > int(m[2]) else m[4]) if m else apply(text)
            return result(values[query[0]])
        stacks = {x.split("=")[0]: list(map(int, x.split("=")[1][1:-1].split(","))) for x in body[0][8:-1].split("|")}
        for _, text in instructions:
            name, op, *rest = text.split(); stack = stacks[name]
            if op == "P": stack.append(int(rest[0]))
            else:
                right, left = stack.pop(), stack.pop(); stack.append(left + right if op == "ADD" else left * right)
        return result(stacks[query[0]][-1])
    if header in (HEADERS["memory"], HEADERS["messages"]):
        active, pending = {}, {}
        events = sorted((int(x.split(" ", 1)[0]), x.split(" ", 1)[1].split()) for x in body[0][:-1].split(";"))
        for _, event in events:
            name, action, *rest = event; value = int(rest[0]) if rest else None
            if action == "SET": active[name] = value
            elif action in ("CLEAR", "REVOKE"): active.pop(name, None)
            elif action == "OFFER": pending[name] = value
            elif action == "ACK" and pending.get(name) == value: active[name] = value; pending.pop(name)
        return result(active[query[0]], kind="integer_or_unknown") if query[0] in active else result("UNKNOWN", "MISSING", "integer_or_unknown")
    if header in (DOCUMENT_HEADER + DOCUMENT_RULE, DOCUMENT_HEADER + LATEST + DOCUMENT_RULE):
        dated = header == DOCUMENT_HEADER + LATEST + DOCUMENT_RULE
        relevant = []
        for row in body[0][:-1].split(";"):
            pieces = row.split()
            if pieces[0] == query[0] and pieces[1] == "A": relevant.append((int(pieces[2][1:]) if dated else 0, int(pieces[-1])))
        latest = max((date for date, _ in relevant), default=None)
        return consensus([value for date, value in relevant if date == latest])
    if header == HEADERS["snapshots"]:
        current = {k: int(v[1:]) for k, v in (x.split("=") for x in body[0][8:-1].split(";"))}
        relevant = []
        for row in body[1][5:-1].split(";"):
            name_rev, flag, value = row.split(); name, rev = name_rev.split("/v")
            if name == query[0] and flag == "SIGNED" and int(rev) == current[name]: relevant.append(int(value))
        return consensus(relevant)
    if header == HEADERS["attestations"]:
        authorized, revoked = set(body[0][11:-1].split(",")), set(body[1][8:-1].split(","))
        relevant = []
        for row in body[2][9:-1].split(";"):
            rid, owner, issuer, value = row.split()
            if owner == query[0] and issuer in authorized and rid not in revoked: relevant.append(int(value))
        return consensus(relevant)
    raise ValueError("unknown reference grammar")


def normalize_response(response):
    if type(response) is not str: return None
    value = response.strip()
    if re.fullmatch(r"0|-?[1-9]\d*", value): return value
    return value.upper() if value.upper() in ("YES", "NO", "UNKNOWN") else None


def score_response(response, case):
    value = normalize_response(response)
    integer = value is not None and re.fullmatch(r"0|-?[1-9]\d*", value) is not None
    kind = case["answer_kind"]
    valid = value in ("YES", "NO") if kind == "choice" else (integer or value == "UNKNOWN") if kind == "integer_or_unknown" else integer
    return {"correct": bool(valid and value == case["gold"]), "format_valid": bool(valid),
            "canonical_output": value, "false_UNKNOWN": value == "UNKNOWN" and case["gold"] != "UNKNOWN",
            "unsupported_canonical_answer": bool(valid and value != "UNKNOWN" and case["gold"] == "UNKNOWN")}


def _build(nonce, fixture):
    rows = []
    for family in FAMILIES:
        for mechanism in (0, 1):
            for index in range(PARENTS_PER_MECHANISM):
                world = _construct(nonce, family, mechanism, index)
                parent = "FO-" + digest([VERSION, nonce, family, mechanism, index])[:28]
                pair = []
                for variant in (0, 1):
                    prompt, gold, status, style = _render(world, nonce, variant)
                    pair.append({"case_id": "FC-" + digest([parent, variant])[:28], "parent_id": parent,
                        "family": family, "mechanism": MECHANISMS[family][mechanism], "mechanism_index": mechanism,
                        "semantic_world_sha256": semantic_world_digest(world), "variant_id": variant, "pair_kind": world["pair_kind"],
                        "difficulty": "hard" if world["hard"] else "easy", "wording_style": style,
                        "query_role": world["query_role"] if variant == 0 or world["pair_kind"] != "QUERY_COUNTERFACTUAL" else world["alternate_role"],
                        "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "gold": gold, "answer_kind": world["kind"], "source_status": status,
                        "max_new_tokens": GENERATION_CAP, "tokenizer_shape_status": "NOT_BOUND_ROOT_GATE_REQUIRED",
                        "exposure": "CPU_FIXTURE_ONLY" if fixture else "FROZEN_POLICY_OPEN_REPLICATION_NOT_MEASURED",
                        "nonce": nonce, "seed": SEED, "generator_version": VERSION,
                        "grammar_role": "SHARED_FROZEN_DOCUMENTS_CONTRACT_NEW_WORLD" if world["type"] == "documents" else "UNRECOGNIZED_NEW_GRAMMAR_OR_MECHANISM",
                        "provenance": {"old_parent_ancestor": None, "old_generator_import": False,
                            "source_independence": "new local implementation/world; shared research author/spec and abstract primitives remain",
                            "human_author_independence": False}})
                changed = pair[0]["gold"] != pair[1]["gold"]
                for row in pair: row["expected_response_change"] = changed
                rows.extend(pair)
    return rows


def build_fixture_bank(nonce=FIXTURE_NONCE):
    if type(nonce) is not str or not nonce.startswith("CPU-FIXTURE-") or nonce == NONCE:
        raise PermissionError("only explicitly separate CPU fixture nonce allowed")
    return _build(nonce, True)


def generate_registered_bank(review_path, policy_freeze_path):
    """Returns the real fixed-nonce bank only after actual review/policy freeze.

    No CLI/file export is implemented. Root owns exact release receipts, actual
    data export, tokenizer/shape checks and native admission; this is not one.
    """
    paths = [Path(review_path).resolve(strict=True), Path(policy_freeze_path).resolve(strict=True)]
    if any(STUDY not in p.parents for p in paths):
        raise PermissionError("review and policy freeze must be V3-local")
    review, freeze = (json.loads(p.read_text(encoding="utf-8")) for p in paths)
    current = {name: sha(STUDY / name) for name in ("fresh_open_bank.py", "fresh_open_verifier.py", "tests/test_fresh_open_bank.py")}
    if review.get("status") != "PASS_FRESH_OPEN_CPU_SOURCE" or review.get("source_hashes") != current:
        raise PermissionError("independent exact fresh source review required")
    if freeze.get("status") != "FROZEN_SELECTED_POLICY_FOR_FRESH_OPEN" or freeze.get("fresh_nonce") != NONCE or freeze.get("fresh_seed") != SEED or freeze.get("fresh_source_hashes") != current or freeze.get("source_review_sha256") != sha(paths[0]):
        raise PermissionError("selected policy/fresh source/nonce freeze required")
    policy = (STUDY / "opposed_gate_policy.py").resolve(strict=True)
    if freeze.get("selected_policy_source_sha256") != sha(policy) or freeze.get("selected_policy") not in ("CONFIDENT_RECURRENCE", "UNCERTAIN_RECURRENCE", "EVIDENCE_CONSISTENCY_GUARD"):
        raise PermissionError("actual selected policy source must be frozen first")
    return _build(NONCE, False)


def blind_predictions(prompt):
    context, question = prompt.rsplit("\n", 1)
    nums = re.findall(r"[-+]?[0-9]+", context)
    names = [n for n in re.findall(r"\b[A-Z]{3}\b", context) if n not in RESERVED]
    def other_query(name):
        # Replace the destination of a reachability query, otherwise its sole
        # owner. Neither the original gold nor a status annotation is read.
        if " to " in question:
            altered = re.sub(r"(?<= to )[A-Z]{3}", name, question, count=1)
        else:
            altered = re.sub(r"(?<=: )[A-Z]{3}", name, question, count=1)
        try: return reference_solve(context + "\n" + altered)["answer"]
        except (ValueError, KeyError, IndexError): return ""
    predictions = {"constant_YES": "YES", "constant_NO": "NO", "constant_UNKNOWN": "UNKNOWN",
        "constant_0": "0", "constant_1": "1", "copy_full_prompt": prompt,
        "copy_first_number": nums[0] if nums else "", "copy_last_number": nums[-1] if nums else "",
        "copy_max_number": str(max(map(int, nums))) if nums else "", "copy_min_number": str(min(map(int, nums))) if nums else "",
        "copy_first_entity": names[0] if names else "", "copy_last_entity": names[-1] if names else "",
        "query_blind_first_entity_solver": other_query(names[0]) if names else "",
        "query_blind_last_entity_solver": other_query(names[-1]) if names else "",
        "document_header_UNKNOWN_else_NO": "UNKNOWN" if context.startswith("DOCUMENTS:") else "NO",
        "any_CLEAR_or_REVOKE_UNKNOWN_else_last_number": "UNKNOWN" if "CLEAR" in context or "REVOKE" in context else nums[-1] if nums else "",
        "rule_RHS_presence": "", "memory_max_value_ignore_state": ""}
    queried = re.search(r"(?:Question|Query): ([A-Z]{3})", question)
    if queried and context.startswith("HORNS:"):
        predictions["rule_RHS_presence"] = "YES" if ">" + queried[1] in context else "NO"
    if queried and context.startswith(("MEMORY:", "MESSAGES:")):
        values = re.findall(r"\b" + queried[1] + r" (?:SET|QUOTE|OFFER|ACK) (-?[0-9]+)", context)
        predictions["memory_max_value_ignore_state"] = str(max(map(int, values))) if values else "UNKNOWN"
    return predictions


def audit_rows(rows):
    """CPU apparatus audit; never candidate scoring or model inference."""
    from .fresh_open_verifier import verify_prompt
    errors, parent_groups, floors = [], defaultdict(list), defaultdict(Counter)
    counts = Counter(row["family"] for row in rows)
    if len(rows) != CASES or len({r["case_id"] for r in rows}) != CASES or counts != dict.fromkeys(FAMILIES, 72):
        errors.append("complete432 cases/sixfamilies differs")
    if len({r["prompt"] for r in rows}) != CASES:
        errors.append("duplicate visible prompt")
    controls = []
    for row in rows:
        parent_groups[row["parent_id"]].append(row)
        if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
            errors.append("prompt hash differs")
        try:
            forward, verified = reference_solve(row["prompt"]), verify_prompt(row["prompt"])
            expected = {"answer": row["gold"], "source_status": row["source_status"], "answer_kind": row["answer_kind"]}
            passed = forward == verified == expected
        except (ValueError, KeyError, IndexError) as exc:
            forward, verified, passed = None, type(exc).__name__, False
        controls.append({"case_id": row["case_id"], "constructor_forward_text_independent_text_agree": passed})
        if not passed: errors.append("positive control mismatch:" + row["case_id"])
        for name, answer in blind_predictions(row["prompt"]).items():
            correct = score_response(answer, row)["correct"]
            floors["overall"][name] += correct; floors[row["family"]][name] += correct
    if len(parent_groups) != PARENTS or len({r["semantic_world_sha256"] for r in rows}) != PARENTS:
        errors.append("complete216 distinct semantic parent worlds differs")
    for parent, pair in parent_groups.items():
        if len(pair) != 2 or {r["variant_id"] for r in pair} != {0, 1} or len({r["family"] for r in pair}) != 1 or len({r["semantic_world_sha256"] for r in pair}) != 1:
            errors.append("parent pair differs:" + parent)
        elif pair[0]["pair_kind"] == "NEUTRAL_VARIANT" and pair[0]["gold"] != pair[1]["gold"]:
            errors.append("neutral variant changes answer")
        elif pair[0]["pair_kind"] == "EVIDENCE_TRANSFER" and {r["source_status"] == "ANSWER" for r in pair} != {False, True}:
            errors.append("evidence pair lacks answerable/unanswerable exchange")
    if max(floors["overall"].values(), default=0) > CASES / 2:
        errors.append("global queryblind/copy/header/skeleton floor above half; no nonce roulette")
    if any(max(floors[f].values(), default=0) == 72 for f in FAMILIES):
        errors.append("perfect family blind shortcut; preparation refuses")
    return {"status": "CPU_APPARATUS_PASS_NOT_NATIVE_OR_SOURCE_REVIEW" if not errors else "CPU_PREPARATION_ERRORS",
        "errors": errors, "cases": len(rows), "parents": len(parent_groups), "models": 0,
        "tokenizer_shapes": "NOT_BOUND_ROOT_GATE_REQUIRED", "positive_controls": controls,
        "family_counts": dict(counts), "mechanism_counts": dict(Counter(r["mechanism"] for r in rows)),
        "difficulty_counts": dict(Counter(r["difficulty"] for r in rows)),
        "query_role_counts_by_family": {f: dict(Counter(r["query_role"] for r in rows if r["family"] == f)) for f in FAMILIES},
        "source_status_counts_by_family": {f: dict(Counter(r["source_status"] for r in rows if r["family"] == f)) for f in FAMILIES},
        "gold_sign_counts_by_numeric_family": {f: dict(Counter("zero" if r["gold"] == "0" else "negative" if r["gold"].startswith("-") else "positive" for r in rows if r["family"] == f)) for f in ("signed_ledgers", "program_state")},
        "floor_correct_counts": {k: dict(v) for k, v in floors.items()},
        "grammar_limits": "Only fresh documents/approved evidence mechanism0 shares frozen DOCUMENTS consensus grammar/primitives. Others are strict unrecognized grammar/mechanism holdouts; not independent human authors."}
