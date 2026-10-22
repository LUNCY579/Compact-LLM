"""OPEN task preparation only; standard library, no model/data-bank imports."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import secrets
import sys

VERSION = "OPEN-TAPS-001"
FAMILIES = (
    ("signed_entries", "DEVELOPMENT", 4),
    ("credit_debit", "OPEN_VALIDATION", 2),
    ("path_derivation", "DEVELOPMENT", 4),
    ("join_derivation", "OPEN_VALIDATION", 2),
    ("register_copy_add", "DEVELOPMENT", 4),
    ("register_mul_reset", "OPEN_VALIDATION", 2),
    ("approved_consensus", "DEVELOPMENT", 4),
    ("latest_consensus", "OPEN_VALIDATION", 2),
)
PACKAGE = Path(__file__).resolve().parent


def rng_for(nonce, seed, family, index):
    key = json.dumps([VERSION, nonce, seed, family, index], separators=(",", ":"))
    return random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest(), "big"))


def names(rng, count):
    result = []
    while len(result) < count:
        value = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(3))
        if value not in result and value not in ("YES", "NOT", "AND"):
            result.append(value)
    return result


def build_case(nonce, seed, family, split, index):
    rng = rng_for(nonce, seed, family, index)
    ids = names(rng, 6)
    query_slot = (index if split == "DEVELOPMENT" else index + 1) % 3
    if family in ("signed_entries", "credit_debit"):
        lines, answers = [], {}
        for name in ids[:3]:
            start, plus, minus, extra = rng.randint(10, 20), rng.randint(22, 29), rng.randint(3, 7), rng.randint(12, 19)
            if family == "signed_entries" and index % 2:
                start, plus, extra = rng.randint(1, 5), -plus, -extra
            answers[name] = start + plus - minus + extra
            if family == "signed_entries":
                entries = [f"{plus:+}", "-" + str(minus), f"{extra:+}"]
                rng.shuffle(entries)  # additive entries commute
                lines.append(f"{name} start {start}; entries {' '.join(entries)}.")
            else:
                lines.append(f"{name} start {start}; credits {plus},{extra}; debit {minus}.")
        rng.shuffle(lines)
        query = ids[query_slot]
        prompt = "Ledgers: sum each account's signed changes.\n" + "\n".join(lines) + f"\nQuestion: {query} final? Reply one integer."
        gold, category = str(answers[query]), "signed_arithmetic"
    elif family in ("path_derivation", "join_derivation"):
        a, b, c, d, e, f = ids
        if family == "path_derivation":
            rules, facts = [f"{a}>{b}", f"{b}>{c}", f"{d}>{e}", f"{e}>{f}"], [a]
        else:
            rules, facts = [f"{a}&{b}>{c}", f"{d}&{e}>{f}"], [a, b, d]
        rng.shuffle(rules)
        rng.shuffle(facts)
        positive = index % 2 == 0
        query = c if positive else f
        prompt = f"Rules derive only their right side when all left symbols are known. Rules: {','.join(rules)}. Facts: {','.join(facts)}.\nQuestion: derive {query}? Reply YES or NO."
        gold, category = ("YES" if positive else "NO"), "deduction_composition"
    elif family in ("register_copy_add", "register_mul_reset"):
        a, b, c = ids[:3]
        initial = {x: rng.randint(2, 5) for x in (a, b, c)}
        if family == "register_copy_add":
            steps = [f"1 {a}={b}+30", f"2 {b}={a}-4", f"3 {c}={b}+7", f"4 {a}={c}-2"]
            # Closed formula independent of the interpreter used by the verifier.
            values = {a: initial[b] + 31, b: initial[b] + 26, c: initial[b] + 33}
        else:
            steps = [f"1 {c}={a}*9", f"2 {b}={c}+5", f"3 {a}={b}+7", f"4 {c}={a}-3"]
            values = {a: initial[a] * 9 + 12, b: initial[a] * 9 + 5, c: initial[a] * 9 + 9}
        rng.shuffle(steps)  # presentation shuffled; step numbers define chronology
        query = (a, b, c)[query_slot]
        prompt = "Registers: " + ",".join(f"{x}={initial[x]}" for x in (a, b, c)) + ". Execute numbered steps in increasing order; each reads current values. Steps: " + ";".join(steps) + f".\nQuestion: {query}? Reply one integer."
        gold, category = str(values[query]), "program_state"
    else:
        entities = ids[:3]
        rng.shuffle(entities)  # status roles assigned symmetrically before query
        good, conflict, missing = entities
        values = rng.sample(range(21, 90), 6)
        records = []
        for name, status, first, second in (
            (good, "approved", values[0], values[0]),
            (conflict, "approved", values[1], values[2]),
            (missing, "draft", values[3], values[4]),
        ):
            if family == "latest_consensus":
                records.extend([f"{name} {status} d2 {first}", f"{name} {status} d2 {second}"])
                if status == "approved":
                    records.append(f"{name} approved d1 {values[5]}")
            else:
                records.extend([f"{name} {status} {first}", f"{name} {status} {second}"])
        rng.shuffle(records)
        # Each context contains all three statuses; query alone selects the endpoint.
        requested = ("answer", "missing", "answer", "conflict")[index] if split == "DEVELOPMENT" else ("answer", "conflict")[index]
        query = {"answer": good, "missing": missing, "conflict": conflict}[requested]
        rule = "Use approved values only; agree=>value, absent or conflict=>UNKNOWN."
        if family == "latest_consensus":
            rule = "Use only the latest dated approved values per entity; agree=>value, absent or conflict=>UNKNOWN."
        prompt = rule + " Records: " + ";".join(records) + f".\nQuestion: {query} value? Reply one integer or UNKNOWN."
        gold, category = (str(values[0]) if requested == "answer" else "UNKNOWN"), "document_reconciliation"
    return {
        "id": f"{VERSION}:{nonce}:{family}:{index}", "parent_id": f"{nonce}:{family}:{index}",
        "generator_family": family, "split": split, "category": category, "prompt": prompt,
        "gold": gold, "answer_kind": ("choice" if category == "deduction_composition"
                                      else "integer_or_unknown" if category == "document_reconciliation" else "integer"),
        "answerability": "unanswerable" if gold == "UNKNOWN" else "answerable",
        "source_status": requested if category == "document_reconciliation" else "NOT_APPLICABLE",
        "provenance": {"generator": VERSION, "nonce": nonce, "seed": seed, "historical_parents": [],
                       "source_kind": "new synthetic nonce world", "human_review": "NOT_REVIEWED"},
        "exposure": "OPEN_SOURCE_VISIBLE_NOT_NATIVE_EVALUATED",
        "independence": "SAME_AUTHOR_SHARED_GENERATOR_NOT_INDEPENDENT_CONFIRMATION",
        "shape": {"target_max_input_tokens": 350, "max_output_tokens": 8,
                  "native_tokenizer_binding": "NOT_RUN_REQUIRED_BEFORE_NATIVE"},
    }


def generate(nonce, seed=17):
    return [build_case(nonce, seed, family, split, index)
            for family, split, count in FAMILIES for index in range(count)]


def solve_prompt(prompt):
    """Separate parser/interpreter: prompt string only, no generator metadata/gold."""
    if not isinstance(prompt, str):
        raise TypeError("prompt string required")
    body, question = prompt.split("\nQuestion: ")
    if body.startswith("Ledgers:"):
        accounts = {}
        for line in body.splitlines()[1:]:
            signed = re.fullmatch(r"([A-Z]{3}) start (\d+); entries ((?:[-+]\d+ ?)+)\.", line)
            grouped = re.fullmatch(r"([A-Z]{3}) start (\d+); credits (\d+),(\d+); debit (\d+)\.", line)
            if signed:
                value = int(signed[2])
                for token in signed[3].split():
                    value += int(token)
                name = signed[1]
            elif grouped:
                name, start, x, y, debit = grouped.groups()
                value = int(start)
                for amount in (int(x), int(y), -int(debit)):
                    value += amount
            else:
                raise ValueError("unparsed ledger")
            if name in accounts:
                raise ValueError("duplicate account")
            accounts[name] = value
        query = re.fullmatch(r"([A-Z]{3}) final\? Reply one integer\.", question)
        return str(accounts[query[1]])
    if body.startswith("Rules derive"):
        rules_raw, facts_raw = re.fullmatch(
            r"Rules derive only their right side when all left symbols are known\. Rules: ([A-Z>&,]+)\. Facts: ([A-Z,]+)\.", body).groups()
        rules = []
        for token in rules_raw.split(","):
            left, right = token.split(">")
            rules.append((set(left.split("&")), right))
        known = set(facts_raw.split(","))
        changed = True
        while changed:
            changed = False
            for requirements, consequence in rules:
                if requirements <= known and consequence not in known:
                    known.add(consequence)
                    changed = True
        query = re.fullmatch(r"derive ([A-Z]{3})\? Reply YES or NO\.", question)
        return "YES" if query[1] in known else "NO"
    if body.startswith("Registers:"):
        initial, instructions = re.fullmatch(
            r"Registers: ([A-Z=0-9,]+)\. Execute numbered steps in increasing order; each reads current values\. Steps: (.*)\.", body).groups()
        state = {}
        for item in initial.split(","):
            name, value = item.split("=")
            state[name] = int(value)
        parsed = []
        for instruction in instructions.split(";"):
            step, dest, src, operator, amount = re.fullmatch(r"(\d+) ([A-Z]{3})=([A-Z]{3})([-+*])(\d+)", instruction).groups()
            parsed.append((int(step), dest, src, operator, int(amount)))
        if len({x[0] for x in parsed}) != len(parsed):
            raise ValueError("duplicate numbered step")
        for _, dest, src, operator, amount in sorted(parsed):
            value = state[src]
            state[dest] = value + amount if operator == "+" else value - amount if operator == "-" else value * amount
        query = re.fullmatch(r"([A-Z]{3})\? Reply one integer\.", question)
        return str(state[query[1]])
    latest = body.startswith("Use only the latest dated")
    rule = ("Use only the latest dated approved values per entity; agree=>value, absent or conflict=>UNKNOWN."
            if latest else "Use approved values only; agree=>value, absent or conflict=>UNKNOWN.")
    if not body.startswith(rule + " Records: "):
        raise ValueError("unrecognized structured prompt")
    evidence = defaultdict(list)
    for record in body.removeprefix(rule + " Records: ").rstrip(".").split(";"):
        name, status, date, value = re.fullmatch(r"([A-Z]{3}) (approved|draft) (?:d(\d+) )?(\d+)", record).groups()
        if status == "approved":
            evidence[name].append((int(date or 0), int(value)))
    query = re.fullmatch(r"([A-Z]{3}) value\? Reply one integer or UNKNOWN\.", question)
    admitted = evidence[query[1]]
    if not admitted:
        return "UNKNOWN"
    latest_date = max(date for date, _ in admitted)
    values = {value for date, value in admitted if not latest or date == latest_date}
    return str(next(iter(values))) if len(values) == 1 else "UNKNOWN"


def normalize_output(response):
    """Gold-independent full-response contract; no substring/last-token extraction."""
    if not isinstance(response, str):
        return None
    text = response.strip()
    if re.fullmatch(r"0|-?[1-9]\d*", text):
        return text
    return text.upper() if text.upper() in ("YES", "NO", "UNKNOWN") else None


def score_response(response, case):
    actual = normalize_output(response)
    integer = actual is not None and re.fullmatch(r"0|-?[1-9]\d*", actual) is not None
    allowed = (actual in ("YES", "NO") if case["answer_kind"] == "choice"
               else actual == "UNKNOWN" or integer if case["answer_kind"] == "integer_or_unknown"
               else integer)
    return {"correct": bool(allowed and actual == case["gold"]), "format_valid": bool(allowed)}


def blind_predictions(prompt):
    context, question = prompt.split("\nQuestion: ")
    numbers = re.findall(r"[-+]?\d+", context)
    candidates = [x for x in re.findall(r"\b[A-Z]{3}\b", context) if x != "YES"]
    substituted = re.sub(r"[A-Z]{3}", candidates[0], question, count=1)
    return {
        "constant_YES": "YES", "constant_NO": "NO", "constant_UNKNOWN": "UNKNOWN",
        "constant_0": "0", "constant_1": "1", "copy_full_prompt": prompt,
        "copy_first_number": numbers[0] if numbers else "",
        "copy_last_number": numbers[-1] if numbers else "",
        "copy_last_entity": candidates[-1] if candidates else "",
        "query_blind_first_entity_solver": solve_prompt(context + "\nQuestion: " + substituted),
    }


def audit(rows):
    counts, floor_counts, per_family = Counter(), Counter(), defaultdict(Counter)
    normalized_prompts = [" ".join(x["prompt"].split()) for x in rows]
    solved = []
    for case in rows:
        prediction = solve_prompt(case["prompt"])
        if prediction != case["gold"]:
            raise AssertionError("independent prompt parser disagrees with generator gold")
        if "target" in case["prompt"].lower():
            raise AssertionError("unique target-marker surface")
        counts[case["split"]] += 1
        for name, value in blind_predictions(case["prompt"]).items():
            correct = score_response(value, case)["correct"]
            floor_counts[name] += correct
            per_family[case["generator_family"]][name] += correct
        solved.append({"id": case["id"], "prediction": prediction, "correct": True})
    no_overlap = len({x["parent_id"] for x in rows}) == len(rows) and len(set(normalized_prompts)) == len(rows)
    family_splits = defaultdict(set)
    for case in rows:
        family_splits[case["generator_family"]].add(case["split"])
    report = {
        "rows": len(rows), "split_counts": dict(counts), "positive_control_correct": len(solved),
        "unique_parents_and_prompts": no_overlap,
        "generator_families_disjoint": all(len(x) == 1 for x in family_splits.values()),
        "blind_correct_counts": dict(floor_counts), "blind_by_family": dict(per_family),
        "blind_global_ceiling": 12, "blind_global_gate": max(floor_counts.values()) <= 12,
        "max_ascii_prompt_bytes": max(len(x["prompt"].encode("ascii")) for x in rows),
        "native_tokens": "NOT_MEASURED; coordinator must bind <=350 before registration",
        "human_review": "NOT_REVIEWED", "native_calls": 0,
    }
    if not no_overlap or not report["generator_families_disjoint"]:
        raise AssertionError("duplicate or split collision")
    return report, solved


def run(out, nonce, seed):
    out = Path(out).resolve()
    if PACKAGE not in out.parents:
        raise PermissionError("output must remain campaign-local")
    out.mkdir(parents=True, exist_ok=False)
    rows = generate(nonce, seed)
    report, solved = audit(rows)
    for name, value in (("bank.json", rows), ("audit.json", report), ("positive_controls.json", solved)):
        with (out / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    manifest = {"status": "OPEN_PREPARATION_ONLY_NOT_NATIVE_ADMITTED", "nonce": nonce, "seed": seed,
                "generator_version": VERSION, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "files": {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in out.iterdir() if x.is_file()},
                "model_loads": 0, "native_seconds": 0, "training": False,
                "next_gate": "native tokenizer shape binding, independent code/data review, complete new baseline/matrix/scorer registration"}
    with (out / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    return 0 if report["blind_global_gate"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--nonce", default=None)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    raise SystemExit(run(args.out, args.nonce or secrets.token_hex(6), args.seed))
