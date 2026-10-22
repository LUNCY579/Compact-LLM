"""Independent public-input audits of opposite questions and requested formats.

A deterministic query-blind policy receives equal inputs on a matched pair.
Distinct truths mean it can score at most one of two rows. This is an instrument
property, not a measured neural capability or an estimate of actual model gains.
"""
from collections import defaultdict
import json
import re
from .common import digest

INSTRUCTIONS = {
    "plain": "Return the answer alone with no surrounding text.",
    "json": 'Return exactly one JSON object with one string field named "answer".',
    "tag": "Return exactly <answer>VALUE</answer> with no surrounding text.",
}

def public_without_query(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("public messages required")
    result, found, fmt = [], 0, None
    for message in messages:
        if set(message) != {"role", "content"} or message["role"] not in ("system", "user"):
            raise ValueError("metadata or assistant answers in policy input")
        content = message["content"]
        if message["role"] == "user" and "TASK_JSON\n" in content:
            if content.count("TASK_JSON\n") != 1 or content.count("\nEND_TASK") != 1:
                raise ValueError("ambiguous task serialization")
            prefix, rest = content.split("TASK_JSON\n", 1)
            payload, suffix = rest.split("\nEND_TASK", 1)
            task = json.loads(payload)
            if "query" not in task:
                raise ValueError("query absent")
            task.pop("query")
            matches = [name for name, instruction in INSTRUCTIONS.items() if prefix.strip() == instruction]
            if len(matches) != 1:
                raise ValueError("visible requested format missing/different")
            fmt = matches[0]; found += 1
            content = prefix + "TASK_JSON\n" + json.dumps(task, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\nEND_TASK" + suffix
        result.append({"role": message["role"], "content": content})
    if found != 1:
        raise ValueError("exactly one public query required")
    return result, fmt

def value_key(row):
    answer = row["canonical_answer"]
    if row["domain"] in ("reasoning", "arithmetic") and re.fullmatch(r"[+-]?\d+", answer):
        return ("integer", int(answer))
    return ("symbol", answer)

def audit_query_crossover(rows):
    parents, errors, domain_counts = defaultdict(list), [], defaultdict(int)
    for row in rows:
        parents[row["parent_id"]].append(row)
    verified = 0
    for parent, pair in parents.items():
        if len(pair) != 2 or {r["variant_id"] for r in pair} != {0, 1}:
            errors.append({"parent_id": parent, "error": "incomplete or duplicate pair"}); continue
        a, b = sorted(pair, key=lambda r: r["variant_id"])
        try:
            wa, fa = public_without_query(a["messages"])
            wb, fb = public_without_query(b["messages"])
            if fa != a["requested_format"] or fb != b["requested_format"]:
                raise ValueError("metadata format differs from visible instruction")
            if digest(wa) != digest(wb) or fa != fb:
                raise ValueError("non-query policy inputs or format differ")
            if value_key(a) == value_key(b):
                raise ValueError("truth unchanged under opposite query")
            if a["domain"] != b["domain"] or a["family_id"] != b["family_id"]:
                raise ValueError("domain/family changed")
            verified += 1; domain_counts[a["domain"]] += 1
        except ValueError as error:
            errors.append({"parent_id": parent, "error": str(error)})
    complete = len(rows) == 2 * len(parents) and not errors and verified == len(parents)
    return {"status": "PASS_QUERYBLIND_PAIR_BOUND_MECHANICS_ONLY" if complete else "BLOCKED_QUERY_CROSSOVER_INSTRUMENT",
        "rows": len(rows), "parents": len(parents), "verified_parents": verified,
        "parents_by_domain": dict(domain_counts), "errors": errors,
        "deterministic_queryblind_pair_accuracy_upper_bound": .5 if complete else None,
        "bound_scope": "same full public input after removing only query; matched wrapper; distinct truths; complete clean-answer scoring",
        "no_eligible_only_subset": True, "native_model_calls": 0, "model_quality_gain_established": False}
