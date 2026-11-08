"""Prompt-only structured positive controls on the already-bound DEV/SMOKE.

No model/tokenizer imports, new task bank, native calls, or protected reads.
Scorer targets are admitted and evaluated only after a prompt-only prediction.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import subprocess
import sys


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
BINDING_RUN = PACKAGE / "runs/binding-001"
SCORER = ROOT / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src/rineya_cross_scale/evaluation/scoring.py"
SCORER_SHA = "5a22e1c1ff7b0a56623931c0782f81dd2b3a9df83ccfe77ce3724d37cf320086"
INTEGER = r"[-+]?\d{1,12}"


@dataclass(frozen=True)
class Solution:
    status: str
    prediction: str | None
    solver: str | None
    witness: dict


def unavailable(reason):
    return Solution("NOT_APPLICABLE", None, None, {"reason": reason})


def success(name, prediction, **witness):
    return Solution("PROMPT_ONLY_SOLVED", str(prediction), name, witness)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def solve_prompt(prompt: str) -> Solution:
    """Accept ONLY model-visible text; no IDs, labels, scorers or bank metadata."""
    if not isinstance(prompt, str):
        raise TypeError("solver accepts only a raw prompt string")
    match = re.fullmatch(
        r"In a counterfactual world, every object that is (\w+) is also (\w+), "
        r"and every (\w+) object is (\w+)\. Object (\w+) is (\w+)\. "
        r"Under those rules, is (\w+) (\w+)\? Answer YES or NO only\.", prompt)
    if match:
        a, b, c, d, entity, initial, query_entity, queried = match.groups()
        reached, edges = {initial}, [(a, b), (c, d)]
        for _ in range(3):
            reached |= {y for x, y in edges if x in reached}
        if entity == query_entity and queried in reached:
            return success("transitive_color", "YES", edges=edges, initial=initial, reached=sorted(reached))
        return unavailable("no proof of queried property; positive-only implication solver")
    if prompt.endswith("Who is youngest? Return only the name."):
        statements = prompt.removesuffix("Who is youngest? Return only the name.").strip().rstrip(".")
        edges = [re.fullmatch(r"(\w+) is older than (\w+)", x) for x in statements.split(". ")]
        if all(edges) and edges:
            relations = [m.groups() for m in edges]
            names = set(x for edge in relations for x in edge)
            reach = {x: set() for x in names}
            for older, younger in relations:
                reach[older].add(younger)
            for _ in names:
                for person in names:
                    reach[person] |= set().union(*(reach[y] for y in list(reach[person]))) if reach[person] else set()
            youngest = [x for x in names if all(x == y or x in reach[y] for y in names)]
            if len(youngest) == 1 and all(x not in reach[x] for x in names):
                return success("age_order", youngest[0], relations=relations, unique_minimum=True)
        return unavailable("age syntax/order does not establish unique youngest")
    handoff = re.fullmatch(r"(\w+) gave the (\w+) to (\w+)\. (.*?)Who has the (\w+) now\? Return only the name\.", prompt)
    if handoff:
        initial, item, holder, remainder, query_item = handoff.groups()
        path = [initial, holder]
        for sentence in remainder.strip().rstrip(".").split(". "):
            step = re.fullmatch(r"(?:Then|Finally) (\w+) gave it to (\w+)", sentence)
            if not step or step[1] != holder:
                return unavailable("handoff does not follow current holder")
            holder = step[2]
            path.append(holder)
        if query_item == item:
            return success("handoff", holder, item=item, path=path)
        return unavailable("query item differs from tracked item")
    arithmetic = re.fullmatch(rf"Compute \(({INTEGER}) × ({INTEGER})\) \+ \(({INTEGER})\)\. Return only the integer\.", prompt)
    if arithmetic:
        a, b, c = map(int, arithmetic.groups())
        return success("signed_product_plus", a * b + c, operands=[a, b, c])
    for pattern in (
        rf"A box held ({INTEGER}) tokens\. ({INTEGER}) were removed\. How many remain\? Return only the integer\.",
        rf"Starting with ({INTEGER}) tokens, take away ({INTEGER})\. State only the number left\.",
        rf"What is ({INTEGER}) minus ({INTEGER})\? Give just the integer answer\.",
    ):
        subtraction = re.fullmatch(pattern, prompt)
        if subtraction:
            a, b = map(int, subtraction.groups())
            return success("subtraction", a - b, operands=[a, b])
    echo = re.fullmatch(r"Return exactly this JSON object and nothing else: (\{.*\})", prompt, re.DOTALL)
    if echo:
        try:
            value = json.loads(echo[1], object_pairs_hook=unique_object,
                               parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        except (ValueError, json.JSONDecodeError):
            return unavailable("invalid or ambiguous JSON payload")
        return success("json_echo", json.dumps(value, separators=(",", ":"), ensure_ascii=False), copied_prompt_payload=True)
    letter = re.fullmatch(r"What is the first letter of the string ([A-Za-z]+)\? Answer in one or two words\.", prompt)
    if letter:
        return success("first_letter", letter[1][0], input_string=letter[1])
    archival = re.fullmatch(
        r"Use only the records below\.\n(.*?)\nQuestion: What is the archival value of the (\w+) target\? Return only the integer\.",
        prompt, re.DOTALL)
    if archival:
        context, query = archival.groups()
        records = [re.fullmatch(r"Record (\d+): (?:unit (\w+)|the (\w+) target) has archival value (\d+)\.", line)
                   for line in context.splitlines()]
        if not all(records):
            return unavailable("unparsed archival evidence")
        matches = [m for m in records if m[3] == query]
        if len(matches) == 1:
            return success("archival_lookup", matches[0][4], query=query,
                           source_record=int(matches[0][1]), records=len(records))
        return unavailable("archival target is missing or ambiguous")
    code = re.fullmatch(
        r"Context: (.*?)\s*Question: What is the (?:access )?code of the (\w+) unit\? "
        r"(Return only the code\.|If the context does not say, answer exactly (\w+)\.)",
        prompt, re.DOTALL)
    if code:
        context, query, instruction, fallback = code.groups()
        clauses = context.strip().rstrip(".").split(". ")
        evidence = []
        for clause in clauses:
            value = re.fullmatch(r"The (\w+) unit has (?:access )?code (\d+)", clause)
            cabinet = re.fullmatch(r"The (\w+) unit is stored in cabinet (\d+)", clause)
            if value:
                evidence.append(value.groups())
            elif not cabinet:
                return unavailable("unparsed code evidence")
        matches = [value for name, value in evidence if name == query]
        if len(matches) == 1:
            return success("code_lookup", matches[0], query=query, evidence=evidence)
        if not matches and fallback:
            return success("closed_context_missing_code", fallback, query=query, evidence=evidence,
                           fallback_from_explicit_prompt_instruction=True)
        return unavailable("code evidence missing/conflicting without admissible fallback")
    return unavailable("unrecognized/factual/language task; no target fallback")


def context_only(prompt):
    """Remove query and answer instruction before blind context attacks."""
    parts = re.split(r"Question:|Who is youngest\?|Who has |Under those rules,|How many remain\?", prompt, maxsplit=1)
    return parts[0] if len(parts) == 2 else ""


def blind_predictions(prompt):
    context = context_only(prompt)
    numbers = re.findall(r"[-+]?\d+", context)
    names = re.findall(r"\b[A-Z][a-z]+\b", context)
    marked = re.findall(r"Record \d+: the \w+ target has archival value (\d+)", context)
    codes = re.findall(r"The \w+ unit has (?:access )?code (\d+)", context)
    literals = re.findall(r"\b(?:YES|NO|UNKNOWN)\b", prompt)
    return {
        "copy_full_prompt": prompt,
        "context_first_number": numbers[0] if numbers else "",
        "context_last_number": numbers[-1] if numbers else "",
        "context_last_capitalized_name": names[-1] if names else "",
        "context_unique_marked_target": marked[0] if len(marked) == 1 else "",
        "context_first_code": codes[0] if codes else "",
        "copy_last_answer_literal": literals[-1] if literals else "",
        **{"constant_" + repr(value): value for value in ("", "YES", "NO", "UNKNOWN", "0", "1")},
    }


def self_checks():
    checks = [
        (solve_prompt("What is 9 minus 14? Give just the integer answer.").prediction == "-5"),
        (solve_prompt("Compute (-3 × 8) + (5). Return only the integer.").prediction == "-19"),
        (solve_prompt("Be factual and brief.").status == "NOT_APPLICABLE"),
        (solve_prompt('Return exactly this JSON object and nothing else: {"x":1,"x":2}').status == "NOT_APPLICABLE"),
        (solve_prompt("A is older than B. B is older than A. Who is youngest? Return only the name.").status == "NOT_APPLICABLE"),
        (solve_prompt("A gave the token to B. Then C gave it to D. Who has the token now? Return only the name.").status == "NOT_APPLICABLE"),
    ]
    try:
        solve_prompt({"raw_prompt": "ignored", "scorer": {"value": "SECRET"}})
    except TypeError:
        checks.append(True)
    else:
        checks.append(False)
    if not all(checks):
        raise AssertionError("prompt-only solver self-check failure")
    return {"passed": len(checks), "kind": "CPU_APPARATUS_FIXTURES_NOT_NEW_TASK_BANK"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def run(out):
    out = Path(out).resolve()
    if PACKAGE not in out.parents:
        raise PermissionError("exclusive output must remain campaign-local")
    out.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((BINDING_RUN / "receipt.json").read_text(encoding="utf-8"))
    audit_path, binding_path = BINDING_RUN / "PROMPT_AUDIT.json", BINDING_RUN / "MODEL_DATA_BINDING.json"
    if sha(audit_path) != receipt["prompt_audit_sha256"] or sha(binding_path) != receipt["binding_sha256"]:
        raise RuntimeError("already-admitted binding evidence changed")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if Path(binding["scorer_path"]).resolve() != SCORER.resolve() or sha(SCORER) != SCORER_SHA:
        raise RuntimeError("pinned scorer mismatch")
    relative = SCORER.relative_to(ROOT).as_posix()
    blob = subprocess.check_output(["git", "-C", str(ROOT), "show", "03db218ceebc61a2bc04f2482cf7e094045e01c1:" + relative])
    if SCORER.read_bytes() != blob:
        raise RuntimeError("scorer differs from admitted integrated-base blob")
    spec = importlib.util.spec_from_file_location("campaign_pinned_scoring", SCORER)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    raw_rows = json.loads(audit_path.read_text(encoding="utf-8"))
    if Counter(row["split"] for row in raw_rows) != {"DEV": 16, "SMOKE": 8}:
        raise RuntimeError("authorized bound surface count differs")
    checks, rows, totals = self_checks(), [], defaultdict(Counter)
    for row in raw_rows:
        prompt = row["raw_prompt"]
        if hashlib.sha256(prompt.encode()).hexdigest() != row["prompt_sha256"]:
            raise RuntimeError("raw prompt hash mismatch")
        if hashlib.sha256(row["rendered_prompt"].encode()).hexdigest() != row["rendered_prompt_sha256"]:
            raise RuntimeError("rendered prompt hash mismatch")
        expected_render = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        if row["rendered_prompt"] != expected_render:
            raise RuntimeError("native rendered prompt does not preserve the exact raw question")
        ids = row["input_token_ids"]
        if len(ids) != row["input_tokens"] or any(type(x) is not int or x < 0 for x in ids):
            raise RuntimeError("bound token-ID shape invalid")
        if row["max_forward_prefix_tokens"] != len(ids) + row["max_new_tokens"] - 1 or row["max_forward_prefix_tokens"] > 4096:
            raise RuntimeError("bound complete prefix shape differs")
        solved = solve_prompt(prompt)  # only this text enters the solver
        result = asdict(solved)
        result.update(case_id=row["case_id"], split=row["split"], family=row["family"],
                      prompt_sha256=row["prompt_sha256"], human_review="NOT_REVIEWED",
                      semantic_review="AGENT_REVIEW_OF_EXPOSED_DEV_NOT_INDEPENDENT",
                      token_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                      input_tokens=len(ids), max_forward_prefix_tokens=row["max_forward_prefix_tokens"],
                      token_audit="PRIOR_EXACT_TOKENIZER_BINDING_REUSED_NOT_RETOKENIZED", truncation=False)
        result["score"] = None if solved.prediction is None else scorer.score_response(solved.prediction, row["scorer"])
        result["format_score"] = None if solved.prediction is None else scorer.score_response(solved.prediction, row["format_scorer"])
        result["blind_scores"] = {name: {"prediction": value, "score": scorer.score_response(value, row["scorer"])["score"]}
                                 for name, value in blind_predictions(prompt).items()}
        attacks = {}
        if solved.prediction is not None:
            kind = row["scorer"]["type"]
            if kind == "keywords":
                attacks["negated_solver_answer"] = "The answer is not " + solved.prediction + "."
            elif kind == "choice":
                attacks["contradictory_choice"] = "NO YES"
            elif kind == "numeric":
                attacks["negated_final_number"] = "The answer is not " + solved.prediction + "."
            elif kind == "json_exact":
                obj = json.loads(solved.prediction)
                if obj.get("ok") is True:
                    obj["ok"] = 1
                    attacks["boolean_to_integer"] = json.dumps(obj)
        result["scorer_attacks"] = {name: {"prediction": value,
            "score": scorer.score_response(value, row["scorer"])["score"],
            "format_score": scorer.score_response(value, row["format_scorer"])["score"]}
            for name, value in attacks.items()}
        totals[row["split"]]["items"] += 1
        totals[row["split"]]["applicable"] += int(solved.prediction is not None)
        totals[row["split"]]["solver_correct"] += int(result["score"] is not None and result["score"]["correct"])
        rows.append(result)
    floors = {}
    for split in ("DEV", "SMOKE"):
        selected = [row for row in rows if row["split"] == split]
        scores = {name: sum(row["blind_scores"][name]["score"] for row in selected)
                  for name in selected[0]["blind_scores"]}
        floors[split] = {"denominator": len(selected), "correct_counts": scores,
                         "best_predefined_single_baseline": max(scores.values()),
                         "selection": "retrospective diagnostic floor only; no per-item oracle mixing"}
    summary = {"status": "PROMPT_ONLY_PARTIAL_POSITIVE_CONTROL_AUDIT",
               "native_model_loads": 0, "native_model_worker_seconds": 0,
               "new_task_bank": False, "human_review": "NOT_REVIEWED",
               "scorer_sha256": SCORER_SHA, "input_audit_sha256": sha(audit_path),
               "binding_sha256": sha(binding_path), "solver_source_sha256": sha(__file__),
               "runtime": {"interpreter": sys.executable, "python": platform.python_version()},
               "self_checks": checks, "counts": dict(totals), "blind_floors": floors,
               "answerability_claim": "Only recognized solver cases; factual/language NOT_APPLICABLE; no all-bank claim",
               "cluster_caution": "three subtraction paraphrases share operands; two archival cases share target/value; not independent evidence"}
    write(out / "rows.json", rows)
    write(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if all(x["applicable"] == x["solver_correct"] for x in totals.values()) else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().out))
