"""Audit stored CONFIRM-002 H-bank generations. Standard library; no generation.

This reports descriptive correctness pairs on a spent bank, without inference.
The CLI executes only the byte-hash-pinned historical boundary_rule module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

FROZEN_RULE_SHA256 = "416cfb6aec8aa28240752d3db2fc1b8af85115c4962a873582765c009adc3bf0"
INSTRUMENT = "CONFIRM-002 boundary scorer"
RUNTIME_KEYS = {"python", "torch", "transformers", "device", "dtype", "decoding",
                "max_new_tokens", "thinking_disabled", "tf32"}


class AuditError(ValueError):
    """Evidence cannot be paired safely."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def reject_constant(value):
    raise AuditError(f"nonfinite JSON constant: {value}")


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def artifact(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return raw, {"path": str(path), "sha256": digest(raw), "bytes": len(raw)}


def index_rows(rows, label):
    require(type(rows) is list and bool(rows), f"{label}: nonempty rows required")
    indexed = {}
    for row in rows:
        require(type(row) is dict, f"{label}: row must be an object")
        key = row.get("id")
        require(type(key) is str and bool(key.strip()) and key == key.strip(),
                f"{label}: canonical nonblank string id required")
        require(key not in indexed, f"{label}: duplicate id {key!r}")
        indexed[key] = row
    return indexed


def same_ids(actual, expected, label):
    missing, extra = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
    require(not missing and not extra,
            f"{label}: incompatible IDs; missing={missing!r}; extra={extra!r}")


def validate_runtime(runtime):
    require(type(runtime) is dict and RUNTIME_KEYS <= set(runtime),
            "missing runtime provenance fields")
    for key in ("python", "torch", "transformers", "device", "dtype", "decoding"):
        require(type(runtime[key]) is str and bool(runtime[key]), f"invalid runtime {key}")
    require(type(runtime["max_new_tokens"]) is int and runtime["max_new_tokens"] == 48,
            "incompatible runtime: expected max_new_tokens=48")
    require(runtime["decoding"] == "greedy" and runtime["dtype"] == "float32",
            "incompatible decoding/dtype provenance")
    require(runtime["thinking_disabled"] is True and runtime["tf32"] is False,
            "incompatible thinking/tf32 provenance")


def summarize(pairs):
    both = sum(p["source"]["correct"] and p["candidate"]["correct"] for p in pairs)
    losses = sum(p["source"]["correct"] and not p["candidate"]["correct"] for p in pairs)
    gains = sum(not p["source"]["correct"] and p["candidate"]["correct"] for p in pairs)
    source_correct = both + losses
    incorrect = len(pairs) - source_correct
    return {"paired_rows": len(pairs), "both_correct": both,
            "both_incorrect": len(pairs) - both - losses - gains,
            "source_correct": source_correct, "candidate_correct": both + gains,
            "source_correct_losses": losses, "source_incorrect_gains": gains,
            "candidate_minus_source_correct": gains - losses,
            "source_correct_loss_denominator": source_correct,
            "source_correct_loss_rate": losses / source_correct if source_correct else None,
            "source_incorrect_gain_denominator": incorrect,
            "source_incorrect_gain_rate": gains / incorrect if incorrect else None,
            "source_attempts": sum(p["source"]["attempt"] for p in pairs),
            "candidate_attempts": sum(p["candidate"]["attempt"] for p in pairs)}


def audit_documents(rows, runs, *, bank_sha256, scorer_sha256, rule,
                    source_arm="SPARSE", candidate_arm="TH25"):
    """Pure deterministic adapter; injected rule is for isolated synthetic tests.

    Real evidence must use audit_paths, which verifies the frozen scorer bytes.
    Reject duplicates before indexing, and require complete *full-bank* coverage.
    """
    require(source_arm != candidate_arm, "source and candidate must differ")
    bank = index_rows(rows, "bank")
    for key, row in bank.items():
        require(row.get("row_type") in ("answerable", "boundary"), f"{key}: invalid row_type")
        require(type(row.get("reference")) is str and type(row.get("family")) is str
                and bool(row["family"]), f"{key}: missing reference/family")
        prompt = row.get("prompt")
        require(type(prompt) is list and bool(prompt) and all(
            type(m) is dict and type(m.get("role")) is str and type(m.get("content")) is str
            for m in prompt), f"{key}: missing prompt provenance")
    answer_ids = sorted(key for key, row in bank.items() if row["row_type"] == "answerable")
    require(bool(answer_ids), "bank has no answerable rows")
    require(type(runs) is list and bool(runs), "nonempty generation receipts required")
    arms, runtime = {}, None
    for run in runs:
        require(type(run) is dict and type(run.get("schema")) is int and run["schema"] == 1
                and run.get("instrument") == INSTRUMENT, "incompatible receipt schema/instrument")
        require(run.get("bank_sha256") == bank_sha256, "incompatible bank sha256 provenance")
        require(run.get("rule_module_sha256") == scorer_sha256,
                "incompatible scorer sha256 provenance")
        validate_runtime(run.get("runtime"))
        if runtime is None:
            runtime = run["runtime"]
        require(run["runtime"] == runtime, "incompatible runtime provenance across receipts")
        require(type(run.get("arms")) is list and bool(run["arms"]), "receipt has no arms")
        for arm in run["arms"]:
            require(type(arm) is dict and type(arm.get("arm")) is str and bool(arm["arm"]),
                    "invalid arm name")
            name = arm["arm"]
            require(name not in arms, f"duplicate arm {name!r}")
            predictions = index_rows(arm.get("predictions"), name)
            same_ids(predictions, bank, name)
            eos = arm.get("eos")
            require(type(eos) is list and bool(eos) and all(type(x) is int and x >= 0 for x in eos)
                    and len(eos) == len(set(eos)), f"{name}: missing/invalid EOS provenance")
            for key, prediction in predictions.items():
                require(prediction.get("family") == bank[key]["family"] and
                        prediction.get("row_type") == bank[key]["row_type"],
                        f"{name}/{key}: incompatible row metadata")
                require(type(prediction.get("prediction")) is str, f"{name}/{key}: missing raw text")
                require(type(prediction.get("credited")) is bool, f"{name}/{key}: nonboolean credited")
                tokens = prediction.get("generated_tokens")
                require(type(tokens) is int and 0 <= tokens <= runtime["max_new_tokens"],
                        f"{name}/{key}: invalid generated_tokens")
                if bank[key]["row_type"] == "answerable":
                    require(type(prediction.get("attempt")) is bool, f"{name}/{key}: nonboolean attempt")
            arms[name] = (arm, predictions)
    require(source_arm in arms and candidate_arm in arms, "missing requested source/candidate arm")
    require(arms[source_arm][0]["eos"] == arms[candidate_arm][0]["eos"],
            "incompatible EOS provenance across compared arms")
    observations, mismatches, summary_mismatches = {}, [], []
    for name in (source_arm, candidate_arm):
        arm, predictions = arms[name]
        rescored = {"boundary_abstain": 0, "n_boundary": len(bank) - len(answer_ids),
                    "answerable_attempt": 0, "answerable_correct": 0, "n_answerable": len(answer_ids)}
        observations[name] = {}
        for key in sorted(bank):
            row, pred = bank[key], predictions[key]
            reply = pred["prediction"]
            strict, credited = rule.score_row(row["row_type"], reply, row["reference"])
            credited = bool(credited)
            if pred["credited"] != credited:
                mismatches.append({"arm": name, "id": key, "field": "credited",
                                   "stored": pred["credited"], "rescored": credited})
            if row["row_type"] == "boundary":
                rescored["boundary_abstain"] += credited
                continue
            first = rule.first_line(reply)
            attempt = not rule.abstains(first)
            if pred["attempt"] != attempt:
                mismatches.append({"arm": name, "id": key, "field": "attempt",
                                   "stored": pred["attempt"], "rescored": attempt})
            rescored["answerable_attempt"] += attempt
            rescored["answerable_correct"] += credited
            observations[name][key] = {"raw_text": reply, "first_line": first,
                "canonical_first_line": rule.canonical(first), "correct": credited,
                "strict_correct": bool(strict), "attempt": attempt,
                "stored_correct": pred["credited"], "stored_attempt": pred["attempt"],
                "generated_tokens": pred["generated_tokens"],
                "at_token_cap": pred["generated_tokens"] == runtime["max_new_tokens"],
                "multiline": len(reply.splitlines()) > 1}
        require(type(arm.get("summary")) is dict and all(
            type(arm["summary"].get(k)) is int for k in rescored), f"{name}: missing/invalid summary")
        for field, value in rescored.items():
            if arm["summary"][field] != value:
                summary_mismatches.append({"arm": name, "field": field,
                                          "stored": arm["summary"][field], "rescored": value})
    pairs = []
    for key in answer_ids:
        left, right = observations[source_arm][key], observations[candidate_arm][key]
        outcome = ("both_correct" if left["correct"] and right["correct"] else
                   "loss" if left["correct"] else "gain" if right["correct"] else "both_incorrect")
        pairs.append({"id": key, "family": bank[key]["family"], "reference": bank[key]["reference"],
                      "prompt": bank[key]["prompt"], "outcome": outcome,
                      "raw_text_changed": left["raw_text"] != right["raw_text"],
                      "canonical_first_line_changed": left["canonical_first_line"] != right["canonical_first_line"],
                      "source": left, "candidate": right})
    return {"schema_version": 1, "status": "DESCRIPTIVE_SPENT_BANK_STORED_OUTPUT_AUDIT",
            "confirmation_claim": False, "model_calls": 0,
            "bank_sha256": bank_sha256, "scorer_sha256": scorer_sha256, "runtime": runtime,
            "source_arm": source_arm, "candidate_arm": candidate_arm,
            "coverage": {"bank_rows": len(bank), "boundary_rows": len(bank) - len(answer_ids),
                         "answerable_pairs": len(pairs), "all_receipt_arms": sorted(arms),
                         "full_bank_ids_complete": True, "duplicate_ids": 0},
            "summary": summarize(pairs),
            "per_family": {family: summarize([p for p in pairs if p["family"] == family])
                           for family in sorted({p["family"] for p in pairs})},
            "stored_flag_mismatches": mismatches, "stored_summary_mismatches": summary_mismatches,
            "format_diagnostics": {
                "raw_text_changed_pairs": sum(p["raw_text_changed"] for p in pairs),
                "canonical_first_line_changed_pairs": sum(p["canonical_first_line_changed"] for p in pairs),
                "changed_text_same_canonical_pairs": sum(p["raw_text_changed"] and not
                    p["canonical_first_line_changed"] for p in pairs),
                "canonical_credit_beyond_strict": {name: sum(o["correct"] and not o["strict_correct"]
                    for o in observations[name].values()) for name in observations}},
            "pairs": pairs,
            "limitations": ["Both confirmation banks are spent; descriptive development analysis only.",
                "No p-values, intervals, significance, noninferiority, or promotion decision is computed.",
                "Correctness is the frozen first-line canonical exact-match rule, not semantic grading.",
                "Receipt runtime identity is checked; raw token IDs and stopping causes are unavailable.",
                "Generation receipts contain no per-arm weight hash or tokenizer/template hash; preflight/build attribution is separate historical evidence."]}


def audit_paths(bank_path, run_paths, scorer_path, *, result_path=None,
                source_arm="SPARSE", candidate_arm="TH25"):
    raw, bank_artifact = artifact(bank_path)
    rows = [decode(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    rule_raw, rule_artifact = artifact(scorer_path)
    require(rule_artifact["sha256"] == FROZEN_RULE_SHA256,
            f"frozen scorer unavailable/incompatible: expected {FROZEN_RULE_SHA256}, got {rule_artifact['sha256']}")
    # Execute the verified bytes directly; no import search and no .pyc writes.
    rule = types.ModuleType("confirm002_frozen_boundary_rule")
    exec(compile(rule_raw, str(scorer_path), "exec"), rule.__dict__)
    runs, sources = [], {"bank": bank_artifact, "frozen_scorer": rule_artifact, "generation_receipts": []}
    for path in run_paths:
        run_raw, run_artifact = artifact(path)
        runs.append(decode(run_raw))
        sources["generation_receipts"].append(run_artifact)
    report = audit_documents(rows, runs, bank_sha256=bank_artifact["sha256"],
        scorer_sha256=rule_artifact["sha256"], rule=rule, source_arm=source_arm, candidate_arm=candidate_arm)
    if result_path is not None:
        result_raw, sources["historical_result"] = artifact(result_path)
        result = decode(result_raw)
        require(type(result) is dict and result.get("instrument") == "CONFIRM-002" and
                result.get("h_bank_sha256") == bank_artifact["sha256"], "incompatible historical result provenance")
        descriptive = result.get("descriptive")
        require(type(descriptive) is dict, "historical result missing descriptive object")
        differences = []
        for label, arm, value in (("answerable_correct", source_arm, report["summary"]["source_correct"]),
                                 ("answerable_correct", candidate_arm, report["summary"]["candidate_correct"]),
                                 ("answerable_attempt", source_arm, report["summary"]["source_attempts"]),
                                 ("answerable_attempt", candidate_arm, report["summary"]["candidate_attempts"])):
            values = descriptive.get(label)
            require(type(values) is dict, f"historical result missing {label} object")
            stored = values.get(arm)
            require(type(stored) is int, f"historical result missing {label}/{arm}")
            if stored != value:
                differences.append({"arm": arm, "field": label, "stored": stored, "rescored": value})
        report["historical_result_mismatches"] = differences
    _, sources["audit_helper"] = artifact(__file__)
    report["source_artifacts"] = sources
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--historical-result", type=Path)
    parser.add_argument("--source-arm", default="SPARSE")
    parser.add_argument("--candidate-arm", default="TH25")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        inputs = [args.bank, args.scorer, *args.runs]
        if args.historical_result:
            inputs.append(args.historical_result)
        require(all(not output.is_relative_to(path.resolve().parent) for path in inputs),
                "derived output must be outside every original evidence/scorer directory")
        report = audit_paths(args.bank, args.runs, args.scorer, result_path=args.historical_result,
                             source_arm=args.source_arm, candidate_arm=args.candidate_arm)
        # Create-only, LF bytes; an existing file or any failed audit is never overwritten.
        with output.open("xb") as stream:
            stream.write((json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    except (AuditError, OSError, UnicodeError, json.JSONDecodeError, TypeError, KeyError) as error:
        print(f"AUDIT_FAILED: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(output), "summary": report["summary"],
                      "stored_flag_mismatches": len(report["stored_flag_mismatches"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
