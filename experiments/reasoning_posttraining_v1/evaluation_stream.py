"""Create-only, hash-chained inference records and explicit same-binding continuation.

Completed cases survive a resource refusal. No optimizer/checkpoint state is resumed;
this facility is restricted to the frozen source-only baseline. Historical nonstreamed
failures cannot be reconstructed and are never described as recovered predictions.
"""
from __future__ import annotations

import math
from pathlib import Path

from experiments.model_quality_gpu_v1.integrity import (
    canonical, digest, new_dir, read_json, seal, unseal, write_new)


def context_for(binding, bank_sha, rows, *, max_sequence, max_new, serialization, cache_policy):
    return {"binding": binding, "bank_sha256": bank_sha, "max_sequence": max_sequence,
            "max_new_tokens": max_new, "serialization": serialization, "cache_policy": cache_policy,
            "rows": [{"id": r["id"], "sha256": digest(r)} for r in rows]}


def validate_record(record, row, max_new, tokenizer=None, eos=None):
    if (record.get("id") != row["id"] or record.get("reference") != row["reference"] or
            record.get("parent_id") != row["parent_id"]):
        raise ValueError("record identity/target differs from the compiled row")
    tokens = record.get("generated_ids_including_terminal")
    if (not isinstance(tokens, list) or not 1 <= len(tokens) <= max_new or
            any(type(t) is not int or not 0 <= t < 151936 for t in tokens)):
        raise ValueError("invalid recorded generated tokens")
    if (type(record.get("terminated")) is not bool or
            record.get("truncated") != (not record["terminated"]) or
            record.get("generated_tokens_including_terminal") != len(tokens)):
        raise ValueError("invalid recorded termination contract")
    ce, targets = record.get("response_ce_sum"), record.get("response_ce_targets")
    if (type(ce) not in (float, int) or not math.isfinite(ce) or ce < 0 or
            type(targets) is not int or targets < 1 or targets != sum(t != -100 for t in row["labels"][1:])):
        raise ValueError("invalid recorded response loss accounting")
    if tokenizer is not None:
        ended = tokens[-1] in eos
        if any(t in eos for t in tokens[:-1]) or record["terminated"] != ended:
            raise ValueError("recorded EOS/termination differs from token stream")
        decoded = tokenizer.decode(tokens[:-1] if ended else tokens, skip_special_tokens=False)
        if record.get("prediction") != decoded:
            raise ValueError("recorded text differs from decoded tokens")
    prediction = record.get("prediction")
    if not isinstance(prediction, str):
        raise ValueError("prediction must be text")
    canonical_correct = record["terminated"] and " ".join(prediction.split()) == " ".join(row["reference"].split())
    strict_correct = record["terminated"] and prediction.strip() == row["reference"].strip()
    if (record.get("canonical_training_target_match") is not canonical_correct or
            record.get("strict_target_text_match") is not strict_correct):
        raise ValueError("recorded correctness differs from its text contract")


def read_prefix(directory, context, rows, tokenizer=None, eos=None):
    directory = Path(directory)
    failure = directory.parent / "failure.json"
    if failure.exists() and "changed source parameters" in read_json(failure).get("message", ""):
        raise ValueError("cannot reuse predictions from a failed source-state integrity check")
    if unseal(read_json(directory / "START.json")) != {"schema": 1, "context": context}:
        raise ValueError("resume requires the identical source, bank, runtime, code, plan and checkout")
    paths = sorted((directory / "records").glob("*.json"))
    if len(paths) > len(rows):
        raise ValueError("record stream exceeds its registered bank")
    records, previous = [], digest(context)
    for i, path in enumerate(paths):
        if path.name != f"{i:06d}.json":
            raise ValueError("record stream has a missing or reordered position")
        raw = read_json(path)
        body = unseal(raw)
        if (body.get("index") != i or body.get("previous") != previous or
                body.get("row_sha256") != context["rows"][i]["sha256"]):
            raise ValueError("record chain or row binding differs")
        validate_record(body["record"], rows[i], context["max_new_tokens"], tokenizer, eos)
        records.append(body["record"])
        previous = raw["sha256"]
    return records


def evaluate_streamed(model, rows, tokenizer, eos, label, reserve, guard, *,
                      directory, context, evaluator, resume=None):
    prefix = read_prefix(resume, context, rows, tokenizer, eos) if resume else []
    directory = new_dir(Path(directory))
    (directory / "records").mkdir()
    start = seal({"schema": 1, "context": context})
    reserve({"new_disk_bytes": len(canonical(start)) + 1})
    write_new(directory / "START.json", start)
    records, previous = [], digest(context)

    def append(record):
        nonlocal previous
        i = len(records)
        validate_record(record, rows[i], context["max_new_tokens"], tokenizer, eos)
        envelope = seal({"index": i, "row_sha256": context["rows"][i]["sha256"],
                         "previous": previous, "record": record})
        reserve({"new_disk_bytes": len(canonical(envelope)) + 1})
        write_new(directory / "records" / f"{i:06d}.json", envelope)
        records.append(record)
        previous = envelope["sha256"]

    # Copy validated prefixes into the new attempt. Original evidence is immutable.
    for record in prefix:
        append(record)
    if len(prefix) < len(rows):
        evaluator(model, rows[len(prefix):], tokenizer, eos, label, reserve, guard,
                  max_new=context["max_new_tokens"], on_record=append,
                  release_unused_cache=context["cache_policy"] == "release_unused_between_cases")
    if len(records) != len(rows):
        raise ValueError("evaluator did not commit exactly the registered remaining rows")
    targets = sum(r["response_ce_targets"] for r in records)
    result = {"scope": "REGISTERED_SOURCE_BASELINE", "label": label,
              "rows": len(records), "records": records,
              "canonical_correct": sum(r["canonical_training_target_match"] for r in records),
              "terminated": sum(r["terminated"] for r in records),
              "response_ce": sum(r["response_ce_sum"] for r in records) / targets,
              "loss_bearing_tokens": targets,
              "generated_tokens_actual": sum(r["generated_tokens_including_terminal"] for r in records),
              "generation_allowance_reserved": (len(rows) - len(prefix)) * context["max_new_tokens"],
              "generation_allowance_inherited": len(prefix) * context["max_new_tokens"],
              "reused_predictions": len(prefix), "fresh_predictions": len(rows) - len(prefix),
              "resume_source": str(resume) if resume else None,
              "stream_sha256": previous, "cache_policy": context["cache_policy"],
              "semantic_generalization_or_preservation_claim": False}
    complete = {"rows": len(records), "last_record_sha256": previous,
                "context_sha256": digest(context), "source_state_final_check": "CONTROLLER_REQUIRED"}
    reserve({"new_disk_bytes": len(canonical(complete)) + 1})
    write_new(directory / "COMPLETE.json", complete)
    return result
