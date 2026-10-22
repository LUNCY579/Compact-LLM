"""Read-only source-proxy and Horn floors on the exact existing open v2 inputs.
No package teacher/model imports, no corpus writes or native admission decision.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json

BASE = Path("C:/Users/clclc/BusinessProject/Titan-Joint-Quality-Pilot-v2-runs/data-001")
EXPECTED = {
    "train.jsonl": "6c82a98a4d66c64d15aa813dc1afa3f20898996c00f2d8590158a8eec8e1b11d",
    "pilot.prompts.jsonl": "1ba852d8b0bb78d42c543a876e24a38987284095ff17f3ec9887bd7882d46bf8",
    "pilot.references.jsonl": "51b3b96d8e557ddcf046efbb7a8afc81d1b49bbadd1289fb41023a7936aced2d",
}
buffers = {name: (BASE / name).read_bytes() for name in EXPECTED}
assert all(hashlib.sha256(buffers[name]).hexdigest() == sha for name, sha in EXPECTED.items())

def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate public JSON key")
        out[key] = value
    return out

def public_task(messages):
    assert len(messages) == 2 and [message["role"] for message in messages] == ["system", "user"]
    text = messages[1]["content"]
    assert text.count("TASK_JSON\n") == text.count("\nEND_TASK") == 1
    payload, tail = text.split("TASK_JSON\n", 1)[1].split("\nEND_TASK", 1)
    assert not tail
    return json.loads(payload, object_pairs_hook=unique)

train = [json.loads(line, object_pairs_hook=unique) for line in buffers["train.jsonl"].decode("utf-8").splitlines()]
prompt_rows = [json.loads(line, object_pairs_hook=unique) for line in buffers["pilot.prompts.jsonl"].decode("utf-8").splitlines()]
prompts = {row["case_id"]: row["messages"] for row in prompt_rows}
references = [json.loads(line, object_pairs_hook=unique) for line in buffers["pilot.references.jsonl"].decode("utf-8").splitlines()]
authority = Counter()
authority_example = None
for line, row in enumerate(train, 1):
    if row["domain"] != "grounding":
        continue
    task = public_task(row["messages"])
    query = task["query"]
    # The predictor never reads fact.source or task.authoritative_source.
    proxy_values = {fact["value"] for fact in task["facts"]
                    if fact["entity"] == query["entity"] and fact["attribute"] == query["attribute"]
                    and fact["value"].startswith("v")}
    prediction = next(iter(proxy_values)) if len(proxy_values) == 1 else "UNKNOWN"
    # Gold is consulted only after the query-aware, source-blind prediction.
    authority["rows"] += 1
    authority["original_correct"] += prediction == row["canonical_answer"]
    authority["original_predicted_UNKNOWN"] += prediction == "UNKNOWN"
    # The falsifier changes only the named authority in an in-memory task.
    altered = {**task, "authoritative_source": "rumor"}
    proper_values = {fact["value"] for fact in altered["facts"]
                     if fact["entity"] == query["entity"] and fact["attribute"] == query["attribute"]
                     and fact["source"] == altered["authoritative_source"]}
    altered_truth = next(iter(proper_values)) if len(proper_values) == 1 else "UNKNOWN"
    authority["switched_truth_r_prefix"] += altered_truth.startswith("r")
    authority["switched_sourceblind_correct"] += prediction == altered_truth
    if authority_example is None:
        authority_example = {
            "case_id": row["case_id"], "train_jsonl_line": line,
            "query": query, "original_named_authority": task["authoritative_source"],
            "old_gold": row["canonical_answer"], "sourceblind_prediction": prediction,
            "switched_named_authority": "rumor", "proper_switched_truth": altered_truth,
            "queried_facts": [fact for fact in task["facts"]
                              if fact["entity"] == query["entity"] and fact["attribute"] == query["attribute"]],
        }

horn = {"copied_fact_else_UNKNOWN": Counter(), "one_original_fact_round": Counter()}
for row in references:
    if row["domain"] != "grounding":
        continue
    task = public_task(prompts[row["case_id"]])
    copied_prediction = "SUPPORTED" if task["query"] in task["facts"] else "UNKNOWN"
    # One simultaneous round uses ORIGINAL facts, never newly inferred atoms.
    one_round = set(task["facts"]) | {
        rule["conclusion"] for rule in task["rules"]
        if all(premise in task["facts"] for premise in rule["premises"])
    }
    round_prediction = "SUPPORTED" if task["query"] in one_round else "UNKNOWN"
    for name, prediction in (("copied_fact_else_UNKNOWN", copied_prediction), ("one_original_fact_round", round_prediction)):
        counts = horn[name]
        counts["rows"] += 1
        counts["correct"] += prediction == row["canonical_answer"]
        counts["false_abstentions"] += prediction == "UNKNOWN" and row["canonical_answer"] != "UNKNOWN"
        counts["unsupported_answers"] += prediction == "SUPPORTED" and row["canonical_answer"] == "UNKNOWN"

assert authority["rows"] == 300
assert all(counts["rows"] == 64 for counts in horn.values())
print(json.dumps({
    "scope": "EXISTING_OPEN_V2_QUERY_AWARE_CONTROLS_AND_IN_MEMORY_AUTHORITY_FALSIFIER",
    "authority_sourceblind": dict(authority), "authority_example": authority_example,
    "horn_controls": {name: dict(counts) for name, counts in horn.items()},
    "input_sha256": {name: hashlib.sha256(blob).hexdigest() for name, blob in buffers.items()},
    "original_gold_changed": False, "data_or_source_modified": False,
    "bank_generated": False, "files_written": False, "native_model_calls": 0,
    "queryblind_bound_scope": "These controls read query; the existing stateless query-blind <=.5 bound does not apply.",
    "native_admission_frozen": False,
}))
