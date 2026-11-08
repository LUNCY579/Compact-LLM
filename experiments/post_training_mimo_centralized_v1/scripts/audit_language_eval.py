"""Scoped comparison of language DEV/CAL prompts against two pinned TRAIN corpora.

Exact prompt and user-sequence equality are checked. Lexical neighbours are review
aids, never semantic-independence or unknown-pretraining-exclusion certificates.
No protected evaluation text, student responses, model or tokenizer is accessed.
"""
from __future__ import annotations
import argparse
import heapq
import json
from pathlib import Path
import re
import time
import unicodedata
from . import e1_data as d

SOURCES = (
    ("V1_TRAIN", "Rineya_Short_Response_Behavior_Corpus_v1.0/data/sft/rineya_short_response_new_train.jsonl",
     "20d662836e9343d2024486ff89dece0b6a618f53a6c9b8449445aa0340a72dbb", 10000),
    ("V2_TRAIN", "Rineya_General_Reasoning_and_Natural_Communication_Corpus_v2.0/data/sft/rineya_v2_new_all_train.jsonl",
     "dc68d9400419e2158144f8832ca52b005628f9b24862f95667e482e34998faee", 32000),
)
STOP = frozenset("a an the i me my we our you your it its to of in on at for with and or but is are was were be been this that these those can could would should will do does did have has had please just now then from as by so if not about into what how when which who all some any than also".split())


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def prompt_keys(messages):
    d.require(type(messages) is list and messages and all(type(m) is dict and set(m) == {"role", "content"}
        and m["role"] in ("system", "user", "assistant") and type(m["content"]) is str and m["content"] for m in messages),
        "message schema")
    d.require(messages[-1]["role"] == "user", "prompt must end in user")
    plain = [m for m in messages if m["role"] != "system"]
    return {"exact_conversation": d.digest(messages), "without_system": d.digest(plain),
            "normalized_conversation": d.digest([{**m, "content": normalized(m["content"])} for m in plain]),
            "normalized_user_sequence": d.digest([normalized(m["content"]) for m in plain if m["role"] == "user"])}


def wordset(messages):
    return set(re.findall(r"[a-z]+", normalized(" ".join(m["content"] for m in messages if m["role"] != "system")))) - STOP


def neighbours(messages, training, limit=3):
    words = wordset(messages)
    def candidates():
        for row in training:
            common = words & row["words"]
            if common:
                yield (len(common) / len(words | row["words"]), row["source"], row["id"], row)
    best = heapq.nlargest(limit, candidates(), key=lambda x: (x[0], x[1], x[2]))
    return [{"jaccard_wordset": score, "source": source, "id": row_id,
             "messages": row["messages"], "source_row_sha256": row["source_row_sha256"],
             "template_family": row["template_family"]} for score, source, row_id, row in best]


def audit(bank, output):
    started = time.perf_counter(); d.require(not output.exists(), "write-once output exists")
    package = Path(__file__).resolve().parents[1]; root = package.parents[1]
    bank = d.safe_path(root, bank.resolve().relative_to(root).as_posix())
    def bank_file(name):
        return d.safe_path(root, (bank / name).relative_to(root).as_posix())
    receipt, receipt_id = d.read_json(bank_file("DATA_RECEIPT.json"))
    d.require(receipt["status"] == "AUTHORED_LANGUAGE_CANDIDATES_UNQUALIFIED", "candidate bank binding")
    candidates = []; bindings = {}
    for role, count in (("DEV", 64), ("CAL", 16)):
        path = bank_file(role + ".jsonl"); raw = path.read_bytes()
        ident = {"sha256": d.hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        d.require(ident == receipt["files"][path.name], "candidate file drift")
        rows = [d.parse(line) for line in raw.splitlines()]
        d.require(len(rows) == count and all(r["role"] == role for r in rows), "candidate role/count")
        candidates.extend(rows); bindings[path.name] = ident
    d.require(len({r["parent_id"] for r in candidates}) == 80, "candidate parent reuse")
    training = []; sources = {}; keys = {name: {} for name in prompt_keys(candidates[0]["messages"])}
    for name, relative, pinned, expected_count in SOURCES:
        path = d.safe_path(root, relative); ident = d.identity(path)
        d.require(ident["sha256"] == pinned, "TRAIN corpus drift")
        count = 0; consumed_hash = d.hashlib.sha256(); consumed_bytes = 0
        with path.open("rb") as stream:
            for raw in stream:
                consumed_hash.update(raw); consumed_bytes += len(raw)
                row = d.parse(raw); count += 1
                d.require(row["metadata"]["split"] == "train" and row["messages"][-1]["role"] == "assistant", "TRAIN-only boundary")
                prompt = row["messages"][:-1]
                for kind, key in prompt_keys(prompt).items():
                    keys[kind].setdefault(key, []).append({"source": name, "id": row["id"]})
                training.append({"source": name, "id": row["id"], "messages": prompt,
                    "words": wordset(prompt), "source_row_sha256": d.digest(row),
                    "template_family": row["metadata"].get("base_template_family", row["metadata"].get("template_family"))})
        d.require(count == expected_count and d.identity(path) == ident and
                  {"sha256": consumed_hash.hexdigest(), "bytes": consumed_bytes} == ident,
                  "TRAIN count/consumed bytes/concurrent drift")
        sources[name] = {"path": relative, **ident, "rows": count}
    collisions = []; within = []; seen = {k: {} for k in keys}; review_rows = []
    for candidate in candidates:
        parent = candidate["parent_id"]
        for kind, key in prompt_keys(candidate["messages"]).items():
            if key in keys[kind]:
                collisions.append({"parent_id": parent, "kind": kind, "matches": keys[kind][key]})
            if key in seen[kind]:
                within.append({"parent_id": parent, "kind": kind, "other_parent": seen[kind][key]})
            seen[kind][key] = parent
        review_rows.append({"parent_id": parent, "role": candidate["role"],
            "candidate_prompt_sha256": d.digest(candidate["messages"]),
            "neighbours": neighbours(candidate["messages"], training)})
    d.require(all(d.identity(bank_file(name)) == ident for name, ident in bindings.items()), "candidate changed during audit")
    output.mkdir(parents=True)
    with (output / "NEIGHBOUR_REVIEW.jsonl").open("xb") as f:
        for row in review_rows: f.write(d.canonical(row) + b"\n")
    result = {"status": "SCOPED_EXACT_AND_LEXICAL_COMPARISON_NOT_SEMANTIC_QUALIFICATION",
        "candidate_receipt": receipt_id, "candidate_files": bindings, "training_sources": sources,
        "candidate_parents": len(candidates), "TRAIN_rows_compared": len(training),
        "comparison_keys": list(keys), "exact_matches": collisions, "within_candidate_matches": within,
        "neighbours_per_parent_max": 3, "neighbour_records": sum(len(r["neighbours"]) for r in review_rows),
        "neighbour_file": d.identity(output / "NEIGHBOUR_REVIEW.jsonl"), "script": d.identity(Path(__file__)),
        "semantic_review_completed": False, "semantic_freshness_qualified": False,
        "unknown_pretraining_exclusion": False, "all_Titan_training_sources_covered": False,
        "protected_payloads_read": False, "native_qualified": False, "training_eligibility": False,
        "model_calls": 0, "wall_seconds": time.perf_counter() - started,
        "limits": "Two pinned original TRAIN corpora only. Wordset neighbours are finite lexical review aids and can miss paraphrases and shared latent templates."}
    with (output / "EXCLUSION_RECEIPT.json").open("xb") as f: f.write(d.canonical(result) + b"\n")
    print(json.dumps({k: result[k] for k in ("candidate_parents", "TRAIN_rows_compared", "neighbour_records")}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); audit(args.bank, args.output)
