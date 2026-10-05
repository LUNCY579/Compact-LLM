"""All-row native tokenizer compiler; no old MQ plans, permits or selection."""
from __future__ import annotations

from collections import defaultdict
import random

from .integrity import digest, seal, unseal
from . import EXPERIMENT, SOURCE_CORE, SOURCE_WEIGHT_SHA256
from .bank import REVIEW_KINDS, _sha, validate_rows, validate_review_principals
from .contract import DATASET_OWNER, Design, validate_design


def _ids(value):
    return isinstance(value, list) and bool(value) and all(type(t) is int and 0 <= t < 151936 for t in value)


def _compile_row(row, tokenizer, design):
    messages = row["messages"]
    kwargs = {"enable_thinking": False}
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **kwargs)
    eos, final = tokenizer.eos_token, messages[-1]["content"]
    if not isinstance(eos, str) or not eos or type(tokenizer.eos_token_id) is not int:
        raise ValueError("Explicit source EOS required")
    if full != prefix + final + eos + "\n":
        raise ValueError("Native final-answer training/generation template differs")
    ids = tokenizer.encode(full, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
    eos_ids = tokenizer.encode(eos, add_special_tokens=False)
    native_full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, **kwargs)
    native_prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, **kwargs)
    if (not _ids(ids) or not _ids(prompt_ids) or ids != native_full or prompt_ids != native_prefix
            or ids[:len(prompt_ids)] != prompt_ids or eos_ids != [tokenizer.eos_token_id]):
        raise ValueError("Rendered/native full/prefix/EOS token identities disagree")
    start = len(prompt_ids)
    ends = [i for i in range(start, len(ids)) if ids[i] == tokenizer.eos_token_id]
    if len(ends) != 1 or ends[0] < start:
        raise ValueError("Exactly one final answer EOS required")
    end = ends[0]
    if (tokenizer.decode(ids[end + 1:], skip_special_tokens=False) != "\n"
            or (end == start and not (final == "" and row["empty_target_verified"]))):
        raise ValueError("Unexpected suffix or unverified empty answer")
    if row["split"] == "train":
        if len(ids) > design.max_sequence:
            raise ValueError("TRAIN exceeds 512; no truncation or subset")
    elif len(prompt_ids) + design.max_new_tokens > design.eval_sequence or end - start + 1 > design.max_new_tokens:
        raise ValueError("DEV prompt+64 or target+EOS exceeds frozen budget")
    return {"id": row["id"], "family": row["family"], "parent_id": row["parent_id"],
            "parent_size": row["parent_size"], "pair_id": row["pair_id"], "category": row["category"],
            "split": row["split"], "ids": ids, "prompt_ids": prompt_ids, "answer_start": start,
            "answer_end": end, "reference": final, "raw_sha256": digest(row),
            "evaluation": row["evaluation"], "metadata": row}


def targets(row, objective):
    """Unshifted position labels; shared loss code shifts causally exactly once."""
    if objective == "whole":
        return [-100] + row["ids"][1:]
    if objective == "response":
        return [token if row["answer_start"] <= i <= row["answer_end"] else -100
                for i, token in enumerate(row["ids"])]
    raise ValueError("Only whole/final-response objectives are registered")


def _atomic_components(rows):
    """Union parent groups and pair links, independent of category or outcome."""
    roots = list(range(len(rows)))
    def find(i):
        while roots[i] != i:
            roots[i] = roots[roots[i]]
            i = roots[i]
        return i
    def union(i, j):
        roots[find(j)] = find(i)
    seen = {}
    for i, row in enumerate(rows):
        for key in (("parent", row["parent_id"]), ("pair", row["pair_id"])):
            if key[1] is None:
                continue
            if key in seen:
                union(i, seen[key])
            seen[key] = i
    blocks = defaultdict(list)
    for i in range(len(rows)):
        blocks[find(i)].append(i)
    return sorted([sorted(block, key=lambda i: rows[i]["id"]) for block in blocks.values()],
                  key=lambda b: rows[b[0]]["id"])


def _ordered_and_groups(rows, design):
    blocks = _atomic_components(rows)
    random.Random(design.order_seed).shuffle(blocks)
    ordered, groups, current, used = [], [], [], 0
    for block in blocks:
        size = sum(len(rows[i]["ids"]) for i in block)
        if size > design.update_tokens:
            raise ValueError("Atomic parent/pair exceeds update budget; no splitting")
        if current and used + size > design.update_tokens:
            groups.append(current)
            current, used = [], 0
        indices = list(range(len(ordered), len(ordered) + len(block)))
        ordered.extend(rows[i] for i in block)
        current.extend(indices)
        used += size
    if current:
        groups.append(current)
    return ordered, groups


def compile_plan(bank, tokenizer, source_identity, code_identity, fixture=False):
    design = Design()
    if type(fixture) is not bool or bank.get("fixture_only") is not fixture:
        raise ValueError("Fixture admission cannot authorize a native plan")
    audit = validate_rows({"train": bank["train"], "dev": bank["dev"]}, fixture)
    if not fixture:
        if (source_identity.get("core_id") != SOURCE_CORE or source_identity.get("weight_sha256") != SOURCE_WEIGHT_SHA256
                or source_identity.get("parameters") != 596049920):
            raise ValueError("Exact ungated Supa 1.7 source identity required")
        if len(bank.get("reviews", [])) != 4:
            raise ValueError("Native plans require actual independently attributable owner reviews")
    if not isinstance(code_identity, dict) or not _sha(code_identity.get("sha256")):
        raise ValueError("Pinned consumer code identity required")
    compiled = [_compile_row(r, tokenizer, design) for r in bank["train"]]
    train, groups = _ordered_and_groups(compiled, design)
    dev = [_compile_row(r, tokenizer, design) for r in sorted(bank["dev"], key=lambda r: r["id"])]
    count = sum(len(row["ids"]) for row in train)
    if count > design.processed_budget or len(groups) > design.max_updates:
        raise ValueError("ALL200 exceeds 25k/32-update budget; no prefix selection/refill")
    body = {"schema": 1, "experiment": EXPERIMENT, "kind": "fixture" if fixture else "native_plan",
            "design": design.as_dict(), "source_identity": source_identity, "code_identity": code_identity,
            "inputs": {"owner_manifest_path": bank.get("manifest_path"), "owner_manifest_sha256": bank.get("manifest_sha256"),
                       "owner_export_binding_sha256": bank.get("export_binding_sha256")},
            "tokenizer": {"chat_template_sha256": digest(tokenizer.chat_template), "eos_token_id": tokenizer.eos_token_id,
                          "newline_ids": tokenizer.encode("\n", add_special_tokens=False)},
            "admission": {"fixture_only": fixture, "owner": bank["owner"], "reviews": bank.get("reviews", [])},
            "audit": audit, "processed_tokens": count, "train": train, "dev": dev, "groups": groups}
    result = seal(body)
    validate_plan(result)
    return result


def validate_plan(value):
    body = unseal(value)
    required = {"schema", "experiment", "kind", "design", "source_identity", "code_identity", "inputs",
                "tokenizer", "admission", "audit", "processed_tokens", "train", "dev", "groups"}
    if set(body) != required or body["schema"] != 1 or body["experiment"] != EXPERIMENT or body["kind"] not in {"fixture", "native_plan"}:
        raise ValueError("Wrong sealed RQSFT plan scope")
    design = validate_design(body["design"])
    fixture = body["kind"] == "fixture"
    if (not isinstance(body["admission"], dict) or body["admission"].get("fixture_only") is not fixture
            or body["admission"].get("owner") != DATASET_OWNER):
        raise ValueError("Plan admission/evidence kind differs")
    source = body["source_identity"]
    if not isinstance(source, dict) or not isinstance(body["code_identity"], dict) or not _sha(body["code_identity"].get("sha256")):
        raise ValueError("Source/consumer code identity required")
    if not fixture and (source.get("core_id") != SOURCE_CORE or source.get("weight_sha256") != SOURCE_WEIGHT_SHA256
                        or source.get("parameters") != 596049920 or len(body["admission"].get("reviews", [])) != 4):
        raise ValueError("Native plan is not exact-source/review-bound")
    if not fixture:
        inputs, reviews = body["inputs"], body["admission"]["reviews"]
        if (not isinstance(inputs, dict) or not _sha(inputs.get("owner_manifest_sha256"))
                or not _sha(inputs.get("owner_export_binding_sha256"))
                or any(not isinstance(r, dict) for r in reviews)
                or {r.get("kind") for r in reviews} != REVIEW_KINDS):
            raise ValueError("Native owner-export/independent-review binding incomplete")
        for review in reviews:
            if (not _sha(review.get("sha256")) or review.get("subject_sha256") != inputs["owner_export_binding_sha256"]
                    or not review.get("reviewer_id") or review.get("reviewer_id") == review.get("exporter_id")
                    or not review.get("exporter_id") or review.get("attribution", {}).get("principal") != review["reviewer_id"]
                    or not review.get("attribution", {}).get("record_reference")):
                raise ValueError("Native independently attributable review summary differs")
        validate_review_principals(reviews)
    banks = {}
    eos = body["tokenizer"].get("eos_token_id")
    if type(eos) is not int or not 0 <= eos < 151936 or not _ids(body["tokenizer"].get("newline_ids")):
        raise ValueError("Pinned EOS/newline token identities required")
    for split in ("train", "dev"):
        rows = body[split]
        if not isinstance(rows, list):
            raise ValueError("Compiled rows required")
        banks[split] = []
        for row in rows:
            raw = row.get("metadata")
            if not isinstance(raw, dict) or row.get("raw_sha256") != digest(raw):
                raise ValueError("Compiled row raw metadata/hash changed")
            for key in ("id", "family", "parent_id", "parent_size", "pair_id", "category", "split", "evaluation"):
                if row.get(key) != raw.get(key):
                    raise ValueError("Compiled row differs from reviewed raw metadata")
            ids, start, end = row.get("ids"), row.get("answer_start"), row.get("answer_end")
            if (not _ids(ids) or type(start) is not int or type(end) is not int or not 0 < start <= end < len(ids)
                    or row.get("prompt_ids") != ids[:start] or ids[end] != eos
                    or row.get("reference") != raw["messages"][-1]["content"]
                    or sum(t == eos for t in ids[start:]) != 1
                    or ids[end + 1:] != body["tokenizer"]["newline_ids"]):
                raise ValueError("Compiled causal prefix/final EOS contract changed")
            if end == start and not (row["reference"] == "" and raw.get("empty_target_verified") is True):
                raise ValueError("EOS-only supervision requires verified intentional empty target")
            if split == "train" and len(ids) > design.max_sequence:
                raise ValueError("Training row exceeds 512")
            if split == "dev" and (len(row["prompt_ids"]) + design.max_new_tokens > design.eval_sequence
                                    or end - start + 1 > design.max_new_tokens):
                raise ValueError("DEV context/answer budget exceeded")
            banks[split].append(raw)
    audit = validate_rows(banks, fixture)
    if body["audit"] != audit:
        raise ValueError("Overlap/quota audit changed")
    expected_rows, expected_groups = _ordered_and_groups(sorted(body["train"], key=lambda r: r["id"]), design)
    if [r["id"] for r in expected_rows] != [r["id"] for r in body["train"]] or expected_groups != body["groups"]:
        raise ValueError("Seeded atomic all-row order/grouping changed")
    count = sum(len(r["ids"]) for r in body["train"])
    if (type(body["processed_tokens"]) is not int or count != body["processed_tokens"]
            or count > design.processed_budget or len(body["groups"]) > design.max_updates):
        raise ValueError("All-row token/update accounting differs or exceeds bounds")
    return body
