"""Auditable candidate build, exhaustive checks, blind packets and research release.

Commands write new paths only. No training or model execution is implemented.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys

from common import (ROOT, canonical, child_path, digest, file_digest, read_rows,
                    source_identity, strict_json, write_json, write_rows)

def config_identity():
    return {str(p.relative_to(ROOT)).replace("\\", "/"): file_digest(p)
            for p in sorted((ROOT / "config").glob("*.json"))}

def mutation_gate(result):
    """Validate actual per-check witnesses, not only their reported totals."""
    if result.get("source_valid") is not True or type(result.get("checks")) is not list:
        return False
    checks = result["checks"]
    names = [c.get("name") for c in checks]
    if len(names) != len(set(names)) or not all(type(n) is str for n in names):
        return False
    actual = [c for c in checks if c.get("applicable") is True]
    if len(actual) < 5 or any(type(c.get("passed")) is not bool for c in actual):
        return False
    if any(c.get("applicable") is not True and
           (c.get("applicable") is not False or c.get("passed") is not None or not c.get("detail"))
           for c in checks):
        return False
    passed = sum(c["passed"] for c in actual)
    return (result.get("applicable_checks") == len(actual) and
            result.get("passed_checks") == passed and
            result.get("failed_checks") == len(actual) - passed and passed == len(actual))

def load_batch(batch):
    batch = child_path(batch)
    manifest = strict_json((batch / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_identity"] != source_identity():
        raise ValueError("producer source changed; rebuild into a new version")
    if manifest["config_identity"] != config_identity():
        raise ValueError("producer policy changed; rebuild into a new version")
    if file_digest(batch / "canonical.jsonl") != manifest["canonical_sha256"]:
        raise ValueError("canonical bytes changed")
    rows = list(read_rows(batch / "canonical.jsonl"))
    if len(rows) != manifest["rows"]:
        raise ValueError("row count changed")
    for row in rows:
        raw = {k: v for k, v in row.items() if k != "row_sha256"}
        if row["row_sha256"] != digest(raw):
            raise ValueError("row hash changed")
    return batch, manifest, rows

def build(out, seed):
    from generator import build_rows
    out = child_path(out)
    if out.exists():
        raise ValueError("output already exists; immutable version required")
    rows = build_rows(seed=seed, per_family=12)
    for row in rows:
        row["row_sha256"] = digest(row)
    out.mkdir(parents=True)
    write_rows(out / "canonical.jsonl", rows)
    manifest = {"schema_version": 1, "seed": seed, "rows": len(rows),
                "parents": len({r["parent_id"] for r in rows}),
                "counts_by_split": dict(Counter(r["split"] for r in rows)),
                "counts_by_family": dict(Counter(r["family"] for r in rows)),
                "counts_by_profile": dict(Counter(r["profile"] for r in rows)),
                "source_identity": source_identity(),
                "config_identity": config_identity(),
                "canonical_sha256": file_digest(out / "canonical.jsonl"),
                "status": "CANDIDATE_REVIEW_REQUIRED",
                "historical_semantic_overlap": "NOT_AUDITED",
                "structural_holdout": "NOT_ESTABLISHED_SAME_SIX_FAMILIES",
                "native_model_runs": 0, "training_authorized": False,
                "model_gain": "UNMEASURED", "retention": "UNMEASURED"}
    write_json(out / "manifest.json", manifest)
    print(canonical(manifest))

def audit(batch, out):
    from verifier import verify_row, mutation_checks
    batch, manifest, rows = load_batch(batch)
    out = child_path(out)
    if out.exists():
        raise ValueError("audit output exists")
    seen = set()
    parents = set()
    prompts = set()
    records = []
    for row in rows:
        defects = list(verify_row(row))
        for key, value, inventory in [("row_id", row["row_id"], seen),
                                      ("parent_id", row["parent_id"], parents),
                                      ("prompt_hash", digest(row["messages"]), prompts)]:
            if value in inventory:
                defects.append("duplicate_" + key)
            inventory.add(value)
        if row["split"] not in {"TRAIN", "CAL", "DEV"}:
            defects.append("closed_or_unknown_split")
        mutations = mutation_checks(row)
        # Applicable checks require actual booleans and matching witness counts.
        if not mutation_gate(mutations):
            defects.append("negative_control_failure")
        records.append({"row_id": row["row_id"], "row_sha256": row["row_sha256"],
                        "defects": defects, "mutation_checks": mutations,
                        "passed": not defects})
    write_rows(out / "rows.jsonl", records)
    result = {"schema_version": 1, "canonical_sha256": manifest["canonical_sha256"],
              "source_identity": source_identity(), "rows_checked": len(records),
              "rows_passed": sum(r["passed"] for r in records),
              "mutation_check_count": sum(r["mutation_checks"]["applicable_checks"] for r in records),
              "status": "PASS_DATA_MECHANICS" if all(r["passed"] for r in records) else "FAIL",
              "row_receipts_sha256": file_digest(out / "rows.jsonl"),
              "native_training_ready": False}
    write_json(out / "summary.json", result)
    print(canonical(result))
    if result["status"] == "FAIL":
        raise SystemExit(1)

def packets(batch, out):
    batch, manifest, rows = load_batch(batch)
    out = child_path(out)
    if out.exists():
        raise ValueError("packet output exists")
    public = [{"row_id": r["row_id"], "row_sha256": r["row_sha256"],
               "family": r["family"], "profile": r["profile"],
               "messages": r["messages"]} for r in rows]
    # Never supply target/world/private reasoning to the blind solver packet.
    for role, actor in [("semantic", "/root/long_context_research"),
                        ("adversarial", "/root/quality_retention_design")]:
        write_rows(out / (role + ".jsonl"), public)
    assignments = {"schema_version": 1, "canonical_sha256": manifest["canonical_sha256"],
                   "rubric_version": "RIQF-RUBRIC-001", "rows": len(rows),
                   "assignments": [{"role": role, "reviewer_id": actor,
                       "packet_sha256": file_digest(out / (role + ".jsonl")),
                       "row_ids": [r["row_id"] for r in rows]}
                      for role, actor in [("semantic", "/root/long_context_research"),
                                         ("adversarial", "/root/quality_retention_design")]]}
    write_json(out / "assignments.json", assignments)
    print(canonical({"packets": str(out), "rows_per_role": len(rows)}))

def validate_assignments(rows, assignments):
    allowed = {r["row_id"]: r for r in rows}
    for assignment in assignments["assignments"]:
        if (type(assignment.get("role")) is not str or
            type(assignment.get("reviewer_id")) is not str or not assignment["reviewer_id"].strip()):
            raise ValueError("review role and actor identity must be nonempty strings")
    expected = {x["role"]: x for x in assignments["assignments"]}
    if set(expected) != {"semantic", "adversarial"} or len(assignments["assignments"]) != 2:
        raise ValueError("exactly two required review roles")
    if type(assignments.get("rows")) is not int or assignments["rows"] != len(rows):
        raise ValueError("assignment row count mismatch")
    for assignment in expected.values():
        if (set(assignment.get("row_ids", [])) != set(allowed) or
            len(assignment.get("row_ids", [])) != len(rows)):
            raise ValueError("assignment row IDs mismatch")
    return expected

def validate_reviews(rows, assignments, review_files):
    from verifier import verify_answer
    allowed = {r["row_id"]: r for r in rows}
    expected = validate_assignments(rows, assignments)
    policy = strict_json((ROOT / "config/quality_policy.json").read_text(encoding="utf-8"))
    roles = {}
    for path in review_files:
        receipts = list(read_rows(path))
        if not receipts:
            raise ValueError("empty review file")
        role = receipts[0]["role"]
        if role not in expected or role in roles:
            raise ValueError("unknown/duplicate review role")
        seen = set()
        for receipt in receipts:
            rid = receipt["row_id"]
            if rid not in allowed or rid in seen:
                raise ValueError("unknown/duplicate reviewed row")
            row = allowed[rid]
            assignment = expected[role]
            if receipt["role"] != role or receipt["reviewer_id"] != assignment["reviewer_id"]:
                raise ValueError("unassigned reviewer")
            if receipt["reviewer_id"] in set(policy["generator_identity_aliases"]) | {row["author_id"]}:
                raise ValueError("author cannot review own row")
            fields = {"row_id", "row_sha256", "role", "reviewer_id", "packet_sha256", "rubric_version",
                      "reviewer_model_identity", "blind_answer", "disposition", "findings", "evidence",
                      "method", "limitations"}
            if set(receipt) != fields:
                raise ValueError("review receipt fields mismatch")
            if receipt["row_sha256"] != row["row_sha256"]:
                raise ValueError("stale review")
            if receipt["packet_sha256"] != assignment["packet_sha256"]:
                raise ValueError("unbound reviewer input")
            if receipt.get("rubric_version") != assignments["rubric_version"]:
                raise ValueError("unbound rubric")
            if receipt.get("disposition") != "accept" or receipt.get("findings") != []:
                raise ValueError("unresolved review finding")
            if not receipt.get("method") or not receipt.get("limitations"):
                raise ValueError("missing method or review limitations")
            if not receipt.get("evidence") or not receipt.get("reviewer_model_identity"):
                raise ValueError("missing concrete evidence or model identity")
            if (type(receipt["evidence"]) is not list or
                not all(type(e) is str and len(e.strip()) >= 12 for e in receipt["evidence"]) or
                not all(type(receipt[k]) is str and receipt[k].strip() for k in
                        ["method", "reviewer_model_identity", "blind_answer"])):
                raise ValueError("typed evidence and review methods required")
            limitations = receipt["limitations"]
            if not ((type(limitations) is str and len(limitations.strip()) >= 12) or
                    (type(limitations) is list and limitations and
                     all(type(x) is str and len(x.strip()) >= 12 for x in limitations))):
                raise ValueError("explicit review limitations required")
            if not verify_answer(row, receipt["blind_answer"]):
                raise ValueError("independent blind answer disagrees")
            seen.add(rid)
        if seen != set(allowed):
            raise ValueError("incomplete row coverage")
        roles[role] = receipts[0]["reviewer_id"]
    if set(roles) != set(expected) or len(set(roles.values())) < 2:
        raise ValueError("two distinct reviewer roles required")
    return {"roles": roles, "reviewed_rows_per_role": len(rows),
            "review_file_hashes": {str(p): file_digest(p) for p in review_files},
            "authenticity": "LOCAL_BYTE_BINDING_NOT_CRYPTOGRAPHIC_ACTOR_ATTESTATION",
            "reviewer_calibration": "UNQUALIFIED_SAME_MODEL_PROCEDURAL_INDEPENDENCE"}

def validate_profile(rows, profile, measured, policy):
    if len(measured) != len(rows):
        raise ValueError("incomplete tokenizer rows")
    expected_files = {"tokenizer.json": policy["tokenizer_sha256"],
                      "tokenizer_config.json": policy["tokenizer_config_sha256"]}
    args = profile.get("template_args")
    if (type(args) is not dict or set(args) != {"enable_thinking", "add_generation_prompt"} or
        args["enable_thinking"] is not False or args["add_generation_prompt"] is not True):
        raise ValueError("template arguments must match exact Boolean contract")
    if (profile.get("tokenizer_files") != expected_files or
        profile.get("chat_template_sha256") != policy["chat_template_sha256"] or
        profile.get("runtime") != {k: policy["tokenizer_runtime"][k] for k in ["transformers", "tokenizers"]} or
        type(profile.get("python")) is not str or
        not profile["python"].split() or
        profile["python"].split()[0] != policy["tokenizer_runtime"]["python"]):
        raise ValueError("tokenizer/template/runtime identity differs from frozen policy")
    for field in ["model_framework_imported", "model_weights_read", "native_long_context_qualified", "training_ready"]:
        if profile.get(field) is not False:
            raise ValueError("unexpected native/model qualification claim")
    for row, record in zip(rows, measured):
        if any(record.get(k) != row[k] for k in ["row_id", "row_sha256", "family", "split", "profile"]):
            raise ValueError("tokenizer row identity changed")
        for key in ["prompt_tokens", "completion_tokens_including_eos", "total_tokens", "padding_tokens", "reserved_response_budget"]:
            if type(record.get(key)) is not int:
                raise ValueError("token counts must be exact integers")
        prompt = record["prompt_tokens"]; target = record["completion_tokens_including_eos"]
        total = record["total_tokens"]
        bucket = next((n for n in [512,1024,2048,4096,8192,16384,32768,40960] if total <= n), "over_declared")
        if (prompt < 1 or not 2 <= target <= policy["max_completion_tokens_including_eos"] or
            total != prompt + target or record["padding_tokens"] != 0 or
            record["reserved_response_budget"] != 128 or
            record.get("ce512_length_fit_only") is not (total <= 512) or
            type(record.get("profile_bucket")) is not type(bucket) or record["profile_bucket"] != bucket):
            raise ValueError("invalid token budget/length-fit metadata")
        for key in ["truncated", "packing", "native_training_eligible"]:
            if record.get(key) is not False:
                raise ValueError("unexpected target transformation or eligibility claim")
        for key in ["prompt_ids_sha256", "completion_ids_sha256", "input_ids_sha256", "labels_sha256", "rendered_prompt_sha256"]:
            value = record.get(key)
            if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid token identity hash")
    observed = {"rows": len(rows), "ce512_length_fit_rows": sum(r["ce512_length_fit_only"] for r in measured),
                "minimum_total_tokens": min(r["total_tokens"] for r in measured),
                "maximum_total_tokens": max(r["total_tokens"] for r in measured),
                "input_tokens": sum(r["total_tokens"] for r in measured),
                "supervised_tokens": sum(r["completion_tokens_including_eos"] for r in measured)}
    for key, value in observed.items():
        if type(profile.get(key)) is not int or profile[key] != value:
            raise ValueError("token summary disagrees with per-row ledger")
    if profile.get("length_bucket_counts") != dict(Counter(str(r["profile_bucket"]) for r in measured)):
        raise ValueError("length histogram disagrees")

def release(batch, audit_dir, profile_dir, packet_dir, review_files, out):
    from verifier import verify_row, mutation_checks
    batch, manifest, rows = load_batch(batch)
    audit_dir, profile_dir, packet_dir, out = map(child_path, [audit_dir, profile_dir, packet_dir, out])
    if out.exists():
        raise ValueError("release output exists")
    audit_summary = strict_json((audit_dir / "summary.json").read_text(encoding="utf-8"))
    if (audit_summary["status"] != "PASS_DATA_MECHANICS" or
        audit_summary["canonical_sha256"] != manifest["canonical_sha256"] or
        audit_summary["source_identity"] != source_identity() or
        audit_summary["rows_passed"] != len(rows) or
        audit_summary["row_receipts_sha256"] != file_digest(audit_dir / "rows.jsonl")):
        raise ValueError("missing or stale passing audit")
    # Recompute deterministic gates rather than trusting a self-declared summary.
    audit_rows = list(read_rows(audit_dir / "rows.jsonl"))
    if len(audit_rows) != len(rows):
        raise ValueError("incomplete mechanical audit")
    for row, record in zip(rows, audit_rows):
        if (record["row_id"] != row["row_id"] or record["row_sha256"] != row["row_sha256"] or
            record.get("defects") != [] or record.get("passed") is not True or verify_row(row)):
            raise ValueError("mechanical row gate failed")
        actual_checks = mutation_checks(row)
        if not mutation_gate(actual_checks):
            raise ValueError("recomputed negative controls failed")
        if actual_checks != record["mutation_checks"]:
            raise ValueError("mechanical audit checks changed")
    profile = strict_json((profile_dir / "summary.json").read_text(encoding="utf-8"))
    if (profile["canonical_sha256"] != manifest["canonical_sha256"] or
        profile["status"] != "PASS_TOKENIZATION_ONLY" or
        profile["source_identity"] != source_identity() or
        profile["rows"] != len(rows) or
        profile["rows_sha256"] != file_digest(profile_dir / "rows.jsonl")):
        raise ValueError("missing or stale tokenizer profile")
    if profile.get("policy_sha256") != file_digest(ROOT / "config/quality_policy.json"):
        raise ValueError("tokenizer policy changed")
    measured = list(read_rows(profile_dir / "rows.jsonl"))
    policy = strict_json((ROOT / "config/quality_policy.json").read_text(encoding="utf-8"))
    validate_profile(rows, profile, measured, policy)
    for filename, expected in profile["tokenizer_files"].items():
        if file_digest(Path(policy["tokenizer_path"]) / filename) != expected:
            raise ValueError("tokenizer source drift after profiling")
    assignments = strict_json((packet_dir / "assignments.json").read_text(encoding="utf-8"))
    if assignments["canonical_sha256"] != manifest["canonical_sha256"]:
        raise ValueError("stale review assignments")
    validate_assignments(rows, assignments)  # Validate roles BEFORE role-derived paths.
    for a in assignments["assignments"]:
        if a["packet_sha256"] != file_digest(packet_dir / (a["role"] + ".jsonl")):
            raise ValueError("review packet changed")
        actual_public = list(read_rows(packet_dir / (a["role"] + ".jsonl")))
        expected_public = [{"row_id": r["row_id"], "row_sha256": r["row_sha256"],
                            "family": r["family"], "profile": r["profile"],
                            "messages": r["messages"]} for r in rows]
        if actual_public != expected_public:
            raise ValueError("review packet differs from canonical public problem")
    reviews = validate_reviews(rows, assignments, [child_path(p) for p in review_files])
    train = [r for r in rows if r["split"] == "TRAIN"]
    write_rows(out / "sft_train.candidate.messages.jsonl",
               [{"messages": r["messages"] + [{"role": "assistant", "content": r["target"]}]} for r in train])
    write_rows(out / "train_lineage.jsonl", [{"row_id": r["row_id"], "parent_id": r["parent_id"],
               "row_sha256": r["row_sha256"], "family": r["family"], "profile": r["profile"]} for r in train])
    result = {"schema_version": 1, "status": "REVIEWED_RESEARCH_PILOT",
              "canonical_sha256": manifest["canonical_sha256"], "rows": len(rows),
              "train_rows": len(train), "review_evidence": reviews,
              "audit_sha256": file_digest(audit_dir / "summary.json"),
              "tokenization_sha256": file_digest(profile_dir / "summary.json"),
              "candidate_sft_sha256": file_digest(out / "sft_train.candidate.messages.jsonl"),
              "source_identity": source_identity(), "training_ready": False,
              "training_authorized": False, "historical_semantic_overlap": "NOT_AUDITED",
              "human_review": "NOT_PERFORMED", "native_model_execution": "NOT_PERFORMED",
              "structural_transfer": "NOT_MEASURED", "model_gain": "UNMEASURED",
              "retention": "UNMEASURED",
              "remaining_gates": ["history/source overlap admission", "calibrated language review where applicable",
                  "registered data/runtime integration", "native context and training qualification",
                  "baseline capture and held-out acquisition/retention comparison"]}
    write_json(out / "release_manifest.json", result)
    print(canonical(result))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build"); p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=2026093001)
    p = sub.add_parser("audit"); p.add_argument("--batch", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("packets"); p.add_argument("--batch", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("release")
    for name in ["batch", "audit-dir", "profile-dir", "packet-dir", "out"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--reviews", nargs=2, required=True)
    args = vars(parser.parse_args()); command = args.pop("command")
    if command == "release": args["review_files"] = args.pop("reviews")
    {"build": build, "audit": audit, "packets": packets, "release": release}[command](**args)

if __name__ == "__main__":
    main()
