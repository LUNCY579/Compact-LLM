#!/usr/bin/env python3
"""Read-only, standard-library verification of the portable Titan offline archive.

Usage: python -B verify_bundle.py [--root ARCHIVE_ROOT] [--hash-only]
The default checks inventory, byte lengths, SHA-256 and frozen offline facts.
--hash-only skips scientific snapshot checks (useful for small verifier tests).
No archived code is imported or executed. PASS is not model-quality evidence.
Exit 0 means verified; exit 2 means blocked/failed. Output goes to stdout only.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

MANIFEST = "PUBLICATION_MANIFEST.json"
CONTROLS = {"README.md", "PUBLICATION_REVIEW.md", "verify_bundle.py", "test_verify_bundle.py", MANIFEST,
            ".gitattributes", ".gitignore"}
DENIED_DIRS = {".git", ".cache", ".pytest_cache", "__pycache__", "node_modules", ".venv", "venv"}
DENIED_EXTS = {".bin", ".pt", ".pth", ".onnx", ".safetensors", ".npy", ".npz", ".pkl", ".pickle",
               ".exe", ".dll", ".so", ".dylib", ".pyc", ".pyo", ".zip", ".gz", ".tar", ".7z"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class SummaryParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def relative(value):
    require(isinstance(value, str) and value and "\\" not in value and ":" not in value, "invalid POSIX path")
    path = PurePosixPath(value)
    require(path.parts and not path.is_absolute() and path.as_posix() == value and
            all(part not in {".", ".."} for part in path.parts), "unsafe path: " + value)
    return path


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def load_json(path):
    def invalid_constant(value):
        raise ValueError("nonfinite JSON value: " + value)
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=reject_duplicate_keys,
                      parse_constant=invalid_constant)


def check_member(root, path):
    info = path.lstat()
    require(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) &
            getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024), "link/reparse point: " + str(path))
    require(path.resolve().is_relative_to(root), "member escapes archive root: " + str(path))
    require(stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode), "nonregular member: " + str(path))
    parts = path.relative_to(root).parts
    require(not any(p.lower() in DENIED_DIRS or p.lower().startswith(".env") for p in parts),
            "environment/cache member: " + str(path))
    if stat.S_ISREG(info.st_mode):
        require(path.suffix.lower() not in DENIED_EXTS, "binary/archive member: " + str(path))
        payload = path.read_bytes()
        require(b"\x00" not in payload, "binary NUL content: " + str(path))
        payload.decode("utf-8-sig")
        return payload


def verify_inventory(root):
    require(root.is_dir() and not root.is_symlink(), "archive root must be an ordinary directory")
    check_member(root, root)
    check_member(root, root / MANIFEST)
    manifest = load_json(root / MANIFEST)
    require(isinstance(manifest, dict), "manifest must be a JSON object")
    controls = manifest.get("control_files", [])
    require(isinstance(controls, list) and all(isinstance(x, str) and x in CONTROLS for x in controls),
            "unsupported control_files (only the fixed publication controls are allowed)")
    require(len(controls) == len(set(controls)), "duplicate control file")
    files, exclusions = manifest["files"], manifest["exclusions"]
    require(isinstance(files, list) and files and isinstance(exclusions, list), "invalid file inventories")
    included, excluded = {}, set()
    for entries, target in ((files, included), (exclusions, excluded)):
        for entry in entries:
            origin, original = relative(entry["original_root"]), relative(entry["original_relative_path"])
            require(len(origin.parts) == 1, "original_root must be a directory name")
            name = str(PurePosixPath("bundle") / origin / original)
            require(re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None and
                    type(entry["bytes"]) is int and entry["bytes"] >= 0, "invalid hash/byte receipt: " + name)
            require(name not in target, "duplicate inventory member: " + name)
            if target is included:
                require(str(relative(entry["path"])) == name, "path/origin mismatch: " + name)
                target[name] = entry
            else:
                require(isinstance(entry["reason"], str) and entry["reason"].strip(), "missing exclusion reason")
                target.add(name)
    require(not set(included) & excluded, "member both included and excluded")
    expected = set(included) | set(controls) | {MANIFEST}
    directories = {str(parent) for name in expected for parent in PurePosixPath(name).parents if str(parent) != "."}
    observed = set()
    for current, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            path = Path(current) / name
            payload = check_member(root, path)
            member = path.relative_to(root).as_posix()
            if payload is None:
                require(member in directories, "untracked directory: " + member)
                continue
            require(member in expected, "untracked file: " + member)
            observed.add(member)
            if member in included:
                receipt = included[member]
                require(len(payload) == receipt["bytes"], "byte length mismatch: " + member)
                require(hashlib.sha256(payload).hexdigest() == receipt["sha256"], "SHA-256 mismatch: " + member)
    require(observed == expected, "missing archive members: " + ", ".join(sorted(expected - observed)))
    return manifest, set(included)


def verify_science(root, included, manifest):
    source = "bundle/Titan-Joint-Quality-Pilot-v3/"
    runs = "bundle/Titan-Joint-Quality-Pilot-v3-runs/"
    def read(name):
        require(name in included, "required frozen snapshot absent from inventory: " + name)
        return load_json(root / name)
    def values(record, expected, label):
        for key, value in expected.items():
            require(record.get(key) == value and type(record.get(key)) is type(value), label + ": " + key)
    values(manifest, {"schema": "TITAN_OFFLINE_PUBLICATION_V1", "status": "REVIEWED_OFFLINE_ARCHIVE_UNADMITTED",
        "publication_date": "2026-10-09", "native_model_calls": 0, "native_authorized": False,
        "confirmation_materialized": False, "model_quality_gain_established": False}, "publication claim guards")
    protocol = read(source + "PROTOCOL.json")
    values(protocol, {"native_authorized": False, "training_authorized": False,
        "confirmation_authorized": False, "native_command": None, "fixture_results_are_capability_evidence": False,
        "native_authority_change": False, "no_serving_changes": True}, "offline protocol")
    require("UNADMITTED_OFFLINE" in protocol["status"], "protocol must remain unadmitted/offline")
    handoff = read(runs + "handoff-002/OWNER_HANDOFF.json")
    values(handoff, {"status": "HELD_OWNER_REVIEW_AND_NATIVE_INTEGRATION_REQUIRED", "native_model_calls": 0,
        "owner_messages_by_this_implementation": 0, "executable_native_command": None,
        "quality_improvement_established": False, "complete_exposure_mechanics_proof": False,
        "data_review_status": "DRAFT_UNREVIEWED_NOT_FROZEN"}, "held handoff")
    values(handoff["contract"], {"native_authorized": False, "training_authorized": False,
        "confirmation_authorized": False, "native_model_calls": 0}, "handoff authority")
    values(handoff["prospective_confirmation"], {"materialized": False, "read_authorized": False}, "confirmation")
    plan = read(runs + "exposure-002/EXPOSURE_PLAN.json")
    values(plan, {"status": "PROSPECTIVE_OFFLINE_EXPOSURE_PLAN", "native_authorized": False,
        "training_authorized": False, "source_objective_qualified": False, "selected_rows": 200,
        "selected_parent_count": 100, "processed_tokens_per_arm": 53913, "updates": 28,
        "truncation": False, "padding": False, "capability_evidence": False,
        "same_sequences_order_groups": True, "same_reasoning_targets": True}, "final exposure")
    groups = plan["groups"]
    require(len(groups) == 28, "update count")
    rows = [row for group in groups for row in group["rows"]]
    require(len(rows) == 200 and set(Counter(row["parent_id"] for row in rows).values()) == {2}, "paired row inventory")
    require(len({row["parent_id"] for row in rows}) == 100, "parent count")
    case_ids = [row["case_id"] for row in rows]
    require(case_ids == plan["selected_case_ids"] and len(set(case_ids)) == 200, "selection order/uniqueness")
    role_tokens = Counter()
    for index, group in enumerate(groups):
        group_rows = group["rows"]
        for row in group_rows:
            ids, mask = row["ids"], row["label_mask"]
            require(len(ids) == len(mask) == row["processed_tokens"] <= 1024 and
                    all(type(x) is bool for x in mask), "row length/mask contract")
            require(sum(mask) == row["response_eos_tokens"] > 0, "response-plus-EOS labels")
            require(0 < row["response_start"] < row["response_end"] <= len(ids) and
                    row["eos_position"] == row["response_end"] - 1 and ids[row["eos_position"]] == 151645 and
                    mask == [row["response_start"] <= i < row["response_end"] for i in range(len(ids))],
                    "response-plus-EOS mask boundaries")
            role_tokens[row["role"]] += len(ids)
        total = sum(row["processed_tokens"] for row in group_rows)
        joint = sum(sum(row["label_mask"]) for row in group_rows)
        reasoning = sum(sum(row["label_mask"]) for row in group_rows if row["role"] == "reasoning")
        require(total == group["processed_tokens"] <= 2048 and reasoning > 0, "group cap/reasoning labels")
        require(group["group_index"] == index and group["planned_denominator"] ==
                group["joint_label_tokens"] == joint == plan["planned_denominators"][index] and
                group["reasoning_label_tokens"] == group["reasoning_only_label_tokens"] == reasoning,
                "common denominator must sum all planned labels")
    require(sum(role_tokens.values()) == 53913 and dict(role_tokens) == plan["processed_token_counts_by_role"], "processed tokens")
    for role, target in protocol["exposure"]["role_shares"].items():
        share = role_tokens[role] / 53913
        require(abs(share - target) <= .05 and abs(share - plan["processed_token_shares_by_role"][role]) < 1e-12, "role shares")
    require(sum(plan["planned_denominators"]) == plan["joint_label_tokens"] == 1653 and
            sum(group["reasoning_only_label_tokens"] for group in groups) == plan["reasoning_only_label_tokens"] == 785,
            "label totals")
    audit = read(runs + "ACTUAL_SOURCE_REPAIR_AUDIT_001.json")
    values(audit, {"source_aware_math_correct": 300, "old_prefix_sourceblind_correct": 150, "rows": 300,
        "parents": 150, "source_blind_projected_pair_equal_count": 150, "same_world_distinct_truth_query_pair_count": 150,
        "non_G_rows_exactly_preserved": 1700, "all_case_order_preserved": True, "pilot_bytes_exactly_preserved": True,
        "native_model_calls": 0, "model_quality_gain_established": False, "native_admission": False,
        "confirmation_reads": 0}, "actual repair/preservation")
    final = read(runs + "FINAL_MECHANICAL_VERIFICATION_002.json")
    values(final, {"rows": 200, "parents": 100, "updates": 28, "processed_tokens_per_arm": 53913,
        "every_group_has_reasoning_labels": True, "common_planned_denominators": True,
        "data_bytes_identical_to_initial_repaired_epoch": True, "initial_001_sources_and_receipts_preserved": True,
        "native_authorized": False, "native_model_calls": 0, "model_quality_established": False}, "mechanical receipt")
    data = read(runs + "data-002/DATA_MANIFEST.json")
    values(data, {"native_authorized": False, "training_authorized": False, "native_model_calls": 0,
        "confirmation_materialized": False, "confirmation_authorized": False, "frozen": False,
        "overlap_review": "UNREVIEWED"}, "unreviewed data")
    require(data["teacher"]["independence_review"] == "UNREVIEWED", "teacher independence review")
    review = read("bundle/Titan-Quality-Patterns-v1-runs/selected-authority-v3-independent-001.json")
    values(review, {"status": "OFFLINE_SOURCE_AND_ACTUAL_G_MECHANICS_CONFIRMED_UNADMITTED"}, "independent review")
    values(review["boundaries"], {"native_model_calls": 0, "training_calls": 0, "protected_or_confirmation_reads": 0,
        "owner_messages": 0, "native_admission_frozen": False, "scientific_model_quality_proved": False}, "review boundaries")
    values(review["qualification_receipt_review"], {"native_qualification": False, "qualification_counts": None,
        "complete_native_forecast": None}, "native qualification remains open")
    for relative_name, digest in final["final_source_files_sha256"].items():
        for prefix in (source, runs + "source-epoch-002/"):
            name = prefix + str(relative(relative_name))
            require(name in included and hashlib.sha256((root / name).read_bytes()).hexdigest() == digest,
                    "canonical source epoch receipt: " + name)
    def raw(name):
        require(name in included, "canonical open input absent: " + name)
        return (root / name).read_bytes()
    old_runs = "bundle/Titan-Joint-Quality-Pilot-v2-runs/"
    for name, digest in data["artifact_sha256"].items():
        current = raw(runs + "data-002/" + str(relative(name)))
        require(hashlib.sha256(current).hexdigest() == digest and current == raw(runs + "data-001/" + name),
                "canonical repaired input epochs: " + name)
        if name.startswith("pilot."):
            require(current == raw(old_runs + "data-001/" + name), "PILOT input preservation: " + name)
    old_train = raw(old_runs + "data-001/train.jsonl").splitlines()
    new_train = raw(runs + "data-002/train.jsonl").splitlines()
    require(len(old_train) == len(new_train) == 2000, "open TRAIN count")
    preserved = 0
    bank = {}
    for old_line, new_line in zip(old_train, new_train):
        old_row, new_row = json.loads(old_line), json.loads(new_line)
        require(old_row["case_id"] == new_row["case_id"], "canonical TRAIN case order")
        bank[new_row["case_id"]] = new_row
        if old_row["role"] != "grounding":
            require(old_line == new_line, "non-G public input/target preservation")
            preserved += 1
    require(preserved == 1700 and all(row["source_record"] == bank[row["case_id"]] for row in rows),
            "canonical non-G inputs and selected public source records")


def main(argv=None):
    parser = SummaryParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--hash-only", action="store_true", help="skip frozen scientific facts; still enforce archive safety")
    try:
        args = parser.parse_args(argv)
        root = args.root.absolute()
        info = root.lstat()
        require(not root.is_symlink() and not getattr(info, "st_file_attributes", 0) &
                getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024), "archive root may not be a link/reparse point")
        root = root.resolve()
        manifest, included = verify_inventory(root)
        if not args.hash_only:
            verify_science(root, included, manifest)
        mode = "hash/inventory only" if args.hash_only else "hash/inventory and offline snapshot facts"
        print(f"PASS: {len(included)} included files; {len(manifest['exclusions'])} exclusions; {mode}.")
        print("No archived code executed. Fixture/mechanical PASS is not model-quality evidence or native authorization.")
        return 0
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, IndexError) as error:
        print("BLOCKED/FAIL: " + str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
