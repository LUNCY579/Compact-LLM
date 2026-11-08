"""Export reviewed language candidates and unassigned CAL templates, write once.

This stdlib-only command never imports a tokenizer, model, native runtime or grader.
All inputs and both implementation files must be tracked and clean. Exact bytes,
Git HEAD and cleanliness are checked again before output creation. Only the frozen
source labels below are read; no CHECK/FINAL payload is opened. Reviewer receipts
identify assistant source reviews, never human approval or qualified grading.

API: prepare(output). CLI: python -m ...scripts.prepare_language_eval --output DIR.
A failed write preserves any partial new directory; retry requires another path.
No existing file or directory is overwritten, and no Git mutation is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from . import language_eval_inputs as language


REGISTRATION = "manifests/LANGUAGE_AUTHORING_REGISTRATION.json"
FREEZE = "manifests/LANGUAGE_WORLD_SPEC_FREEZE.json"
PROPOSAL = "manifests/LANGUAGE_DEV_PROPOSAL.json"
SPEC_REVIEW = "receipts/LANGUAGE_WORLD_SPEC_REVIEW.json"
AMENDMENT = "AMENDMENT_006.md"
SPEC_FILES = tuple(language.SOURCE_DIRECTORY + "SPECS_" + part + ".json" for part in ("A", "B"))
PAYLOAD_FILES = tuple(language.SOURCE_DIRECTORY + "PAYLOADS_" + part + ".json" for part in ("A", "B"))
REVIEW_FILES = ("receipts/LANGUAGE_PAYLOAD_REVIEW_A.json", "receipts/LANGUAGE_PAYLOAD_REVIEW_B.json")
INPUT_FILES = (AMENDMENT, REGISTRATION, FREEZE, PROPOSAL, SPEC_REVIEW) + SPEC_FILES + PAYLOAD_FILES + REVIEW_FILES


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _identity(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _same(actual, expected, label):
    _require(language.canonical(actual) == language.canonical(expected), label + " drift")


def _path(package, label):
    """Fixed labels must resolve to themselves; reject redirects before reading."""
    _require(label in INPUT_FILES, "unregistered source label")
    path = package / label
    _require(path.resolve() == path, "source symlink or redirected path")
    return path


def _raw(path):
    before = path.stat()
    value = path.read_bytes()
    after = path.stat()
    _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
             and len(value) == after.st_size, "source changed during read: " + str(path))
    return value


def _git_read(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True)


def _tracked_clean(root, relative):
    subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", "--", relative],
                   check=True, capture_output=True)
    _require(not _git_read(root, "status", "--porcelain", "--", relative).strip(),
             "critical source is uncommitted: " + relative)


def _fixed_registration(registration, specs, proposal_identity):
    _require(type(registration) is dict, "registration object required")
    fixed = {
        "schema_version": 1, "status": "ACCEPTED_DATA_ONLY_AUTHORING_NOT_GRADING_OR_NATIVE_EXECUTION",
        "amendment": AMENDMENT, "proposal_sha256": proposal_identity["sha256"],
        "spec_freeze_required_before_rendering": True, "DEV_parent_count": 64, "CAL_parent_count": 16,
        "CAL_unique_candidate_controls": 96, "CAL_repeats_per_template": 16,
        "packet_templates": [{"template_id": name, "seed": seed} for name, seed in language.TEMPLATES.items()],
        "prompt_cap_including_template": 896, "reference_control_cap_including_EOS": 128,
        "truncation": False, "system_message": None, "qualified_reference_score_vectors": None,
        "qualified_graders": [], "human_approval": False, "native_launch_authorized": False, "training_eligibility": False,
        "CHECK": {"reserved_parent_slots": 128, "payloads": 0, "status": "CLOSED_COUNT_ONLY"},
        "FINAL": {"reserved_parent_slots": None, "payloads": 0, "status": "CLOSED_UNSPECIFIED"},
    }
    for field, expected in fixed.items():
        _require(field in registration, "registration field missing: " + field)
        _same(registration[field], expected, "registration " + field)
    slots = [{**{field: spec[field] for field in ("parent_id", "role", "stratum", "conversation_form", "author_id")},
              "spec_source": spec["spec_source_path"],
              "payload_source": spec["spec_source_path"].replace("/SPECS_", "/PAYLOADS_")} for spec in specs]
    _same(registration.get("slots"), slots, "fixed registration slots")


def _frozen_specs(freeze, review, specs, input_identities):
    _require(type(freeze) is dict and type(review) is dict, "spec freeze/review objects required")
    fixed = {"status": "FROZEN_LANGUAGE_WORLD_SPECS_BEFORE_RENDERING", "payloads_authored_at_freeze": 0,
             "native_calls": 0, "qualified_labels": False, "training_eligibility": False, "CHECK_FINAL_payloads_opened": False}
    for field, expected in fixed.items():
        _same(freeze.get(field), expected, "spec freeze " + field)
    _same(freeze.get("registration"), input_identities[REGISTRATION], "frozen registration identity")
    _same(freeze.get("assistant_review"), input_identities[SPEC_REVIEW], "frozen spec-review identity")
    _same(freeze.get("source_files"), {label: input_identities[label] for label in SPEC_FILES}, "frozen spec source identities")
    fields = ("parent_id", "role", "stratum", "conversation_form", "author_id", "world_spec_sha256", "spec_file_sha256")
    _same(freeze.get("worlds"), [{field: spec[field] for field in fields} for spec in specs], "all 80 frozen world identities")
    for field, expected in {"status": "ASSISTANT_SPEC_REVIEW_NOT_QUALIFIED_GRADING", "reviewer_id": "/root",
                            "remaining_blocking_spec_findings": [], "native_calls": 0,
                            "qualified_graders": 0, "reference_score_vectors": None}.items():
        _same(review.get(field), expected, "spec review " + field)
    _same(review.get("reviewed_parents"), [spec["parent_id"] for spec in specs], "spec-review population")
    _same(review.get("source_files"), {Path(label).name: input_identities[label] for label in SPEC_FILES}, "spec-review file identities")


def _payload_reviews(reviews, rows, input_identities):
    for reviewer_part, payload_part, label in (("A", "B", REVIEW_FILES[0]), ("B", "A", REVIEW_FILES[1])):
        review = reviews[label]
        _require(type(review) is dict, "payload review object required")
        payload_label = language.SOURCE_DIRECTORY + "PAYLOADS_" + payload_part + ".json"
        fixed = {"status": "ASSISTANT_SOURCE_REVIEW_NOT_QUALIFIED_GRADING",
                 "reviewer_id": language.AUTHORS[reviewer_part], "reviewed_payload_file": payload_label,
                 "reviewed_payload_file_sha256": input_identities[payload_label]["sha256"],
                 "blocking_findings": [], "native_grading_qualification": False, "human_approval": False}
        for field, expected in fixed.items():
            _require(field in review, "payload review field missing: " + field)
            _same(review[field], expected, "payload review " + field)
        parents = review.get("reviewed_parent_ids")
        expected_parents = {row["parent_id"] for row in rows if row["payload_source_path"] == payload_label}
        _require(type(parents) is list and len(parents) == 40 and all(type(parent) is str for parent in parents)
                 and len(set(parents)) == 40 and set(parents) == expected_parents,
                 "payload review must cover exactly its 40 source parents")


def _write(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)


def prepare(output):
    """Publish one immutable source export; never grant native/grading authority."""
    started = time.perf_counter()
    output = Path(output).resolve()
    _require(not output.exists(), "write-once output directory already exists")
    package = Path(__file__).resolve().parents[1]
    root = package.parents[1]
    code_commit = _git_read(root, "rev-parse", "HEAD").strip()
    _require(re.fullmatch(r"[0-9a-f]{40}", code_commit) is not None, "full source Git commit required")
    paths = [Path(__file__).resolve(), Path(language.__file__).resolve()] + [_path(package, label) for label in INPUT_FILES]
    source_files, source_paths, raw_inputs = {}, {}, {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        _tracked_clean(root, relative)
        raw = _raw(path)
        source_files[relative] = _identity(raw)
        source_paths[relative] = path
        if path.is_relative_to(package):
            label = path.relative_to(package).as_posix()
            if label in INPUT_FILES:
                raw_inputs[label] = raw
    input_identities = {label: _identity(raw) for label, raw in raw_inputs.items()}
    documents = {label: language._parse(raw) for label, raw in raw_inputs.items() if label.endswith(".json")}
    specs = language.validate_specs([{"source_path": label, "raw_bytes": raw_inputs[label]} for label in SPEC_FILES])
    _fixed_registration(documents[REGISTRATION], specs, input_identities[PROPOSAL])
    _frozen_specs(documents[FREEZE], documents[SPEC_REVIEW], specs, input_identities)
    rows = language.validate_payloads(specs, [{"source_path": label, "raw_bytes": raw_inputs[label]} for label in PAYLOAD_FILES])
    _payload_reviews(documents, rows, input_identities)
    outputs = {role + ".jsonl": b"".join(language.canonical(row) + b"\n" for row in rows if row["role"] == role)
               for role in ("DEV", "CAL")}
    packet_bindings = {}
    for template, seed in language.TEMPLATES.items():
        packet = language.build_cal_packet(rows, seed=seed, template_id=template)
        language._packet(packet)
        outputs["packets/" + template + ".rating_view.json"] = language.canonical(packet["rating_view"]) + b"\n"
        outputs["coordinator/" + template + ".mapping.json"] = language.canonical(packet["coordinator_mapping"]) + b"\n"
        packet_bindings[template] = {"seed": seed, "items": 112, "unique_controls": 96, "hidden_repeats": 16,
                                     "rating_view_sha256": packet["coordinator_mapping"]["rating_view_sha256"],
                                     "mapping_sha256": packet["coordinator_mapping"]["mapping_sha256"]}
    for relative, identity in source_files.items():
        path = source_paths[relative]
        if path.relative_to(package).as_posix() in INPUT_FILES:
            _path(package, path.relative_to(package).as_posix())
        _same(_identity(_raw(path)), identity, "source changed before publication: " + relative)
        _tracked_clean(root, relative)
    _require(_git_read(root, "rev-parse", "HEAD").strip() == code_commit, "source Git HEAD changed during preparation")
    dirty = _git_read(root, "status", "--porcelain")
    output.mkdir(parents=True, exist_ok=False)
    (output / "packets").mkdir()
    (output / "coordinator").mkdir()
    for label, raw in outputs.items():
        _write(output / label, raw)
    files = {label: _identity(_raw(output / label)) for label in outputs}
    for label, raw in outputs.items():
        _same(files[label], _identity(raw), "published output " + label)
    receipt = {
        "schema_version": 1, "status": "AUTHORED_LANGUAGE_CANDIDATES_UNQUALIFIED",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "code_commit": code_commit,
        "overall_dirty_status": dirty, "source_files": source_files, "source_inventory_sha256": language.digest(source_files),
        "input_files": input_identities, "rows_sha256": language.digest(rows), "files": files,
        "counts": {"worlds": 80, "DEV_parents": 64, "CAL_parents": 16, "CAL_unique_controls": 96,
                   "packet_templates": 2, "rating_items_per_template": 112, "hidden_repeats_per_template": 16,
                   "actual_rating_records": 0}, "packet_templates": packet_bindings,
        "model_calls": 0, "native_calls": 0, "native_model_calls": 0, "programmatic_model_calls": 0,
        "training_updates": 0, "teacher_calls": 0,
        "model_call_scope": "Calls made by this export script only; excludes assistant source authorship and peer review.",
        "assistant_authoring_review_cost": "UNMETERED_BY_THIS_SCRIPT; assistant authored source/control texts, not native Titan responses",
        "native_launch_authorized": False, "training_eligibility": False, "grading_qualified": False,
        "reference_labels_qualified": False, "human_approval": False, "source_correct_membership": False,
        "tokenizer_fit_measured": False, "held_out_access": {"CHECK": False, "FINAL": False},
        "wall_seconds": time.perf_counter() - started,
        "wall_seconds_scope": "Export work only; excludes unmeasured assistant authorship and peer review.",
        "limitations": "Assistant source review only. Packets are unassigned; coordinator separation is a workflow boundary, not cryptographic protection. No native, tokenizer or grading qualification.",
    }
    _write(output / "DATA_RECEIPT.json", language.canonical(receipt) + b"\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare(args.output)
    print(json.dumps({key: receipt[key] for key in ("status", "counts", "native_launch_authorized", "grading_qualified")}))


if __name__ == "__main__":
    main()
