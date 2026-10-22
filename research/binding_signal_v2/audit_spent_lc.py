"""Closed, post-delivery text-contact audit against the already-spent LC suite.

Never exposes LC text, answers, predictions or outcomes. The frozen candidate is
an input, never rewritten. This uses the documented native text-contact rule;
it does not reconstruct the exclusion manifest's unspecified stopword algorithm.
"""
from pathlib import Path
import importlib.util
import json

import audit_open_exclusions as opened

BASE = Path("C:/Users/clclc/BusinessProject")
RECORDS = BASE / "rineya-dataset-retention-records"
CANDIDATE = RECORDS / "binding-signal-prep-20260920/candidate-v2"
MANIFEST = RECORDS / "lcr-exclusion-manifest/LCR_EXCLUSION_MANIFEST.json"
MANIFEST_SHA = "54b02b35c37b1cc842dbf413aac56980d3d3ade203f3ea5092bb332416914858"
CLONE = BASE / "ChatBubble-retention-doselora-resume-20260920"
PANEL = CLONE / "Rineya_Long_Context_Demo_v1/data/all_prompts.jsonl"
GUARD = CLONE / "experiments/reasoning_posttraining_v1/text_contact.py"


def main():
    out = CANDIDATE / "SPENT_LC_TEXT_CONTACT_AUDIT.json"
    if out.exists():
        raise FileExistsError("Audit output is create-only")
    if opened.sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError("Approved exclusion-aid identity changed")
    manifest = opened.read_json(MANIFEST)
    expected = manifest["source_sha256"]
    if opened.sha(PANEL) != expected:
        raise ValueError("Spent LC prompt suite differs from the approved source")
    build = opened.read_json(CANDIDATE / "BUILD.json")
    if build["record"] != "BINDING_SIGNAL_V2_DATA_READINESS":
        raise ValueError("Only the already-frozen v2 draft is admitted to this audit")
    frozen = {name: opened.sha(CANDIDATE / name) for name in ("TRAIN.jsonl", "DEV.jsonl", "BUILD.json")}
    if any(frozen[name] != digest for name, digest in build["files_sha256"].items()):
        raise ValueError("Candidate differs from its pre-audit BUILD")
    spec = importlib.util.spec_from_file_location("binding_v2_spent_lc_contact", GUARD)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    if guard.POLICY != "user_text_contact_v1_id_8gram_digit_blinded_stem":
        raise ValueError("Unexpected native text-contact policy")
    guard_sha = opened.sha(GUARD)
    panel = opened.read_rows(PANEL)
    if len(panel) != manifest["source_rows"] or len(panel) != 256:
        raise ValueError("Spent LC row inventory changed")
    panel_ids = {str(row[key]) for row in panel for key in ("id", "scenario_id") if row.get(key)}
    results = {}
    for split in ("train", "dev"):
        rows = opened.read_rows(CANDIDATE / (split.upper() + ".jsonl"))
        if len(rows) != build["rows"][split]:
            raise ValueError("Frozen candidate count changed")
        contact = opened.compare(rows, panel, guard)
        ids = {str(row[key]) for row in rows for key in ("id", "parent_id", "base_world_id") if row.get(key)}
        collisions = [{"candidate_id": row["candidate_id"], "reasons": row["reasons"]}
                      for row in contact["collisions"]]
        results[split] = {"rows": len(rows), "contact_rows": contact["collision_rows"],
                          "reason_counts": contact["reason_counts"], "candidate_contacts": collisions,
                          "row_or_scenario_id_intersection": sorted(ids & panel_ids)}
    if opened.sha(PANEL) != expected or opened.sha(GUARD) != guard_sha:
        raise ValueError("Audit source/guard changed during inspection")
    if any(opened.sha(CANDIDATE / name) != digest for name, digest in frozen.items()):
        raise ValueError("Candidate changed during audit")
    blocked = any(value["contact_rows"] or value["row_or_scenario_id_intersection"] for value in results.values())
    record = {
        "schema": 1, "record": "BINDING_V2_SPENT_LC_POST_DELIVERY_CONTACT_AUDIT",
        "status": "BLOCKED_TEXT_CONTACT" if blocked else "PASS_DOCUMENTED_NATIVE_TEXT_CONTACT_ONLY",
        "candidate_files_sha256": frozen,
        "approved_manifest_path": str(MANIFEST), "approved_manifest_sha256": MANIFEST_SHA,
        "source_prompt_path": str(PANEL), "source_prompt_sha256": expected, "source_rows": len(panel),
        "guard_path": str(GUARD), "guard_sha256": guard_sha, "guard_policy": guard.POLICY,
        "auditor_sha256": opened.sha(Path(__file__)), "shared_comparison_code_sha256": opened.sha(Path(opened.__file__)),
        "scope": "Frozen candidate checked by an isolated auditor against the already-spent LC prompt suite; no raw source text returned to the builder.",
        "checks": results, "candidate_rows_modified": 0, "model_calls": 0,
        "answer_key_read": False, "predictions_or_outcomes_read": False, "reserved_any_read_files_opened": 0,
        "manifest_stopword_8gram_check": "UNVERIFIED_ORIGINAL_NORMALIZATION_UNAVAILABLE",
        "manifest_stopword_8gram_overlap": None,
        "reserved_bank_fingerprint_coverage": "UNVERIFIED_NO_APPROVED_INDICES",
        "semantic_contamination_certificate": False,
        "training_permission": False, "evaluation_permission": False, "quality_claim": False,
        "release_permitted": False,
    }
    opened.write_report(out, record)
    print(json.dumps({"status": record["status"], "output": str(out),
                      "contact_rows": {split: value["contact_rows"] for split, value in results.items()}}))
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
