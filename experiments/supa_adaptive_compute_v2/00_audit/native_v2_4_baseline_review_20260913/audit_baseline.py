"""Score-blind v2.4 versus preserved v2.3 A0 audit; standard library only.

Reuses the previous independent audit's unchanged readers and full receipt
checks. No tensors, checkpoint bytes, efficacy scores or other arms are read.
Without --write, only stdout is produced. Receipt creation is exclusive.
"""
from datetime import datetime, timezone
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


OUTPUT = Path(__file__).resolve().parent
HELPER = OUTPUT.parent / "native_v2_3_baseline_review/audit_baseline.py"
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("preserved_baseline_audit", HELPER)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

# The helper's literal "v2.3" label enables its then-current source check.
# These numeric labels deliberately skip that obsolete conditional; this wrapper
# explicitly rehashes all current v2.4 sources below. Preserved v2.3 sources are
# reconciled against the v2.3 registration and execution receipts, not HEAD.
audit.REGISTRATIONS = {
    "2.3": "86e768e08bf3576dd8ce80ec939b7e876fb5fb7449d41a3aa51c497ce6607b44",
    "2.4": "081a8f6bf3c7396c97f13e3d4a7d55bf7311b37dcc0b0bc7bfd32478fa44241e",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.write:
        for name in ("receipt.json", "audit.md"):
            if (OUTPUT / name).exists():
                raise FileExistsError(f"refusing to replace {name}")
    expected, dataset_path, dataset_sha = audit.read_expected()
    runs, excluded = audit.inventory()
    counts = {version: {"complete_cases": sum(len(value["example_ids"]) for _, value in entries),
                        "complete_shards": len(entries)} for version, entries in runs.items()}
    if any(value != {"complete_cases": 128, "complete_shards": 30} for value in counts.values()):
        print(json.dumps({"status": "INCOMPLETE", "inventory": counts, "receipt_written": False}))
        return 2
    inspected = {version: audit.inspect_version(version, runs[version], expected, dataset_path, dataset_sha)
                 for version in audit.REGISTRATIONS}
    reg_path = (audit.ROOT / "native_armr/evidence/00_preregistration"
                / audit.REGISTRATIONS["2.4"] / "registration.json")
    current_registration = audit.load(reg_path)
    for name, digest in current_registration["sources"].items():
        audit.require(audit.sha(audit.REPO / name) == digest, "current_v2_4_sources", name)

    old_rows, old_logits, old_summary = inspected["2.3"]
    new_rows, new_logits, new_summary = inspected["2.4"]
    audit.require(old_summary["qualified_runtime"] == new_summary["qualified_runtime"]
                  and old_summary["checkpoint_files"] == new_summary["checkpoint_files"],
                  "cross_version_identity", "same runtime and canonical checkpoint receipts")
    differences = {key: [] for key in ("token_ids", "gold_nll", "logit_file_bytes")}
    nll_max_abs_difference = 0.0
    for identifier in sorted(old_rows):
        for key in ("token_ids", "gold_nll"):
            if old_rows[identifier][key] != new_rows[identifier][key]:
                differences[key].append(identifier)
        nll_max_abs_difference = max(nll_max_abs_difference,
                                     abs(old_rows[identifier]["gold_nll"] - new_rows[identifier]["gold_nll"]))
        if old_logits[identifier] != new_logits[identifier]:
            differences["logit_file_bytes"].append(identifier)
    comparison = {
        "n": 128,
        "identical_counts": {key: 128 - len(value) for key, value in differences.items()},
        "difference_example_ids": differences,
        "maximum_absolute_nll_difference": nll_max_abs_difference,
        "comparison_rule": "Exact token lists and parsed NLL values; SHA256 equality of saved logit file bytes; no tolerances",
    }
    for path in (HELPER, Path(__file__)):
        audit.sha(path)
    receipt = {
        "status": "PASS" if not any(differences.values()) else "EVIDENCE_VALID_WITH_RAW_DIFFERENCES",
        "scope": "Complete A0 architecture baseline evidence and raw-output reproducibility only",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "wall_seconds": time.perf_counter() - started,
        "checks_passed": sum(audit.checks.values()),
        "checks_by_category": dict(audit.checks),
        "dataset_manifest_sha256": dataset_sha,
        "current_source_files_rehashed": len(current_registration["sources"]),
        "baselines": {"v2.3": old_summary, "v2.4": new_summary},
        "comparison": comparison,
        "excluded_noncomplete_receipts": {"v" + key: value for key, value in excluded.items()},
        "file_sha256": audit.file_hashes,
        "boundaries": {
            "scoring_or_oracle_imported": False, "accuracy_computed": False,
            "response_or_answer_fields_consulted": False, "torch_or_numpy_imported": False,
            "model_or_checkpoint_loaded": False, "gpu_used": False,
            "confirmation_bank_opened": False, "cross_scale_data_opened": False,
            "git_used": False, "old_evidence_modified": False,
        },
        "limitations": [
            "Saved-evidence audit only: no fresh forward, checkpoint-byte reread or live tensor inspection.",
            "Weight preservation and runtime facts are reconciled before/after execution and qualification receipts.",
            "Case files are bound by exact opaque JSON equality to SHA-verified results rows; their SHA values are newly inventoried.",
            "Logit archive bytes are hashed without deserializing tensors or recomputing logit metrics.",
            "Current v2.4 sources are rehashed; v2.3 source identity is checked against preserved registration and execution manifests, not historical Git blobs.",
            "Qualification report, artifact hashes and all 40 pass/gate receipts per version are checked; numerical qualification metrics are not recomputed.",
            "Bank and result field projection discards answers, responses and scores; opaque byte reconciliation traverses their serialized bytes without semantic consultation.",
            "A0 identity does not qualify other arms, complete the architecture gate, or establish efficacy.",
        ],
    }
    if args.write:
        note = (
            "# Independent v2.4 A0 baseline identity review\n\n"
            "Both v2.3 and v2.4 contain 128 unique architecture cases across the same 30 populated "
            "deterministic shards. Shards 15 and 23 are empty in the frozen bank. All case IDs, "
            "metadata and ordering match the bank and manifests.\n\n"
            f"Exact raw identity: {comparison['identical_counts']['token_ids']}/128 token sequences, "
            f"{comparison['identical_counts']['gold_nll']}/128 reference NLL values, and "
            f"{comparison['identical_counts']['logit_file_bytes']}/128 saved logit archive SHA256 values. "
            f"Maximum absolute NLL difference: {nll_max_abs_difference}. "
            "No tolerances were applied. Any mismatching IDs are listed in receipt.json. "
            "No accuracy, efficacy or candidate selection was calculated.\n\n"
            "Both sets of manifests bind their respective frozen registration, configuration, sources, "
            "dataset and passed GPU qualification. Canonical checkpoint and state hashes, zero trainable "
            "backbone parameters/gradients, before/after preservation, and deterministic FP32 eager "
            "runtime receipts reconcile within and across versions. "
            f"All {receipt['current_source_files_rehashed']} current v2.4 source files were rehashed. "
            "The preserved v2.3 source map was reconciled against its own manifests.\n\n"
            f"Validation: {receipt['checks_passed']} checks; {len(audit.file_hashes)} distinct files hashed. "
            "The standard-library audit reuses the unchanged previous review helper and records both "
            "audit-script hashes. Run `python audit_baseline.py` for a read-only repeat; `--write` "
            "is accepted only when the two exclusive output files do not exist.\n\n"
            "Limits: this is a saved-evidence audit, not a new model execution or checkpoint reread. "
            "Saved logit tensors are not loaded. Qualification metrics are not recomputed. "
            "Scores and answer/response fields are discarded during JSON projection and only "
            "traversed as opaque bytes for integrity reconciliation. Confirmation and Track B "
            "data remain unopened. This review does not open the five-arm architecture or capability gate.\n"
        )
        with (OUTPUT / "receipt.json").open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(receipt, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        with (OUTPUT / "audit.md").open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(note)
    print(json.dumps({"status": receipt["status"], "checks_passed": receipt["checks_passed"],
                      "files_hashed": len(audit.file_hashes), "comparison": comparison,
                      "receipt_written": args.write, "wall_seconds": receipt["wall_seconds"]}))
    return 0 if receipt["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
