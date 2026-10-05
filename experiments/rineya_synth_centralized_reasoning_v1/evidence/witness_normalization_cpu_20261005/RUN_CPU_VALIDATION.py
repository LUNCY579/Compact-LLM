"""Bounded stdlib-only validation/preservation receipt for this staged patch."""
import difflib
import hashlib
import json
from pathlib import Path
import pickletools
import re
import subprocess
import sys
import time
import zipfile

STAGE = Path(__file__).resolve().parent
PACKAGE = STAGE.parents[1]


def file_record(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    source_names = ("backend_qual_native.py", "backend_qual_core.py",
                    "study_torch_backend.py", "native_mechanics.py")
    closure = [PACKAGE / name for name in source_names]
    for tag in ("A_attempt2", "B_attempt1"):
        raw = PACKAGE / "artifacts" / ("backendq003_worker_ram_" + tag)
        preserved = PACKAGE / "evidence" / ("backendq003_worker_ram_" + tag + "_20261005")
        names = ["worker_result.json", "SUPERVISOR_RECEIPT.json"]
        names += (["stage_detached_c9_witness.json", "stage_detached_witness10.json",
                   "stage_checkpoint_c9_descriptor.json"] if tag.startswith("A")
                  else ["stage_restored_c9.json"])
        closure += [directory / name for directory in (raw, preserved) for name in names]
        closure += [preserved / "source_snapshots" / "experiments" /
                    "rineya_synth_centralized_reasoning_v1" / name for name in source_names]
    before = {str(path): file_record(path) for path in closure}
    original = STAGE / "original_backend_qual_native.py"
    staged = STAGE / "backend_qual_native.py"
    assert original.read_bytes() == (PACKAGE / "backend_qual_native.py").read_bytes()
    before_lines = original.read_text(encoding="utf-8").splitlines(keepends=True)
    after_lines = staged.read_text(encoding="utf-8").splitlines(keepends=True)
    diff = "".join(difflib.unified_diff(before_lines, after_lines,
                                      fromfile="a/backend_qual_native.py",
                                      tofile="b/backend_qual_native.py"))
    (STAGE / "witness_normalization.diff").write_text(diff, encoding="utf-8", newline="\n")
    symbolic = {}
    a_leaf = PACKAGE / "artifacts/backendq003_worker_ram_A_attempt2"
    b_leaf = PACKAGE / "artifacts/backendq003_worker_ram_B_attempt1"
    a_w9 = (a_leaf / "stage_detached_c9_witness.json").read_bytes()
    b_c9 = (b_leaf / "stage_restored_c9.json").read_bytes()
    assert a_w9 == b_c9
    symbolic["A_W9_and_failed_B_restored_witness_byte_identical"] = True
    symbolic["witness_file"] = file_record(a_leaf / "stage_detached_c9_witness.json")
    symbolic["witness_internal_sha256"] = json.loads(a_w9)["sha256"]
    for name in ("stage_detached_c9_witness.json", "stage_detached_witness10.json"):
        raw = (a_leaf / name).read_bytes()
        parsed = json.loads(raw)
        live = json.loads(raw)
        live["optimizer_groups"][0]["betas"] = tuple(live["optimizer_groups"][0]["betas"])
        normalize = json.loads(raw)
        normalize["optimizer_groups"][0]["betas"] = list(live["optimizer_groups"][0]["betas"])
        assert live != parsed and normalize == parsed
        canonical = lambda value: json.dumps(value, ensure_ascii=True, sort_keys=True,
                                             separators=(",", ":"), allow_nan=False).encode("ascii")
        assert canonical(live) == canonical(normalize)
        symbolic[name] = {"only_reconstructed_betas_tuple_vs_list_inequality": True,
                          "old_vs_normalized_canonical_bytes_identical": True,
                          "raw_file": file_record(a_leaf / name)}
    with zipfile.ZipFile(a_leaf / "qualification_C9.pt") as archive:
        member = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        operations = list(pickletools.genops(archive.read(member)))
        beta_index = next(i for i, (_, argument, _) in enumerate(operations) if argument == "betas")
        sequence = operations[beta_index:beta_index + 6]
        assert sequence[2][1] == .9 and sequence[3][1] == .999 and sequence[4][0].name == "TUPLE2"
        symbolic["C9_betas_pickle_opcodes_no_deserialization"] = [
            {"opcode": opcode.name, "argument": argument, "offset": offset}
            for opcode, argument, offset in sequence]
    started = time.perf_counter()
    command = [sys.executable, "-B", str(STAGE / "tests/test_witness_normalization.py")]
    completed = subprocess.run(command, cwd=PACKAGE, text=True, capture_output=True, timeout=90)
    duration = time.perf_counter() - started
    (STAGE / "cpu_validation.stdout.log").write_text(completed.stdout, encoding="utf-8", newline="\n")
    (STAGE / "cpu_validation.stderr.log").write_text(completed.stderr, encoding="utf-8", newline="\n")
    after = {str(path): file_record(path) for path in closure}
    match = re.search(r"Ran (\d+) tests", completed.stderr)
    tests = int(match.group(1)) if match else 0
    unchanged = before == after
    receipt = {"schema_version": 1,
               "status": "PASS_STAGED_CPU_WITNESS_REPRESENTATION_CONTROLS" if
                    completed.returncode == 0 and tests == 5 and unchanged else "FAILED_STAGED_CPU_CONTROLS",
               "scope": "SYNTH-CRS-001_CPU_FAKE_AND_STDLIB_SERIALIZATION_ONLY",
               "command": command, "returncode": completed.returncode,
               "tests_run": tests, "negative_C9_subcontrols": ["beta", "tensor_hash", "rng", "cursor"],
               "elapsed_seconds": duration, "native_execution_performed": False,
               "framework_model_GPU_load_performed": False,
               "real_checkpoint_deserialized_or_restored": False,
               "study_warmstart_authorized": False, "new_native_attempt_authorized": False,
               "source_or_launch_gate_integration_performed": False,
               "original_native_receipts_and_source_closure_unchanged": unchanged,
               "preserved_closure_before": before, "preserved_closure_after": after,
               "staged_artifact_hashes": {str(path): file_record(path) for path in (
                    original, staged, STAGE / "tests/test_witness_normalization.py", Path(__file__),
                    STAGE / "witness_normalization.diff", STAGE / "cpu_validation.stdout.log",
                    STAGE / "cpu_validation.stderr.log")},
               "symbolic_metadata_diagnosis": symbolic,
               "source_only_successor_binding_review": {
                    "review_status": "DRAFT_REQUIRES_INDEPENDENT_REVIEW_NO_NATIVE_AUTHORITY",
                    "backend_qual_native_source_hash_changes": True,
                    "StudyTorchBackend_and_raw_checkpoint_mechanics_unchanged": True,
                    "serialized_W9_W10_witness_bytes_unchanged_by_normalization": True,
                    "normal_new_scope_rebuild_changes_recipe_sha256": True,
                    "same_recipe_predecessor_and_C9_metadata_checks_remain_closed": True,
                    "reuse_of_successful_A_C9_requires_explicit_reviewed_source_only_amendment": True,
                    "required_review_conditions": [
                        "Exact old and new adapter source hashes and four-line representation diff",
                        "Preserved successful A/C9/W9/W10 and failed B receipts remain immutable",
                        "Independent proof that raw checkpoint state, optimizer restore, native mechanics and serialized witnesses are unchanged",
                        "Old checkpoint semantic recipe remains separately bound from amended executing source provenance",
                        "Fresh actual source/runtime/host checks and original resource/numerical/full-witness checks",
                        "Owner-scoped approval before any successor B native attempt; no study warmstart authority"],
                    "warning": "No unchanged-config retry or silent same-recipe waiver is justified by these CPU controls"}}
    target = STAGE / "CPU_VALIDATION_001.json"
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": receipt["status"], "tests_run": tests,
                      "returncode": completed.returncode, "closure_unchanged": unchanged,
                      "receipt": str(target), "staged_adapter": file_record(staged)}, indent=2))
    return 0 if receipt["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    sys.exit(main())
