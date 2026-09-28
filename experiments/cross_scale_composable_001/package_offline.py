"""Create first-pass receipts once, after all offline source edits are complete."""
import datetime
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np

from . import mechanics as m
from . import receipts as r

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
E = STUDY / "evidence"
PARENT = "07e53b0e7d8ed1dd3243493b3a25b97b3988f4fd"
BRANCH = "research/cross-scale-orthogonal-composable-skills-v1"


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def main():
    r.require(git("rev-parse", "HEAD") == PARENT, "packaging must precede first study commit")
    r.require(git("branch", "--show-current") == BRANCH, "wrong study branch")
    r.require(not (E / "WORKTREE_RECEIPT.json").exists(), "receipt bundle already exists")
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    r.write_once(E / "WORKTREE_RECEIPT.json", {
        "schema": 1, "recorded_utc": timestamp, "repository": "C:/Users/clclc/BusinessProject/Chat-Bubble",
        "worktree": ROOT.as_posix(), "branch": BRANCH, "parent_ref": "codex/dataset-retention-audit-v1",
        "parent_commit": PARENT, "fetch_origin_prune_succeeded_before_preflight": True,
        "observed_precreation": {"local_branch_exists": False, "remote_branch_exists": False,
                                 "worktree_path_exists": False},
        "preflight_evidence": "Successful commands and output recorded in task conversation, before git worktree add.",
        "creation_command": "git -C C:/Users/clclc/BusinessProject/Chat-Bubble worktree add C:/Users/clclc/BusinessProject/ChatBubble-cross-scale-composable-v1 -b " + BRANCH + " codex/dataset-retention-audit-v1",
        "existing_research_branches_modified": False,
        "final_commit_receipt": "External records namespace FINAL_SOURCE_RECEIPT.json, written after clean commit; avoids self-reference."})
    command = [sys.executable, "-m", "unittest", "discover", "-s",
               "experiments/cross_scale_composable_001", "-t", ".", "-v"]
    start = time.perf_counter()
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    log = (done.stdout + done.stderr).encode("utf-8")
    with (E / "CPU_TEST_OUTPUT.txt").open("xb") as stream:
        stream.write(log)
    count = re.search(r"Ran (\d+) tests?", log.decode("utf-8"))
    tests = {"status": "COMPLETE" if done.returncode == 0 and count else "FAILED",
             "completed": done.returncode == 0 and count is not None, "returncode": done.returncode,
             "tests_run": int(count.group(1)) if count else None, "elapsed_seconds": elapsed,
             "command": command, "output_sha256": r.file_digest(E / "CPU_TEST_OUTPUT.txt"),
             "native_model_calls": 0, "optimizer_steps": 0}
    r.write_once(E / "CPU_TESTS.json", tests)
    r.complete_child(done.returncode, tests)
    fixture_command = [sys.executable, "-m", "experiments.cross_scale_composable_001.fixture_run",
                       "--out", str(E / "FIXTURE_RUN.json")]
    child = subprocess.run(fixture_command, cwd=ROOT, capture_output=True, text=True)
    r.require(child.returncode == 0, "fixture child failed: " + child.stderr)
    fixture = json.loads((E / "FIXTURE_RUN.json").read_text(encoding="utf-8"))
    r.complete_child(child.returncode, fixture)
    calibration = json.loads((E / "CALIBRATION_BINDING.v2.json").read_text(encoding="utf-8"))
    r.validate_calibration(calibration)
    architecture = {
        "schema": 1, "student": {"name": "Supa 1.7 devfreeze", "parameters": 596049920,
            "layers": 28, "residual_width": 1024, "ffn_width": 3072, "q_heads": 16, "kv_heads": 8, "head_dim": 128,
            "model_sha256": "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b",
            "size_eligibility": "User explicitly accepted rounded 596M; compact roughly sub-billion preference."},
        "donor": {"name": "Qwen3-4B", "parameters": 4022468096, "layers": 36, "residual_width": 2560,
            "ffn_width": 9728, "q_heads": 32, "kv_heads": 8, "head_dim": 128,
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c", "deliverable": False,
            "skilled_adapter_sha256": "e7ab0919192dd10845c5f0769ce942028737750367c736e8bff4664fa020164a"},
        "asset_inventory_sha256": r.file_digest(E / "BINDING_ASSETS.json"),
        "layer_map": {"rule": "nearest normalized depth; exact ties to lower index; zero-based post-block",
            "donor_to_student": list(m.normalized_depth_map(36, 28)),
            "student_to_donor": list(m.normalized_depth_map(28, 36)),
            "capture_pairs_student_donor": [[0, 0], [13, 17], [27, 35]],
            "functional_equivalence_claim": False},
        "tokenizer_assets_identical": True, "raw_weight_copy_compatible": False,
        "donor_runtime_status": "NF4 double-quantized base/BF16 compute binding pending native qualification"}
    r.write_once(E / "ARCHITECTURE_MAP.json", architecture)
    r.write_once(E / "REGISTRATION.json", {
        "schema": 1, "study": "cross-scale-composable-001", "version": 1, "registered_utc": timestamp,
        "stage": "A_B_OFFLINE_ONLY", "native_authority": False,
        "protocol": {"path": "experiments/cross_scale_composable_001/PREREGISTRATION.md",
                     "sha256": r.file_digest(STUDY / "PREREGISTRATION.md")},
        "hypotheses": ["H1 cross-scale skill direction", "H2 orthogonal projection", "H3 composition",
                       "H4 norm/rank/budget specificity", "H5 functional vs parameter alignment"],
        "arms": ["S0", "S1", "X0", "X1", "X2", "O0", "R0"],
        "active_calibration": {"file": "CALIBRATION_BINDING.v2.json", "sha256": r.file_digest(E / "CALIBRATION_BINDING.v2.json")},
        "architecture_sha256": r.file_digest(E / "ARCHITECTURE_MAP.json"),
        "primary_stage_D_projection_variant": "S1-norm-restored; raw projection diagnostic only",
        "cross_draw_net_floor": 20, "required_report": ["net", "nd", "exact_p"],
        "pending_execution_amendments": ["native runtime/token binding and source-clean grant", "protected independent basis",
            "open-dev S1 selection rule binding", "complete evaluation banks/populations and forward-token budgets",
            "independent A/B donors and rows", "replication seeds and multiplicity families"],
        "native_model_calls_before_registration": 0, "scientific_improvement_claim": False})
    source_paths = [p.relative_to(ROOT).as_posix() for p in STUDY.iterdir() if p.is_file()]
    source_paths += [p.relative_to(ROOT).as_posix() for p in ROOT.glob("CROSS_SCALE_*.md")]
    source_paths += [".gitattributes", "experiments/rab001_exploration/torchfree.py"]
    source = r.implementation_digest(ROOT, source_paths)
    source.update(schema=1, parent_commit=PARENT, source_branch=BRANCH,
                  source_commit_resolution="External FINAL_SOURCE_RECEIPT.json binds this inventory to the clean study commit.",
                  generated_evidence_excluded_to_avoid_circular_hash=True,
                  assets_inventory_sha256=r.file_digest(E / "BINDING_ASSETS.json"))
    r.write_once(E / "SOURCE_BINDING.json", source)
    r.write_once(E / "ENVIRONMENT.json", {"python": sys.version, "executable": sys.executable,
        "executable_sha256": r.file_digest(sys.executable), "numpy": np.__version__, "platform": platform.platform(),
        "environment": str(ROOT / ".venv-cross-scale"), "experiment_specific": True,
        "packages_installed": ["numpy==2.4.6"], "pinned_native_environments_mutated": False})
    r.write_once(E / "TRANSPORT_RECEIPT.json", {"evidence_kind": "SYNTHETIC_CPU_FIXTURE_ONLY", **fixture["transport"]})
    for name, receipt in fixture["deltas"].items():
        r.write_once(E / "DELTA_RECEIPTS" / (name + ".json"), receipt)
    for name in ("composition", "orthogonal_composition", "reverse_composition"):
        r.write_once(E / "COMPOSITION_RECEIPTS" / (name + ".json"),
                     {"evidence_kind": "SYNTHETIC_CPU_FIXTURE_ONLY", **fixture[name]})
    r.write_once(E / "COMPARABILITY.json", {"fixture_bindings": "PASSED_WITH_REQUIRED_REFUSALS",
        "native_comparisons": "BLOCKED_ON_COMPARABILITY", "native_behavioral_endpoints_opened": False,
        "rank_control_demonstration": fixture["norm_controls"],
        "reason": "No native captures/student deltas/probe output bindings; protected independent basis not yet qualified."})
    r.write_once(E / "RESULTS.json", {"schema": 1, "mechanical_tests": tests,
        "fixture_sha256": r.file_digest(E / "FIXTURE_RUN.json"), "native_model_calls": 0,
        "native_tokenizer_calls": 0, "optimizer_steps": 0, "behavioral_results": None,
        "scientific_improvement_claim": False})
    r.write_once(E / "STATUS.json", {"first_pass": "OFFLINE_IMPLEMENTATION_AND_FIXTURES_COMPLETE",
        "native": "NOT_AUTHORIZED_NOT_RUN", "scientific_claim": "UNOPENED",
        "smallest_next_proposal": "32-prompt sequential student/donor-base/donor-skilled capture:96 forwards,0 optimizer steps",
        "actual_transfer_test": "After C qualification: S0 vs X0 paired student training with reciprocal norm controls",
        "authorized_next_action": "Commit this offline study; no automatic native execution",
        "dose_lane": "UNMODIFIED; expired envelope not reused; no new wait"})
    interpretation = ("MECHANICALLY ESTABLISHED\nSource/data audit, registered residual transport, projection, norm controls, "
        "composition, and CPU fixtures. See CPU_TESTS.json.\n\nSCIENTIFICALLY UNTESTED\nAll student skill-transfer, "
        "retention, and composability hypotheses. No improvement claim opened.\n\nBLOCKED\nNative authority, runtime "
        "and token qualification, independent protected basis, full evaluation budget, A/B donors, and resource gates.\n\n"
        "AUTHORIZED NEXT ACTION\nOffline commit and review. Proposed native96-forward representation session requires "
        "fresh explicit authority; actual transfer requires matched S0/X0 training and probes.\n")
    with (E / "INTERPRETATION.md").open("xb") as stream:
        stream.write(interpretation.encode("utf-8"))
    inventory = {p.relative_to(E).as_posix(): r.file_digest(p) for p in E.rglob("*") if p.is_file()}
    r.write_once(E / "BUNDLE_MANIFEST.json", {"files": inventory, "sha256": r.digest(inventory)})
    print(json.dumps({"status": "COMPLETE", "tests": tests["tests_run"], "source_digest": source["sha256"],
                      "evidence_files": len(inventory) + 1, "native_model_calls": 0}))


if __name__ == "__main__":
    main()
