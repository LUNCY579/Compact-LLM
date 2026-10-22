"""Bounded reviewer metadata addendum; no runtime/data/scientific imports."""
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil

V3 = Path(__file__).resolve().parents[2]; CHECK = V3 / "reviews/fresh_worker_reader_check_001"
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
prior_path = V3 / "reviews/fresh_worker_reader_source_review_001.json"
if sha(prior_path) != "9b2a8c351e666ea0239d05bfbcecee32a4853bcd27bf9a05018ac736a90588fa": raise RuntimeError("prior review changed")
prior = json.loads(prior_path.read_text(encoding="utf-8"))
held = {"fresh_analysis.py": "a986b3affad1e9f1084b45c63fe2ce4c0927fc3f2fa52ab070fe07c2ae1988aa",
        "tests/test_fresh_analysis.py": "6b593e6a5cfd98537fc1179067e3bfcba7540a7555721d5409e8aee6000fa9e2",
        "fresh_scope.py": "22e08e1c342fcd068544d642952d7448c0b7d9de9fe659c8d3231e3eb47e634e",
        "prepare_fresh_admission.py": "8a9f6a16f34171f77e49ff7acd429c8f1b05908c524edc76d7e1975cb8ba58ff",
        "fresh_readiness.py": "b9966132fec1fd1ca3d3bcc1a835a52174c02a55d9d87aa0dddca338e12e9182"}
if any(sha(V3 / name) != value for name, value in held.items()): raise RuntimeError("new held source drift")
unchanged = {}
for b in prior["source_bindings"]:
    if b["path"] not in ("fresh_analysis.py", "tests/test_fresh_analysis.py"):
        if sha(V3 / b["path"]) != b["sha256"]: raise RuntimeError("worker/adapter drift")
        unchanged[b["path"]] = b["sha256"]
for name in ("FRESH_ANALYSIS_PLAN_DRAFT_001.md", "FRESH_NATIVE_DESIGN_DRAFT_002.md", "opposed_gate_policy.py", "fresh_entry.py", "probe_native.py", "observer.py"):
    value = prior["source_helper_protocol_bindings"][name]
    if sha(V3 / name) != value: raise RuntimeError("science/helper/plan drift")
    unchanged[name] = value
v1 = V3 / "fresh_admission_v1.json"
if sha(v1) != "cca385584023b45211a2a482885306e3505c59bd0b59b22b5f246282a6fe3b90": raise RuntimeError("V1 not preserved")
snap = CHECK / "source_snapshot_final_002"; snap.mkdir(exist_ok=False)
for name, value in held.items():
    target = snap / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(V3 / name, target)
    if sha(target) != value: raise RuntimeError("snapshot drift")
old_reader = CHECK / "source_snapshot_final_001/fresh_analysis.py"
old_test = CHECK / "source_snapshot_final_001/tests/test_fresh_analysis.py"
if sha(old_reader) != "d07a99c3a8de65a09b5a405f6bc253f4859574c231f13a057a96790eaf9b45de" or sha(old_test) != "18698dd0068e393b80960367474c5b49332b6d3188af07110482b939e4c26eaf": raise RuntimeError("old d07/1869 snapshots differ")
diff_path = CHECK / "reader_d07_to_a986_002.diff"
with diff_path.open("x", encoding="utf-8") as f:
    f.write("".join(difflib.unified_diff(old_reader.read_text(encoding="utf-8").splitlines(True), (V3 / "fresh_analysis.py").read_text(encoding="utf-8").splitlines(True), fromfile="held_d07", tofile="held_a986")))
parent_path = V3 / "audits/fresh_supervisor_teardown_checks_001.json"
if sha(parent_path) != "ff3a3c871983e400431bfb71520c4e75e03f3bad5f51376d8ffa8931d2059632": raise RuntimeError("parent check receipt changed")
parent = json.loads(parent_path.read_text(encoding="utf-8"))
if parent["status"] != "PASS_MOCKED_FAILURE_ACCOUNTING_AND_LOCK_TEARDOWN_WITH_LIMITS" or any(sha(V3 / name) != value for name, value in parent["bindings"].items()): raise RuntimeError("parent check source differs")
proof_names = ("forecast_v2_regressions_003.xml", "test_forecast_v2_regression_003.py", "reader_d07_to_a986_002.diff")
report = {
 "schema": "v3_independent_fresh_worker_reader_source_review_addendum_v2",
 "created_utc": datetime.now(timezone.utc).isoformat(),
 "status": "PASS_FRESH_WORKER_READER_SOURCE_ONLY_NOT_NATIVE_PERMISSION", "errors": [], "unresolved_source_findings": [],
 "scope": "Bounded other-author reader forecast-duration/formula fix and V2 producer-consumer paths only. Reuses prior source001 review for unchanged worker/adapter/scientific/statistical behavior; no broader raw-history/corpus/model analysis.",
 "prior_source_review": {"path": "reviews/fresh_worker_reader_source_review_001.json", "sha256": sha(prior_path),
     "source_epoch": "d07/1869 +V1; historical, not current reader qualification"},
 "new_source_hashes": held, "unchanged_source_hashes_rechecked": unchanged,
 "resolved_findings": [{"id": "WR03", "identified_by": "Independent full-packet peer before any model attempt",
    "prior_bug": "d07 raw_gate required forecast durations finite but allowed negative remaining_forward/nonforward, which could discount a declared full-run forecast.",
    "fix": "All six forecast quantities finite/nonnegative; observed warm forward strictlypositive;33744*1.5*M forward,4104*per-row nonforward and total sums reconciled; fixedcaps/factor/reserve/deadline remain exact.",
    "numeric_scope": "math.isclose rel1e-12/abs1e-9 only reconciles ordinary FP64 arithmetic/JSON formula ordering; no FP32 kernel identity tolerance or threshold/scientific adjustment.",
    "validation": "17 targeted CPU checks including combined negative-forward/nonforward witness, individual negative durations, formula/sum/factor/reserve failures, valid complete synthetic456x9raw proof and exactV2 consumer paths."}],
 "V2_interfaces": {"reader_admission": "fresh_admission_v2.json", "reader_exact_review": "reviews/fresh_dispatch_review_002.json",
    "builder_output": "fresh_admission_v2.json", "builder_review_receipt": "reviews/fresh_dispatch_review_002.json",
    "builder_source_review": "reviews/fresh_worker_reader_source_review_002.json",
    "source_review_producer_enum": "PASS_FRESH_WORKER_READER_SOURCE_ONLY_NOT_NATIVE_PERMISSION", "producer_errors": [],
    "scope_and_readiness": "exactV2 admission only; identical syntheticV1 packet rejected by exact path guard",
    "copied_packet": "Exact V2 admission SHA and copied review SHA/PASS/admissionSHA checks remain; prior001/generic/outside review names are refused by reader helper.",
    "new_admission_created_by_reviewer": False},
 "preserved_inputs": {"V1_admission": "fresh_admission_v1.json", "V1_sha256": sha(v1),
    "prior_reader_snapshot": str(old_reader.relative_to(V3)).replace("\\", "/"), "prior_reader_sha256": sha(old_reader),
    "prior_test_snapshot": str(old_test.relative_to(V3)).replace("\\", "/"), "prior_test_sha256": sha(old_test)},
 "scientific_contract": {"primary": "EVIDENCE_CONSISTENCY_GUARD", "comparator": "UNIT_PRIOR_FROZEN_ALPHA1", "practical_margin": .05,
    "bootstrap": "20k Random730217/indices499,19499 conditional authored-parent resampling; unchanged",
    "preservation": "all observed K0/unit correct-case and new endpoint harms zero; family/copy-panel restrictions unchanged",
    "items": 456, "task_rows": 4104, "qualification_API_cap": 852, "task_API_cap": 33744, "model_API_cap": 34596,
    "no_policy_count_margin_threshold_or_fitted_candidate_changes": True},
 "independent_targeted_checks": {"tests": 17, "author_affected_fixtures": 9, "reviewer_added": 8, "deselected_unaffected": 8,
    "failures": 0, "errors": 0, "skips": 0, "seconds": 6.60,
    "command": "borrowed exact Python -B -m pytest -q -p no:cacheprovider; explicitnewV3-local temp_forecast003/JUnit; -k forecast/review_path/full_raw_matrix/consumer/builder/scope_exact",
    "proof_hashes": {"reviews/fresh_worker_reader_check_001/" + name: sha(CHECK / name) for name in proof_names},
    "scope": "Synthetic rawfixture, mocked source/review/runtime and path guards only; no real readiness/GPU/Job/model/tokenizer/admission/corpus access."},
 "relied_on_checks": [{"path": "audits/fresh_supervisor_teardown_checks_001.json", "sha256": sha(parent_path), "status": parent["status"],
    "source_sha256": parent["bindings"]["fresh_supervisor.py"], "tests": 7,
    "attribution": "Native engineer independently tested ROOT parent source with CPU mocks; author of reader is not independent for reader, but is independent of ROOT parent. Reused scoped7checks, not rerun here; current source/test hashes verified. Actual Job/CUDA containment still pending."},
    {"path": "reviews/fresh_worker_reader_source_review_001.json", "sha256": sha(prior_path),
     "attribution": "Earlier49CPU checks and source inspection reused only for unchanged reader scientific/statistical branches and worker/adapter/helper/plan; superseded forecast/V1 parts explicitly replaced."}],
 "author_role_exclusion": "Reviewer authored bank/verifier/release gate; their independence remains excluded and peer550 source acceptance is inherited only through unchanged prior review. No actual fresh labels/outcomes inspected or scored.",
 "models_GPU_actual_tokenizers_candidate_scores_protected_data_calls": 0,
 "candidate_source_edits_by_reviewer": False, "native_permission": False,
 "remaining_required_gates": ["Freeze exact currentV2 admission and independent full-packet PASS_FRESH_K01_FP32_DISPATCH bound to it; oldV1 freeze/review cannot authorize current reader.",
    "Coordinator actual contained CUDAFP32 qualification/full current-source/data/runtime/assets/state/resource/complete forecast, then sole complete raw matrix.",
    "Independent terminal/raw/cost/quality recomputation before any efficacy/preservation/source/speed conclusion."],
}
target = V3 / "reviews/fresh_worker_reader_source_review_002.json"
with target.open("x", encoding="utf-8") as f: json.dump(report, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")
md = """# Fresh worker and reader source review addendum002

PASS_FRESH_WORKER_READER_SOURCE_ONLY_NOT_NATIVE_PERMISSION, errors[] and no unresolved source findings within this bounded review. The current held reader is a986b3af, tests6b593e6a. The prior001/9b2a8c35 receipt and its d07/1869 source snapshots remain historical and immutable.

The full-packet peer found that d07 accepted negative forecast durations before any model attempt. The owner now requires every forecast component finite and nonnegative, strictlypositive observed warm time, the registered33744*1.5*M forward formula,4104-row overhead formula and reconciled total. Counts, factor, reserves, primary/statistical margin and candidate behavior are unchanged. This fixes forecast validation; it does not create a measured quality result.

17 targeted CPU checks passed in6.60s:9 affected source fixtures plus8 added reviewer forecast/path regressions. The combined negative-duration witness now refuses. Exact V2 scope/readiness/reader paths and builder's new source002 producer interface pass synthetic metadata checks; reader rejects prior001/generic/outside review names and checks copied review/admission hashes. No actual readiness, GPU, Job, model, tokenizer, scientific cases or fresh candidate scores were accessed.

V1 admissioncca38558 remains preserved. Builder/scope/readiness/reader use fresh_admission_v2.json and reader/builder use reviews/fresh_dispatch_review_002.json; no old grant fallback exists. Worker52262/adapterb0ac/scientific/statistical source is unchanged and relies on earlier qualified review001 checks, with current hashes rechecked.

ROOT's current supervisor3e81 relies additionally on the native engineer's independent scoped7 mocked teardown checks ff3a3c87; source/test hashes were verified, not rerun here. Actual WindowsJob/CUDA containment remains a future gate. This reviewer authored the bank/verifier/release gate and claims no independence for that corpus source.

This PASS is source preparation only. Exact immutable V2 admission plus independent full-packet dispatch review, actual native qualification/full forecast, complete terminal raw matrix and independent offline recomputation remain required before a quality or preservation conclusion.
"""
with target.with_suffix(".md").open("x", encoding="utf-8") as f: f.write(md)
print(json.dumps({"status": report["status"], "json_sha256": sha(target), "md_sha256": sha(target.with_suffix(".md")), "tests": 17, "native_permission": False}))
