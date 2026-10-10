"""Reviewer metadata emitter only; imports no model runtime or scientific bank."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

V3 = Path(__file__).resolve().parents[2]
CHECK = V3 / "reviews/fresh_worker_reader_check_001"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

held = {
    "fresh_native.py": "52262b1ab7639ac0e564746af3bd0d4065b3c7cdcf57b7549569f308884428de",
    "opposed_gate_generation.py": "b0ac061de15808a551fccbda3a3b3b1c8ed539de33ca47dbc865598857e660f1",
    "tests/test_fresh_native.py": "12cb34f55e85d9df68df140afee8c432ac637afdc67814e023e14e0d43fb29e0",
    "tests/test_opposed_gate_generation.py": "4eafef960c30dbe8aba2882e3df7609a67186a9ff2df16e9f45ae092c73813ce",
    "fresh_analysis.py": "d07a99c3a8de65a09b5a405f6bc253f4859574c231f13a057a96790eaf9b45de",
    "tests/test_fresh_analysis.py": "18698dd0068e393b80960367474c5b49332b6d3188af07110482b939e4c26eaf",
}
if any(sha(V3 / n) != h for n, h in held.items()): raise RuntimeError("held affected source drift")
snapshot = CHECK / "source_snapshot_final_001"
snapshot.mkdir(exist_ok=True)
bindings = []
for name, value in held.items():
    target = snapshot / name; target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists(): shutil.copyfile(V3 / name, target)
    if sha(target) != value: raise RuntimeError("snapshot differs")
    bindings.append({"path": name, "sha256": value, "snapshot": target.relative_to(V3).as_posix()})
base = json.loads((CHECK / "worker_source_bindings_001.json").read_text(encoding="utf-8"))["bindings"]
root_changes = {"fresh_supervisor.py": "3e81f147c69ec5d4abe424c143be44e9891ba811622fc93eed4d185801615c4a",
                "prepare_fresh_admission.py": "4b758f8b6c01409279f278d5e7e45167023f05e09f1f642e57e5d6f0bb8519f4"}
if any(sha(V3 / n) != h for n, h in base.items() if n not in root_changes): raise RuntimeError("checked worker/helper/root/plan drift")
if any(sha(V3 / n) != h for n, h in root_changes.items()): raise RuntimeError("root change epoch differs")
current_base = {**base, **root_changes}
prep_path = V3 / "reviews/fresh_parent_preparation_review_002.json"
prep = json.loads(prep_path.read_text(encoding="utf-8"))
if sha(prep_path) != "0af0f19972c2d9541c01b7b89b694a4db42cfe1e033c024129be88e6d98e047b": raise RuntimeError("prep review drift")
if any(sha(V3 / b["path"]) != b["sha256"] for b in prep["bindings"] if b["path"] != "fresh_supervisor.py"): raise RuntimeError("prior prep source drift")
bank_path = V3 / "reviews/fresh_bank_source_review_001.json"
bank = json.loads(bank_path.read_text(encoding="utf-8"))
if sha(bank_path) != "55013d23632d121f2874cfe0ee63935742f8f27086d3d1b71f7b5129fe943868" or bank["status"] != "PASS_FRESH_OPEN_CPU_SOURCE" or bank["errors"] != []: raise RuntimeError("peer bank acceptance drift")
if any(sha(V3 / n) != h for n, h in bank["source_hashes"].items()): raise RuntimeError("peer bank source drift")
proof_files = ("worker_adapter_fixtures_001.xml", "reader_regressions_002.xml", "reader_witness_001.xml",
               "test_reader_gate_witness_001.py", "test_reader_gate_regression_002.py", "worker_source_bindings_001.json")
proofs = {"reviews/fresh_worker_reader_check_001/" + n: sha(CHECK / n) for n in proof_files}
old = "versions/fresh_analysis_pre_independent_review_001/"
if sha(V3 / (old + "fresh_analysis.py")) != "7d303e8ca53b8eaf3ecf42e1dc156801c1a6269cf15e7aa1fa6bbff26e33e69b" or sha(V3 / (old + "tests/test_fresh_analysis.py")) != "eacc23bee191eee39731e4481e9c54bcbf73b43d106eefb6772693ae961362dd": raise RuntimeError("old source preservation differs")
report = {
 "schema": "v3_independent_fresh_worker_reader_source_review_v1",
 "created_utc": datetime.now(timezone.utc).isoformat(),
 "status": "PASS_FRESH_WORKER_READER_SOURCE_ONLY_NOT_NATIVE_PERMISSION",
 "errors": [], "unresolved_source_findings": [],
 "role": "Independent review of other-author worker/greedy adapter/terminal reader and ROOT preparation interfaces. Reviewer authored the fresh bank/verifier/release gate; those sources and actual corpus validity are excluded from independent acceptance here.",
 "model_or_GPU_or_actual_tokenizer_calls": 0,
 "actual_scientific_cases_or_fresh_candidate_outcomes_read": False,
 "protected_data_or_native_TEST_or_old_VAL_access": False,
 "candidate_source_edits_by_reviewer": False,
 "source_bindings": bindings, "source_helper_protocol_bindings": current_base,
 "root_parent_limited_epoch": {"prior_independently_reviewed_supervisor_sha256": base["fresh_supervisor.py"],
    "current_supervisor_sha256": root_changes["fresh_supervisor.py"],
    "status": "CURRENT_PARENT_TEARDOWN_CHANGE_NOT_ACCEPTED_BY_THIS_WORKER_READER_RECEIPT",
    "scope": "ROOT added finally-charge/Job-close-error handling/verified termination/unknown-lock retention after prior prep002. Source delta inspected; independent new parent fixtures and peer full-packet acceptance remain required. Prior prep002 supervisor snapshot remains68680. Ten other prep002 bindings rechecked unchanged.",
    "current_builder_sha256": root_changes["prepare_fresh_admission.py"],
    "builder_interface": "New worker-reader prerequisite status/errors[] exact; no native admission created by reviewer."},
 "count_contract": {"scientific": 432, "preservation": 24, "items": 456, "paths_per_item": 9,
    "rows": 4104, "task_API_cap": 33744, "qualification_API_cap": 852, "model_API_cap": 34596,
    "qualification_formula": "67 + 8*(24+24+24+25) + 9", "task_formula": "456*74; conservative both-confidence-arm maxima",
    "new_prefix_qualification": "67API/65full attempts/64complete/1expectedpartial/2precompute caches/4921complete positions",
    "max_driver": "9attempts/8complete/1expectedpartial at357", "task_output_cap": 8, "qualification_output_cap": 24,
    "maximum_original_tokens": 350, "maximum_forward_prefix": 357, "state_adjoints": 0, "gold_calls": 0},
 "source_checks": [
    "Native loader requires exact fresh admission/review/current source, model/SMOKE binding, runtime/interpreter/Qwen, initial6GiB floors and prompt-only source/token equality before one model load. Runtime source cache is exclusive in result/hf_cache; canonical/sibling source or assets are not edited.",
    "Exact three raw and seven token fields; labels/gold/status/category/parent/family do not enter worker decision API. Input-only guard/header consume raw_prompt; confidence policies consume raw_prompt plus actual live original-input pilot p; opaque metadata is added after generation. Rich offline scientific files are not opened by the worker.",
    "Fresh first67 calls invoke the old source-bound helper under NEW parent clock and global ledger, without invoking consumed passive admission/run/export paths. Prefix helper451 internal cap is a stricter local ceiling for67 qualification calls, not reusable old launch authority.",
    "Greedy adapter uses original full prefixes, ones mask, use_cache=False and rejects returned cache through immutable gpu helper. It wraps actual L12/L13 unit recurrence with one-shot L0-L2 anchors, FP32/eval/frozen parameter/config/tokenizer signatures, restored hooks/owners and unchanged RMS/positions/head implementation. Analytic CPU bypass rejects the actual canonical model class.",
    "Each confidence arm constructs its own actual K0 pilot. Opaque once-only handle binds instance/source/state/tokenizer/prefix and cannot be forged/reused. K0 reuses that actual first vector; K1 pays pilot plus selected original-prefix forward and every own continuation. Discarded pilot tokens/EOS never dictate K1 generation.",
    "Global START/terminal IDs account for prefix/reference/session/cache/partial compute; row call IDs partition task calls. EOS is included in T, no unused cap continuation, every pilot/head/position remains charged. Expected qualification partial positions stay UNKNOWN; unexpected failure/cache/safety stops cannot produce complete task evidence.",
    "Qualified helper/source/stat/readout code UTF8 decoding is explicit. Worker pre-review bytes are preserved by owner; affected held worker and adapter bytes unchanged after R fixes.",
    "Source final receipt checks residentb23 before/after, ten canonical assets and customcode, no gradients/source drift, exact4104 rows/global positions, actual primary-repeat identity, independent complementary confidence pilots and both frozen random streams. These are future native assertions, not native proof generated in this review.",
    "Full forecast charges33744task calls plus every control/pilot/parser/nonforward overhead and qualification already elapsed;1.5 measured extrapolation and1800 internal reserve fit15600 parent/13800 body. Arbitrary host stalls are not mathematically bounded. Parent anti-retry/Job/lock/redirector resource behavior remains prior-prep source qualified, with actual execution pending.",
    "Terminal reader imports stdlib only before labels, checks terminal exit0/copied admission/current full dispatch map/independent dispatch review/required reader+plan+scorer/native membership, native source/runtime/assets/state/resources/ledger and all raw cells first. Only then imports immutable offline scorer/verifier and opens exact pinned label/proof paths.",
    "Revised reader independently rejoins all five actual admission-pinned prompt/token export files, exact schema/source/raw and rendered hashes/token IDs/counts/order/full456 runtime projection before labels; source source-map membership is mandatory.",
    "Revised reader requires finite nonnegative native/body/supervisor/ledger/parent/forward/cache/partial/case timings and resource values plus aware ordered event/terminal study timestamps; missing or detached proof fails before labels. Exceptions null primary without survivor-only scoring.",
    "Primary stays GUARD versus prior-frozen unit alpha1: equal two-variant parent/twelve mechanism/six-family weights,20k Random730217 draws and sorted499/19499 endpoints. Eligibility requires delta>=.05, lower>0, zero ALL observed K0/unit correct-case losses/new format/unsupported/falseUNKNOWN harms, no negative family/count endpoints, zero copy-panel baseline-correct loss. No secondary promotion, equivalence or general-source/speed claim.",
    "Own measured whole-response K0/unit retrospective oracle, random17/29 mean before aggregation, named separate recognized/unrecognized groups, all endpoints/denominators, solver/blind/copy floors and instrumented costs remain descriptive and source qualified. Conditional parent uncertainty is not independent human/source confirmation.",
    "Prospective builder4b758 only prepares exact new admission; actual producer prep002 status/unresolved[] and data-review/source-review status/errors[] interfaces match code, reader/test/plan existence and actual hash membership are checked. New supervisor3e81 teardown delta after prep002 is not independently accepted by this worker-reader receipt and needs peer full-packet/new mock checks."
 ],
 "independent_CPU_checks": {"total_current_PASS": 49, "worker_adapter": {"tests": 29, "seconds": 3.29},
    "reader_and_independent_regressions": {"tests": 20, "source_fixtures": 9, "reviewer_added": 11, "seconds": 7.41},
    "failures": 0, "errors": 0, "skips": 0,
    "command_contract": "borrowed exact Python -B -m pytest -q -p no:cacheprovider; new explicit V3-local reviews basetemp and JUnit outputs",
    "limits": "Torch CPU analytic Host fixtures for worker/adapter only; stdlib synthetic reader data and mock source/hash/runtime/asset plumbing. No real Job, CUDA/model/weights/tokenizer/corpus or efficacy replay.",
    "proof_hashes": proofs},
 "resolved_findings": [
    {"id": "WR01", "old_source": "7d303e8c", "finding": "Missing runtime-population to actual admitted prompt/token export join before labels", "witness": "Synthetic complete raw bundle accepted absent export paths without opening them", "resolution": "d07 exact five-path/hash/dispatch/schema/ordered456projection join; owner9fixtures and reviewer missing-export NULL-before-label regression pass"},
    {"id": "WR02", "old_source": "7d303e8c", "finding": "Missing finite/nonnegative parent resource and clock/cost predicates", "witness": "NaN parent RSS/host/workerclock and matched negative terminal/ledger/body values were accepted", "resolution": "d07 finite/nonnegative/aware ordered clocks/cost/resources; owner mutation fixture plus10 independent parent/body/terminal corruption regressions pass"}
 ],
 "historical_witnesses": {"tests": 2, "seconds": .80, "expected_defects_reproduced": True,
    "scope": "Held7d only; not current-PASS tests or source/cryptographic bypass. Old witness files remain unchanged.",
    "preserved_old_reader": old + "fresh_analysis.py", "old_sha256": "7d303e8ca53b8eaf3ecf42e1dc156801c1a6269cf15e7aa1fa6bbff26e33e69b",
    "preserved_old_tests": old + "tests/test_fresh_analysis.py", "old_tests_sha256": "eacc23bee191eee39731e4481e9c54bcbf73b43d106eefb6772693ae961362dd"},
 "relied_on_prior_reviews": [
    {"path": "reviews/fresh_parent_preparation_review_002.json", "sha256": sha(prep_path), "status": prep["status"],
     "scope": "Our earlier independent ROOT parent/export/panel/statplan R01-R05 source review and39CPU fixtures reused only for that epoch; ten current bound hashes unchanged. Supervisor68680 changed to3e81 after prep002 and is excluded from current independent acceptance here. No actual Job/dispatch proof."},
    {"path": "reviews/fresh_bank_source_review_001.json", "sha256": sha(bank_path), "status": bank["status"],
     "scope": "Other-reviewer550 bank/verifier/release source+fixture acceptance reused; current3hashes rechecked. Our authored bank/verifier is explicitly excluded from independent assessment here; actual scientificdata review is a separate required gate."}
 ],
 "remaining_required_gates": ["Independent new-supervisor teardown/accounting mock checks and peer exact full-packet source acceptance; prep00268680 cannot qualify current3e81 by itself.",
    "Actual fresh corpus/tokenizer/export/provenance execution acceptance by independent peer, without this reviewer claiming own-bank independence.",
    "Final exact source/reader/test/plan/data/protocol admission freeze and independent full packet PASS_FRESH_K01_FP32_DISPATCH bound to that admission.",
    "Coordinator actual contained first-time CUDA FP32 mechanics/maximum/full-parity/cache/exception/head/pilot/state/assets/source/resource qualification and complete forecast before task matrix.",
    "Successful terminal complete raw matrix, actual cost/resource/source/state joins and independent offline recomputation before any quality/preservation/source/speed conclusion."],
 "native_permission": False, "actual_admission_present_at_review": (V3 / "fresh_admission_v1.json").exists(),
 "source_or_efficacy_confirmation": False,
}
target = V3 / "reviews/fresh_worker_reader_source_review_001.json"
with target.open("x", encoding="utf-8") as f: json.dump(report, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")
md = """# Fresh worker and reader independent source review

PASS_FRESH_WORKER_READER_SOURCE_ONLY_NOT_NATIVE_PERMISSION. No unresolved source findings remain. The held worker52262b1a, adapterb0ac061d and revised terminal readerd07a99c3 satisfy the prospective scientific/source/count contracts; their exact source/test snapshots and hashes are in the JSON receipt.

49 CPU checks passed:29 worker/adapter tests on analytic Torch CPU hosts and20 synthetic stdlib reader checks, including11 added independent regressions. No actual model, CUDA, tokenizer, scientific fresh-case/outcome or protected-data access occurred. No candidate source was edited by this reviewer.

Two reader gaps were reproduced on old7d303e8c with synthetic full raw records: missing admitted prompt/token joins and invalid clock/resource values. The owner preserved old7d/eacc source/test bytes, then fixed both without changing the primary or analysis plan. The revised reader rejoins the exact five frozen prompt/token files and ordered456-item runtime projection before labels, rejects nonfinite/negative/disordered proof clocks and costs, and leaves primary NULL on failed proof. Original defect witnesses remain preserved, separate from current passing regressions.

The conservative physical budget is852 qualification plus33744 task calls=34596,456items*9paths=4104rows. Fresh first67 mechanics calls do not reuse a consumed passive grant. Every live confidence pilot, chosen original-prefix forward and own continuation is counted. Native residentstate/assets/head/anchors/cache/fullparity/resource/forecast assertions remain future execution gates.

The fixed guard needs at least5pp over frozenunit and a positive lower conditional95% endpoint from20k stratified parent draws. All observed A0/unit correct-case and new format/unsupported/falseUNKNOWN harms must be zero; family counts and copy-panel retention must also pass. Secondary policies cannot replace primary. Conditional authored-world uncertainty, visible-copy/full-solver floors and instrumented timing do not establish independent human/source transfer, neural reasoning or production speed.

Scope attribution: earlier ROOT preparation review002/0af0f199 and39CPU checks qualify their historical source epoch; ten other bindings remain unchanged. ROOT changed supervisor68680 to3e81 for teardown accounting after that review. Its source delta was inspected, but current supervisor independent mock and full-packet acceptance remain required and are excluded from this worker/reader PASS. Builder4b758 now consumes this receipt's exact producer enum/errors[]. Peer bank source review55013d23 is reused with its current3hashes rechecked. This reviewer authored the bank/verifier/release gate, so those sources and actual corpus validity are excluded from this receipt's independent claim.

This source review is not native permission. Actual corpus execution acceptance, immutable admission plus independent full-packet dispatch review, actual contained CUDA qualification/full forecast and complete terminal/raw independent recomputation remain required.
"""
mdpath = target.with_suffix(".md")
with mdpath.open("x", encoding="utf-8") as f: f.write(md)
print(json.dumps({"status": report["status"], "json_sha256": sha(target), "md_sha256": sha(mdpath), "current_tests": 49, "native_permission": False}))
