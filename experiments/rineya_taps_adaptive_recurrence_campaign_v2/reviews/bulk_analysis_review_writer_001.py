"""Freeze independent bulk analyzer review after bounded source/synthetic tests."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

C = Path(__file__).resolve().parent.parent
EXPECTED = "3defe511cd8a189f75e0e0529a56b782b965f40095c139d3a0c603b2bbf0d3db"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    target = C / "reviews/bulk_analysis_review_receipt.json"
    if target.exists() or sha(C / "bulk_analysis.py") != EXPECTED: raise RuntimeError("exclusive final analyzer/source differs")
    tests_path = C / "reviews/bulk_analysis_cpu_junit_002.xml"
    suites = list(ET.parse(tests_path).getroot().iter("testsuite"))
    counts = {key: sum(int(s.get(key, "0")) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    if counts != {"tests": 23, "failures": 0, "errors": 0, "skipped": 0}: raise RuntimeError("independent full CPU suite differs")
    probe_path = C / "reviews/bulk_analysis_probes_final_001.json"
    probes = read(probe_path)
    if probes["analyzer_sha256"] != EXPECTED or len(probes["tests"]) != 4 or not all(row["passed"] for row in probes["tests"]): raise RuntimeError("independent causal/endpoint probes differ")
    if sha(C / "bulk_native.py") != "1ed52ccbb546d826caec2bb48b3c7fa76dd5a136f1a1d7e3fae421b4871c46aa": raise RuntimeError("frozen worker changed")
    if sha(C / "bulk_admission_v1.json") != "a303cd65c353a6d12e6544503bbf210c054ae919f795577d8b01e96224464aca": raise RuntimeError("frozen admission changed")
    source_names = ("bulk_analysis.py", "tests/test_bulk_analysis.py", "analysis.py", "gpu_analysis.py", "open_analysis.py", "bulk_bank.py", "open_bank.py",
                    "reviews/bulk_analysis_probes_001.py", "reviews/bulk_analysis_review_writer_001.py")
    result = {"schema_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(), "reviewer": "/root/adversarial_review",
        "status": "PASS_BULK_ANALYZER_SOURCE_ONLY", "source_sha256": {name: sha(C / name) for name in source_names},
        "independent_CPU_tests": {"counts": counts, "pytest_seconds": 21.82, "junit_sha256": sha(tests_path)},
        "independent_bounded_probes": {"passed": 4, "receipt_sha256": sha(probe_path), "checks": [row["probe"] for row in probes["tests"]]},
        "resolved_source_review_issues": [
            {"id": "BA01", "required_gate": "actual parent source-manifest schema", "disposition": "Real sources() coverage checked; all84 admitted file hashes still verified locally. Real-schema empty CLI regression passes."},
            {"id": "BA02", "required_gate": "two distinct cache-rejection contracts", "disposition": "Exact True/no-object and False/object start/terminal coverage now required."},
            {"id": "BA03", "required_gate": "fixed scientific decision threshold", "disposition": "Registered practical_margin.05 and per-contrast meets_margin explicit; preservation/uncertainty separate."},
            {"id": "BA04", "required_gate": "deferred paired readout accounting", "disposition": "Separate3168-record finite nongold/zero-extra-forward journal read/hash/audit; otherwise explicitly NOT_ANALYZED; standalone latencyNULL."},
            {"id": "BA05", "required_gate": "causal stochastic-control consistency", "disposition": "Same case/shared prior token prefix/effective FP32 scale cannot produce differing next token; identical tokenIDs require equal decoding/stopping. Distinct draws remain allowed."}],
        "FP32_unit_branch_evidence": {"scheduler": "experiments/rineya_taps_adaptive_recurrence_v1/scheduler.py:152", "random_control": "control.py:74",
            "apply_update": "experiments/rineya_taps_adaptive_recurrence_v1/scheduler.py:192",
            "finding": "Unit/fixed scalars are ones_like/full_like FP32 progress; random draws use existing alpha dtype. Exact-unit branch is tested on alpha Tensor after casting, so effective FP32 equality preserves branch identity."},
        "reviewed_contract": {"physical_rows": 3296, "scientific_rows": 3168, "deterministic_repetitions": 1, "A5_repetitions": 2,
            "constant_grid": [.8, .9, 1., 1.1, 1.1473425965417514, 1.2], "oracle_nonphysical_rows": 288, "frozen_validation_aliases": 96,
            "primary_validation_parent_units": 48, "parent_stage_lane": "BULK_GPU_FP32_SCREEN", "worker_stage_lane": "BULK_GPU_FP32_INTERVENTION",
            "primary_validation_comparator": "DEV-frozen global six-constant winner; validation envelope/oracle diagnostic",
            "aggregation": "A5 seeds within cases, both variants within parent, equal six-category weights",
            "preservation": "Each case/variant loss remains visible even when parent means cancel; canonical format/UNKNOWN source-status endpoints independently regraded",
            "costs": "Physical rows only; oracle/alias copies excluded; absent totalsNULL, observed worker cost retained; partial computeNULL; deferred readout adds0model calls, standalone latencyNULL",
            "primary_incomplete_values": None, "headroom_capture": "unclipped;NULL for nonpositive/incomplete finite-grid headroom"},
        "native_model_loads": 0, "native_calls": 0, "new_native_outcomes_read": False,
        "limitations": ["This freezes analysis source before scientific measurement; real terminal/raw replay remains future independent work",
            "Synthetic qualification/review/worker statuses are CPU fixtures, not native qualification",
            "Shared authored generator families and48 held-out world parents permit conditional exploratory interpretation only; no broad confirmation/equivalence",
            "Metadata-known evidence variants never become model inputs; semantic natural-language abstention beyond canonicalUNKNOWN remains unassessed",
            "Controller revision/model training and empirical allocation-efficiency improvement remain unclaimed",
            "Pinned deterministic no-state-carry contract underlies same-prefix/alpha consistency; any breach suppresses primary claims rather than relabeling it stochastic",
            "Late A5 phase/instrumented shared-device timing do not estimate production speed or deterministic timing-repeat variance"],
        "unresolved_analyzer_blockers": []}
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": result["status"], "analyzer_sha256": EXPECTED, "independent_tests": 23, "independent_probes": 4}))


if __name__ == "__main__": main()
