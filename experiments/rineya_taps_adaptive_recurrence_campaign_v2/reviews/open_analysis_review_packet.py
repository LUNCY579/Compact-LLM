"""Freeze independent OPEN analyzer source review before native outcome use."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

C = Path(__file__).resolve().parent.parent
EXPECTED_SOURCE = "c072790ea7d5ee4864fbeef762509fb24c0b9f9c99f7481b47eee37c28695576"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    target = C / "reviews/open_analysis_review_receipt.json"
    if target.exists() or sha(C / "open_analysis.py") != EXPECTED_SOURCE:
        raise RuntimeError("exclusive receipt/final reviewed source differs")
    evidence = {}
    for name, count in (("open_analysis_final_cpu_junit.xml", 8), ("open_analysis_endpoint_junit.xml", 2)):
        path = C / "reviews" / name
        suites = list(ET.parse(path).getroot().iter("testsuite"))
        totals = {key: sum(int(s.get(key, "0")) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
        if totals != {"tests": count, "failures": 0, "errors": 0, "skipped": 0}:
            raise RuntimeError("independent analyzer CPU tests differ")
        evidence[name] = {"counts": totals, "sha256": sha(path)}
    path = C / "reviews/open_analysis_review_probes_final.json"
    probes = json.loads(path.read_text(encoding="utf-8"))
    if probes["analyzer_sha256"] != EXPECTED_SOURCE or len(probes["checks"]) != 12 or not all(row["passed"] for row in probes["checks"]):
        raise RuntimeError("independent adversarial probes differ")
    source_paths = ["open_analysis.py", "tests/test_open_analysis.py", "analysis.py", "gpu_analysis.py", "open_bank.py",
                    "reviews/test_open_analysis_independent.py", "reviews/open_analysis_review_probes.py", "reviews/open_analysis_review_packet.py"]
    result = {"schema_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(), "reviewer": "/root/adversarial_review",
              "status": "PASS_OPEN_ANALYZER_SOURCE_ONLY", "native_model_loads": 0, "native_outcome_read": False,
              "source_sha256": {name: sha(C / name) for name in source_paths}, "CPU_tests": evidence,
              "independent_adversarial_probes": {"count": 12, "passed": 12, "receipt_sha256": sha(path)},
              "supersedes_for_analyzer_use": "reviews/open_analysis_initial_review_receipt.json",
              "resolved_required_issues": ["OA01", "OA02", "OA03", "OA04", "OA05", "OA06"],
              "additional_lineage_corrections": ["Original model/runtime and SMOKE metadata checked against admitted hashes before use",
                                                 "Imported analyzer/helper/scorer sources hashed in emitted manifest"],
              "endpoint_and_cost_contract": {"physical_rows": 608, "constant_grid_points": 6,
                  "DEVELOPMENT_parents": 16, "OPEN_VALIDATION_parents": 8, "generator_clusters_per_split": 4,
                  "oracle_reused_rows": 48, "frozen_comparator_aliases": 16, "aliases_oracle_extra_generation": 0,
                  "validation_primary_fixed_comparator": "frozen DEVELOPMENT alpha",
                  "validation_fixed_envelope_and_oracle": "retrospective diagnostics only",
                  "A5_repeats": "within-parent averages, not independent units",
                  "non_A5_repeat_divergence": "suppresses complete/primary comparisons",
                  "partial_matrix_primary_values": None, "empty_measurement_subtotals": None,
                  "headroom_capture": "unclipped; null with nonpositive/incomplete finite-grid headroom"},
              "source_review_limitations": ["Approval applies only to these analyzer/helper/scorer source hashes; final raw replay remains independent work",
                  "Synthetic qualification statuses are fixtures; no native OPEN qualification or efficacy is established here",
                  "Canonical strict scalar/UNKNOWN endpoints; semantic natural-language abstention and reasoning beyond answer contract remain NOT_ASSESSED",
                  "SMOKE score schema is checked; primary new-bank scorer is independently regraded, while inherited SMOKE scoring retains source-bound worker provenance",
                  "Source-visible eight validation parents/four variants share four code categories with development; no independent confirmation or broad intelligence claim",
                  "Timing is synchronized and instrumented; no production speed claim"],
              "unresolved_analyzer_blockers": []}
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": result["status"], "analyzer_sha256": EXPECTED_SOURCE, "CPU_tests": 10, "adversarial_probes": 12}))


if __name__ == "__main__":
    main()
