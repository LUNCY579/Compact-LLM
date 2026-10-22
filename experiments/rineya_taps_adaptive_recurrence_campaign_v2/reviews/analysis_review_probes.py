"""Independent synthetic analyzer and prompt-only solver probes; no native input."""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import analysis, solver_audit


def main():
    test_path = ROOT / "experiments/rineya_taps_adaptive_recurrence_campaign_v2/tests/test_analysis.py"
    spec = importlib.util.spec_from_file_location("review_synthetic_analysis_fixtures", test_path)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    checks = {}
    base = fixtures.report()
    checks["complete_synthetic_reference_recomputed"] = base["complete"]
    rows, metadata, native, worker = fixtures.synthetic_matrix()
    row = next(r for r in rows if r["arm"] == "A0" and r["split"] == "DEV" and r["seed"] == 29)
    row["token_ids"] = [5, 99]
    changed = fixtures.report(rows, metadata, native, worker)
    checks["independent_intact_repeat_divergence_blocks_primary"] = not changed["complete"] and not changed["pairs"] and changed["constant_oracle"]["headroom"] is None
    rows, metadata, native, worker = fixtures.synthetic_matrix()
    physical = [r for r in rows if r["physical_run"]]
    traces = fixtures.synthetic_traces(rows)
    traces[0]["alpha"] = .9
    checks["scaled_ordinary_first_pass_refused"] = "ordinary first-pass trajectory was scaled" in analysis.validate_traces(physical, traces)
    traces = fixtures.synthetic_traces(rows)
    traces[-1]["case_id"] = "wrong-provenance"
    checks["trajectory_provenance_refused"] = "trajectory/outcome provenance differs" in analysis.validate_traces(physical, traces)
    checks["empty_matrix_primary_is_null"] = analysis.summarize([], [], metadata, {}, {}, {})["constant_oracle"]["headroom"] is None
    checks["capture_above_one_and_negative_kept"] = analysis.grid_headroom(1., .25, .5, True)["capture"] == 3 and analysis.grid_headroom(0, .5, 1, True)["capture"] == -1
    checks["solver_signature_only_prompt"] = list(inspect.signature(solver_audit.solve_prompt).parameters) == ["prompt"]
    probes = [
        ("Compute (-11 × 7) + (13). Return only the integer.", "-64"),
        ("What is 16 minus 23? Give just the integer answer.", "-7"),
        ("In a counterfactual world, every object that is blue is also green, and every green object is red. Object Z is blue. Under those rules, is Z red? Answer YES or NO only.", "YES"),
        ("Iris is older than Juno. Juno is older than Kira. Who is youngest? Return only the name.", "Kira"),
        ("Iris gave the key to Juno. Then Juno gave it to Kira. Who has the key now? Return only the name.", "Kira"),
        ("Context: The Nera unit has code 321. Question: What is the code of the Nera unit? Return only the code.", "321"),
        ("Context: The Nera unit has code 321. Question: What is the code of the Vela unit? If the context does not say, answer exactly UNKNOWN.", "UNKNOWN"),
    ]
    solver_outputs = []
    for prompt, expected in probes:
        result = solver_audit.solve_prompt(prompt)
        passed = result.prediction == expected
        checks["unseen_prompt_solver_" + str(len(solver_outputs))] = passed
        solver_outputs.append({"prompt": prompt, "expected_from_independent_fixture": expected, "prediction": result.prediction, "witness": result.witness})
    checks["cycle_refused"] = solver_audit.solve_prompt("Iris is older than Juno. Juno is older than Iris. Who is youngest? Return only the name.").status == "NOT_APPLICABLE"
    checks["conflicting_code_refused"] = solver_audit.solve_prompt("Context: The Nera unit has code 321. The Nera unit has code 654. Question: What is the code of the Nera unit? Return only the code.").status == "NOT_APPLICABLE"
    checks["facts_have_no_target_fallback"] = solver_audit.solve_prompt("Which blood cells primarily carry oxygen? Answer concisely.").status == "NOT_APPLICABLE"
    try:
        solver_audit.solve_prompt({"prompt": "What is 9 minus 4? Give just the integer answer.", "scorer": {"value": 999}})
    except TypeError:
        checks["target_bearing_object_rejected"] = True
    else:
        checks["target_bearing_object_rejected"] = False
    checks["query_blind_unique_marker_floor_exposed"] = solver_audit.blind_predictions("Use only the records below.\nRecord 000: the Nera target has archival value 44.\nQuestion: What is the archival value of the Vela target? Return only the integer.")["context_unique_marked_target"] == "44"
    require = all(checks.values())
    result = {"status": "PASS_SYNTHETIC_REVIEW_PROBES" if require else "FAIL", "checks": checks,
              "analysis_source_sha256": hashlib.sha256(Path(analysis.__file__).read_bytes()).hexdigest(),
              "solver_source_sha256": hashlib.sha256(Path(solver_audit.__file__).read_bytes()).hexdigest(),
              "native_outcomes_read": False, "native_model_loads": 0,
              "solver_fixture_outputs": solver_outputs,
              "limitations": "Fixtures do not create a new task bank or native performance evidence"}
    out = Path(__file__).with_name("analysis_review_probes.json")
    with out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"status": result["status"], "checks": len(checks), "failures": [k for k, v in checks.items() if not v]}))
    if not require:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
