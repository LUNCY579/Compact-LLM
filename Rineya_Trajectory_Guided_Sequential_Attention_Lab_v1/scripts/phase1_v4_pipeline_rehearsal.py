"""De-risk the v4 scale-up by rehearsing the whole registered pipeline.

v4 changes the regime by an order of magnitude in both directions: 5,000 ->
~50,000 training examples and 2,500 -> 20,000 updates, on a band the registered
pipeline has never generated, with no split_overrides and a competence gate in
front of the runner. Discovering a defect in any of that during the confirmatory
run would waste ~18 GPU-hours and, worse, produce failure records inside the
evidence chain.

So: run the identical sequence at toy scale first.

PREPARE -> COMPETENCE GATE -> TRAIN -> BARRIER -> SINGLE EVALUATE -> VERIFY ->
ANALYZE, with a real GO report the runner must consult, the real in-band dataset
shape, and the real verifier. Nothing here is evidentiary -- the config declares
a distinct study id and the scale is meaningless -- but every code path is the
one v4 will take.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
WORK = Path(os.environ.get("TGSA_REHEARSAL_DIR", LAB / ".rehearsal"))

sys.path.insert(0, str(LAB))
from tgsa.phase1_competence import ArmCompetence, CompetenceThresholds, evaluate_competence

BAND = {
    "direct_lookup": {"min_nodes": 5, "max_nodes": 8, "min_hops": 1, "max_hops": 1},
    "pointer_chase": {"min_nodes": 5, "max_nodes": 8, "min_hops": 2, "max_hops": 2,
                      "reject_trivial": True},
    "state_update": {"min_steps": 2, "max_steps": 3, "reject_trivial": True},
}


def run(label: str, argv: list[str], *, cwd: Path = LAB) -> int:
    print(f"\n=== {label} ===", flush=True)
    started = time.time()
    completed = subprocess.run(argv, cwd=str(cwd), capture_output=True, text=True)
    elapsed = time.time() - started
    if completed.returncode != 0:
        print(f"FAILED rc={completed.returncode} ({elapsed:.1f}s)", flush=True)
        print(completed.stdout[-3000:], flush=True)
        print(completed.stderr[-4000:], flush=True)
    else:
        print(f"ok ({elapsed:.1f}s)", flush=True)
        tail = completed.stdout.strip().splitlines()[-4:]
        for line in tail:
            print(f"  {line}", flush=True)
    return completed.returncode


def write_competence(path: Path) -> None:
    """A genuine GO report. The runner must actually consult and accept it."""

    arms = [
        ArmCompetence(
            arm=name, holdout_accuracy=0.30, seen_accuracy=0.32, answer_nll=2.0,
            distinct_predictions=30,
            per_family_accuracy={"pointer_chase": 0.30, "state_update": 0.29},
            per_seed_holdout_accuracy=[0.29, 0.30, 0.31],
        )
        for name in ("cascade", "history_control", "learned_delta")
    ]
    report = evaluate_competence(
        arms,
        thresholds=CompetenceThresholds(
            minimum_margin_over_trivial=0.030, minimum_absolute_accuracy=0.070,
            maximum_answer_nll_fraction_of_uniform=0.97, minimum_distinct_predictions=4,
            maximum_family_accuracy=0.95, maximum_generalization_gap=0.10,
            maximum_seed_coefficient_of_variation=0.35, minimum_seed_runs=3,
        ),
        trivial_baselines={"uniform": 0.03125, "eval_majority": 0.039},
        uniform_answer_nll=math.log(32),
        primary_families=("pointer_chase", "state_update"),
        # Synthetic, like every other number in this rehearsal. A real run
        # computes this from the holdout's candidate sets.
        family_null={"pointer_chase": 0.1558, "state_update": 0.0312},
    )
    assert report["verdict"] == "GO", report["reasons"]
    report["regime"] = {
        "band": "easy_nontrivial", "band_spec": BAND,
        "train_examples": 2400, "steps": 150,
        "embed_dim": 96, "num_layers": 2, "num_heads": 6,
        "batch_size": 64, "learning_rate": 1e-3,
        "development_seeds": [1, 2, 3], "train_split_seed": 1_234_567,
        "holdout_split_seed": 987_654_321, "holdout_examples": 300,
        # The builder requires this, so without it the rehearsal fails at its
        # first step -- which is exactly what happened when the builder was
        # hardened and the rehearsal was not re-run.
        "split_generation_order": ["holdout", "train"],
        # ...and this makes the resulting artifact unusable for a real launch.
        # A rehearsal writes a genuine GO report to a fixed path in order to
        # exercise the gate; require_go refuses anything carrying this marker.
        "development_only": True,
        "note": "REHEARSAL ONLY -- toy scale, not evidentiary",
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    competence = WORK / "competence.json"
    power = WORK / "power.json"
    config = WORK / "study.json"
    write_competence(competence)
    power.write_text(json.dumps({"recommended_seed_count": 3}), encoding="utf-8")

    if run("BUILD CONFIG", [
        str(PYTHON), str(LAB / "scripts" / "build_phase1_v4_config.py"),
        "--competence", str(competence), "--power", str(power),
        "--output", str(config),
        "--study-id", "tgsa-phase1-v4-pipeline-rehearsal",
        "--test-per-family", "120", "--dev-per-family", "60",
        "--allow-development-only",
    ]):
        return 1

    # The gate is resolved relative to the config's own directory.
    print(f"\ncompetence report sits beside the config: {competence.name}", flush=True)

    if run("PREPARE DATA", [
        str(PYTHON), str(LAB / "scripts" / "prepare_phase1_data.py"),
        "--config", str(config), "--output-dir", str(WORK / "dataset"), "--force",
    ]):
        return 1

    # --- the gate must actually be able to stop a launch ------------------
    print("\n=== NEGATIVE CONTROL: require_go must refuse this very report ===", flush=True)
    from tgsa.phase1_competence import require_go as _require_go
    try:
        _require_go(competence)  # deliberately WITHOUT allow_development_only
    except ValueError as error:
        if "development_only" not in str(error):
            print(f"FAILED: refused, but not for the marker reason: {error}", flush=True)
            return 1
        print("ok -- the rehearsal's own GO report cannot certify a launch", flush=True)
    else:
        print("FAILED: require_go accepted a development-only report", flush=True)
        return 1

    print("\n=== NEGATIVE CONTROL: the runner must refuse a NO_GO ===", flush=True)
    broken = json.loads(competence.read_text(encoding="utf-8"))
    broken["verdict"] = "NO_GO"
    broken["checks"]["all_arms_beat_trivial"] = False
    broken["reasons"] = ["rehearsal negative control"]
    competence.write_text(json.dumps(broken), encoding="utf-8")
    refused = subprocess.run(
        [str(PYTHON), str(LAB / "scripts" / "run_phase1_model_study.py"),
         "--config", str(config), "--dataset-dir", str(WORK / "dataset"),
         "--output-dir", str(WORK / "refused"), "--phase", "train"],
        cwd=str(LAB), capture_output=True, text=True,
    )
    if refused.returncode == 0:
        print("FAILED: the runner launched on a NO_GO report", flush=True)
        return 1
    if "NO_GO" not in (refused.stdout + refused.stderr):
        print("FAILED: refused, but not for the gate reason:", flush=True)
        print((refused.stdout + refused.stderr)[-2000:], flush=True)
        return 1
    if (WORK / "refused").exists():
        print("FAILED: the runner created an output root before refusing", flush=True)
        return 1
    print("ok -- refused, for the right reason, before creating an output root", flush=True)
    write_competence(competence)  # restore the GO report

    study = WORK / "study_out"
    for phase in ("train", "barrier", "evaluate"):
        if run(f"{phase.upper()}", [
            str(PYTHON), str(LAB / "scripts" / "run_phase1_model_study.py"),
            "--config", str(config), "--dataset-dir", str(WORK / "dataset"),
            "--output-dir", str(study), "--phase", phase,
        ]):
            return 1

    if run("VERIFY", [
        str(PYTHON), str(LAB / "scripts" / "verify_phase1_evidence.py"),
        "--config", str(config), "--output-dir", str(study),
        "--dataset-dir", str(WORK / "dataset"), "--expect-confirmatory", "true",
    ]):
        return 1

    if run("ANALYZE", [
        str(PYTHON), str(LAB / "scripts" / "analyze_phase1_model_study.py"),
        "--config", str(config), "--output-dir", str(study),
        "--dataset-dir", str(WORK / "dataset"),
    ]):
        return 1

    analysis = json.loads((study / "analysis.json").read_text(encoding="utf-8"))
    print("\n=== REHEARSAL SUMMARY ===", flush=True)
    print(f"decision           {analysis.get('decision')}", flush=True)
    print(f"study_id           {analysis.get('study_id')}", flush=True)
    hierarchy = analysis.get("hypothesis_hierarchy", {})
    for key in ("h1a", "h1b", "h1c"):
        entry = hierarchy.get(key)
        if isinstance(entry, dict):
            print(f"{key:18s} passed={entry.get('passed')}", flush=True)
    print("\nThe pipeline executes end to end in the v4 shape. Nothing above is "
          "evidentiary: 150 steps on 2,400 examples cannot support any claim.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
