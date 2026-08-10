from __future__ import annotations

import json
from pathlib import Path

from scripts.phase_9_student_distillation.select_procedural_recovery_v211 import CandidateInput, select
from scripts.phase_9_student_distillation.validate_rag_reasoning_corpus import validate

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/phase_9_student_distillation/procedural_recovery_v211"
CONFIG = ROOT / "configs/phase_9_procedural_recovery_v211.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def summary(*, proc: float, supported: float = 0.63, faith: float = 0.98,
            abstain: float = 0.81, conflict: float = 1.0, social: float = 1.0,
            valid: bool = True) -> dict:
    return {
        "evaluation_valid": valid,
        "conditions": {
            "real_rag": {
                "procedure_correctness": proc,
                "supported_answer_correctness": supported,
                "evidence_faithfulness": faith,
                "correct_abstention_rate": abstain,
                "conflict_safe_handling_rate": conflict,
                "social_usable_rate": social,
            }
        },
    }


def test_v211_corpus_is_valid_large_and_real_world_diverse() -> None:
    packet = validate(CORPUS)
    assert packet["status"] == "PASS", packet["problems"]
    assert packet["reviewed_rows"] == 6560
    assert packet["split_counts"] == {"train": 4976, "dev": 792, "test": 792}
    assert packet["knowledge_documents"] == 5040
    assert packet["procedures"] == 2320

    manifest = json.loads((CORPUS / "manifests/dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["procedure_family_count"] == 24
    families = set(manifest["procedure_families"])
    for expected in {"incident", "rate_limit", "inventory", "shipping", "access", "failover", "maintenance", "retention"}:
        assert expected in families
    assert manifest["profile_counts"] == {
        "boundary": 800,
        "clean_binding": 1440,
        "hard_collision": 2880,
        "moderate_distractor": 1440,
    }


def test_v211_stage_a_is_small_hard_collision_burst() -> None:
    rows = read_jsonl(CORPUS / "curriculum_views/train_stage_a_short_binding.reviewed.jsonl")
    assert len(rows) == 768
    assert all(row.get("decision") == "answer" for row in rows)
    assert all("hard_collision" in (row.get("scenario_flags") or []) for row in rows)
    assert all(row.get("curriculum_role") == "stage_a_short_binding" for row in rows)
    assert all(row["messages"][-1]["role"] == "assistant" for row in rows)


def test_v211_stage_b_is_preservation_first_and_replays_verified_original_procedure() -> None:
    rows = read_jsonl(CORPUS / "curriculum_views/train_stage_b_preservation_first.reviewed.jsonl")
    assert len(rows) == 4820
    roles: dict[str, int] = {}
    for row in rows:
        roles[row.get("curriculum_role", "")] = roles.get(row.get("curriculum_role", ""), 0) + 1
    preservation = sum(v for k, v in roles.items() if k.startswith("preservation_"))
    assert preservation == 3120
    assert preservation / len(rows) > 0.64
    assert roles["preservation_factual_answer"] == 1200
    assert roles["preservation_factual_boundary"] == 700
    assert roles["preservation_procedure_answer_verified"] == 800
    assert roles["preservation_procedure_boundary"] == 300
    assert roles["preservation_social"] == 120
    assert roles["new_clean_moderate"] == 900
    assert roles["new_hard_residual"] == 350
    assert roles["new_boundary"] == 450
    # Unlike v2.1, original procedure behavior is explicitly replayed rather than excluded wholesale.
    proc_preserve = [r for r in rows if r.get("curriculum_role") == "preservation_procedure_answer_verified"]
    assert proc_preserve and all(r.get("mode") == "procedure" for r in proc_preserve)
    assert all(r["messages"][-1]["role"] == "assistant" for r in rows)


def test_v211_config_and_runner_encode_conservative_checkpoint_ladder() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["training"]["stage_a"]["max_steps"] == 60
    assert cfg["training"]["stage_b1"]["max_steps"] == 60
    assert cfg["training"]["stage_b2"]["max_steps"] == 60
    assert cfg["training"]["stage_a"]["learning_rate"] == 1e-5
    assert cfg["training"]["stage_b1"]["learning_rate"] == 5e-6
    assert cfg["training"]["stage_b2"]["learning_rate"] == 3e-6
    assert cfg["selection"]["fallback"] == "query_aware_only"
    assert cfg["retrieval"]["selection_policy"] == "query_aware_v21"

    runner = (ROOT / "scripts/phase_9_student_distillation/run_phase_9_procedural_recovery_v211.ps1").read_text(encoding="utf-8")
    assert "AllowAggressiveTraining" in runner
    assert "total recovery steps" in runner
    assert "Stage A] Original Phase 9 DEV" in runner
    assert "Stage B1] Original Phase 9 DEV" in runner
    assert "Stage B2] Original Phase 9 DEV" in runner
    assert "select_procedural_recovery_v211.py" in runner
    assert "sealed_selected_original_test" in runner
    assert "query_aware_v21" in runner
    assert "--resume" in runner
    assert "RequireResearchReady" in runner
    # DEV ladder stays soft; only sealed TEST uses the hard research-ready gate.
    pre_sealed, sealed_block = runner.split("if ($RunSealedTest)", 1)
    assert "-RequireResearchReady" not in pre_sealed
    assert sealed_block.count("-RequireResearchReady") >= 2


def test_v211_selector_rejects_training_that_overwrites_original_path() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    baseline_orig = summary(proc=0.95, supported=0.63, faith=0.98)
    baseline_chal = summary(proc=0.01)
    harmful = summary(proc=0.23, supported=0.49, faith=0.84)
    challenge_gain = summary(proc=0.16)
    packet = select(
        config=cfg,
        baseline_original=baseline_orig,
        baseline_challenge=baseline_chal,
        baseline_adapter="old_adapter",
        trained=[
            CandidateInput("stage_a", harmful, challenge_gain, "a"),
            CandidateInput("stage_b1", harmful, challenge_gain, "b1"),
            CandidateInput("stage_b2", harmful, challenge_gain, "b2"),
        ],
    )
    assert packet["selected_candidate"] == "query_aware_only"
    assert packet["selected_adapter"] == "old_adapter"
    assert packet["status"].startswith("QUERY_AWARE_ONLY_SELECTED")


def test_v211_selector_accepts_small_residual_gain_when_preservation_cost_is_lower() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    baseline_orig = summary(proc=0.95, supported=0.63, faith=0.98)
    baseline_chal = summary(proc=0.01)
    stage_a_orig = summary(proc=0.94, supported=0.625, faith=0.975)
    stage_a_chal = summary(proc=0.18)
    stage_b_orig = summary(proc=0.90, supported=0.60, faith=0.94)
    stage_b_chal = summary(proc=0.19)
    packet = select(
        config=cfg,
        baseline_original=baseline_orig,
        baseline_challenge=baseline_chal,
        baseline_adapter="old",
        trained=[
            CandidateInput("stage_a", stage_a_orig, stage_a_chal, "a"),
            CandidateInput("stage_b1", stage_b_orig, stage_b_chal, "b1"),
            CandidateInput("stage_b2", stage_b_orig, stage_b_chal, "b2"),
        ],
    )
    assert packet["selected_candidate"] == "stage_a"
    assert packet["selected_adapter"] == "a"
    assert "stage_a" in packet["pareto_frontier"]
