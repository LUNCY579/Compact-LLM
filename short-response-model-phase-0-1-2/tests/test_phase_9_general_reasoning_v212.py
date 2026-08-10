from __future__ import annotations

import json
from pathlib import Path

from scripts.phase_9_student_distillation.select_general_reasoning_v212 import Candidate, select

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "phase_9_student_distillation" / "general_reasoning_v212"
CONFIG = ROOT / "configs" / "phase_9_general_reasoning_v212.json"


def _read(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _rag(proc=.95, supported=.63, faith=.98, abstain=.81, conflict=1.0, social=1.0):
    return {
        "evaluation_valid": True,
        "procedure_correctness": proc,
        "supported_answer_correctness": supported,
        "evidence_faithfulness": faith,
        "correct_abstention_rate": abstain,
        "conflict_safe_handling_rate": conflict,
        "social_usable_rate": social,
    }


def _reasoning(overall=.30, macro=.30, hard=.20, abstain=.80, short=1.0):
    return {
        "evaluation_valid": True,
        "overall_correctness": overall,
        "macro_family_correctness": macro,
        "hard_correctness": hard,
        "abstention_correctness": abstain,
        "short_usable_rate": short,
    }


def test_packaged_v212_corpus_is_large_balanced_and_disjoint():
    manifest=json.loads((CORPUS/"manifests"/"dataset_manifest.json").read_text())
    assert manifest["canonical_rows"] == 13200
    assert manifest["families"] == 33
    assert manifest["split_counts"] == {"train":10560,"dev":1320,"test":1320}
    assert manifest["stage_b_general_rows"] == 3600
    assert manifest["stage_b_preservation_rows"] == 2400
    splits={s:_read(CORPUS/"reviewed_splits"/f"{s}.reviewed.jsonl") for s in ("train","dev","test")}
    ids={s:{r["id"] for r in rows} for s,rows in splits.items()}
    assert not (ids["train"] & ids["dev"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["dev"] & ids["test"])
    assert len({r["reasoning_family"] for r in splits["dev"]}) == 33


def test_targets_are_short_and_no_chain_of_thought_is_trained():
    rows=_read(CORPUS/"reviewed_splits"/"dev.reviewed.jsonl")
    assert rows
    for row in rows[:300]:
        assert len(row["messages"]) == 3
        assert row["messages"][-1]["content"] == row["approved_answer"]
        assert len(row["approved_answer"].split()) <= 12
        assert "rationale" not in row
        assert isinstance(row["verification"], dict)


def test_selector_keeps_input_when_reasoning_gain_does_not_pay_for_regression():
    cfg=json.loads(CONFIG.read_text())
    base=Candidate("input_student","",_rag(),{"evaluation_valid":True,"procedure_correctness":.02},_reasoning())
    bad=Candidate("bad","bad",_rag(proc=.70,supported=.50,faith=.80,abstain=.60),{"evaluation_valid":True,"procedure_correctness":.10},_reasoning(overall=.45,macro=.45,hard=.35,abstain=.60))
    out=select(cfg,base,[bad])
    assert out["selected_candidate"] == "input_student"


def test_selector_accepts_broad_gain_when_preservation_is_near_flat():
    cfg=json.loads(CONFIG.read_text())
    base=Candidate("input_student","",_rag(),{"evaluation_valid":True,"procedure_correctness":.02},_reasoning())
    good=Candidate("good","adapter",_rag(proc=.95,supported=.64,faith=.975,abstain=.80),{"evaluation_valid":True,"procedure_correctness":.05},_reasoning(overall=.52,macro=.51,hard=.44,abstain=.80))
    out=select(cfg,base,[good])
    assert out["selected_candidate"] == "good"
