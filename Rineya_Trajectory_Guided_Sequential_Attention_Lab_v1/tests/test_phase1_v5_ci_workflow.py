"""Static contract checks for the dedicated TGSA v5 workflow."""

from __future__ import annotations

import re
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tgsa-v5.yml"


def test_v5_workflow_exists_and_is_scoped_to_successor_branch() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "research/tgsa-v5-competence-and-mechanism" in text
    assert "run_full" not in text
    assert "self-hosted" not in text


def test_v5_workflow_references_only_existing_configs_and_schemas() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for relative in re.findall(
        r"Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1/"
        r"((?:configs|evidence)/[A-Za-z0-9_.-]+\.json)",
        text,
    ):
        assert (LAB_ROOT / relative).is_file(), relative


def test_v5_workflow_never_runs_registered_plan_or_destroys_persistent_evidence() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "--plan \"${SMOKE_PLAN}\"" in text
    assert "--plan \"${REGISTERED_PLAN}\"" not in text
    assert "rm -rf full_evidence" not in text
    assert "mechanism_test\"] is False" in text
    assert "rineya_retrofit\"] is False" in text
