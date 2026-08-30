"""The v4 documents state numbers that live in code. They must agree.

Three separate edits this session changed a threshold and had to be chased
through the gate module, the pilot report, the preregistration, and the runbook
by hand. A stale number in a preregistration is worse than a stale number in a
comment: the document is the thing a reader trusts when the code has moved on.

These tests fail when a document and the code disagree, so the drift is caught
at commit time rather than by a reader months later.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
if str(LAB / "scripts") not in sys.path:
    sys.path.insert(0, str(LAB / "scripts"))

from tgsa.phase1_competence import COMPETENCE_SCHEMA_VERSION, CompetenceThresholds

PILOT = (LAB / "PHASE1_V4_PILOT_REPORT.md").read_text(encoding="utf-8")
PREREG = (LAB / "PHASE1_V4_PREREGISTRATION.md").read_text(encoding="utf-8")
RUNBOOK = (LAB / "PHASE1_V4_RUNBOOK.md").read_text(encoding="utf-8")

# The thresholds the frozen re-gate uses. Everything else must agree with these.
from regate_competence_report import FROZEN  # noqa: E402


@pytest.mark.parametrize(
    "value,label",
    [
        (FROZEN.minimum_margin_over_trivial, "0.030"),
        (FROZEN.minimum_absolute_accuracy, "0.070"),
        (FROZEN.maximum_generalization_gap, "0.10"),
        (FROZEN.maximum_seed_coefficient_of_variation, "0.35"),
    ],
)
def test_the_pilot_report_threshold_table_matches_the_code(value: float, label: str) -> None:
    assert f"{value:.3f}".rstrip("0").rstrip(".") in label or f"{value}" in label
    assert label in PILOT, f"threshold {label} is not in the pilot report's frozen table"


def test_the_minimum_seed_runs_is_stated_in_both_documents() -> None:
    assert FROZEN.minimum_seed_runs == 3
    assert "≥ **3**" in PILOT or "at least 3" in PILOT or "(3)" in PILOT
    assert "3" in PREREG


def test_no_document_still_quotes_the_withdrawn_sd_bound() -> None:
    """maximum_seed_sd was replaced by a coefficient-of-variation bound."""

    assert not hasattr(CompetenceThresholds(
        minimum_margin_over_trivial=0.03, minimum_absolute_accuracy=0.07), "maximum_seed_sd")
    for name, text in (("pilot", PILOT), ("prereg", PREREG), ("runbook", RUNBOOK)):
        assert "maximum_seed_sd" not in text, f"{name} still names the withdrawn threshold"
        assert "between-seed SD, per arm" not in text, f"{name} still gates on raw SD"


def test_the_prereg_gate_table_lists_every_check() -> None:
    """A check that exists in code but not in the prereg is an unregistered gate."""

    from tgsa.phase1_competence import (
        ArmCompetence,
        evaluate_competence,
    )
    import math

    arms = [
        ArmCompetence(arm=name, holdout_accuracy=0.3, seen_accuracy=0.31, answer_nll=2.0,
                      distinct_predictions=30,
                      per_family_accuracy={"pointer_chase": 0.3, "state_update": 0.3},
                      per_seed_holdout_accuracy=[0.29, 0.30, 0.31])
        for name in ("cascade", "history_control", "learned_delta")
    ]
    report = evaluate_competence(
        arms, thresholds=FROZEN,
        trivial_baselines={"uniform": 0.03125, "eval_majority": 0.039},
        uniform_answer_nll=math.log(32),
        primary_families=("pointer_chase", "state_update"),
    )
    for check in report["checks"]:
        assert check in PREREG, f"check {check!r} is enforced by code but not registered"


def test_the_schema_version_is_not_stale_in_the_pilot_report() -> None:
    suffix = COMPETENCE_SCHEMA_VERSION.rsplit(".", 1)[-1]
    assert f"at `{suffix}`" in PILOT or f"to `{suffix}`" in PILOT, (
        f"pilot report does not mention the current schema suffix {suffix!r}"
    )


def test_the_escalation_table_matches_the_ladder_code() -> None:
    from phase1_v4_escalation import (
        ESCALATED_EXAMPLES,
        ESCALATED_STEPS,
        MEASUREMENT_CAP,
    )

    assert f"**{ESCALATED_STEPS // 1000}k**" in PREREG
    assert f"**{ESCALATED_EXAMPLES // 1000}k**" in PREREG
    assert f"**{ESCALATED_EXAMPLES:,}** examples" in PREREG
    assert f"**{ESCALATED_STEPS:,}** updates" in PREREG
    assert "capped at **four**" in PREREG and MEASUREMENT_CAP == 4


def test_every_runbook_command_names_a_script_that_exists() -> None:
    referenced = set(re.findall(r"python (scripts/[\w./-]+\.py)", RUNBOOK))
    assert referenced, "no script commands found in the runbook; the regex is wrong"
    missing = sorted(name for name in referenced if not (LAB / name).is_file())
    assert not missing, f"runbook references scripts that do not exist: {missing}"


def test_the_runbook_power_command_uses_the_v4_stratum_count() -> None:
    """v3's 17 strata would report the power of a study nobody is going to run."""

    assert "--stratum-count 3 " in RUNBOOK
    assert "--stratum-count 17" not in RUNBOOK


# --------------------------------------------------------------------------
# The README is what a reader trusts first
# --------------------------------------------------------------------------

README = (LAB / "README.md").read_text(encoding="utf-8")


def test_the_readme_test_count_is_not_stale() -> None:
    """It claimed 22 tests long after the suite passed 250."""

    import subprocess

    collected = subprocess.run(
        [sys.executable, "-m", "pytest", str(LAB / "tests"), "--collect-only", "-q"],
        capture_output=True, text=True, cwd=str(LAB),
    ).stdout
    count = int(re.search(r"(\d+) tests collected", collected).group(1))
    claimed = re.search(r"(\d+) tests\.", README)
    assert claimed, "README no longer states a test count"
    assert int(claimed.group(1)) == count, (
        f"README claims {claimed.group(1)} tests, suite collects {count}"
    )


def test_the_readme_does_not_overclaim() -> None:
    """The v3 result must not be readable as evidence about the mechanism."""

    assert "not evidence against TGSA" in README
    assert "Anything about the TGSA mechanism." in README
    assert "non-evidentiary" in README


def test_the_readme_repository_map_lists_files_that_exist() -> None:
    referenced = set(re.findall(r"^\| `([\w./-]+\.md)` \|", README, flags=re.MULTILINE))
    assert referenced, "no document rows found in the repository map"
    missing = sorted(name for name in referenced if not (LAB / name).is_file())
    assert not missing, f"repository map lists files that do not exist: {missing}"
