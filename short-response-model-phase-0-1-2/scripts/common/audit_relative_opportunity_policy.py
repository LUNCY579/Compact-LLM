from __future__ import annotations

"""Static audit preventing absolute promotion floors from returning.

This does not ban numerical experiment settings such as layer counts, active
ratios, seeds, or diagnostic corruption detectors. It targets only automatic
promotion controls in official phase configuration, wrappers, and gate code.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONFIGS = [
    ROOT / "configs/phase_2_progressive_reduction.yaml",
    ROOT / "configs/phase_2c_targeted_recovery.yaml",
    ROOT / "configs/phase_4b_structural_reduction.yaml",
    ROOT / "configs/sparse_runtime_pilot.yaml",
]
WRAPPERS = [
    ROOT / "scripts/phase_2_layer_selection/run_phase_2_structural_reduction.ps1",
    ROOT / "scripts/phase_2c_targeted_recovery/run_phase_2c.ps1",
    ROOT / "scripts/phase_4_model_soups/run_phase_4.ps1",
]
SOURCE_DIRS = [ROOT / "scripts", ROOT / "Phase2B_Layer_Combination_Playground_v1.0"]

PROHIBITED_CONFIG_KEYS = re.compile(
    r"(?im)^\s*(?:min(?:imum)?|max(?:imum)?)[_-](?:quality|semantic|complete|intelligibility|"
    r"category|reference|nll|major|critical|hard|score|recall|vram|runtime|speedup)[\w_-]*\s*:"
)
PROHIBITED_WRAPPER_PARAMS = re.compile(
    r"(?i)\$(?:Min|Minimum|Max|Maximum)(?:Quality|Semantic|Complete|Intelligibility|Category|"
    r"Reference|Nll|Major|Critical|Hard|Score|Recall|Vram|Runtime|Speedup)"
)
PROHIBITED_SOURCE_COMPARISON = re.compile(
    r"(?i)(quality_pass_rate|semantic_pass_rate|complete_utterance_rate|intelligibility_pass_rate|"
    r"hard_failure_case_count|critical_failure_count|major_case_count|mean_quality_score|"
    r"worst_decile_mean_quality_score)\s*(?:<=|>=|<|>)\s*(?:\d|[A-Za-z_])"
)


def audit() -> list[str]:
    findings: list[str] = []
    for path in CONFIGS:
        text = path.read_text(encoding="utf-8")
        for match in PROHIBITED_CONFIG_KEYS.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            findings.append(f"{path.relative_to(ROOT)}:{line}: prohibited promotion threshold key")
    for path in WRAPPERS:
        text = path.read_text(encoding="utf-8")
        for match in PROHIBITED_WRAPPER_PARAMS.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            findings.append(f"{path.relative_to(ROOT)}:{line}: prohibited PowerShell promotion threshold")
    for directory in SOURCE_DIRS:
        for path in directory.rglob("*.py"):
            if path.name in {"audit_relative_opportunity_policy.py"}:
                continue
            text = path.read_text(encoding="utf-8")
            for match in PROHIBITED_SOURCE_COMPARISON.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{path.relative_to(ROOT)}:{line}: direct absolute quality comparison")
    return findings


def main() -> None:
    findings = audit()
    if findings:
        print("Relative-opportunity policy audit FAILED:")
        for finding in findings:
            print(f"- {finding}")
        raise SystemExit(1)
    print("Relative-opportunity policy audit passed: no absolute promotion floors found.")


if __name__ == "__main__":
    main()
