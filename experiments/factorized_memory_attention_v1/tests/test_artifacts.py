from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_artifacts_are_present_and_json_is_valid() -> None:
    required = (
        "PROTOCOL.md",
        "HISTORY_AUDIT.json",
        "PARAMETER_BUDGET.json",
        "MANIFEST.json",
        "QUALIFICATION_RECEIPT.json",
        "evaluation_schema.json",
        "configs/fma_rank32_supa.json",
        "scripts/probe_resources.py",
        "scripts/smoke_generate.py",
        "scripts/qualify.py",
    )
    for relative in required:
        assert (ROOT / relative).is_file(), relative
    for relative in (
        "HISTORY_AUDIT.json",
        "PARAMETER_BUDGET.json",
        "MANIFEST.json",
        "QUALIFICATION_RECEIPT.json",
        "evaluation_schema.json",
    ):
        json.loads((ROOT / relative).read_text(encoding="utf-8"))
