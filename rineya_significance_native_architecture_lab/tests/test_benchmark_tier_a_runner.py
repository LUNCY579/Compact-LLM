from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jsonschema


def test_tier_a_runner_writes_honest_complete_run(tmp_path: Path) -> None:
    package = Path(__file__).resolve().parents[1]
    output = tmp_path / "run"
    completed = subprocess.run(
        [
            sys.executable,
            str(package / "scripts" / "run_tier_a.py"),
            "--output",
            str(output),
            "--seeds",
            "17",
            "--evidence-depths",
            "0.1,0.9",
            "--filler-count",
            "8",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    for relative in (
        "manifest.json",
        "summary.json",
        "cases.jsonl",
        "traces",
        "statistics",
        "figures/data/exponent_sweep.csv",
    ):
        assert (output / relative).exists()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["model_inference_performed"] is False
    assert summary["gpu_used"] is False
    assert "Rineya reasoning or retrieval quality improves" in summary["claims_not_supported"]

    manifest_schema = json.loads((package / "schemas" / "manifest.schema.json").read_text(encoding="utf-8"))
    case_schema = json.loads((package / "schemas" / "case.schema.json").read_text(encoding="utf-8"))
    trace_schema = json.loads((package / "schemas" / "trace.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(json.loads((output / "manifest.json").read_text(encoding="utf-8")), manifest_schema)
    first_case = json.loads((output / "cases.jsonl").read_text(encoding="utf-8").splitlines()[0])
    jsonschema.validate(first_case, case_schema)
    for trace_manifest in (output / "traces").glob("*/trace_manifest.json"):
        jsonschema.validate(json.loads(trace_manifest.read_text(encoding="utf-8")), trace_schema)
