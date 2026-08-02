from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "data/evaluation/quality_v4_4_comprehensive_calibration.jsonl"
BENCHMARK = ROOT / "data/phase_2_structural_reduction/probe_150_thorough_v4_4.jsonl"
VERIFIER = ROOT / "scripts/common/verify_quality_evaluation_v4_4.py"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v43_comprehensive_packet_covers_supported_surface_and_passes() -> None:
    packet = read_jsonl(PACKET)
    benchmark = read_jsonl(BENCHMARK)
    official_categories = {str(row["category"]) for row in benchmark}
    covered_categories = {
        tag.split(":", 1)[1]
        for item in packet
        for tag in item.get("coverage_tags", [])
        if tag.startswith("category:")
    }
    assert len(packet) == 620
    assert len(official_categories) == 31
    assert official_categories <= covered_categories
    completed = subprocess.run(
        [sys.executable, str(VERIFIER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "620/620 cases" in completed.stdout
    assert "61 scored labels" in completed.stdout
    assert "150/150 real collapses" in completed.stdout
