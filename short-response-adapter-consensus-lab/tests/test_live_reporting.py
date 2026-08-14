from __future__ import annotations

from pathlib import Path

import torch

from consensus_lab.evaluation.direct import (
    _effective_summary_every,
    _release_unused_device_memory,
    update_live_report,
)


def summary(method: str, correctness: float, case_count: int) -> dict[str, object]:
    return {
        "dataset": "dev",
        "committee": "full_k7",
        "method": method,
        "committee_size": 7,
        "case_count": case_count,
        "correctness": correctness,
        "mean_output_tokens": 2.0,
        "mean_latency_seconds": 1.0,
        "peak_gpu_vram_bytes": 1024,
        "mean_agreement": 0.5,
        "false_abstention_rate": 0.0,
    }


def test_incomplete_live_report_suppresses_unequal_n_ranking(tmp_path: Path) -> None:
    path = tmp_path / "live.md"
    update_live_report(
        path=path,
        run_name="frozen_run",
        completed=150,
        total=200,
        summaries=[summary("hard", 0.2, 100), summary("soft", 0.9, 50)],
        warnings=[],
    )
    text = path.read_text(encoding="utf-8")
    assert "Current best" not in text
    assert "unequal N" in text
    assert "Interim condition summaries (not method-comparable)" in text


def test_complete_live_report_reports_best_condition(tmp_path: Path) -> None:
    path = tmp_path / "live.md"
    update_live_report(
        path=path,
        run_name="frozen_run",
        completed=200,
        total=200,
        summaries=[summary("hard", 0.2, 100), summary("soft", 0.3, 100)],
        warnings=[],
    )
    text = path.read_text(encoding="utf-8")
    assert "Current best: `full_k7 / soft`" in text
    assert "## Completed summaries" in text


def test_cuda_cleanup_only_releases_cuda_cache(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("released"))
    _release_unused_device_memory(torch.device("cpu"))
    assert calls == []
    _release_unused_device_memory(torch.device("cuda:0"))
    assert calls == ["released"]


def test_summary_cadence_caps_quadratic_rebuilds() -> None:
    assert _effective_summary_every(25, 4050) == 25
    assert _effective_summary_every(25, 16200) == 81
    assert _effective_summary_every(25, 48600) == 243
