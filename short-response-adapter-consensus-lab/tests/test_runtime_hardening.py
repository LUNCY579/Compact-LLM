from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from consensus_lab.evaluation.summary import summary_projection
from consensus_lab.experiments.config import CommitteeSpec, ExperimentConfig, filter_committees_by_size
from consensus_lab.run_provenance import begin_runtime_invocation, finish_runtime_invocation
from experiments.phase2_mechanism import run_remaining_pipeline
from experiments.phase2_mechanism.audit_phase2_completion import (
    historical_capture_generator_is_well_formed,
)
from experiments.phase2_mechanism.reconcile_completed_runtime_manifest import reconcile_manifest


def test_runtime_manifest_rejects_identity_drift_and_marks_unclean_invocation(tmp_path: Path) -> None:
    output = tmp_path / "run.jsonl"
    manifest = tmp_path / "run.jsonl.runtime.json"
    identity = {"identity_sha256": "identity-a", "identity": {"config": "a"}}
    first, _ = begin_runtime_invocation(
        manifest_path=manifest,
        output_path=output,
        identity=identity,
        runtime_controls={"max_new_rows": 1},
    )
    output.write_text(json.dumps({"row": 1}) + "\n", encoding="utf-8")
    second, packet = begin_runtime_invocation(
        manifest_path=manifest,
        output_path=output,
        identity=identity,
        runtime_controls={"max_new_rows": 1},
    )
    assert first != second
    assert packet["invocations"][0]["status"] == "INTERRUPTED_OR_UNCLEAN_EXIT"
    completed = finish_runtime_invocation(
        manifest_path=manifest,
        invocation_id=second,
        output_path=output,
        expected_rows=1,
    )
    assert completed["status"] == "COMPLETE"
    with pytest.raises(RuntimeError, match="different source/config/model/runtime identity"):
        begin_runtime_invocation(
            manifest_path=manifest,
            output_path=output,
            identity={"identity_sha256": "identity-b", "identity": {}},
            runtime_controls={},
        )


def test_runtime_manifest_refuses_after_the_fact_backfill(tmp_path: Path) -> None:
    output = tmp_path / "legacy.jsonl"
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refuse provenance backfill"):
        begin_runtime_invocation(
            manifest_path=tmp_path / "legacy.jsonl.runtime.json",
            output_path=output,
            identity={"identity_sha256": "a", "identity": {}},
            runtime_controls={},
        )


def test_summary_projection_discards_large_token_traces() -> None:
    row = {
        "run_name": "run",
        "dataset": "d",
        "committee": "c",
        "method": "hard_plurality",
        "steps": [{"large": "trace"}],
        "output_token_ids": [1, 2, 3],
        "output": "answer",
        "correct": True,
    }
    projected = summary_projection(row)
    assert projected["output"] == "answer"
    assert projected["correct"] is True
    assert "steps" not in projected
    assert "output_token_ids" not in projected


def test_committee_filter_changes_effective_config_identity() -> None:
    raw = {
        "name": "test",
        "committees": [
            {"name": "k1", "experts": ["a"]},
            {"name": "k2", "experts": ["a", "b"]},
        ],
    }
    config = ExperimentConfig(
        name="test",
        backbone=Path("backbone"),
        experts=[],
        committees=[CommitteeSpec("k1", ["a"], "x"), CommitteeSpec("k2", ["a", "b"], "x")],
        methods=["hard_plurality"],
        datasets={"dev": Path("dev.jsonl")},
        max_sequence_length=8,
        max_new_tokens=2,
        cache_mode="reference",
        eos_policy="ordinary",
        eos_confidence_threshold=0.5,
        minimum_tokens=1,
        adaptive_core_experts=[],
        adaptive_agreement_threshold=0.67,
        seed=1,
        raw=raw,
        config_sha256="original",
    )
    filtered = filter_committees_by_size(config, 1)
    assert [committee.name for committee in filtered.committees] == ["k1"]
    assert filtered.config_sha256 != config.config_sha256
    assert [packet["name"] for packet in filtered.raw["committees"]] == ["k1"]


def test_gpu_stage_runs_in_bounded_fresh_workers(tmp_path: Path, monkeypatch: Any) -> None:
    output = tmp_path / "batched.jsonl"
    monkeypatch.setattr(run_remaining_pipeline, "STATE", tmp_path / "state.json")
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, cwd: Path, check: bool) -> None:
        calls.append(command)
        before = run_remaining_pipeline.jsonl_count(output)
        add = min(2, 5 - before)
        with output.open("a", encoding="utf-8") as handle:
            handle.write("{}\n" * add)

    monkeypatch.setattr(run_remaining_pipeline.subprocess, "run", fake_run)
    state: dict[str, Any] = {"status": "RUNNING"}
    run_remaining_pipeline.run_gpu_stage_batched(
        name="fake_gpu_stage",
        command=["python", "run.py"],
        output_path=output,
        expected_rows=5,
        batch_rows=2,
        cooldown_seconds=0,
        max_consecutive_failures=3,
        state=state,
    )
    assert run_remaining_pipeline.jsonl_count(output) == 5
    assert len(calls) == 3
    assert all(command[-2:] == ["--max-new-rows", "2"] for command in calls)


def test_completed_output_can_close_stale_runtime_manifest_without_rewriting_rows(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run.jsonl"
    manifest = output.with_suffix(output.suffix + ".runtime.json")
    identity = "a" * 64
    rows = [
        {
            "dataset": "dev",
            "committee": "k1",
            "method": "hard_plurality",
            "example_id": str(index),
            "runtime_identity_sha256": identity,
        }
        for index in range(2)
    ]
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    raw_before = output.read_bytes()
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "RUNNING",
                "output_path": str(output.resolve()),
                "identity_sha256": identity,
                "identity": {},
                "expected_rows": 2,
                "observed_rows": 1,
                "invocations": [
                    {
                        "invocation_id": "final",
                        "status": "STARTED",
                        "pid": 12345,
                        "rows_before": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    packet, changed = reconcile_manifest(
        manifest_path=manifest,
        output_path=output,
        expected_rows=2,
        process_checker=lambda _pid: False,
        observed_at_utc="2026-08-14T00:00:00+00:00",
    )
    assert changed is True
    assert output.read_bytes() == raw_before
    assert packet["status"] == "COMPLETE"
    assert packet["completion_basis"] == "POST_HOC_VERIFIED_COMPLETE"
    assert packet["invocations"][-1]["status"] == "INTERRUPTED_OR_UNCLEAN_EXIT"
    assert packet["invocations"][-1]["rows_added_before_unclean_exit"] == 1
    assert packet["verified_output"]["rows"] == 2

    unchanged, changed_again = reconcile_manifest(
        manifest_path=manifest,
        output_path=output,
        expected_rows=2,
        process_checker=lambda _pid: False,
    )
    assert changed_again is False
    assert unchanged == packet


def test_historical_boundary_capture_keeps_original_generator_hash() -> None:
    packet = {
        "generator": {
            "path": "experiments/phase2_mechanism/make_selected_runtime_boundaries.py",
            "sha256": "b" * 64,
        }
    }
    assert historical_capture_generator_is_well_formed(packet)
    packet["generator"]["sha256"] = "not-a-sha"
    assert not historical_capture_generator_is_well_formed(packet)
