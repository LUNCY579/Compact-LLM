from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from consensus_lab.decoding.decoder import GenerationResult
from consensus_lab.evaluation.direct import EVALUATION_PROTOCOL_VERSION, run_direct_experiment
from consensus_lab.experiments.config import CommitteeSpec, ExperimentConfig
from consensus_lab.io import read_jsonl
from consensus_lab.paths import LabPaths


class FakeCommittee:
    experts: list[Any] = []

    def subset(self, names: list[str]) -> "FakeCommittee":
        return self

    def metadata(self) -> dict[str, int]:
        return {
            "backbone_parameter_count": 10,
            "loaded_parameter_count": 11,
            "committee_adapter_parameter_count": 1,
            "active_parameter_count_per_expert_forward": 11,
            "adapter_storage_bytes": 4,
        }

    def encode_messages(self, messages: list[dict[str, Any]], *, max_sequence_length: int) -> torch.Tensor:
        return torch.tensor([[1]])


def test_interrupted_run_resumes_without_duplicate_rows(tmp_path: Path, monkeypatch: Any) -> None:
    official = tmp_path / "short-response-model-phase-0-1-2"
    lab = tmp_path / "short-response-adapter-consensus-lab"
    official.mkdir()
    lab.mkdir()
    dataset = official / "dev.jsonl"
    rows = [
        {
            "id": f"case-{index}",
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            "answer": "answer",
            "decision": "answer",
        }
        for index in range(2)
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = ExperimentConfig(
        name="resume_test",
        backbone=official,
        experts=[],
        committees=[CommitteeSpec("fake", [], "test")],
        methods=["hard_plurality"],
        datasets={"dev": dataset},
        max_sequence_length=8,
        max_new_tokens=2,
        cache_mode="reference",
        eos_policy="ordinary",
        eos_confidence_threshold=0.5,
        minimum_tokens=1,
        adaptive_core_experts=[],
        adaptive_agreement_threshold=0.67,
        seed=7,
        raw={},
        config_sha256="test-config",
    )
    calls = 0

    def fake_generate(*args: Any, **kwargs: Any) -> GenerationResult:
        nonlocal calls
        calls += 1
        return GenerationResult(
            output_text="answer",
            output_token_ids=[2],
            steps=[
                {
                    "agreement": 1.0,
                    "vote_entropy": 0.0,
                    "mean_pairwise_js": None,
                    "tie": False,
                    "eos_vote_fraction": 1.0,
                    "winner_is_eos": True,
                }
            ],
            stop_reason="eos",
            latency_seconds=0.01,
            time_to_first_final_token_seconds=0.01,
            forward_passes=1,
            adapter_evaluations=1,
            peak_gpu_vram_bytes=None,
            peak_cpu_rss_bytes=100,
            cache_mode="test",
        )

    monkeypatch.setattr("consensus_lab.evaluation.direct.generate_reference", fake_generate)
    paths = LabPaths(tmp_path, lab, official)
    output = lab / "results.jsonl"
    kwargs = {
        "paths": paths,
        "config": config,
        "committee": FakeCommittee(),
        "output_path": output,
        "summary_json": lab / "summary.json",
        "summary_csv": lab / "summary.csv",
        "live_report": lab / "live.md",
        "limit": None,
        "resume": True,
    }
    run_direct_experiment(**kwargs)
    run_direct_experiment(**kwargs)
    written = read_jsonl(output)
    assert calls == 2
    assert len(written) == 2
    assert len({row["example_id"] for row in written}) == 2
    assert all(row["evaluation_protocol_version"] == EVALUATION_PROTOCOL_VERSION for row in written)
