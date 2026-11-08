"""Numerical GPU admission and real tiny-decoder bridge path; no efficacy claims."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

from cross_scale.src.bridge_fit import fit_initial, fit_delta_bridge
from cross_scale.src.delta_alignment import donor_delta
from cross_scale.src.depth_pair_screen import evaluate_bridge
from cross_scale.src.gpu_qualification import (bridge_gradients, fixture_manifest, qualification_fixtures,
                                               tensor_agreement, validate_qualification_report)
from shared.integrity import state_hash


def gpu_config():
    return json.loads((Path(__file__).resolve().parents[1]/"configs/alignment_gpu_v2_1.json").read_text())


def test_qualification_fixture_selection_is_balanced_scenario_unique_and_order_invariant():
    rows = [{"family": f"f{family}", "example_id": f"e{family}_{partition}_{index}",
             "scenario_id": f"s{family}_{partition}_{index}", "alignment_partition": partition}
             for family in range(10) for partition in ("fit", "holdout") for index in range(3)]
    fixtures = qualification_fixtures(rows, gpu_config())
    assert len(fixtures) == 8
    assert [row["alignment_partition"] for row in fixtures] == ["fit", "holdout"]*4
    assert all(row["example_id"].endswith("_0") for row in fixtures)
    assert fixtures == qualification_fixtures(list(reversed(rows)), gpu_config())
    with pytest.raises(ValueError, match="eight"):
        qualification_fixtures(rows[:6], gpu_config())


@pytest.mark.parametrize("change", ["relative", "cosine", "absolute", "nan", "shape"])
def test_numerical_qualification_fails_closed(change):
    expected, actual = torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([[1., 0.], [0., 1.]])
    thresholds = deepcopy(gpu_config()["qualification"])
    if change == "relative":
        actual *= 1.0003
    elif change == "cosine":
        actual[0, 1] = .01
        thresholds.update(maximum_relative_l2=1., maximum_absolute_error=1.)
    elif change == "absolute":
        expected *= 1000
        actual = expected+.003
    elif change == "nan":
        actual[0, 0] = float("nan")
    else:
        actual = actual[:1]
    if change in ("nan", "shape"):
        with pytest.raises((ValueError, FloatingPointError)):
            tensor_agreement(expected, actual, thresholds)
    else:
        assert not tensor_agreement(expected, actual, thresholds)["pass"]


def test_qualification_accepts_exact_zero_and_small_fp32_roundoff():
    thresholds = gpu_config()["qualification"]
    assert tensor_agreement(torch.zeros(3, 2), torch.zeros(3, 2), thresholds)["pass"]
    values = torch.randn(4, 3, generator=torch.Generator().manual_seed(62))
    assert tensor_agreement(values, values*1.000001, thresholds)["pass"]


def test_cuda_terminal_memory_failure_preserves_failed_receipt(tmp_path, monkeypatch):
    from argparse import Namespace
    from cross_scale.scripts import run_alignment as caller
    from shared import manifests, runtime
    from shared.hashing import write_json
    config = gpu_config()
    root = tmp_path/"repo/experiments/programme/cross_scale"
    monkeypatch.setattr(caller, "ROOT", root)
    monkeypatch.setattr(caller, "PROGRAMME", root.parent)
    monkeypatch.setattr(caller, "configure", lambda path: config)
    registration = {"sources": {"toy.py": "immutable"}}
    monkeypatch.setattr(manifests, "verify_registration", lambda *args: registration)
    monkeypatch.setattr(runtime, "configure_runtime", lambda config: None)
    def exhausted(config):
        raise MemoryError("intentional terminal CUDA budget failure")
    monkeypatch.setattr(runtime, "cuda_memory", exhausted)
    monkeypatch.setattr(caller, "baseline_receipt", lambda config, registration, arm:
                        {"status": "STARTED", "track": "cross_scale", "arm": arm})
    monkeypatch.setattr(caller, "read_capture", lambda *args, **kwargs: ({"dataset_manifest_sha256": "data"}, None))
    monkeypatch.setattr(caller, "collect_completed_pairs", lambda *args: ([], []))
    monkeypatch.setattr(caller, "select_matrix", lambda *args: {"status": "toy"})
    evidence = root/"evidence/01_alignment_matrix/toy"
    write_json(evidence/"capture.json", {"toy": True})
    with pytest.raises(RuntimeError, match="terminal resource"):
        caller.execute(Namespace(action="aggregate", config=tmp_path/"config.json", run_id="toy", attempt=0))
    output = evidence/"aggregate/attempt_000"
    result = json.loads((output/"run_manifest.json").read_text())
    assert result["status"] == "FAILED"
    assert result["cuda_memory_error"]["type"] == "MemoryError"
    assert (output/"failure.json").is_file()


@pytest.fixture
def complete_qualification_report():
    from cross_scale.scripts.run_alignment import calibration_rows
    from cross_scale.src.donor_layers import DONOR_RECORD, strip_calibration_rows
    from shared.hashing import sha256_file
    config = gpu_config()
    rows = calibration_rows(Path(__file__).resolve().parents[2]/"data/generated/v2/cross_scale/manifest.json")
    selected = qualification_fixtures(rows, config)
    fixture_rows = [{**row, "alignment_partition": selected[index]["alignment_partition"]}
                    for index, row in enumerate(strip_calibration_rows(selected))]
    comparisons = []
    def append(label, width, gradient=False):
        for name, count in (("all", 1),) if gradient else (
                ("all_tokens", 2), ("ordinary_tokens", 1), ("special_tokens", 1), ("final_prompt_token", 1)):
            comparisons.append({"label": label, "stratum": name, "shape": [count, width],
                "relative_l2": 0., "maximum_absolute_error": 0., "mean_token_cosine": 1., "pass": True})
    for name, depths, width in (("supa", [8, 12, 16, 19], 1024),
                                ("donor", [11, 12, 15, 16, 19, 20, 23, 24], 2560)):
        for depth in depths:
            for row in selected:
                append(f"{name}_L{depth}_{row['example_id']}", width)
    for supa_index in (8, 12, 16, 19):
        for donor_index in (12, 16, 20, 24):
            for row in selected:
                if row["alignment_partition"] == "holdout":
                    append(f"S{supa_index}_D{donor_index}_B0_delta_{row['example_id']}", 2560)
    for donor_index in (12, 16, 20, 24):
        for objective in ("B2", "B3"):
            for key, width in (("up", 1024*2560), ("bias", 2560)):
                append(f"S8_D{donor_index}_{objective}_gradient_{key}", width, gradient=True)
    record = json.loads(DONOR_RECORD.read_text())
    weights = record["weights"] if isinstance(record["weights"], list) else record["weights"]["files"]
    files = {row["name"]: row["sha256"] for row in weights}
    files["config.json"] = record["config"]["sha256"]
    report = {"fixtures": fixture_manifest(fixture_rows), "thresholds": config["qualification"],
              "qualification_pass": True, "generation_count": 0, "comparisons": comparisons,
              "same_gpu_donor_delta_repeat_bit_identical": True,
              "same_gpu_streamed_capture_repeat_bit_identical": True,
              "same_gpu_bridge_gradient_repeat_bit_identical": True,
              "supa_files": {"model.safetensors": config["checkpoint_sha256"]},
              "donor_identity": {"revision": config["donor_revision"], "record_sha256": sha256_file(DONOR_RECORD), "files": files},
              "layer_integrity": [{"donor_layer": index, "state_sha256": "a"*64,
                    "cpu_rope_sha256": "b"*64, "gpu_rope_sha256": "b"*64,
                    "CPU_GPU_parameter_bytes_equal": True, "frozen_unchanged": True,
                    "backbone_gradients": 0, "trainable_backbone_parameters": 0} for index in (12, 16, 20, 24)]}
    return report, config, rows


def test_full_qualification_report_reconstructs_656_comparisons(complete_qualification_report):
    report, config, rows = complete_qualification_report
    assert len(report["comparisons"]) == 656
    assert validate_qualification_report(report, config, rows)


@pytest.mark.parametrize("damage", ["omitted", "duplicate", "failed_metric", "width", "partition", "checkpoint", "gradient", "repeat"])
def test_qualification_validator_does_not_trust_saved_pass(complete_qualification_report, damage):
    report, config, rows = complete_qualification_report
    if damage == "omitted": report["comparisons"].pop()
    elif damage == "duplicate": report["comparisons"].append(report["comparisons"][0])
    elif damage == "failed_metric": report["comparisons"][0]["relative_l2"] = .01
    elif damage == "width": report["comparisons"][0]["shape"][1] = 1
    elif damage == "partition": report["fixtures"]["rows"][0]["alignment_partition"] = "holdout"
    elif damage == "checkpoint": report["donor_identity"]["files"]["config.json"] = "wrong"
    elif damage == "gradient": report["layer_integrity"][0]["backbone_gradients"] = 1
    else: report["same_gpu_streamed_capture_repeat_bit_identical"] = False
    with pytest.raises(ValueError):
        validate_qualification_report(report, config, rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="local CUDA needed for tiny real-decoder device test")
def test_tiny_cuda_delta_fit_and_heldout_metrics_keep_artifacts_on_cpu():
    from shared.runtime import configure_runtime
    config = gpu_config()
    configure_runtime(config)
    config["delta_steps"] = 3  # Internal toy fixture only, never a registered run.
    architecture = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=36, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        layer_types=["full_attention"]*36)
    architecture._attn_implementation = "eager"
    layer = Qwen3DecoderLayer(architecture, 12).eval().requires_grad_(False)
    rope = Qwen3RotaryEmbedding(architecture).eval().requires_grad_(False)
    generator = torch.Generator().manual_seed(101)
    prompts = []
    for index in range(4):
        small, big = torch.randn(4, 8, generator=generator), torch.randn(4, 16, generator=generator)
        with torch.no_grad():
            delta = donor_delta(layer, rope, big.unsqueeze(0))[0]
        prompts.append({"small": small, "big": big, "delta": delta,
                        "ids": torch.tensor([[0, 1, 2, 3]]), "scenario_id": f"s{index}"})
    initial = fit_initial(torch.cat([p["small"] for p in prompts[:2]]),
                          torch.cat([p["big"] for p in prompts[:2]]), "B0", config, ["s0"]*4+["s1"]*4)
    before = state_hash(layer)
    cpu_gradients = bridge_gradients(initial, layer, rope, prompts[0], config, False)
    layer.to("cuda:0"); rope.to("cuda:0")
    gpu_gradients = bridge_gradients(initial, layer, rope, prompts[0], config, False)
    for key in ("up", "bias"):
        assert tensor_agreement(cpu_gradients[key].reshape(1, -1), gpu_gradients[key].reshape(1, -1), config["qualification"])["pass"]
    result, history = fit_delta_bridge(initial, layer, rope, prompts[:2], config, token_normalized=False)
    assert len(history) == 3 and result.up.device.type == "cuda" and result.down.device.type == "cuda"
    assert state_hash(layer) == before
    assert all(p["small"].device.type == "cpu" for p in prompts)
    report = evaluate_bridge(result, layer, rope, prompts[2:], [0], config)
    assert report["delta"]["ordinary_tokens"]["tokens"] == 6
    assert not list(result.parameters())
