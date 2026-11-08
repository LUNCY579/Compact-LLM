"""Tiny tensors only: calibration arithmetic, label boundary and frozen replay."""
import json
import math

import pytest
import torch
from torch import nn

from native_armr.src import control_calibration as calibration
from native_armr.src.operators import FrozenSwiGLU
from shared.hashing import canonical_sha256, sha256_file, write_json
from shared.integrity import state_hash


def record(index, native_energy, control_energy, tokens=2):
    return {"scenario_id": f"scenario_{index}", "example_id": f"example_{index}",
            "metrics": {"ordinary": {"native_energy": native_energy, "control_energy": control_energy,
                       "token_count": tokens, "hidden_rms": 1., "native_delta_rms": 2., "control_delta_rms": 4.}}}


def test_registered_config_and_label_free_projection():
    config = calibration.load_config()
    assert config["device"] == "cpu" and config["control"]["seed"] == 7301
    prompt = "A synthetic test prompt."
    row = {"track": "native_armr", "split": "calibration", "variant": "original",
           "example_id": "native_armr_calibration_tiny_0003_original", "scenario_id": "tiny_0003",
           "family": "tiny", "template_id": "tiny.one", "prompt": prompt,
           "prompt_sha256": canonical_sha256(prompt), "alignment_partition": "holdout"}
    class PoisonLabels(dict):
        def __getitem__(self, key):
            if key in {"answer", "scorer", "facts", "question", "latent_sha256"}:
                raise AssertionError("label field accessed")
            return super().__getitem__(key)
    projected = calibration.prompt_projection(PoisonLabels(row))
    assert set(projected) == set(calibration.PROMPT_FIELDS)
    assert calibration.prompt_projection({**row, "answer": "changed", "scorer": object()}) == projected
    for updates in ({"track": "cross_scale"}, {"split": "architecture_dev"},
                    {"split": "confirmatory_dev"}, {"alignment_partition": "fit"},
                    {"prompt": "tampered"}, {"variant": "paraphrase"}):
        with pytest.raises(ValueError):
            calibration.prompt_projection({**row, **updates})


def test_special_and_added_ids_are_excluded_before_energy_reduction():
    ids = torch.tensor([1, 2, 3, 4])
    mask = calibration.ordinary_mask(ids, {1, 4})
    hidden = torch.tensor([[1000., 1000.], [1., 1.], [2., 2.], [1000., 1000.]])
    native = torch.tensor([[99999., 99999.], [2., 2.], [2., 2.], [99999., 99999.]])
    control = native * 2
    metrics = calibration.scenario_energies(hidden, native, control, mask)
    assert metrics["ordinary"]["native_energy"] == 2.5
    assert metrics["ordinary"]["control_energy"] == 10.
    assert metrics["ordinary"]["token_count"] == 2
    assert metrics["all"]["native_energy"] > metrics["ordinary"]["native_energy"]
    assert metrics["special_control"]["token_count"] == 2
    assert metrics["final_prompt"]["token_count"] == 1


def test_denominator_floor_is_per_token_and_empty_special_is_explicit():
    hidden = torch.tensor([[0., 0.], [2., 2.]])
    delta = torch.ones_like(hidden)
    metrics = calibration.scenario_energies(hidden, delta, delta, torch.tensor([True, True]))
    assert metrics["ordinary"]["native_energy"] == pytest.approx((1e8 + .25) / 2)
    assert metrics["special_control"] is None


def test_equal_scenario_weight_is_not_pooled_token_weight():
    fit = calibration.aggregate_partition([record(0, 1., 4., 1), record(1, 9., 4., 100)])
    assert fit["native_energy"] == 5.
    assert fit["ordinary_tokens"] == 101
    assert calibration.fit_alpha(fit) == math.sqrt(5 / 4)
    holdout = calibration.aggregate_partition([record(2, 1e20, .00001)])
    assert calibration.fit_alpha(fit) == math.sqrt(5 / 4)
    assert holdout["native_energy"] != fit["native_energy"]
    with pytest.raises(ValueError, match="unique scenario"):
        calibration.aggregate_partition([record(0, 1, 2), record(0, 3, 4)])


@pytest.mark.parametrize("native,control", [(0., 1.), (1., 0.), (-1., 1.), (1., float("nan")),
                                           (float("inf"), 1.), (1e308, 1e-308)])
def test_invalid_scale_never_clips_or_falls_back(native, control):
    with pytest.raises(ValueError):
        calibration.fit_alpha({"native_energy": native, "control_energy": control})


def test_bad_masks_and_nonfinite_inference_fail():
    with pytest.raises(ValueError, match="ordinary"):
        calibration.ordinary_mask(torch.tensor([1, 2]), {1, 2})
    hidden = torch.ones(3, 4)
    with pytest.raises(ValueError, match="finite CPU FP32"):
        calibration.scenario_energies(hidden, hidden * float("nan"), hidden, torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="aligned"):
        calibration.scenario_energies(hidden, hidden, hidden, torch.ones(2, dtype=torch.bool))


def test_replay_uses_exact_original_positions_mask_rope_and_order():
    calls = []
    class Layer(nn.Module):
        def __init__(self, index, delta):
            super().__init__()
            self.index, self.delta = index, delta
            self.weight = nn.Parameter(torch.tensor(float(delta)), requires_grad=False)
            self.eval()
        def forward(self, hidden, **kwargs):
            assert not torch.is_grad_enabled()
            calls.append((self.index, kwargs))
            return hidden + self.weight
    leading, trailing = Layer(12, 2), Layer(13, 3)
    hidden = torch.randn(1, 5, 4)
    mask, positions, rope = torch.zeros(1, 1, 5, 5), torch.arange(5).unsqueeze(0), (torch.ones(1, 5, 4), torch.zeros(1, 5, 4))
    before = [state_hash(layer) for layer in (leading, trailing)]
    delta = calibration.replay_native_delta(hidden, leading, trailing, attention_mask=mask,
        position_ids=positions, position_embeddings=rope)
    assert torch.allclose(delta, torch.full_like(delta, 5))
    assert [index for index, _ in calls] == [12, 13]
    for _, kwargs in calls:
        assert kwargs["attention_mask"] is mask
        assert kwargs["position_ids"] is positions
        assert kwargs["position_embeddings"] is rope
        assert kwargs["use_cache"] is False
    for layer, digest in zip((leading, trailing), before):
        receipt = calibration._module_receipt(layer, digest)
        assert receipt["backbone_gradients"] == 0 and receipt["trainable_parameters"] == 0


def test_tiny_actual_hf_replay_matches_fixed_block_hooks(monkeypatch):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from shared.hidden_capture import positions_mask
    from armr.recurrence import FixedBlock
    torch.manual_seed(17)
    config = Qwen3Config(vocab_size=16, hidden_size=8, intermediate_size=12,
                        num_hidden_layers=14, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=4, layer_types=["full_attention"] * 14)
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    ids = torch.tensor([[1, 4, 6]])
    raw = []
    handle = model.model.layers[13].register_forward_hook(lambda module, args, out: raw.append(out.clone()))
    with torch.inference_mode():
        model(ids, use_cache=False)
        hidden = raw.pop()
        positions, mask = positions_mask(hidden)
        delta = calibration.replay_native_delta(hidden, model.model.layers[12], model.model.layers[13],
            attention_mask=mask, position_ids=positions, position_embeddings=model.model.rotary_emb(hidden, positions))
        raw.clear()
        with FixedBlock(model, layers=[12, 13], applications=2):
            model(ids, use_cache=False)
        assert torch.equal(raw[-1] - hidden, delta)
    handle.remove()
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    # Run the production streamed prefix against tiny in-memory checkpoint
    # tensors. This verifies raw post-L13 equivalence without a real checkpoint.
    import rineya_cross_scale.streaming_forward as streaming
    class TinyWeights:
        def __init__(self, path):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def close(self):
            pass
        def rows(self, key, indices):
            return model.state_dict()[key][indices].clone()
        def tensor(self, key):
            return model.state_dict()[key].clone()
    monkeypatch.setattr(streaming, "StreamingWeights", TinyWeights)
    receipts = []
    states, _ = calibration._stream_prefix(None, [ids], config, model.model.rotary_emb,
                                           lambda *args: None, receipts)
    assert torch.equal(states[0], hidden)
    assert [r["layer"] for r in receipts] == list(range(14))


def test_seeded_map_is_unchanged_by_measurement_and_alpha_does_not_touch_weights():
    one = FrozenSwiGLU(8, 16, seed=7301, std=.02, rms_epsilon=1e-6)
    two = FrozenSwiGLU(8, 16, seed=7301, std=.02, rms_epsilon=1e-6)
    assert state_hash(one) == state_hash(two)
    before = state_hash(one)
    hidden = torch.randn(4, 8)
    with torch.inference_mode():
        control = one(hidden)
    metrics = calibration.scenario_energies(hidden, control * 3, control, torch.ones(4, dtype=torch.bool))
    fit = calibration.aggregate_partition([{**record(0, 0, 0), "metrics": metrics}])
    assert calibration.fit_alpha(fit) == pytest.approx(3.)
    assert calibration._module_receipt(one, before)["backbone_gradients"] == 0


def test_exclusive_writes_and_artifact_hash_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "ROOT", tmp_path)
    path = tmp_path / "native_armr/evidence/00_control_calibration/run/report.json"
    write_json(path, {"status": "FAILED"})
    with pytest.raises(FileExistsError):
        write_json(path, {"status": "COMPLETE"})
    with pytest.raises(RuntimeError, match="hash/path"):
        calibration.verify_calibration_artifact(path, "0" * 64, require_committed=False)
    with pytest.raises(RuntimeError, match="science/freeze"):
        calibration.verify_calibration_artifact(path, sha256_file(path), require_committed=False)


def test_stream_module_mutation_and_gradients_rejected():
    module = nn.Linear(2, 2).eval().requires_grad_(False)
    before = state_hash(module)
    with torch.no_grad():
        module.weight.add_(1)
    with pytest.raises(RuntimeError, match="mutated"):
        calibration._module_receipt(module, before)
    module.weight.grad = torch.ones_like(module.weight)
    with pytest.raises(RuntimeError, match="not frozen"):
        calibration._module_receipt(module, state_hash(module))


@pytest.fixture
def synthetic_artifact(tmp_path, monkeypatch):
    """Build 128 scalar-only fake receipts to exercise the complete verifier."""
    config = calibration.load_config()
    monkeypatch.setattr(calibration, "ROOT", tmp_path)
    monkeypatch.setattr(calibration, "REPO", tmp_path)
    config_path = tmp_path / "native_armr/configs/control_calibration_v2_3.json"
    write_json(config_path, config)
    source_path = tmp_path / "source.py"
    source_path.write_text("# tiny synthetic fixture\n", encoding="utf-8")
    sources = {"source.py": sha256_file(source_path)}
    monkeypatch.setattr(calibration, "calibration_sources", lambda: sources)
    monkeypatch.setattr(calibration, "load_config", lambda path: config)
    data = {}
    for kind in ("dataset_manifest", "calibration_data"):
        target = tmp_path / "data" / (kind + ".json")
        write_json(target, {"fixture": True})
        data[kind + "_path"] = target.relative_to(tmp_path).as_posix()
        data[kind + "_sha256"] = sha256_file(target)
    rows, cases = [], []
    for index in range(128):
        row = {"example_id": f"native_armr_calibration_tiny_{index:04d}_original",
               "scenario_id": f"native_armr_calibration_tiny_{index:04d}", "family": "tiny",
               "template_id": "tiny.synthetic", "prompt": "Tiny prompt " + str(index),
               "alignment_partition": "holdout" if index % 4 == 3 else "fit"}
        row["prompt_sha256"] = canonical_sha256(row["prompt"])
        rows.append(row)
        case = {key: value for key, value in row.items() if key != "prompt"}
        case.update(token_ids=[1, 3], full_prompt_tokens=2, retained_prompt_tokens=2,
                    metrics=record(index, 4., 16., 1)["metrics"])
        cases.append(case)
    monkeypatch.setattr(calibration, "calibration_prompts", lambda path: (rows, data))
    contract = {"config": config, "source_files": sources,
                "calibration_config_path": config_path.relative_to(tmp_path).as_posix(),
                "calibration_config_sha256": sha256_file(config_path), **data,
                **{partition + "_example_ids": [r["example_id"] for r in rows if r["alignment_partition"] == partition]
                   for partition in ("fit", "holdout")}}
    registration = {"contract": contract, "status": "FROZEN", "registration_sha256": canonical_sha256(contract)}
    registration_path = tmp_path / "native_armr/evidence/00_control_calibration_registration/fake/registration.json"
    write_json(registration_path, registration)
    directory = tmp_path / "native_armr/evidence/00_control_calibration/fake"
    for case in cases:
        write_json(directory / "cases" / (canonical_sha256(case["example_id"]) + ".json"), case)
    fit, holdout = [calibration.aggregate_partition([c for c in cases if c["alignment_partition"] == partition])
                    for partition in ("fit", "holdout")]
    alpha = calibration.fit_alpha(fit)
    write_json(directory / "fit_scale.json", {"alpha": alpha, "fit": fit, "formula": calibration.FORMULA})
    for partition in (fit, holdout):
        partition["scaled_control_energy"] = alpha ** 2 * partition["control_energy"]
    frozen = {"state_sha256_before": "a" * 64, "state_sha256_after": "a" * 64,
              "trainable_parameters": 0, "backbone_gradients": 0, "parameters": 2,
              "buffer_sha256_before": {}, "buffer_sha256_after": {}}
    report = {"schema_version": calibration.SCHEMA, "status": "COMPLETE", "track": "native_armr",
              "control": calibration.CONTROL, "formula": calibration.FORMULA, "label_free": True,
              "answer_labels_used": False, "generated_tokens": 0, "holdout_selects_alpha": False,
              "alpha_search": False, "alpha_clipping": False, "trainable_parameters": 0,
              "backbone_gradients": 0, "canonical_unchanged": True, "device": "cpu", "dtype": "float32",
              "threads": 4, "use_cache": False, "alpha": alpha, "fit": fit, "holdout": holdout,
              "registration_path": registration_path.relative_to(tmp_path).as_posix(),
              "registration_sha256": registration["registration_sha256"],
              **{key: value for key, value in contract.items() if key not in {"config", "fit_example_ids", "holdout_example_ids"}},
              "case_files": {p.name: sha256_file(p) for p in sorted((directory / "cases").glob("*.json"))},
              "special_control_token_ids": [1],
              "layer_state_receipts": [{"stage": stage, "layer": index, **frozen} for stage, index in
                  [("ordinary_prefix", i) for i in range(14)] + [("replay", 12), ("replay", 13)]],
              "rotary_state_receipt": frozen, "control_state_receipt": {**frozen, "parameters": 31457280},
              "control_state_sha256": "a" * 64,
              "checkpoint_files_before": {"model.safetensors": config["checkpoint_sha256"]},
              "checkpoint_files_after": {"model.safetensors": config["checkpoint_sha256"]},
              "fit_scale_sha256": sha256_file(directory / "fit_scale.json")}
    path = directory / "report.json"
    write_json(path, report)
    write_json(directory / "started_manifest.json", {"status": "STARTED"})
    write_json(directory / "run_manifest.json", {"status": "COMPLETE", "report_sha256": sha256_file(path),
                                                "registration_sha256": registration["registration_sha256"]})
    return path, report


def test_complete_artifact_reconstructs_and_checks_commit_boundary(synthetic_artifact, monkeypatch):
    path, report = synthetic_artifact
    checked = []
    monkeypatch.setattr(calibration, "_require_committed", lambda candidate: checked.append(candidate))
    assert calibration.verify_calibration_artifact(path, sha256_file(path))["alpha"] == .5
    assert len(checked) == 137  # 128 cases, 4 bound inputs, 5 run/registration files.


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(alpha=.75), "alpha differs"),
    (lambda r: r["holdout"].update(native_energy=1.), "aggregation mismatch"),
    (lambda r: r["layer_state_receipts"][0].update(backbone_gradients=1), "has gradients"),
    (lambda r: r["checkpoint_files_after"].update(extra="changed"), "checkpoint mutation"),
    (lambda r: r.update(holdout_selects_alpha=True), "science/freeze"),
])
def test_artifact_rejects_result_and_state_tampering(synthetic_artifact, mutation, match):
    path, report = synthetic_artifact
    mutation(report)
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match=match):
        calibration.verify_calibration_artifact(path, sha256_file(path), require_committed=False)
