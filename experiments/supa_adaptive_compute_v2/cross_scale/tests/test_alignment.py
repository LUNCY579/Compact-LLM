"""Tiny tensors and actual tiny HF decoders: mechanics, never efficacy evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

from shared.hashing import canonical_sha256
from shared.legacy import legacy_graft
from cross_scale.src.bridge_fit import cross_scenario_permutation, fit_initial, fit_delta_bridge, weighted_ridge
from cross_scale.src.delta_alignment import donor_delta, normalized_loss, stratified_metrics, token_masks, token_metrics
from cross_scale.src.depth_pair_screen import OBJECTIVES, depth_pairs, run_pair, select_matrix, validate_bundle
from cross_scale.src.graft import CrossScaleOperator


@pytest.fixture
def config():
    value = json.loads((Path(__file__).resolve().parents[1]/"configs/alignment_v2.json").read_text())
    value["delta_steps"] = 3  # Only internal toy tests shorten fitting; no registration/run is created.
    return value


@pytest.fixture
def tiny_donor():
    torch.manual_seed(21)
    architecture = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=36, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, layer_types=["full_attention"]*36)
    architecture._attn_implementation = "eager"
    layer = Qwen3DecoderLayer(architecture, 12).eval().requires_grad_(False)
    rope = Qwen3RotaryEmbedding(architecture).eval().requires_grad_(False)
    return layer, rope, architecture


def toy_prompts(layer, rope):
    generator = torch.Generator().manual_seed(26)
    rows = []
    for index in range(4):
        small = torch.randn(4, 8, generator=generator)
        big = torch.randn(4, 16, generator=generator)
        with torch.no_grad():
            delta = donor_delta(layer, rope, big.unsqueeze(0))[0]
        rows.append({"small": small, "big": big, "delta": delta,
                     "ids": torch.tensor([[0, 1, 2, 3]]), "scenario_id": f"s{index}"})
    return rows


def toy_bundle(config, prompts):
    return {"track": "cross_scale", "role": "calibration",
        "rows": [{"example_id": f"e{i}", "scenario_id": f"s{i}", "family": "toy", "prompt": f"toy {i}"} for i in range(4)],
        "train_prompt_mask": [True, True, False, False], "heldout_scenarios": ["s2", "s3"],
        "input_ids": [p["ids"] for p in prompts], "special_ids": [0],
        "small": {i: [p["small"].clone() for p in prompts] for i in config["supa_layers"]},
        "donor": {depth: [(p["big"] if depth == 11 else p["big"]+p["delta"]).clone() for p in prompts]
                  for j in config["donor_layers"] for depth in (j-1, j)}}


def test_token_metrics_do_not_hide_ordinary_error():
    target = torch.tensor([[1., 0.], [100., 0.]])
    predicted = torch.tensor([[0., 0.], [100., 0.]])
    metrics = token_metrics(predicted, target)
    assert metrics["mean_token_relative_mse"] == pytest.approx(.5)
    assert metrics["global_relative_mse"] == pytest.approx(1/10001)
    assert metrics["mean_token_cosine"] == pytest.approx(.5)
    assert normalized_loss(predicted, target, token_normalized=True) == pytest.approx(.5)
    assert normalized_loss(predicted, target, token_normalized=False) == pytest.approx(1/10001)
    masks = token_masks([torch.tensor([[2, 9]])], {9})
    report = stratified_metrics(predicted, target, masks)
    assert report["ordinary_tokens"]["mean_token_relative_mse"] == 1
    assert report["special_tokens"]["mean_token_relative_mse"] == 0
    assert report["final_prompt_token"]["tokens"] == 1


def test_empty_special_stratum_is_explicit_and_final_per_prompt():
    masks = token_masks([torch.tensor([[1, 2]]), torch.tensor([[3]])], [])
    assert masks["final_prompt_token"].tolist() == [False, True, True]
    report = stratified_metrics(torch.ones(3, 2), torch.ones(3, 2), masks)
    assert report["special_tokens"] == {"tokens": 0}
    with pytest.raises(ValueError):
        token_masks([torch.tensor([], dtype=torch.long)], [])


def test_weighted_ridge_matches_independent_weighted_normal_equations():
    x = torch.tensor([[1., -1.], [2., 2.], [-2., 3.], [4., -1.]])
    y = torch.tensor([[1.], [2.], [1.], [50.]])
    weight, bias = weighted_ridge(x, y, .01)
    a, b = x.double(), y.double()
    w = b.square().mean(-1).clamp_min(1e-8).reciprocal()
    w /= w.mean()
    xm, ym = (a*w[:, None]).sum(0)/w.sum(), (b*w[:, None]).sum(0)/w.sum()
    centered = a-xm
    gram = centered.T@torch.diag(w)@centered
    expected = torch.linalg.solve(gram+.01*torch.trace(gram)/2*torch.eye(2), centered.T@torch.diag(w)@(b-ym))
    assert torch.allclose(weight.double(), expected)
    assert torch.allclose(bias.double(), ym-xm@expected)


@pytest.mark.parametrize("counts", [(2, 2), (3, 2, 1), (4, 1, 1, 2), (1, 1, 1)])
def test_shuffled_pair_control_preserves_all_tokens_and_deranges_scenarios(counts):
    scenarios = [str(i) for i, count in enumerate(counts) for _ in range(count)]
    for seed in range(8):
        perm = cross_scenario_permutation(scenarios, seed)
        assert sorted(perm.tolist()) == list(range(len(scenarios)))
        assert all(scenarios[i] != scenarios[j] for i, j in enumerate(perm))
        assert torch.equal(perm, cross_scenario_permutation(scenarios, seed))
    with pytest.raises(ValueError):
        cross_scenario_permutation(["one"]*4+["two"], 0)


def test_random_bridge_dimensions_reproducibility_and_rng_isolation(config):
    small, big = torch.randn(12, 8), torch.randn(12, 16)
    state = torch.random.get_rng_state().clone()
    first = fit_initial(small, big, "BR", config, ["a"]*6+["b"]*6)
    second = fit_initial(small, big, "BR", config, ["a"]*6+["b"]*6)
    assert first.up.shape == (8, 16) and first.down.shape == (16, 8)
    assert not list(first.parameters())
    assert all(torch.equal(a, b) for a, b in zip(first.buffers(), second.buffers()))
    assert torch.equal(state, torch.random.get_rng_state())
    with pytest.raises(ValueError):
        fit_initial(small, big[:3], "B0", config, ["a"]*12)


@pytest.mark.parametrize("normalized", [False, True])
def test_delta_optimization_updates_only_bridges_and_preserves_donor(config, tiny_donor, normalized):
    from armr.provenance import state_hash
    layer, rope, _ = tiny_donor
    prompts = toy_prompts(layer, rope)
    small, big = torch.cat([p["small"] for p in prompts[:2]]), torch.cat([p["big"] for p in prompts[:2]])
    initial = fit_initial(small, big, "B1" if normalized else "B0", config, ["a"]*4+["b"]*4)
    before = state_hash(layer)
    result, history = fit_delta_bridge(initial, layer, rope, prompts[:2], config, token_normalized=normalized)
    assert len(history) == config["delta_steps"]
    assert not torch.equal(result.up, initial.up)
    assert torch.equal(result.down, initial.down)
    assert state_hash(layer) == before
    assert all(not p.requires_grad and p.grad is None for p in layer.parameters())
    assert all(p["small"].grad is None and p["big"].grad is None for p in prompts)
    next(layer.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        fit_delta_bridge(initial, layer, rope, prompts[:2], config, token_normalized=normalized)


def test_pair_executes_all_six_controls_on_heldout_full_prompts(config, tiny_donor):
    layer, rope, _ = tiny_donor
    bundle = toy_bundle(config, toy_prompts(layer, rope))
    result, artifacts = run_pair(bundle, config, 8, 12, layer, rope)
    assert list(result["objectives"]) == list(OBJECTIVES) == list(artifacts)
    assert result["native_replay_bit_identical"]
    for objective, values in result["objectives"].items():
        assert values["optimizer_steps"] == (3 if objective in {"B2", "B3"} else 0)
        assert values["heldout"]["delta"]["ordinary_tokens"]["tokens"] == 6
        assert values["heldout"]["delta"]["final_prompt_token"]["tokens"] == 2
    with pytest.raises(ValueError, match="registered depth"):
        run_pair(bundle, config, 8, 16, layer, rope)
    bundle["donor"][12][0][0, 0] += .1
    with pytest.raises(ValueError, match="replay"):
        run_pair(bundle, config, 8, 12, layer, rope)


@pytest.mark.parametrize("mutation", ["ids", "duplicate", "scenario", "nan", "width", "trainable", "answers", "depth"])
def test_bundle_rejects_leakage_bad_shapes_and_gradients(config, tiny_donor, mutation):
    layer, rope, _ = tiny_donor
    bundle = toy_bundle(config, toy_prompts(layer, rope))
    if mutation == "ids": bundle["input_ids"].pop()
    elif mutation == "duplicate": bundle["rows"][1]["example_id"] = "e0"
    elif mutation == "scenario": bundle["rows"][2]["scenario_id"] = "s0"
    elif mutation == "nan": bundle["small"][8][0][0, 0] = float("nan")
    elif mutation == "width": bundle["small"][8][0] = torch.ones(4, 7)
    elif mutation == "trainable": bundle["small"][8][0].requires_grad_(True)
    elif mutation == "answers": bundle["rows"][0]["answer"] = "leak"
    else: del bundle["donor"][11]
    with pytest.raises(ValueError): validate_bundle(bundle, config)


def matrix_records(config):
    metric = {"tokens": 8, "mean_token_cosine": .4, "mean_token_relative_mse": 2.,
              "global_relative_mse": 2., "predicted_rms": 1., "target_rms": 1., "residual_magnitude_ratio": 1.}
    strata = {name: deepcopy(metric) for name in ("all_tokens", "ordinary_tokens", "special_tokens", "final_prompt_token")}
    strata["ordinary_tokens"]["tokens"] = 6
    strata["special_tokens"]["tokens"] = 2
    strata["final_prompt_token"]["tokens"] = 2
    return [{"supa_layer": i, "donor_layer": j, "config_sha256": canonical_sha256(config),
             "donor_state_before": "frozen", "donor_state_after": "frozen", "native_replay_bit_identical": True,
             "trainable_backbone_parameters": 0, "backbone_gradient_tensors": 0,
             "objectives": {o: {"heldout": {kind: deepcopy(strata) for kind in ("up", "down", "delta")}} for o in OBJECTIVES}}
            for i, j in depth_pairs(config)]


def test_full_matrix_coverage_gates_and_stable_tie_break(config):
    records = matrix_records(config)
    assert len(records)*len(OBJECTIVES) == 96
    assert select_matrix(records, config)["status"] == "MECHANISM_NO_GO"
    for row in records:
        for name in ("B1", "B3"):
            for metric in row["objectives"][name]["heldout"]["delta"].values():
                metric.update(mean_token_cosine=.7, mean_token_relative_mse=1.)
    decision = select_matrix(list(reversed(records)), config)
    assert decision["status"] == "MECHANISM_GO"
    assert [(r["supa_layer"], r["donor_layer"], r["objective"]) for r in decision["selected"]] == [(8, 12, "B1")]
    assert decision["large_generation_licensed"] is False
    for broken in (records[:-1], records+[records[0]]):
        with pytest.raises(ValueError): select_matrix(broken, config)
    records[0]["objectives"].pop("BS")
    with pytest.raises(ValueError): select_matrix(records, config)


@pytest.mark.parametrize("mutation", ["nan", "negative_error", "gradients", "native_replay"])
def test_matrix_cannot_hide_numerical_or_integrity_failure(config, mutation):
    records = matrix_records(config)
    if mutation == "nan": records[0]["objectives"]["BR"]["heldout"]["delta"]["ordinary_tokens"]["mean_token_cosine"] = float("nan")
    elif mutation == "negative_error": records[0]["objectives"]["B0"]["heldout"]["up"]["ordinary_tokens"]["global_relative_mse"] = -1
    elif mutation == "gradients": records[0]["backbone_gradient_tensors"] = 1
    else: records[0]["native_replay_bit_identical"] = False
    with pytest.raises(ValueError): select_matrix(records, config)


def test_residual_delta_bias_cancellation_gate_zero_restore_and_frozen_supa(tiny_donor):
    layer, rope, architecture = tiny_donor
    legacy = legacy_graft()
    bridge = legacy.Bridges(torch.eye(16), torch.zeros(16), torch.eye(16), torch.full((16,), 300.))
    hidden = torch.randn(1, 4, 16)
    expert = legacy.ResidualExpert(layer, architecture, bridge, .2)
    positions, mask = legacy.positions_mask(hidden)
    with torch.no_grad():
        expected = hidden+.2*donor_delta(layer, expert.rope, bridge.project(hidden))
        assert torch.equal(expert(hidden, attention_mask=mask, position_ids=positions), expected)
        expert.gate = 0
        assert expert(hidden, attention_mask=None, position_ids=None) is hidden
    architecture.num_hidden_layers = 28
    architecture.layer_types = ["full_attention"]*28
    model = Qwen3ForCausalLM(architecture).eval().requires_grad_(False)
    ids = torch.tensor([[1, 4, 5]])
    operator = CrossScaleOperator(expert, 12)
    with torch.inference_mode():
        base = model(ids, use_cache=False).logits
        with operator.context(model):
            assert torch.equal(model(ids, use_cache=False).logits, base)
        expert.gate = .1
        with operator.context(model):
            assert not torch.equal(model(ids, use_cache=False).logits, base)
        assert torch.equal(model(ids, use_cache=False).logits, base)
    assert not model._forward_pre_hooks
    next(model.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"): operator.context(model)


def test_donor_guard_rejects_rms_and_deadline(config, tiny_donor):
    layer, rope, _ = tiny_donor
    hidden = torch.randn(1, 4, 16)
    config["hidden_rms_ratio_bounds"] = [2., 3.]
    with pytest.raises(FloatingPointError, match="RMS"): donor_delta(layer, rope, hidden, config=config)
    config["max_prompt_seconds"] = 1e-12
    with pytest.raises(TimeoutError): donor_delta(layer, rope, hidden, config=config)


def test_shuffled_control_uses_same_correspondence_for_both_maps(config):
    small, big = torch.randn(8, 3), torch.randn(8, 5)
    scenarios = ["one"]*4+["two"]*4
    perm = cross_scenario_permutation(scenarios, config["seed"])
    expected = legacy_graft().Bridges.fit(small, big[perm], config["ridge_relative"])
    actual = fit_initial(small, big, "BS", config, scenarios)
    assert all(torch.equal(a, b) for a, b in zip(actual.buffers(), expected.buffers()))


@pytest.mark.parametrize("check", ["absolute", "baseline", "mse", "null", "all_tokens", "scale"])
def test_each_prespecified_mechanism_gate_can_veto_selection(config, check):
    records = matrix_records(config)
    target = records[0]["objectives"]["B3"]["heldout"]["delta"]
    for metric in target.values():
        metric.update(mean_token_cosine=.7, mean_token_relative_mse=1.)
    assert select_matrix(records, config)["selected"]
    ordinary = target["ordinary_tokens"]
    if check == "absolute": ordinary["mean_token_cosine"] = .59
    elif check == "baseline": records[0]["objectives"]["B0"]["heldout"]["delta"]["ordinary_tokens"]["mean_token_cosine"] = .65
    elif check == "mse": ordinary["mean_token_relative_mse"] = 1.61
    elif check == "null": records[0]["objectives"]["BS"]["heldout"]["delta"]["ordinary_tokens"]["mean_token_cosine"] = .61
    elif check == "all_tokens": target["all_tokens"]["mean_token_cosine"] = .49
    else: ordinary["residual_magnitude_ratio"] = 4.01
    assert select_matrix(records, config)["status"] == "MECHANISM_NO_GO"


@pytest.mark.parametrize("failure", [None, "source", "oversize", "stream_mismatch"])
def test_capture_qualification_source_integrity_and_weights_only_roundtrip(config, tiny_donor, tmp_path, monkeypatch, failure):
    from shared import hidden_capture, model_loading
    from armr import provenance
    from cross_scale.src import donor_layers
    _, _, architecture = tiny_donor
    model = Qwen3ForCausalLM(architecture).eval().requires_grad_(False)

    class Tokenizer:
        all_special_ids = [0]
        def get_vocab(self): return {str(i): i for i in range(32)}
        def get_added_vocab(self): return {}
        def apply_chat_template(self, *args, **kwargs): return "toy prompt"
        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.ones((1, 257 if failure == "oversize" else 3), dtype=torch.long)}

    tokenizer = Tokenizer()
    monkeypatch.setattr(model_loading, "load_supa", lambda value: (model, tokenizer,
        {"model_path": str(tmp_path), "torch": torch.__version__}))
    monkeypatch.setattr(provenance, "checkpoint_files", lambda path: {"model": "frozen"})
    monkeypatch.setattr(donor_layers, "donor_identity", lambda value: (tmp_path, {"checkpoint": "frozen"}))
    monkeypatch.setattr(donor_layers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    calls = []
    def sources(track):
        calls.append(track)
        return {"source.py": "changed" if failure == "source" and len(calls) > 1 else "frozen"}
    monkeypatch.setattr(donor_layers, "source_files", sources)

    def stream(path, encoded, layers, limit, progress=None):
        captured = {i: [] for i in layers}
        for ids in encoded:
            handles = [model.model.layers[i].register_forward_hook(
                lambda module, args, out, index=i: captured[index].append(out[0].clone())) for i in layers]
            try:
                with torch.no_grad(): model(ids, use_cache=False)
            finally:
                for handle in handles: handle.remove()
        if failure == "stream_mismatch": captured[layers[0]][0] += 1
        if progress: progress(max(layers), max(layers))
        return captured
    monkeypatch.setattr(hidden_capture, "capture_native", stream)
    rows = [{"example_id": f"e{i}", "scenario_id": f"s{i}", "family": "toy", "prompt": f"toy {i}",
             "answer": "must not enter bundle", "track": "cross_scale", "split": "calibration",
             "alignment_partition": "fit" if i == 0 else "holdout"} for i in range(2)]
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            donor_layers.capture_bundle(rows, config)
    else:
        result = donor_layers.capture_bundle(rows, config)
        assert result["sources"] == {"source.py": "frozen"}
        assert result["canonical_fixture_bit_identical"] is True
        assert type(result["supa_identity"]["torch"]) is str
        assert all("answer" not in row for row in result["rows"])
        path = tmp_path/"tiny.pt"
        torch.save(result, path)
        assert torch.load(path, weights_only=True)["source_sha256"] == canonical_sha256(result["sources"])
