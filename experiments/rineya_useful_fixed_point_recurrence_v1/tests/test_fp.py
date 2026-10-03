from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.rineya_useful_fixed_point_recurrence_v1.core import (
    InjectionInterface, finite_recurrence, orthogonal_injection, kv_geometry)
from experiments.rineya_useful_fixed_point_recurrence_v1.diagnostics import trajectory_metrics, logit_metrics
from experiments.rineya_useful_fixed_point_recurrence_v1.host import FixedPointHost, HostError
from experiments.rineya_useful_fixed_point_recurrence_v1.protocol import (
    admit_splits, paired_report, preflight, new_run, write_json, sha256, verify_bundle)
from experiments.rineya_useful_fixed_point_recurrence_v1.training import fit_interface, TrainingBudget
from experiments.rineya_useful_fixed_point_recurrence_v1.depth import DepthPrior


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(17)
    torch.set_num_threads(1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_projection_formula_dtype_and_zero(dtype):
    v = torch.tensor([[3., 4.], [2., -1.]], dtype=dtype)
    q = torch.tensor([[1., 0.], [0., 0.]], dtype=dtype)
    out = orthogonal_injection(v, q)
    assert out.dtype == dtype
    assert torch.allclose(out[0], torch.tensor([1., 4.], dtype=dtype), atol=1e-6)
    assert torch.equal(out[1], v[1])


def test_projection_gradcheck():
    v, q = [torch.randn(2, 3, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    assert torch.autograd.gradcheck(orthogonal_injection, (v, q))


@pytest.mark.parametrize("eps", [0, -1, float("nan"), float("inf")])
def test_projection_invalid_epsilon(eps):
    with pytest.raises(ValueError):
        orthogonal_injection(torch.ones(2), torch.ones(2), eps)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_rejected(bad):
    with pytest.raises(ValueError):
        orthogonal_injection(torch.tensor([bad]), torch.ones(1))


def test_projection_is_per_token():
    v, q = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    together = orthogonal_injection(v, q)
    for i in range(2):
        for j in range(3):
            assert torch.equal(together[i, j], orthogonal_injection(v[i, j], q[i, j]))


def test_matched_interfaces_zero_start_and_opening():
    add, ortho = InjectionInterface(8, 2, "additive"), InjectionInterface(8, 2, "orthogonal")
    ortho.load_state_dict(add.state_dict())
    h, x = torch.randn(3, 8), torch.randn(3, 8)
    for module in (add, ortho):
        assert torch.equal(module(h, x), h)
        module(h, x).square().sum().backward()
        assert module.blend.grad.abs() > 0
        assert torch.equal(module.up.weight.grad, torch.zeros_like(module.up.weight))
    assert sum(p.numel() for p in add.parameters()) == sum(p.numel() for p in ortho.parameters())
    with torch.no_grad():
        add.blend.fill_(.4)
        ortho.blend.fill_(.4)
    assert not torch.equal(add(h, x), ortho(h, x))


@pytest.mark.parametrize("width,rank,mode", [(0, 1, "additive"), (3, 4, "additive"), (4, 2, "bad")])
def test_interface_validation(width, rank, mode):
    with pytest.raises(ValueError):
        InjectionInterface(width, rank, mode)


def test_tbptt_keeps_live_anchor_gradient():
    anchor = torch.ones(2, 3, requires_grad=True)
    module = InjectionInterface(3, 1, "additive")
    with torch.no_grad():
        module.blend.fill_(.5)
    output = finite_recurrence(lambda h: .8*h, torch.zeros_like(anchor), anchor,
                               steps=4, grad_window=1, interface=module)
    output.sum().backward()
    assert anchor.grad.abs().sum() > 0
    assert module.blend.grad.abs() > 0


def test_full_window_gradient_equivalence():
    def get(window):
        weight = torch.tensor(.8, requires_grad=True)
        y = finite_recurrence(lambda h: weight*h, torch.ones(2, 3), torch.zeros(2, 3),
                              steps=3, grad_window=window)
        y.sum().backward()
        return y.detach(), weight.grad
    a, b = get(None), get(3)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert get(1)[1] != a[1]


@pytest.mark.parametrize("steps,window,alpha", [(-1, None, 1), (True, None, 1), (1, 0, 1), (1, None, 3)])
def test_recurrence_bounds(steps, window, alpha):
    with pytest.raises(ValueError):
        finite_recurrence(lambda h: h, torch.ones(2), torch.ones(2), steps=steps, grad_window=window, alpha=alpha)


def test_kv_honest_geometry():
    assert kv_geometry(4, 2, 5)["full_banks"] == 12
    value = kv_geometry(28, 2, 2)
    assert value["full_banks"] == 30 and value["terminal_banks"] == 28
    assert value["kv_fraction_saved"] == pytest.approx(1/15)
    assert value["measured"] is False


@pytest.mark.parametrize("shape", [(2, 3), (1, 2, 3)])
def test_gate_collapse_not_correctness(shape):
    h = torch.ones(shape)
    r = trajectory_metrics(h, h, 2*h, h, h, h)
    assert r["stationary_fraction"] == 1
    assert r["suppressed_update_fraction"] == 1
    assert r["raw_update_norm"] > 0 and r["applied_update_norm"] == 0
    assert r["correctness_inferred"] is False


def test_masked_diagnostics_and_logits():
    h = torch.ones(1, 2, 3)
    after = h.clone()
    after[:, 1] *= 2
    mask = torch.tensor([[True, False]])
    r = trajectory_metrics(h, h, after, after, h, h, mask)
    assert r["tokens"] == 1 and r["relative_state_change"] == 0
    with pytest.raises(ValueError):
        trajectory_metrics(h, h, h, h, h, h, torch.zeros_like(mask))
    l = torch.randn(2, 5)
    assert logit_metrics(l, l)["kl_reference_to_candidate"] == 0
    assert logit_metrics(l, -l)["kl_reference_to_candidate"] >= 0


class Layer(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
        self.weight = nn.Parameter(torch.eye(4))
        self.calls = []
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        self.calls.append((self.index, None if position_ids is None else position_ids.clone()))
        return hidden_states + .01*torch.tanh(hidden_states @ self.weight + self.index/28)


class Qwen3AbstainForCausalLM(nn.Module):
    """Shape-only CPU fixture: not a checkpoint or real Qwen implementation."""
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Layer(i) for i in range(28)])
        self.embedding = nn.Embedding(8, 4)
        self.config = SimpleNamespace(layer_types=["full_attention"]*28)
    def forward(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None):
        h = self.embedding(input_ids)
        for i, layer in enumerate(self.model.layers):
            # Distinct signatures expose accidental reuse of last-layer positions.
            pos = torch.arange(input_ids.shape[1]).unsqueeze(0) + i*100
            h = layer(h, attention_mask=attention_mask, position_ids=pos, use_cache=use_cache)
        return h


@pytest.fixture
def model():
    return Qwen3AbstainForCausalLM().eval().requires_grad_(False)


@pytest.mark.parametrize("applications,alpha", [(1, 1.), (2, 0.)])
def test_host_identity_and_restoration(model, applications, alpha):
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        baseline = model(ids)
        with FixedPointHost(model, applications=applications, alpha=alpha) as host:
            result = model(ids, use_cache=False)
            assert host.calls == 2*applications
        restored = model(ids)
    assert torch.equal(baseline, result) and torch.equal(baseline, restored)
    assert not model._forward_pre_hooks
    assert "forward" not in model.model.layers[12].__dict__


def test_host_complete_block_and_per_layer_positions(model):
    ids = torch.tensor([[1, 2]])
    with FixedPointHost(model) as host, torch.no_grad():
        out = model(ids, attention_mask=torch.ones_like(ids), use_cache=False)
    assert out.shape == (1, 2, 4) and len(host.records) == 1
    for i in (12, 13):
        calls = model.model.layers[i].calls
        assert len(calls) == 2 and torch.equal(calls[0][1], calls[1][1])
    assert len(model.model.layers[0].calls) == 1


@pytest.mark.parametrize("kwargs", [{}, {"use_cache": True}, {"use_cache": False, "past_key_values": object()}])
def test_host_rejects_cache_and_implicit_cache(model, kwargs):
    with pytest.raises(HostError), FixedPointHost(model), torch.no_grad():
        model(torch.tensor([[1]]), **kwargs)
    assert not model._forward_pre_hooks


def test_host_blocks_gradients_and_overlap(model):
    with pytest.raises(HostError), FixedPointHost(model):
        model(torch.tensor([[1]]), use_cache=False)
    with FixedPointHost(model):
        with pytest.raises(HostError):
            with FixedPointHost(model):
                pass
    assert not model._forward_pre_hooks


def test_host_zero_injection_matches_raw(model):
    ids = torch.tensor([[1, 2]])
    with torch.no_grad():
        with FixedPointHost(model):
            raw = model(ids, use_cache=False)
        for mode in ("additive", "orthogonal"):
            with FixedPointHost(model, interface=InjectionInterface(4, 2, mode)):
                assert torch.equal(raw, model(ids, use_cache=False))


def test_host_gradients_to_external_interface_only(model):
    interface = InjectionInterface(4, 2)
    with FixedPointHost(model, interface=interface, training_interface=True):
        loss = model(torch.tensor([[1, 2]]), use_cache=False).square().mean()
        loss.backward()
    assert interface.blend.grad.abs() > 0
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("applications", [0, 3, True, 2.0])
def test_host_no_deeper_native_lane(model, applications):
    with pytest.raises(ValueError):
        FixedPointHost(model, applications=applications)


def test_host_existing_forward_and_failure_restore(model):
    layer = model.model.layers[13]
    original = layer.forward
    def broken(*args, **kwargs):
        raise RuntimeError("fixture failure")
    layer.forward = broken
    with pytest.raises(RuntimeError), FixedPointHost(model), torch.no_grad():
        model(torch.tensor([[1]]), use_cache=False)
    assert layer.forward is broken and not model._forward_pre_hooks
    layer.forward = original


def test_training_freezes_base_and_has_matched_budget():
    base = nn.Linear(4, 4).eval().requires_grad_(False)
    before = copy.deepcopy(base.state_dict())
    h, q, target = torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)
    module = InjectionInterface(4, 2, "additive")
    def closure(step):
        return (base(module(h, q))-target).square().mean(), 3
    report = fit_interface(module, base, closure, train_split="TRAIN", budget=TrainingBudget(max_steps=3))
    assert report["optimizer_steps"] == 3 and report["supervised_tokens"] == 9
    assert all(torch.equal(before[k], v) for k, v in base.state_dict().items())
    assert report["history"][-1]["blend"] != 0
    for split in ("DEV", "RETENTION", "TEST", "FINAL"):
        with pytest.raises(ValueError):
            fit_interface(module, base, closure, train_split=split, budget=TrainingBudget())


def test_training_detects_dead_gradient_and_nonfinite():
    module = InjectionInterface(4, 2)
    base = nn.Linear(4, 4).eval().requires_grad_(False)
    with pytest.raises(RuntimeError, match="nonzero"):
        fit_interface(module, base, lambda _: (module.blend*0, 1), train_split="TRAIN", budget=TrainingBudget(max_steps=1))
    with pytest.raises(FloatingPointError):
        fit_interface(module, base, lambda _: (module.blend*float("nan"), 1), train_split="TRAIN", budget=TrainingBudget(max_steps=1))


def test_depth_prior_resume_and_gradient():
    prior = DepthPrior()
    assert prior.sample() in (1, 2, 3)
    with pytest.raises(RuntimeError):
        prior.sample()
    prior.objective(2.).backward()
    assert torch.isfinite(prior.logits.grad).all()
    resumed = DepthPrior()
    resumed.load_state_dict(copy.deepcopy(prior.state_dict()))
    assert resumed.sample() == prior.sample()
    with pytest.raises(ValueError):
        DepthPrior(support=(1, 2), target_mean=1.5)


def row(key, split, prompt, cluster=None):
    return dict(case_id=key, split=split, prompt=prompt, cluster_id=cluster or key)


def test_split_dedup_and_cluster_protection():
    assert len(admit_splits([row("a", "TRAIN", "one"), row("b", "DEV", "two")])) == 2
    for data in ([row("a", "TRAIN", "One"), row("b", "DEV", " one ")],
                 [row("a", "TRAIN", "one", "c"), row("b", "DEV", "two", "c")],
                 [row("a", "FINAL", "one")]):
        with pytest.raises(ValueError):
            admit_splits(data)


def outcomes(seed=17):
    return [dict(case_id=str(i), cluster_id=str(i), seed=seed, arm="orthogonal", split="DEV",
                 baseline_correct=a, candidate_correct=b)
            for i, (a, b) in enumerate([(False, True), (False, True), (True, False), (True, True)])]


def test_paired_cluster_report_and_no_seed_pooling():
    result = paired_report(outcomes(), bootstrap_samples=100)
    assert result["net_correction_rate"] == .25
    assert result["corrections"] == 2 and result["damages"] == 1
    mixed = outcomes(); mixed[0]["seed"] = 29
    with pytest.raises(ValueError):
        paired_report(mixed)
    invalid = outcomes(); invalid[0]["candidate_correct"] = 1
    with pytest.raises(ValueError):
        paired_report(invalid)
    single = outcomes();
    for r in single:
        r["cluster_id"] = "same"
    assert paired_report(single)["paired_cluster_bootstrap_95"] is None


def test_immutable_artifacts_and_bundle(tmp_path):
    directory = new_run(tmp_path / "run")
    write_json(directory / "a.json", {"ok": True})
    with pytest.raises(FileExistsError):
        write_json(directory / "a.json", {})
    with pytest.raises(FileExistsError):
        new_run(directory)
    pins = {"a.json": sha256(directory / "a.json")}
    verify_bundle(directory, pins)
    with pytest.raises(ValueError):
        verify_bundle(directory, {"../outside": "0"*64})
    (directory / "extra").write_text("not pinned")
    with pytest.raises(ValueError):
        verify_bundle(directory, pins)


def test_preflight_default_blocks_and_native_does_not_load(tmp_path):
    result = preflight({})
    assert result["status"] == "BLOCKED" and result["native_loads"] == 0
    from experiments.rineya_useful_fixed_point_recurrence_v1.native import run_native
    result = run_native({}, repo=tmp_path, checkpoint=tmp_path/"missing-model", cases_path=tmp_path, out=tmp_path/"blocked")
    assert result["status"] == "BLOCKED"
    assert json.loads((tmp_path/"blocked"/"status.json").read_text())["status"] == "BLOCKED"


def test_cli_preflight_and_no_native_train_command():
    package = Path(__file__).resolve().parents[1]
    root = package.parents[1]
    result = subprocess.run([sys.executable, "-m", "experiments.rineya_useful_fixed_point_recurrence_v1",
                             "preflight", "--permit", str(package/"configs/native_screen.example.json")],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "BLOCKED"


def test_nonexact_bundle_does_not_inspect_unrelated_symlinks(tmp_path):
    (tmp_path/"source.py").write_text("x=1\n")
    (tmp_path/"unrelated").symlink_to(tmp_path/"absent")
    pins = {"source.py": sha256(tmp_path/"source.py")}
    verify_bundle(tmp_path, pins, exact=False)
    with pytest.raises(ValueError):
        verify_bundle(tmp_path, pins)
    (tmp_path/"alias.py").symlink_to(tmp_path/"source.py")
    with pytest.raises(ValueError):
        verify_bundle(tmp_path, {"alias.py": pins["source.py"]}, exact=False)


def test_depth_support_and_pending_checkpoint_rejected():
    original = DepthPrior(support=(1, 2, 3))
    other = DepthPrior(support=(1, 2, 4))
    with pytest.raises(ValueError):
        other.load_state_dict(copy.deepcopy(original.state_dict()))
    original.sample()
    with pytest.raises(RuntimeError):
        original.state_dict()


def test_host_rejects_positional_cache(model):
    with pytest.raises(HostError), FixedPointHost(model), torch.no_grad():
        model(torch.tensor([[1]]), None, False, object())


def test_host_rejects_float64_native_and_large_batches(model):
    with pytest.raises(HostError), FixedPointHost(model), torch.no_grad():
        model(torch.tensor([[1], [2]]), use_cache=False)
    model.double()
    with pytest.raises(HostError), FixedPointHost(model), torch.no_grad():
        model(torch.tensor([[1]]), use_cache=False)


def test_host_emergency_norm_guard_and_restoration(model):
    layer = model.model.layers[13]
    def huge(hidden_states, **kwargs):
        return 100*hidden_states
    layer.forward = huge
    with pytest.raises(HostError, match="RMS"), FixedPointHost(model), torch.no_grad():
        model(torch.tensor([[1]]), use_cache=False)
    assert layer.forward is huge
    assert "_fp_owner" not in model.model.layers[12].__dict__


def test_host_tensor_output_only(model):
    model.model.layers[13].forward = lambda hidden_states, **kwargs: (hidden_states,)
    with pytest.raises(ValueError), FixedPointHost(model), torch.no_grad():
        model(torch.tensor([[1]]), use_cache=False)


def test_train_guard_aborts_without_mutation():
    module = InjectionInterface(4, 2)
    saved = copy.deepcopy(module.state_dict())
    base = nn.Linear(4, 4).eval().requires_grad_(False)
    def guard():
        raise RuntimeError("resource stop")
    with pytest.raises(RuntimeError, match="resource"):
        fit_interface(module, base, lambda _: (module.blend.square(), 1), guard=guard,
                      train_split="TRAIN", budget=TrainingBudget(max_steps=1))
    assert all(torch.equal(saved[k], v) for k, v in module.state_dict().items())


def test_no_native_output_inside_checkpoint(tmp_path):
    from experiments.rineya_useful_fixed_point_recurrence_v1.native import run_native
    with pytest.raises(ValueError, match="checkpoint tree"):
        run_native({}, repo=tmp_path, checkpoint=tmp_path, cases_path=tmp_path, out=tmp_path/"bad")
    assert not (tmp_path/"bad").exists()


def test_report_cli(tmp_path):
    package = Path(__file__).resolve().parents[1]
    rows = tmp_path/"outcomes.jsonl"
    rows.write_text("\n".join(json.dumps(x) for x in outcomes()))
    command = [sys.executable, "-m", "experiments.rineya_useful_fixed_point_recurrence_v1", "report",
               "--rows", str(rows), "--out", str(tmp_path/"report")]
    result = subprocess.run(command, cwd=package.parents[1], capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)["net_correction_rate"] == .25
