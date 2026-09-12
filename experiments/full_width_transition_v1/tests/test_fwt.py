"""CPU mechanics on random tiny weights. These are NOT official-source results."""
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import pytest
import torch
from experiments.architecture_benchmark_001.budget import Config, presets
from experiments.architecture_benchmark_001.model import DenseReference
from experiments.full_width_transition_v1.model import FullWidthTransition, augmentation_budget, matrix
from experiments.full_width_transition_v1.checks import active_checks, compare, objective
from experiments.full_width_transition_v1 import guards as g
from experiments.full_width_transition_v1.source import key_map, state_digest


@pytest.fixture(autouse=True)
def seed():
    torch.set_num_threads(1)
    torch.manual_seed(17)


@pytest.fixture
def source():
    c = Config(vocab_size=41, hidden_size=16, intermediate_size=32,
               num_attention_heads=4, num_key_value_heads=2, head_dim=8,
               encoder_layers=4, decoder_layers=0, architecture="plain", max_position_embeddings=128)
    return DenseReference(c).eval()


@pytest.fixture
def ids():
    return torch.tensor([[1, 2, 3, 4, 5, 6, 7]])


def active(model):
    with torch.no_grad():
        for read in model.reads:
            read.o_proj.weight.normal_(std=0.01)
    return model.eval()


@pytest.mark.parametrize("mode", ["M", "P"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_initial_identity_and_no_aliasing(source, ids, mode, dtype):
    source = source.to(dtype)
    before = state_digest(source)
    model = FullWidthTransition.from_source(source, split=2, mode=mode)
    with torch.inference_mode():
        assert torch.equal(model(ids).logits, source(ids).logits)
    for key, value in source.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])
        assert value.data_ptr() != model.state_dict()[key].data_ptr()
    assert state_digest(source) == before


@pytest.mark.parametrize("mode", ["M", "P"])
def test_active_paths_cache_prefix_padding_packing(source, ids, mode):
    model = active(FullWidthTransition.from_source(source, split=2, mode=mode))
    result = active_checks(model, ids)
    assert result["cache_pass"] and result["causal_pass"], result
    assert len(result["cache_grid"]) == 10


@pytest.mark.parametrize("mode", ["M", "P"])
def test_zero_output_initialization_wakes_upstream(source, ids, mode):
    model = FullWidthTransition.from_source(source, split=2, mode=mode)
    parameters = model.freeze_backbone()
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    model.train()
    with torch.no_grad():
        teacher = source(ids).logits
    optimizer.zero_grad()
    loss, metrics = objective(model(ids).logits, teacher, ids)
    assert metrics["kl"] == 0.0
    loss.backward()
    assert torch.count_nonzero(model.reads[0].o_proj.weight.grad)
    assert not torch.count_nonzero(model.memory.k_proj.weight.grad)
    optimizer.step()
    optimizer.zero_grad()
    objective(model(ids).logits, teacher, ids)[0].backward()
    assert torch.count_nonzero(model.memory.k_proj.weight.grad)
    assert torch.count_nonzero(model.reads[0].q_proj.weight.grad)
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith(("memory.", "reads.")))


@pytest.mark.parametrize("mode", ["M", "P"])
def test_checkpointing_matches_values_and_gradients(source, ids, mode):
    a = active(FullWidthTransition.from_source(source, split=2, mode=mode))
    b = copy.deepcopy(a)
    for model in (a, b):
        model.freeze_backbone()
        model.train()
    b.gradient_checkpointing = True
    la, lb = a(ids).logits, b(ids).logits
    assert torch.equal(la, lb)
    la.square().mean().backward()
    lb.square().mean().backward()
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        if pa.requires_grad:
            torch.testing.assert_close(pa.grad, pb.grad, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("mode", ["M", "P"])
def test_save_reload_and_create_only(source, ids, tmp_path, mode):
    a = active(FullWidthTransition.from_source(source, split=2, mode=mode))
    path = tmp_path / "checkpoint"
    a.save(path)
    b = FullWidthTransition.load(path)
    with torch.inference_mode():
        assert torch.equal(a(ids).logits, b(ids).logits)
    with pytest.raises(FileExistsError):
        a.save(path)


@pytest.mark.parametrize("corruption", ["extra", "nan", "shape", "dtype"])
def test_checkpoint_corruption_rejected(source, tmp_path, corruption):
    from safetensors.torch import load_file, save_file
    path = tmp_path / "c"
    FullWidthTransition.from_source(source, split=2, mode="M").save(path)
    state = load_file(str(path / "model.safetensors"))
    key = "reads.0.o_proj.weight"
    if corruption == "extra":
        state["unaccounted"] = torch.zeros(1)
    elif corruption == "nan":
        state[key][0, 0] = float("nan")
    elif corruption == "shape":
        state[key] = state[key][:1].clone()
    else:
        state[key] = state[key].double()
    save_file(state, str(path / "model.safetensors"))
    with pytest.raises(ValueError):
        FullWidthTransition.load(path)


def test_full_width_counts_without_allocating_weights():
    config = presets()["B28_1024_official_shape"]
    counts = []
    for mode in ("M", "P"):
        with torch.device("meta"):
            model = FullWidthTransition(config, 14, mode)
        counts.append(sum(p.numel() for p in model.parameters()))
        added = sum(p.numel() for name, p in model.named_parameters() if name.startswith(("memory.", "reads.")))
        assert added == 60_834_688
        assert len(model.encoder) == 28 and len(model.reads) == 14 and len(model.decoder) == 0
    assert counts == [656_884_608, 656_884_608]
    m = matrix()
    assert m["B28"]["parameters"] == 596_049_920
    assert m["C28_FW_deferred"]["parameters"] == 568_786_304
    assert m["D40_FW_deferred"]["parameters"] == 784_821_248
    assert m["C40_FW_deferred"]["parameters"] == 744_973_952
    assert m["M28"]["kv_banks"] == 29 > m["B28"]["kv_banks"]


def test_exact_parameter_inventory_match(source):
    a = FullWidthTransition.from_source(source, split=2, mode="M")
    b = FullWidthTransition.from_source(source, split=2, mode="P")
    assert set(a.state_dict()) == set(b.state_dict())
    for key in a.state_dict():
        assert torch.equal(a.state_dict()[key], b.state_dict()[key])
    assert len(a.freeze_backbone()) == len(b.freeze_backbone())
    assert sum(p.numel() for p in a.parameters() if p.requires_grad) == augmentation_budget(source.config, 2)["added_parameters"]


def test_control_has_no_extra_cross_token_read(source):
    c = source.config
    model = active(FullWidthTransition.from_source(source, split=2, mode="M"))
    read = model.reads[0]
    x = torch.randn(1, 1, c.hidden_size)
    k = torch.randn(1, c.num_key_value_heads, 3, c.head_dim)
    v = torch.randn_like(k)
    altered = v.clone()
    altered[:, :, 0] += 10
    qp, kp = torch.tensor([[2]]), torch.tensor([[0, 1, 2]])
    qv, kv = torch.ones_like(qp).bool(), torch.ones_like(kp).bool()
    qs, ks = torch.zeros_like(qp), torch.zeros_like(kp)
    for mode in ("M", "P"):
        a = read(x, (k, v), qp, kp, qv, kv, qs, ks, mode)
        b = read(x, (k, altered), qp, kp, qv, kv, qs, ks, mode)
        assert torch.equal(a, b) == (mode == "P")


@pytest.mark.parametrize("mutation", ["weights", "arm", "ablation", "owner"])
def test_stale_or_foreign_cache_rejected(source, ids, mutation):
    a = FullWidthTransition.from_source(source, split=2, mode="M")
    with torch.inference_mode():
        cache = a(ids[:, :3], use_cache=True).cache
    if mutation == "weights":
        with torch.no_grad():
            a.reads[0].o_proj.weight.add_(0.01)
    elif mutation == "arm":
        a.mode = "P"
    elif mutation == "ablation":
        a.read_enabled = False
    else:
        a = FullWidthTransition.from_source(source, split=2, mode="M")
    with torch.inference_mode(), pytest.raises(ValueError):
        a(ids[:, 3:4], past=cache, use_cache=True)


@pytest.mark.parametrize("case", ["float_ids", "empty", "bad_id", "bad_mask", "bad_segment", "past_without_cache", "cache_in_training", "bad_positions", "right_pad_tail", "arithmetic_override"])
def test_bad_forward_inputs_fail(source, ids, case):
    model = FullWidthTransition.from_source(source, split=2, mode="M")
    kwargs = {}
    if case == "float_ids":
        ids = ids.float()
    elif case == "empty":
        ids = ids[:, :0]
    elif case == "bad_id":
        ids[0, 0] = source.config.vocab_size
    elif case == "bad_mask":
        kwargs["attention_mask"] = torch.full_like(ids, 2)
    elif case == "bad_segment":
        kwargs["segment_ids"] = torch.zeros_like(ids).float()
    elif case == "past_without_cache":
        with torch.inference_mode():
            kwargs["past"] = model(ids, use_cache=True).cache
    elif case == "cache_in_training":
        model.train()
        kwargs["use_cache"] = True
    elif case == "bad_positions":
        kwargs["position_ids"] = torch.full_like(ids, 128)
    elif case == "right_pad_tail":
        kwargs["attention_mask"] = torch.ones_like(ids)
        kwargs["attention_mask"][:, -1] = 0
        kwargs["last_position_only"] = True
    else:
        model.memory_accumulation_dtype = torch.float64
    with torch.inference_mode(), pytest.raises(ValueError):
        model(ids, **kwargs)


@pytest.mark.parametrize("split", [0, 4, -1, True, 1.5])
def test_invalid_split(source, split):
    with pytest.raises(ValueError):
        FullWidthTransition.from_source(source, split=split, mode="M")


def test_non_plain_donor_rejected(source):
    with pytest.raises(ValueError):
        FullWidthTransition(replace(source.config, architecture="ced", encoder_layers=2, decoder_layers=2), 2, "M")


def test_tail_only_and_branch_ablation(source, ids):
    model = active(FullWidthTransition.from_source(source, split=2, mode="M"))
    with torch.inference_mode():
        full = model(ids, use_cache=True)
        tail = model(ids, use_cache=True, last_position_only=True)
        assert compare(tail.logits, full.logits[:, -1:])["pass"]
        assert len(tail.cache.encoder) == 4 and tail.cache.memory[0].shape[2] == ids.shape[1]
        model.read_enabled = False
        assert torch.equal(model(ids).logits, source(ids).logits)


def test_comparison_does_not_replace_elementwise_by_argmax():
    a = torch.tensor([[1000.0, 0.02]])
    b = torch.tensor([[1000.0, 0.0]])
    assert torch.equal(a.argmax(-1), b.argmax(-1))
    assert not compare(a, b)["pass"]
    assert not compare(torch.tensor([float("nan")]), torch.zeros(1))["pass"]
    assert not compare(torch.zeros(2), torch.zeros(1))["pass"]


def test_source_map_inventory(source):
    assert set(key_map(source.config).values()) == set(source.state_dict())
    assert len(key_map(presets()["B28_1024_official_shape"])) == 310


def test_pinned_dependencies():
    assert g.check_dependencies()["base_commit"] == "aa9bdcd1ff805066b9566230006542704f8b9321"


def test_paired_25_step_engineering_on_tiny_random_weights(source, ids):
    from experiments.full_width_transition_v1.run import engineering_pair
    batches = [(ids + i) % 41 for i in range(8)]
    checks, report = engineering_pair(source, batches, split=2, recipe=g.protocol()["engineering"])
    assert all(checks.values()), checks
    assert len(report["M"]["trajectory"]) == len(report["P"]["trajectory"]) == 25
    assert report["M"]["order"] == report["P"]["order"]
    import random
    expected = list(range(8))
    random.Random(17).shuffle(expected)
    assert report["M"]["order"][:8] == expected
    assert expected != list(range(8))
    assert all(all(report[arm]["upstream_gradients"].values()) for arm in ("M", "P"))


def test_changed_smoke_budget_is_rejected(source, ids):
    from experiments.full_width_transition_v1.run import engineering_pair
    recipe = dict(g.protocol()["engineering"], steps=26)
    with pytest.raises(ValueError):
        engineering_pair(source, [ids], split=2, recipe=recipe)


def bound():
    return {"implementation": "unit-test-only", "source": "NOT-OFFICIAL", "runtime": {}, "fixtures": "test"}


def parent():
    return g.receipt("P1", bound(), {k: True for k in g.protocol()["required_checks"]["P1"]}, {"unit_test_only": True})


def persist(tmp_path, value):
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_valid_parent_integrity_and_no_campaign_license(tmp_path):
    result = parent()
    assert g.validate_parent(persist(tmp_path, result), bound()) == result["sha256"]
    assert not any(result["permissions"].values())


@pytest.mark.parametrize("corruption", ["hash", "foreign", "empty", "failed", "bool", "stage", "permission", "binding"])
def test_bad_parent_rejected(tmp_path, corruption):
    result = parent()
    if corruption == "hash":
        result["sha256"] = "bad"
    else:
        if corruption == "foreign":
            result["experiment"] = "RAB-001"
        elif corruption == "empty":
            result["checks"] = {}
        elif corruption == "failed":
            result["checks"]["native_parity"] = False
        elif corruption == "bool":
            result["checks"]["native_parity"] = 1
        elif corruption == "stage":
            result["stage"] = "P2"
        elif corruption == "permission":
            result["permissions"]["registered_training"] = True
        else:
            result["binding"]["source"] = "changed"
        result.pop("sha256")
        result["sha256"] = hashlib.sha256(g.canonical(result)).hexdigest()
    with pytest.raises(ValueError):
        g.validate_parent(persist(tmp_path, result), bound())


def test_fail_closed_empty_checks_and_unknown_stage():
    assert g.receipt("P1", bound(), {}, {})["status"] == "FAIL"
    assert not g.receipt("P1", bound(), {}, {})["licenses_25_step_engineering_only"]
    with pytest.raises(ValueError):
        g.receipt("P3", bound(), {}, {})


def test_dependency_changes_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "REPO", tmp_path)
    dep = tmp_path / "experiments/architecture_benchmark_001/budget.py"
    dep.parent.mkdir(parents=True)
    dep.write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="dependency changed"):
        g.check_dependencies()


def test_implementation_binds_test_and_protocol_changes(tmp_path, monkeypatch):
    old_root = g.ROOT
    (tmp_path / "dependency_lock.json").write_bytes((old_root / "dependency_lock.json").read_bytes())
    monkeypatch.setattr(g, "ROOT", tmp_path)
    before = g.implementation()
    (tmp_path / "new_test.py").write_text("assert True\n", encoding="utf-8")
    assert g.implementation() != before


def test_source_hash_and_symlink_checks(tmp_path, monkeypatch):
    data = b"fixture-not-weights"
    file = tmp_path / "model.safetensors"
    file.write_bytes(data)
    monkeypatch.setattr(g, "source_lock", lambda: {"assets": {file.name: hashlib.sha256(data).hexdigest()}, "weight_bytes": len(data)})
    assert g.audit_source(tmp_path)
    file.write_bytes(b"tampered")
    with pytest.raises(ValueError):
        g.audit_source(tmp_path)
    target = tmp_path / "target"
    target.write_bytes(data)
    file.unlink()
    file.symlink_to(target)
    with pytest.raises(ValueError):
        g.audit_source(tmp_path)


def test_missing_source_fails_before_model_loading(tmp_path):
    with pytest.raises(ValueError, match="source asset"):
        g.audit_source(tmp_path)


def test_runtime_fail_closed(monkeypatch):
    monkeypatch.setattr(g, "runtime", lambda: dict(g.protocol()["runtime"], transformers=None))
    with pytest.raises(ValueError, match="unqualified runtime"):
        g.require_runtime()


def test_output_scope_and_preservation(tmp_path, monkeypatch):
    root, source = tmp_path / "new", tmp_path / "official"
    root.mkdir()
    source.mkdir()
    monkeypatch.setattr(g, "ROOT", root)
    path = g.new_output(root / "outputs" / "attempt-1", source)
    assert path.is_dir()
    with pytest.raises(FileExistsError):
        g.new_output(path, source)
    for bad in (source / "write", root / "outputs", tmp_path / "legacy"):
        with pytest.raises(ValueError):
            g.new_output(bad, source)


@pytest.mark.parametrize("command", ["train", "recover", "evaluate", "compress", "replace", "P3"])
def test_campaign_commands_do_not_exist(command):
    from experiments.full_width_transition_v1.run import main
    with pytest.raises(SystemExit) as error:
        main([command])
    assert error.value.code == 2


def test_parent_required_before_smoke(tmp_path):
    from experiments.full_width_transition_v1.run import main
    with pytest.raises(SystemExit):
        main(["smoke", "--source", str(tmp_path), "--out", str(tmp_path / "out")])


def test_fixture_roles_are_disjoint():
    fixture = json.loads((g.ROOT / "fixtures.json").read_text())
    assert len(fixture["qualification"]) == len(fixture["engineering_train"]) == 8
    assert not set(fixture["qualification"]) & set(fixture["engineering_train"])


@pytest.mark.parametrize("temperature,weight", [(0, 0.5), (float("inf"), 0.5), (2, float("nan")), (2, -1)])
def test_invalid_objective_parameters(temperature, weight):
    with pytest.raises(ValueError):
        objective(torch.zeros(1, 3, 5), torch.zeros(1, 3, 5), torch.ones(1, 3).long(), temperature=temperature, kl_weight=weight)


def test_shape_mismatch_in_objective_rejected():
    with pytest.raises(ValueError):
        objective(torch.zeros(1, 3, 5), torch.zeros(1, 1, 5), torch.ones(1, 3).long())


def test_engineering_length_bound(source):
    from experiments.full_width_transition_v1.run import engineering_pair
    with pytest.raises(ValueError, match="bounds"):
        engineering_pair(source, [torch.ones(1, 65).long()], split=2, recipe=g.protocol()["engineering"])
