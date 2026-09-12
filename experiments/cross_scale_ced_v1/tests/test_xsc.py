from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import copy
import json
import subprocess
import sys
import torch
import pytest

from experiments.architecture_benchmark_001.model import DenseReference
from experiments.cross_scale_ced_v1.config import CrossConfig, cross_budget, official_config, tiny_config
from experiments.cross_scale_ced_v1.model import CrossScale, CrossCache
from experiments.cross_scale_ced_v1.checks import qualify_model, cache_tensors
from experiments.cross_scale_ced_v1.checkpoint import save_interface, load_interface
from experiments.cross_scale_ced_v1.source import key_map, validate_config, state_digest, source_profile, audit_source
from experiments.cross_scale_ced_v1 import guards
from experiments.cross_scale_ced_v1.training import causal_loss, paired_smoke


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(17)


@pytest.fixture
def ids():
    return torch.tensor([[3, 11, 8, 7, 19, 2, 5, 13]])


def active(mode="H", dtype=torch.float32):
    m = CrossScale(tiny_config(mode)).to(dtype=dtype).eval()
    with torch.no_grad():
        for r in m.reads:
            r.o_proj.weight.normal_(0, 0.03)
    return m


@pytest.mark.parametrize("mode", ["H", "P"])
def test_actual_official_meta_budget(mode):
    with torch.device("meta"):
        m = CrossScale(official_config(mode))
    b = m.verify_budget()
    assert b["parameters"] == 5375687808
    assert b["trainable_interface_parameters"] == 757169792
    assert b["kv_banks"] == 65
    assert b["kv_mib_at_8192_tokens_batch1"] == 2080
    assert m.encoder.embed_tokens.weight.shape == (151936, 1024)
    assert m.decoder.embed_tokens.weight.shape == (151936, 2560)
    assert len(m.encoder.encoder) == 28 and len(m.decoder.encoder) == 36
    assert m.reads[0].q_proj.weight.shape == (4096, 2560)
    assert m.memory.k_proj.weight.shape == (1024, 1024)
    m.freeze_backbones()
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == b["trainable_interface_parameters"]


@pytest.mark.parametrize("mode", ["H", "P"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_output_and_disabled_identity(mode, dtype, ids):
    m = CrossScale(tiny_config(mode)).to(dtype=dtype).eval()
    with torch.inference_mode():
        assert torch.equal(m(ids).logits, m.decoder(ids).logits)
    with torch.no_grad():
        for r in m.reads:
            r.o_proj.weight.normal_(0, 0.1)
    m.read_enabled = False
    with torch.inference_mode():
        assert torch.equal(m(ids).logits, m.decoder(ids).logits)


@pytest.mark.parametrize("mode", ["H", "P"])
def test_full_depth_small_width_active_gate(mode):
    m = CrossScale(tiny_config(mode, full_depth=True))
    result = qualify_model(m)
    assert result["passed"] and result["gradient_tensors"] == 148
    assert len(m.encoder.encoder) == 28 and len(m.decoder.encoder) == 36


@pytest.mark.parametrize("mode", ["H", "P"])
def test_complete_tiny_qualification(mode):
    assert qualify_model(CrossScale(tiny_config(mode)))["passed"]


def test_sources_copied_without_alias(ids):
    c = tiny_config()
    e, d = DenseReference(c.encoder), DenseReference(c.decoder)
    before = state_digest(e), state_digest(d)
    m = CrossScale.from_sources(e, d)
    for source, target in ((e, m.encoder), (d, m.decoder)):
        for (n, p), (n2, p2) in zip(source.named_parameters(), target.named_parameters(), strict=True):
            assert n == n2 and torch.equal(p, p2) and p.data_ptr() != p2.data_ptr()
    for block, read in zip(m.decoder.encoder, m.reads, strict=True):
        assert torch.equal(block.q_proj.weight, read.q_proj.weight)
        assert block.q_proj.weight.data_ptr() != read.q_proj.weight.data_ptr()
    m.freeze_backbones()
    m(ids).logits.sum().backward()
    assert (state_digest(e), state_digest(d)) == before
    assert all(p.grad is None for p in m.encoder.parameters())
    assert all(p.grad is None for p in m.decoder.parameters())


@pytest.mark.parametrize("mode", ["H", "P"])
@pytest.mark.parametrize("offset", [0, 32760])
@pytest.mark.parametrize("chunks", [(1,)*8, (3, 2, 3), (1, 7), (8,)])
def test_cache_schedules_and_immutability(mode, offset, chunks, ids):
    m = active(mode)
    pos = torch.arange(8)[None]+offset
    with torch.inference_mode():
        full = m(ids, position_ids=pos).logits
        cache, pieces, start = None, [], 0
        for n in chunks:
            old = [] if cache is None else [x.clone() for x in cache_tensors(cache)]
            out = m(ids[:, start:start+n], position_ids=pos[:, start:start+n], past=cache, use_cache=True)
            if cache is not None:
                assert all(torch.equal(a, b) for a, b in zip(old, cache_tensors(cache), strict=True))
            cache = out.cache; pieces.append(out.logits); start += n
        torch.testing.assert_close(full, torch.cat(pieces, 1), atol=2e-6, rtol=2e-5)
        assert cache.encoder[0][0].shape == (1, 2, 8, 4)
        assert cache.decoder[0][0].shape == (1, 3, 8, 6)
        assert cache.memory[0].shape == (1, 3, 8, 6)


@pytest.mark.parametrize("mode", ["H", "P"])
@pytest.mark.parametrize("kind", ["future", "encoder_future", "segments", "padding", "repeated_positions"])
def test_no_cross_stream_leakage(mode, kind, ids):
    m = active(mode)
    changed = ids.clone()
    kwargs, encoder_ids, start, end = {}, None, 0, 4
    if kind in ("future", "repeated_positions"):
        changed[:, 4:] = 2
        if kind == "repeated_positions":
            kwargs["position_ids"] = torch.zeros_like(ids)
    elif kind == "encoder_future":
        encoder_ids = ids.clone(); encoder_ids[:, 4:] = 2
    elif kind == "segments":
        kwargs["segment_ids"] = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        changed[:, :4] = 2; start, end = 4, 8
    elif kind == "padding":
        kwargs["attention_mask"] = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        changed[:, :4] = 2; start, end = 4, 8
    with torch.inference_mode():
        a = m(ids, **kwargs).logits[:, start:end]
        b = m(changed, encoder_input_ids=encoder_ids, **kwargs).logits[:, start:end]
        assert torch.equal(a, b)


@pytest.mark.parametrize("mode", ["H", "P"])
def test_memory_use_and_same_token_scope(mode, ids):
    m = active(mode)
    changed = ids.clone(); changed[:, 2] = 23
    with torch.inference_mode():
        base = m(ids).logits
        altered = m(ids, encoder_input_ids=changed).logits
        assert torch.equal(base[:, :2], altered[:, :2])
        assert float((base[:, 2:]-altered[:, 2:]).abs().max()) > 1e-6
        diag = m(ids, diagnostics=True).diagnostics
        assert len(diag) == 4 and any(x["read_rms"] > 0 for x in diag)


@pytest.mark.parametrize("mode", ["H", "P"])
def test_gradient_checkpointing_equivalence(mode, ids):
    a = active(mode); b = copy.deepcopy(a)
    a.freeze_backbones(); b.freeze_backbones()
    a.train(); b.train(); b.gradient_checkpointing = True
    za, zb = a(ids).logits, b(ids).logits
    assert torch.equal(za, zb)
    za.square().mean().backward(); zb.square().mean().backward()
    for (n, p), (n2, q) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert n == n2
        if p.requires_grad:
            assert p.grad is not None and q.grad is not None
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
        else:
            assert p.grad is None and q.grad is None


@pytest.mark.parametrize("mutation", ["foreign", "weights", "mode", "shape", "dtype", "nan", "metadata", "bank", "position"])
def test_reject_invalid_cache(mutation, ids):
    m = active()
    with torch.inference_mode():
        cache = m(ids[:, :3], use_cache=True).cache
    if mutation == "foreign":
        m = active()
    elif mutation == "weights":
        with torch.no_grad(): m.reads[0].o_proj.weight.add_(0.1)
    elif mutation == "mode":
        m.read_enabled = False
    elif mutation in ("shape", "dtype", "nan"):
        k, v = cache.memory
        if mutation == "shape": k = k[:, :, :-1]
        if mutation == "dtype": k = k.double()
        if mutation == "nan":
            k = k.clone(); k[0, 0, 0, 0] = float("nan")
        cache = replace(cache, memory=(k, v))
    elif mutation == "metadata":
        cache = replace(cache, segments=cache.segments.float())
    elif mutation == "bank":
        cache = replace(cache, decoder=cache.decoder[:-1])
    elif mutation == "position":
        cache = replace(cache, positions=torch.full_like(cache.positions, -1))
    with torch.inference_mode(), pytest.raises(ValueError):
        m(ids[:, 3:4], past=cache, use_cache=True)


@pytest.mark.parametrize("case", ["ids_dtype", "empty", "range", "encoder_shape", "encoder_range", "mask", "mask_shape", "segments", "position", "flag", "nothing", "cache_grad", "override"])
def test_reject_bad_inputs(case, ids):
    m = active(); kwargs = {}
    if case == "ids_dtype": ids = ids.float()
    if case == "empty": ids = ids[:, :0]
    if case == "range": ids = torch.full_like(ids, 41)
    if case == "encoder_shape": kwargs["encoder_input_ids"] = ids[:, :2]
    if case == "encoder_range": kwargs["encoder_input_ids"] = torch.full_like(ids, -1)
    if case == "mask": kwargs["attention_mask"] = torch.full_like(ids, 2)
    if case == "mask_shape": kwargs["attention_mask"] = torch.ones(2, 8)
    if case == "segments": kwargs["segment_ids"] = ids.float()
    if case == "position": kwargs["position_ids"] = torch.full_like(ids, 40960)
    if case == "flag": kwargs["use_cache"] = 1
    if case == "nothing": kwargs["compute_logits"] = False
    if case == "cache_grad": kwargs["use_cache"] = True
    if case == "override": m.decoder.memory_accumulation_dtype = torch.float64
    with pytest.raises(ValueError): m(ids, **kwargs)


def test_left_padding_cache_and_batch(ids):
    m = active()
    x = torch.cat((ids, ids.flip(1)), 0)
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
    with torch.inference_mode():
        full = m(x, attention_mask=mask).logits
        a = m(x[:, :4], attention_mask=mask[:, :4], use_cache=True)
        b = m(x[:, 4:], attention_mask=mask[:, 4:], past=a.cache, use_cache=True)
        torch.testing.assert_close(full, torch.cat((a.logits, b.logits), 1), atol=2e-6, rtol=2e-5)
        z = m(ids, compute_logits=False, return_hidden=True)
        assert z.logits is None and z.hidden.shape == (1, 8, 20)


@pytest.mark.parametrize("mode", ["H", "P"])
def test_interface_roundtrip_and_create_only(tmp_path, mode, ids):
    m = active(mode); source = {"scope": "synthetic", "sha": "abc"}
    expected = m(ids).logits.detach().clone()
    save_interface(m, tmp_path/"checkpoint", source)
    with pytest.raises(FileExistsError): save_interface(m, tmp_path/"checkpoint", source)
    with torch.no_grad(): m.reads[0].o_proj.weight.zero_()
    load_interface(m, tmp_path/"checkpoint", source)
    assert torch.equal(expected, m(ids).logits)


@pytest.mark.parametrize("case", ["checksum", "binding", "mode", "keys", "nan", "dtype", "shape"])
def test_interface_tamper_rejected_without_partial_mutation(tmp_path, case):
    from safetensors.torch import load_file, save_file
    m = active(); path = tmp_path/"ckpt"; binding = {"scope": "synthetic"}
    save_interface(m, path, binding)
    manifest = guards.read_json(path/"manifest.json")
    if case == "checksum":
        with (path/"interface.safetensors").open("ab") as f: f.write(b"tamper")
    elif case == "binding": binding = {"scope": "other"}
    elif case == "mode": manifest["config"]["mode"] = "P"
    else:
        state = load_file(str(path/"interface.safetensors")); key = next(iter(state))
        if case == "keys": state.pop(key)
        if case == "nan": state[key].flatten()[0] = float("nan")
        if case == "dtype": state[key] = state[key].double()
        if case == "shape": state[key] = state[key].reshape(-1)[:1]
        save_file(state, str(path/"interface.safetensors"))
        manifest["weights_sha256"] = guards.sha256_file(path/"interface.safetensors")
    (path/"manifest.json").write_text(json.dumps(manifest))
    before = state_digest(m)
    with pytest.raises(ValueError): load_interface(m, path, binding)
    assert state_digest(m) == before


@pytest.mark.parametrize("case", ["mode", "vocab", "architecture", "extra"])
def test_config_fail_closed(case):
    c = tiny_config()
    with pytest.raises((ValueError, TypeError)):
        if case == "mode": CrossConfig(c.encoder, c.decoder, "C")
        if case == "vocab": CrossConfig(c.encoder, replace(c.decoder, vocab_size=43))
        if case == "architecture": CrossConfig(replace(c.encoder, architecture="ced", decoder_layers=1), c.decoder)
        if case == "extra": CrossConfig.from_dict(dict(c.to_dict(), arbitrary=True))


def test_config_serialization_and_invalid_bytes():
    c = tiny_config()
    assert CrossConfig.from_dict(c.to_dict()) == c
    for v in (0, -1, True, 1.5):
        with pytest.raises(ValueError): cross_budget(c, v)


def test_pinned_dependencies_and_key_inventory():
    guards.verify_dependencies()
    assert len(key_map(official_config().encoder)) == 310
    assert len(key_map(official_config().decoder)) == 398
    assert source_profile("decoder")["revision"] == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert source_profile("encoder")["assets_sha256"]["tokenizer.json"] == source_profile("decoder")["assets_sha256"]["tokenizer.json"]


def official_raw(role):
    c = getattr(official_config(), role)
    return dict(model_type="qwen3", architectures=["Qwen3ForCausalLM"], hidden_size=c.hidden_size,
                intermediate_size=c.intermediate_size, num_hidden_layers=c.encoder_layers,
                num_attention_heads=c.num_attention_heads, num_key_value_heads=c.num_key_value_heads,
                head_dim=c.head_dim, rms_norm_eps=c.rms_norm_eps, rope_theta=c.rope_theta,
                vocab_size=c.vocab_size, max_position_embeddings=c.max_position_embeddings,
                tie_word_embeddings=True, hidden_act="silu", attention_bias=False,
                attention_dropout=0.0, rope_scaling=None, use_sliding_window=False, sliding_window=None)


@pytest.mark.parametrize("role", ["encoder", "decoder"])
@pytest.mark.parametrize("bad", ["hidden_size", "head_dim", "auto_map", "quantization_config", "abstain_logit_bias", "tie_word_embeddings", "use_sliding_window"])
def test_official_configuration_rejects_wrong_architecture(role, bad):
    raw = official_raw(role)
    assert validate_config(raw, role) == getattr(official_config(), role)
    if bad in ("head_dim", "hidden_size"): raw[bad] = 1
    elif bad == "auto_map": raw[bad] = {"x": "malicious.RemoteCode"}
    elif bad in ("quantization_config", "abstain_logit_bias"): raw[bad] = None
    else: raw[bad] = not raw[bad]
    with pytest.raises(ValueError): validate_config(raw, role)


def test_missing_sources_cannot_pass(tmp_path):
    for role in ("encoder", "decoder"):
        with pytest.raises(ValueError): audit_source(tmp_path, role)


def test_source_hash_and_unexpected_weights_fail(tmp_path, monkeypatch):
    import experiments.cross_scale_ced_v1.source as s
    raw = official_raw("decoder")
    (tmp_path/"config.json").write_text(json.dumps(raw))
    index = {"weight_map": {k: "model.safetensors" for k in key_map(official_config().decoder)}}
    (tmp_path/"model.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path/"model.safetensors").write_bytes(b"synthetic-header-only")
    assets = {p.name: guards.sha256_file(p) for p in tmp_path.iterdir()}
    profile = {"id": "synthetic", "revision": "synthetic", "parameters": 0, "assets_sha256": assets}
    monkeypatch.setattr(s, "source_profile", lambda role: profile)
    assert s.audit_source(tmp_path, "decoder")["id"] == "synthetic"
    (tmp_path/"rogue.bin").write_bytes(b"x")
    with pytest.raises(ValueError): s.audit_source(tmp_path, "decoder")
    (tmp_path/"rogue.bin").unlink()
    (tmp_path/"model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError): s.audit_source(tmp_path, "decoder")


def test_source_symlinks_and_duplicate_json_rejected(tmp_path):
    a = tmp_path/"a"; a.write_text("x"); b = tmp_path/"b"; b.symlink_to(a)
    with pytest.raises(ValueError): guards.regular_file(b)
    x = tmp_path/"x.json"; x.write_text('{"field":1,"field":2}')
    with pytest.raises(ValueError): guards.read_json(x)


def valid_p1(tmp_path, binding):
    exact = {k: 0.0 for k in ("zero_identity_max_abs", "future_leak_max_abs", "repeated_position_causality",
                              "segment_isolation", "padding_isolation", "disabled_identity_max_abs", "checkpoint_max_abs")}
    checks = dict(exact, passed=True, all_masked_finite=True, cache_immutability=True,
                  frozen_backbones_unchanged=True, gradient_tensors=148,
                  gradient_tensor_names=[f"test_{i}" for i in range(148)],
                  cache_schedule_max_abs=0.0, tail_logits_max_abs=0.0, encoder_intervention_max_abs=0.001)
    native = dict(passed=True, cached_native_checked=True, positions=[0, 257, 32760], max_abs=0.0)
    payload = dict(schema=1, experiment="XSC-001", phase="P1", binding=binding, matched_inventory=True,
                   results={"H": checks, "P": checks}, native_encoder=native, native_decoder=native)
    guards.write_json_new(tmp_path/"qualification.json", payload)
    guards.write_json_new(tmp_path/"binding.json", {"binding": binding})
    names = ["qualification.json", "binding.json"]
    for mode in ("H", "P"):
        sub = tmp_path/f"{mode}_active_interface"; sub.mkdir()
        for name in ("manifest.json", "interface.safetensors"):
            # Checksums test fixture only. It never runs the official CLI/source guards.
            (sub/name).write_text("synthetic")
            names.append(f"{mode}_active_interface/{name}")
    body = dict(schema=1, experiment="XSC-001", phase="P1", status="PASS", binding=binding,
                permissions=guards.permissions(), checks={k: True for k in guards.protocol()["P1_checks"]},
                licenses=guards.protocol()["P1_licenses"],
                artifacts={name: guards.sha256_file(tmp_path/name) for name in names})
    p = tmp_path/"receipt.json"; p.write_text(json.dumps(guards.seal_receipt(body)))
    return p, body


@pytest.mark.parametrize("case", ["status", "phase", "binding", "permissions", "check", "license", "artifact", "checksum"])
def test_receipt_rejects_invalid_or_incomplete_evidence(tmp_path, case):
    binding = {"synthetic": True}
    p, body = valid_p1(tmp_path, binding)
    assert guards.validate_p1(p, binding)["status"] == "PASS"
    if case == "status": body["status"] = "SYNTHETIC_PASS_ONLY"
    if case == "phase": body["phase"] = "tiny-qualify"
    if case == "binding": binding = {"synthetic": False}
    if case == "permissions": body["permissions"]["DEV"] = True
    if case == "check": body["checks"].pop("H")
    if case == "license": body["licenses"] = "train"
    if case == "artifact": (tmp_path/"qualification.json").write_text("tamper")
    receipt = guards.seal_receipt(body)
    if case == "checksum": receipt["receipt_sha256"] = "bad"
    p.write_text(json.dumps(receipt))
    with pytest.raises(ValueError): guards.validate_p1(p, binding)


def test_runtime_resource_and_scope_guards(monkeypatch, tmp_path):
    monkeypatch.setattr(guards, "available_ram_bytes", lambda: 2**30)
    with pytest.raises(ValueError): guards.require_resources("qualify")
    with pytest.raises(ValueError): guards.new_directory(tmp_path/"outside", scoped=True)
    monkeypatch.setattr(guards.importlib.metadata, "version", lambda name: guards.protocol()["runtime"][name])
    assert guards.runtime(require=True)["libraries"] == guards.protocol()["runtime"]
    monkeypatch.setattr(guards.importlib.metadata, "version", lambda name: "0.0")
    with pytest.raises(ValueError): guards.runtime(require=True)


def test_write_outputs_create_only(tmp_path):
    p = tmp_path/"new.json"; guards.write_json_new(p, {"ok": True})
    with pytest.raises(FileExistsError): guards.write_json_new(p, {"ok": False})
    assert guards.read_json(p) == {"ok": True}


def test_masked_loss_does_not_train_padding_or_cross_document_targets(ids):
    logits = torch.randn(1, 8, 41, requires_grad=True)
    mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0]])
    seg = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    loss, report = causal_loss(logits, ids, attention_mask=mask, segment_ids=seg)
    assert report["targets"] == 4
    loss.backward()
    for i in (0, 3, 6, 7): assert not logits.grad[:, i].any()
    with pytest.raises(ValueError): causal_loss(logits, ids, attention_mask=torch.zeros_like(ids))


def test_fixed_paired_engineering_smoke(tmp_path, ids):
    m = CrossScale(tiny_config())
    result = paired_smoke(m.encoder, m.decoder, [ids], tmp_path, {"scope": "synthetic"}, synthetic=True)
    assert result["status"] == "SYNTHETIC_PASS_ONLY"
    assert result["paired_inventory_exact"]
    assert not any(result["permissions"].values())
    for mode in ("H", "P"):
        assert result["results"][mode]["steps"] == 25
        assert result["results"][mode]["gradient_tensors"] == 20
        assert (tmp_path/f"{mode}_interface"/"manifest.json").exists()


def test_inspect_never_claims_pretrained_execution():
    p = subprocess.run([sys.executable, "-m", "experiments.cross_scale_ced_v1", "inspect"],
                       cwd=guards.ROOT, capture_output=True, text=True, check=True)
    report = json.loads(p.stdout)
    assert not any(report["permissions"].values())
    assert report["budget"]["parameters"] == 5375687808
    assert report["official_execution"].startswith("not run")


@pytest.mark.parametrize("case", ["scope", "missing_arm", "native", "gradients", "inactive", "leakage"])
def test_receipt_artifact_semantics_not_just_checksums(tmp_path, case):
    binding = {"scope": "synthetic-test-fixture"}
    p, body = valid_p1(tmp_path, binding)
    artifact = guards.read_json(tmp_path/"qualification.json")
    if case == "scope": artifact["phase"] = "tiny-qualify"
    if case == "missing_arm": artifact["results"].pop("P")
    if case == "native": artifact["native_decoder"]["passed"] = False
    if case == "gradients": artifact["results"]["H"]["gradient_tensors"] = 20
    if case == "inactive": artifact["results"]["H"]["encoder_intervention_max_abs"] = 0.0
    if case == "leakage": artifact["results"]["H"]["future_leak_max_abs"] = 1e-9
    (tmp_path/"qualification.json").write_text(json.dumps(artifact))
    body["artifacts"]["qualification.json"] = guards.sha256_file(tmp_path/"qualification.json")
    p.write_text(json.dumps(guards.seal_receipt(body)))
    with pytest.raises(ValueError): guards.validate_p1(p, binding)


def test_h_p_start_from_byte_identical_tensor_states():
    torch.manual_seed(113); h = CrossScale(tiny_config("H"))
    torch.manual_seed(113); p = CrossScale(tiny_config("P"))
    assert h.state_dict().keys() == p.state_dict().keys()
    assert all(torch.equal(t, p.state_dict()[name]) for name, t in h.state_dict().items())


def test_zero_o_initialization_wakes_up_in_the_expected_order(ids):
    m = CrossScale(tiny_config()); m.freeze_backbones()
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    loss, _ = causal_loss(m(ids).logits, ids); loss.backward()
    assert m.reads[0].o_proj.weight.grad.abs().sum() > 0
    assert m.reads[0].q_proj.weight.grad.abs().sum() == 0
    assert m.memory.k_proj.weight.grad.abs().sum() == 0
    opt.step(); opt.zero_grad(set_to_none=True)
    loss, _ = causal_loss(m(ids).logits, ids); loss.backward()
    assert m.memory.k_proj.weight.grad.abs().sum() > 0
    assert m.reads[0].q_proj.weight.grad.abs().sum() > 0


def test_loss_rejects_invalid_coefficients_and_masks(ids):
    logits = torch.randn(1, 8, 41)
    for v in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError): causal_loss(logits, ids, teacher_kl=v)
    with pytest.raises(ValueError): causal_loss(logits, ids, attention_mask=torch.full_like(ids, 2))


def make_streaming_fixture(tmp_path, role, monkeypatch):
    from safetensors.torch import save_file
    import experiments.cross_scale_ced_v1.source as s
    c = getattr(tiny_config(), role)
    donor = DenseReference(c).eval()
    state = {name: donor.state_dict()[target].to(torch.bfloat16).contiguous()
             for name, target in key_map(c).items()}
    raw = official_raw(role)
    for key, value in dict(vocab_size=c.vocab_size, hidden_size=c.hidden_size,
                           intermediate_size=c.intermediate_size, num_hidden_layers=c.encoder_layers,
                           num_attention_heads=c.num_attention_heads, num_key_value_heads=c.num_key_value_heads,
                           head_dim=c.head_dim).items():
        raw[key] = value
    (tmp_path/"config.json").write_text(json.dumps(raw))
    if role == "encoder":
        save_file(state, str(tmp_path/"model.safetensors"))
        (tmp_path/"CORE.json").write_text(json.dumps({"core_id": "synthetic"}))
    else:
        names = [f"model-0000{i}-of-00003.safetensors" for i in range(1, 4)]
        mapping = {key: names[i % 3] for i, key in enumerate(state)}
        for name in names:
            save_file({k: v for k, v in state.items() if mapping[k] == name}, str(tmp_path/name))
        (tmp_path/"model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    profile = dict(id="synthetic", revision="test-only", parameters=sum(p.numel() for p in donor.parameters()),
                   tensor_count=len(state), assets_sha256={p.name: guards.sha256_file(p) for p in tmp_path.iterdir()
                                                           if p.name != "CORE.json"})
    if role == "encoder": profile["core_git_blob_sha1"] = guards.git_blob((tmp_path/"CORE.json").read_bytes())
    monkeypatch.setattr(s, "official_config", tiny_config)
    monkeypatch.setattr(s, "source_profile", lambda role: profile)
    return s, profile, state


@pytest.mark.parametrize("role", ["encoder", "decoder"])
def test_real_safetensors_streaming_loader_on_tiny_sources(tmp_path, role, monkeypatch):
    s, profile, state = make_streaming_fixture(tmp_path, role, monkeypatch)
    model = s.load_reference(tmp_path, role)
    assert sum(p.numel() for p in model.parameters()) == profile["parameters"]
    assert all(p.dtype == torch.float32 and not p.requires_grad for p in model.parameters())
    for key, target in key_map(model.config).items():
        assert torch.equal(state[key].float(), model.state_dict()[target])


@pytest.mark.parametrize("role", ["encoder", "decoder"])
@pytest.mark.parametrize("corruption", ["dtype", "shape", "nan", "extra", "missing"])
def test_streaming_loader_rejects_bad_tensor_inventory(tmp_path, role, corruption, monkeypatch):
    from safetensors.torch import load_file, save_file
    s, profile, state = make_streaming_fixture(tmp_path, role, monkeypatch)
    file = next(tmp_path.glob("*.safetensors"))
    tensors = load_file(str(file)); key = next(iter(tensors))
    if corruption == "dtype": tensors[key] = tensors[key].float()
    if corruption == "shape": tensors[key] = tensors[key].flatten()[:1]
    if corruption == "nan": tensors[key].flatten()[0] = float("nan")
    if corruption == "extra": tensors["unregistered.weight"] = torch.zeros(2, dtype=torch.bfloat16)
    if corruption == "missing": tensors.pop(key)
    save_file(tensors, str(file))
    profile["assets_sha256"][file.name] = guards.sha256_file(file)
    with pytest.raises(ValueError): s.load_reference(tmp_path, role)


@pytest.mark.parametrize("mode", ["H", "P"])
def test_reader_locality_control_is_not_full_model_context_blindness(mode):
    from experiments.cross_scale_ced_v1.model import MemoryRead
    c = tiny_config(mode); r = MemoryRead(c)
    with torch.no_grad(): r.o_proj.weight.normal_(0, 0.1)
    x = torch.randn(1, 3, 20); k = torch.randn(1, 3, 8, 6); v = torch.randn_like(k)
    pos = torch.tensor([[5, 6, 7]]); kp = torch.arange(8)[None]
    valid = torch.ones(1, 3, dtype=torch.bool); kv = torch.ones(1, 8, dtype=torch.bool)
    seg, ks = torch.zeros_like(pos), torch.zeros_like(kp)
    changed = v.clone(); changed[:, :, :5] += 3
    a = r(x, (k, v), pos, kp, valid, kv, seg, ks)
    b = r(x, (k, changed), pos, kp, valid, kv, seg, ks)
    if mode == "P": assert torch.equal(a, b)
    else: assert not torch.equal(a, b)
