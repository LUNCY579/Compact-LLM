"""CPU/stdlib fixtures only. These tests never open a model or a research bank."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import struct
import sys

import pytest
from experiments.scp_native_qualification_v1 import contract as c
from experiments.scp_native_qualification_v1.adapter import (Reservations, ScriptedTransport,
    FixtureTokenizer, HostAdapter, resource_gate)
from experiments.scp_native_qualification_v1.assets import (inventory, read_header, validate_inventory,
    validate_config_delta, audit_subject)
from experiments.scp_native_qualification_v1.evidence import write_fixture_bundle, verify_bundle
from experiments.scp_native_qualification_v1.preflight import validate_checkout_observation
from experiments.scp_native_qualification_v1.__main__ import main


def request(tmp_path):
    r = c.draft()
    r.update(successor_commit="a" * 40, protocol_sha256="b" * 64, asset_report_sha256="c" * 64,
             host_sha256="d" * 64, max_process_rss_bytes=6 * 2**30, evidence_root=str(tmp_path / "evidence"))
    return r


def adapter(steps=((0., 5., 0., 0.), (0., 0., 0., 5.)), *, emit=None, reserve=None, guard=None):
    charges, events = [], []
    reservations = Reservations(reserve or charges.append)
    reservations.qualification()
    host = HostAdapter(ScriptedTransport(tuple(steps)), FixtureTokenizer(), [2, 3], reservations,
                       guard or (lambda: None), emit or events.append)
    return host, reservations, charges, events


MESSAGES = [{"role": "system", "content": "Unscored fixture."}, {"role": "user", "content": "Return a label."}]


def test_inventory_exact():
    assert len(inventory()) == 310
    assert len(inventory(True)) == 220
    assert sum(math.prod(s) for s in inventory().values()) == 596049920
    assert sum(math.prod(s) for s in inventory(True).values()) == 314618880
    assert "model.layers.7.self_attn.q_proj.weight" not in inventory(True)
    assert "model.layers.8.self_attn.q_proj.weight" in inventory(True)


def test_complete_prospective_request_still_not_authority(tmp_path):
    r = request(tmp_path)
    assert c.validate_request(r)["native_execution_permitted"] is False
    r.update(decision="APPROVED_INFERENCE_ONLY", approved_by="fixture-only", authorization_reference="fixture-only")
    assert c.validate_request(r)["permit_consumed"] is False
    with pytest.raises(c.Refusal, match=c.NATIVE_BLOCK):
        c.refuse_native(r)


@pytest.mark.parametrize("permission", c.CLOSED)
def test_closed_permissions(permission, tmp_path):
    r = request(tmp_path); r["permissions"][permission] = True
    with pytest.raises(c.Refusal): c.validate_request(r)


@pytest.mark.parametrize("key", list(c.LIMITS))
def test_no_budget_widening(key, tmp_path):
    r = request(tmp_path); r["limits"][key] += 1
    with pytest.raises(c.Refusal): c.validate_request(r)


@pytest.mark.parametrize("key,value", [("successor_commit", c.PARENT), ("successor_commit", "f" * 64),
    ("host_sha256", None), ("asset_report_sha256", "x" * 64), ("window_tokens", 4096),
    ("generation_requests", True), ("qualification_forwards", 5), ("watchdog_seconds", 601),
    ("native_execution_permitted", True), ("max_process_rss_bytes", True),
    ("max_process_rss_bytes", 0), ("max_process_rss_bytes", float("nan")),
    ("evidence_root", "relative/path"), ("decision", "APPROVED")])
def test_malformed_or_expanded_request(key, value, tmp_path):
    r = request(tmp_path); r[key] = value
    with pytest.raises(c.Refusal): c.validate_request(r)


def test_draft_missing_host_and_old_training_grant(tmp_path):
    with pytest.raises(c.Refusal): c.validate_request(c.draft())
    r = request(tmp_path); r["training_permission"] = True
    with pytest.raises(c.Refusal): c.validate_request(r)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":-Infinity}'])
def test_ambiguous_json(text):
    with pytest.raises(c.Refusal): c.loads(text)


def test_probe_gold_or_changes_refused():
    c.validate_probe(copy.deepcopy(c.PROBE))
    for extra in ("reference", "answer", "gold", "scorer", "task"):
        with pytest.raises(c.Refusal): c.validate_probe({**c.PROBE, extra: "LUMEN"})
    p = dict(c.PROBE); p["user"] += " Changed."
    with pytest.raises(c.Refusal): c.validate_probe(p)


def test_no_native_before_any_callback():
    def fail(*a, **k): raise AssertionError("callback executed")
    with pytest.raises(c.Refusal, match=c.NATIVE_BLOCK):
        HostAdapter(object(), object(), [], object(), fail, fail, native=True)
    with pytest.raises(c.Refusal): HostAdapter(object(), FixtureTokenizer(), [2], Reservations(fail), fail, fail)


def test_cli_launch_does_not_read_grant(tmp_path, monkeypatch, capsys):
    def fail(*a, **k): raise AssertionError("read attempted")
    monkeypatch.setattr(Path, "read_text", fail)
    assert main(["launch", "--request", str(tmp_path / "nonexistent")]) == 2
    assert c.NATIVE_BLOCK in capsys.readouterr().err


def test_zero_training_and_distinct_units():
    host, r, charges, events = adapter()
    caps = [64, 288] + [64, 64, 64, 64, 32] * 2
    results = [host.generate(MESSAGES, cap) for cap in caps]
    assert r.requests == 12 and r.allowance == 928 and r.forward_allowance == 934
    assert sum(x["actual_forward_calls"] for x in results) == 24
    assert all(x["pretrained_calls"] == 0 for x in results)
    assert all(set(x) <= {"evaluation_and_qualification_model_calls", "generated_tokens_all_candidates"} for x in charges)
    with pytest.raises(c.Refusal): host.generate(MESSAGES, 64)


def test_secondary_eos_and_cache_isolation():
    host, _, _, _ = adapter()
    a = host.generate(MESSAGES, 64); b = host.generate(MESSAGES, 64)
    assert a == b
    assert a["generated_ids_including_terminal"] == [1, 3]
    assert a["text"] == "<token:1>" and a["terminated"] is True


def test_non_eos_cap_exhaustion():
    host, _, _, _ = adapter(steps=((0., 5., 0., 0.),) * 32)
    result = host.generate(MESSAGES, 32)
    assert result["output_tokens"] == 32 and result["terminated"] is False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), True, "0"])
def test_all_logits_checked_not_just_argmax(bad):
    host, r, charges, events = adapter(steps=((0., 99., bad, 0.),))
    with pytest.raises(c.Refusal): host.generate(MESSAGES, 64)
    assert r.allowance == 64 and len(charges) == 2 and events[-1]["event"] == "FAILED"
    with pytest.raises(c.Refusal): host.generate(MESSAGES, 64)
    assert len(charges) == 2


@pytest.mark.parametrize("cap", [True, 0, -1, 63, 65, 2048, float("inf")])
def test_bad_caps(cap):
    host, _, charges, _ = adapter()
    with pytest.raises(c.Refusal): host.generate(MESSAGES, cap)
    assert len(charges) == 1


def test_actual_selector_prompt_overflow_refused_without_reservation():
    host, _, charges, _ = adapter()
    messages = copy.deepcopy(MESSAGES); messages[1]["content"] = "candidate " * 2048
    with pytest.raises(c.Refusal, match="OVERLENGTH"): host.generate(messages, 32)
    assert len(charges) == 1


def test_label_fields_refused():
    host, _, _, _ = adapter()
    messages = copy.deepcopy(MESSAGES); messages[1]["reference"] = "leak"
    with pytest.raises(c.Refusal): host.generate(messages, 64)


def test_reservation_failure_prevents_forward():
    calls = []
    def reserve(charges):
        calls.append(charges)
        if len(calls) > 1: raise RuntimeError("exhausted")
    host, _, _, events = adapter(reserve=reserve)
    with pytest.raises(RuntimeError): host.generate(MESSAGES, 64)
    assert host.attempted_forwards == 0 and events[-1]["reserved"] is False


def test_guard_failure_keeps_charge_and_poison():
    def stop(): raise RuntimeError("resource floor")
    host, r, charges, events = adapter(guard=stop)
    with pytest.raises(RuntimeError): host.generate(MESSAGES, 64)
    assert r.allowance == 64 and host.attempted_forwards == 0
    assert events[-1]["reserved"] is True


def test_sink_failure_cannot_be_success():
    def fail(event): raise OSError("disk full")
    host, r, _, _ = adapter(emit=fail)
    with pytest.raises(OSError): host.generate(MESSAGES, 64)
    assert host.failed and r.allowance == 64 and host.attempted_forwards == 0


def snapshot():
    return {"available_host_bytes": 6 * 2**30, "rss_bytes": 2 * 2**30,
            "gpu0": {"free": 4 * 2**30, "total": 8 * 2**30}, "time_unix": 1000.0}


def test_resource_value_not_host_certificate():
    assert resource_gate(snapshot(), before_load=True, max_rss=6 * 2**30, now=1001)["native_execution_permitted"] is False


@pytest.mark.parametrize("kind", ["ram", "gpu", "cuda", "rss", "stale", "future", "nan", "bool"])
def test_resource_refusals(kind):
    s = snapshot()
    if kind == "ram": s["available_host_bytes"] = 1
    if kind == "gpu": s["gpu0"]["free"] = 1
    if kind == "cuda": s["gpu0"] = None
    if kind == "rss": s["rss_bytes"] = 7 * 2**30
    if kind == "stale": s["time_unix"] = 990
    if kind == "future": s["time_unix"] = 1002
    if kind == "nan": s["time_unix"] = float("nan")
    if kind == "bool": s["available_host_bytes"] = True
    with pytest.raises(c.Refusal): resource_gate(s, before_load=True, max_rss=6 * 2**30, now=1001)


def tensor_file(path, header, data=b"\0" * 8):
    payload = c.canonical(header)
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + data)


def test_header_parsing(tmp_path):
    p = tmp_path / "tiny.safetensors"
    h = {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    tensor_file(p, h)
    assert read_header(p) == h
    with pytest.raises(c.Refusal): validate_inventory(h)


@pytest.mark.parametrize("kind", ["gap", "overlap", "dtype", "bool_shape", "trailing", "huge"])
def test_bad_headers(kind, tmp_path):
    p = tmp_path / "bad.safetensors"
    h = {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    if kind == "gap": h["x"]["data_offsets"] = [4, 12]
    if kind == "overlap": h["y"] = copy.deepcopy(h["x"])
    if kind == "dtype": h["x"]["dtype"] = "I64"
    if kind == "bool_shape": h["x"]["shape"] = [True]
    tensor_file(p, h, b"\0" * (9 if kind == "trailing" else 8))
    if kind == "huge": p.write_bytes(struct.pack("<Q", 2**40))
    with pytest.raises(c.Refusal): read_header(p)


@pytest.mark.parametrize("change", ["rename", "shape", "bf16", "extra"])
def test_exact_inventory_not_just_count(change):
    h = {n: {"dtype": "F32", "shape": s, "data_offsets": []} for n, s in inventory(True).items()}
    name = next(iter(h))
    if change == "rename": h["different_name"] = h.pop(name)
    if change == "shape": h[name]["shape"] = [1]
    if change == "bf16": h[name]["dtype"] = "BF16"
    if change == "extra": h["lm_head.weight"] = h[name]
    with pytest.raises(c.Refusal): validate_inventory(h, partial=True)


def config():
    return {"model_type": "qwen3", "hidden_size": 1024, "intermediate_size": 3072,
            "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
            "head_dim": 128, "vocab_size": 151936, "tie_word_embeddings": True,
            "use_sliding_window": False, "attention_dropout": 0.0, "torch_dtype": "bfloat16"}


def test_only_fp32_config_exception():
    source = config(); subject = {**source, "torch_dtype": "float32"}
    validate_config_delta(source, subject)
    for key, value in (("rope_theta", 99), ("auto_map", {"x": "y"}), ("hidden_size", 512), ("torch_dtype", "float16")):
        with pytest.raises(c.Refusal): validate_config_delta(source, {**subject, key: value})


def test_wrong_actual_files_never_become_assets(tmp_path):
    a = tmp_path / "source"; a.mkdir(); (a / "model.safetensors").write_bytes(b"tiny")
    b = tmp_path / "subject"; b.mkdir(); (b / "model.safetensors").write_bytes(b"tiny2")
    p = tmp_path / "partial"; p.write_bytes(b"tiny3")
    with pytest.raises(c.Refusal, match="SHA-256"): audit_subject(a, p, b)


def test_safe_paths_create_only(tmp_path):
    p = tmp_path / "x.json"; c.write_new(p, {"x": 1})
    with pytest.raises(FileExistsError): c.write_new(p, {"x": 2})
    link = tmp_path / "link"; link.symlink_to(p)
    with pytest.raises(c.Refusal): c.file_hash(link)
    with pytest.raises(c.Refusal): c.safe_path(tmp_path / ".." / "x")


@pytest.mark.parametrize("change", ["branch", "parent_head", "dirty", "old_edit", "deleted", "outside"])
def test_checkout_refusals(change):
    branch, head, status = c.BRANCH, "e" * 40, ""
    changed = [["A", "experiments/scp_native_qualification_v1/new.py"]]
    if change == "branch": branch = "main"
    if change == "parent_head": head = c.PARENT
    if change == "dirty": status = "?? untracked"
    if change == "old_edit": changed = [["M", "experiments/sequential_complementary_v1/reasoning.py"]]
    if change == "deleted": changed[0][0] = "D"
    if change == "outside": changed[0][1] = "other/new.py"
    with pytest.raises(c.Refusal): validate_checkout_observation(branch, head, status, changed)


def bundle(tmp_path):
    root = tmp_path / "checkout"; root.mkdir()
    host, _, _, events = adapter()
    result = host.generate(MESSAGES, 64)
    out = tmp_path / "evidence"
    verification = write_fixture_bundle(out, root, events, [result])
    return out, root, verification


def test_fixture_receipt_roundtrip_and_no_overwrite(tmp_path):
    out, root, v = bundle(tmp_path)
    assert v["not_a_native_pass"] and v["completed_requests"] == 1
    assert verify_bundle(out) == v
    with pytest.raises(c.Refusal): write_fixture_bundle(out, root, [], [])
    with pytest.raises(c.Refusal): write_fixture_bundle(root / "evidence", root, [], [])


def test_tampered_bytes_rejected(tmp_path):
    out, _, _ = bundle(tmp_path)
    (out / "EVENTS.json").write_text("[]")
    with pytest.raises(c.Refusal): verify_bundle(out)


def reseal(out, result, events):
    for name, value in (("EVENTS.json", events), ("RESULT.json", result)):
        (out / name).write_bytes(c.canonical(value) + b"\n")
    manifest = {"schema": "SCP_NQ_MANIFEST_V1", "files": {n: c.file_hash(out / n) for n in ("EVENTS.json", "RESULT.json")}}
    (out / "MANIFEST.json").write_bytes(c.canonical(manifest))


@pytest.mark.parametrize("change", ["native", "grant", "counter", "unfinished", "token", "termination", "impl"])
def test_forged_receipts_fail_even_with_recomputed_hashes(change, tmp_path):
    out, _, _ = bundle(tmp_path)
    result = c.loads((out / "RESULT.json").read_text()); events = c.loads((out / "EVENTS.json").read_text())
    if change == "native": result["classification"] = "PASS_NATIVE_WIRING_ONLY"
    if change == "grant": result["native_execution_permitted"] = True
    if change == "counter": result["optimizer_steps"] = True
    if change == "unfinished": events.pop()
    if change == "token": events[2]["token_id"] = 99
    if change == "termination": events[-1]["terminated"] = False; result["results"][0]["terminated"] = False
    if change == "impl": result["implementation_sha256"] = "e" * 64
    result["events_sha256"] = c.digest(events)
    reseal(out, result, events)
    with pytest.raises(c.Refusal): verify_bundle(out)


def test_package_has_no_pretrained_imports_or_execution(tmp_path):
    import ast
    forbidden = {"torch", "transformers", "safetensors"}
    for path in c.PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): assert not any(x.name.split(".")[0] in forbidden for x in node.names)
            if isinstance(node, ast.ImportFrom): assert (node.module or "").split(".")[0] not in forbidden
    assert main(["status"]) == 0


@pytest.mark.parametrize("eos", [None, [], [True], [-1], [2, 2], [[2]], "2"])
def test_bad_eos_configuration(eos):
    with pytest.raises(c.Refusal):
        HostAdapter(ScriptedTransport(((0., 1., 0.),)), FixtureTokenizer(), eos,
                    Reservations(lambda x: None), lambda: None, lambda e: None)


def test_arbitrary_transport_container_refused():
    with pytest.raises(c.Refusal): ScriptedTransport([[0., 1.]])
    with pytest.raises(c.Refusal): ScriptedTransport(())


def test_qualification_reservation_not_reusable():
    r = Reservations(lambda x: None)
    with pytest.raises(c.Refusal): r.generation(64)
    r.qualification()
    with pytest.raises(c.Refusal): r.qualification()


def test_fixture_cli_end_to_end(tmp_path, capsys):
    out = tmp_path / "cli-evidence"
    assert main(["fixture", "--out", str(out)]) == 0
    v = json.loads(capsys.readouterr().out)
    assert v["completed_requests"] == 12 and v["not_a_native_pass"]
    assert main(["verify-fixture", "--directory", str(out)]) == 0
