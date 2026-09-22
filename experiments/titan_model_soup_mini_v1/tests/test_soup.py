import copy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import soup
import score


def tensors(dtype=torch.float32):
    return {"model.embed_tokens.weight": torch.arange(12).reshape(6, 2).to(dtype),
            "model.layers.8.mlp.up_proj.weight": torch.tensor([[1., 2.], [3., 4.]], dtype=dtype)}


def write_pair(tmp_path, dtype=torch.float32, tied=False):
    a, b = tensors(dtype), tensors(dtype)
    b["model.layers.8.mlp.up_proj.weight"] += .5
    if tied:
        b["lm_head.weight"] = b["model.embed_tokens.weight"].clone()
    p, q = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    save_file(a, p)
    save_file(b, q)
    return p, q


@pytest.mark.parametrize("alpha", [0., .25, .5, .75, 1.])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_stream_roundtrip(tmp_path, alpha, dtype):
    a, b = write_pair(tmp_path, dtype)
    before = [soup.sha256(a), soup.sha256(b)]
    out = tmp_path / "out.safetensors"
    receipt = soup.stream_pair(a, b, out, alpha)
    with safe_open(out, framework="pt", device="cpu") as h:
        for key in tensors():
            x, y = tensors()[key], tensors()[key].clone()
            if "layers" in key:
                y += .5
            assert torch.equal(h.get_tensor(key), x + (y-x)*alpha)
            assert h.get_tensor(key).dtype == torch.float32
    assert before == [soup.sha256(a), soup.sha256(b)]
    assert receipt["native_model_qualified"] is False


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), -.1, 1.1, True, "0.5"])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        soup.lerp(torch.ones(2), torch.ones(2), alpha)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite(bad):
    with pytest.raises(ValueError):
        soup.lerp(torch.tensor([bad]), torch.ones(1), .5)


def test_integers_refused():
    with pytest.raises(ValueError):
        soup.lerp(torch.ones(2, dtype=torch.int64), torch.ones(2), .5)


def test_identical_coordinates_are_exact():
    a = torch.tensor([.100000001, -13.9375, 0.000014])
    assert torch.equal(soup.lerp(a, a, .3), a)


def test_delta_and_parent_algebra():
    s = torch.tensor([1., 2.]); dt = torch.tensor([.5, .25]); dl = torch.tensor([.125, -.25])
    t, local = s + dt, s + dt + dl
    assert torch.equal(soup.lerp(t, local, .5), s + dt + .5 * dl)
    assert torch.equal(soup.lerp(s, local, .5), s + .5 * dt + .5 * dl)


def test_soft_mask_identity():
    s = torch.tensor([1., 2.]); d = torch.tensor([.5, .25]); mask = torch.tensor([1., 0.])
    assert torch.equal(soup.lerp(s+d, s+mask*d, .5), s + (mask+.5*(1-mask))*d)


def test_tied_alias_allowed(tmp_path):
    a, b = write_pair(tmp_path, tied=True)
    soup.stream_pair(a, b, tmp_path / "out.safetensors", .5)


def test_unequal_tied_alias_refused(tmp_path):
    a, b = write_pair(tmp_path)
    data = tensors()
    data["lm_head.weight"] = data["model.embed_tokens.weight"] + 1
    save_file(data, b)
    with pytest.raises(ValueError, match="tied-head values"):
        soup.stream_pair(a, b, tmp_path / "out.safetensors", .5)


def test_missing_tensor_refused(tmp_path):
    a, b = write_pair(tmp_path)
    data = tensors(); data.pop("model.layers.8.mlp.up_proj.weight")
    save_file(data, b)
    with pytest.raises(ValueError, match="key/shape"):
        soup.stream_pair(a, b, tmp_path / "out.safetensors", .5)


def test_shape_mismatch_refused(tmp_path):
    a, b = write_pair(tmp_path)
    data = tensors(); data["model.layers.8.mlp.up_proj.weight"] = torch.ones(2, 3)
    save_file(data, b)
    with pytest.raises(ValueError, match="key/shape"):
        soup.stream_pair(a, b, tmp_path / "out.safetensors", .5)


def test_overwrite_refused(tmp_path):
    a, b = write_pair(tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        soup.stream_pair(a, b, a, .5)


def test_chunked_matches_whole(tmp_path, monkeypatch):
    a, b = write_pair(tmp_path)
    monkeypatch.setattr(soup, "CHUNK_ELEMENTS", 2)
    first = soup.stream_pair(a, b, tmp_path / "one.safetensors", .5)
    monkeypatch.setattr(soup, "CHUNK_ELEMENTS", 262144)
    second = soup.stream_pair(a, b, tmp_path / "two.safetensors", .5)
    assert first["weight_sha256"] == second["weight_sha256"]


def config():
    return {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "num_hidden_layers": 28,
            "hidden_size": 1024, "vocab_size": 151936, "tie_word_embeddings": True, "torch_dtype": "bfloat16"}


@pytest.mark.parametrize("key,value", [("architectures", ["Qwen3AbstainForCausalLM"]),
    ("hidden_size", 2048), ("vocab_size", 100), ("num_hidden_layers", 12),
    ("tie_word_embeddings", False), ("auto_map", {}), ("quantization_config", {})])
def test_config_rejects(key, value):
    c = config(); c[key] = value
    with pytest.raises(ValueError): soup.canonical_config(c)


def test_dtype_config_only_is_compatible():
    a = config(); b = config(); b["torch_dtype"] = "float32"
    assert soup.canonical_config(a) == soup.canonical_config(b)


def test_duplicate_json_rejected(tmp_path):
    p = tmp_path / "x.json"; p.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError): soup.read_json(p)


def test_bad_hash(tmp_path):
    p = tmp_path / "file"; p.write_text("foo")
    with pytest.raises(ValueError): soup.checked_file(tmp_path, "file", "0"*64)


def test_unbound_template_refuses():
    with pytest.raises(ValueError, match="unbound path"):
        soup.inspect_manifest(Path(__file__).resolve().parents[1] / "manifest.example.json", "sparse_opallo_50")


def test_unregistered_arm_refuses(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"schema": 1, "experiment": soup.EXPERIMENT, "artifacts": {}}))
    with pytest.raises(ValueError, match="unregistered"):
        soup.inspect_manifest(p, "unlimited_sweep")


def run(tmp_path, name, correct, parents=None):
    meta = {k: "1"*64 for k in score.HASHES}
    meta.update(experiment=soup.EXPERIMENT, suite_id="fixture", split="CAL")
    obj = {"metadata": meta, "rows": [{"id": str(i), "parent_id": str(parents[i] if parents else i),
        "domain": "transition", "correct": v} for i, v in enumerate(correct)]}
    p = tmp_path / name; p.write_text(json.dumps(obj)); return p


def test_paired_counts(tmp_path):
    a = run(tmp_path, "a.json", [True, True, False, False])
    b = run(tmp_path, "b.json", [True, False, True, True])
    r = score.paired_report(a, b)["domains"]["transition"]
    assert (r["source_correct_retained"], r["source_correct_lost"], r["newly_correct"]) == (1, 1, 2)


@pytest.mark.parametrize("field", score.MATCHED)
def test_unmatched_metadata_refused(tmp_path, field):
    a = run(tmp_path, "a.json", [True]); b = run(tmp_path, "b.json", [True])
    obj = json.loads(b.read_text()); obj["metadata"][field] = "2"*64
    b.write_text(json.dumps(obj))
    with pytest.raises(ValueError): score.paired_report(a, b)


def test_missing_rows_refused(tmp_path):
    a = run(tmp_path, "a.json", [True]); b = run(tmp_path, "b.json", [True, False])
    with pytest.raises(ValueError): score.paired_report(a, b)


def test_bool_string_refused(tmp_path):
    a = run(tmp_path, "a.json", ["false"])
    with pytest.raises(ValueError): score.load_run(a)


def test_repeated_parents_no_row_p(tmp_path):
    a = run(tmp_path, "a.json", [True, True], [1, 1]); b = run(tmp_path, "b.json", [False, True], [1, 1])
    r = score.paired_report(a, b)
    assert r["domains"]["transition"]["raw_exact_mcnemar_two_sided_p"] is None
    assert not r["quality_improvement_established"]


@pytest.mark.parametrize("g,l,p", [(0,0,1.), (15,0,2**-14), (1,1,1.), (0,1,1.), (0,5,.0625)])
def test_mcnemar(g,l,p):
    assert score.exact_mcnemar(g,l) == pytest.approx(p)


def data(tmp_path, name, parents, explicit=True):
    path = tmp_path / name
    rows = [{"id": f"misleading-same-prefix-{i}", **({"parent_id": p} if explicit else {})} for i,p in enumerate(parents)]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_real_parent_field_and_clean_audit(tmp_path):
    a = data(tmp_path, "a.jsonl", ["a", "b"])
    b = data(tmp_path, "b.jsonl", ["c"])
    c = data(tmp_path, "c.jsonl", ["d"])
    assert score.audit_splits(a,b,[c])["overlaps"] == {"cal_check":0,"train_cal":0,"train_check":0}


def test_contamination_refused(tmp_path):
    a = data(tmp_path, "a.jsonl", ["a"]); b = data(tmp_path, "b.jsonl", ["b"]); c = data(tmp_path, "c.jsonl", ["a"])
    with pytest.raises(ValueError, match="contamination"): score.audit_splits(a,b,[c])


def test_missing_parent_field_refused(tmp_path):
    a = data(tmp_path, "a.jsonl", ["a"], False)
    with pytest.raises(ValueError, match="explicit parent_id"): score.parent_ids(a)
