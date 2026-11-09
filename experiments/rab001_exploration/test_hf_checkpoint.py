import json
from pathlib import Path

import pytest

from experiments.rab001_exploration.hf_checkpoint import SIDECARS, file_hash, verify_hf_candidate


def fixture(tmp_path):
    source, candidate = tmp_path / "source", tmp_path / "candidate"
    source.mkdir(); candidate.mkdir()
    for name in SIDECARS:
        (source / name).write_text(name)
        (candidate / name).write_text(name)
    (source / "model.safetensors").write_bytes(b"source identity fixture")
    (candidate / "model.safetensors").write_bytes(b"candidate identity fixture")
    receipt = {"record": "CHECKPOINT_MATERIALIZATION", "layout": "huggingface",
               "source_weight_sha256": file_hash(source / "model.safetensors"),
               "out_weight_sha256": file_hash(candidate / "model.safetensors"),
               "checkpoint_manifest_sha256": "recorded checkpoint identity"}
    (candidate / "MATERIALIZATION.json").write_text(json.dumps(receipt))
    return source, candidate


def test_verified_identity_keeps_source_and_candidate_distinct(tmp_path):
    source, candidate = fixture(tmp_path)
    identity = verify_hf_candidate(candidate, source)
    assert identity["source_weight_sha256"] != identity["weight_sha256"]
    assert identity["redundant_research_export_created"] is False
    assert set(identity["sidecars_sha256"]) == set(SIDECARS)


@pytest.mark.parametrize("name", SIDECARS)
def test_changed_source_configuration_or_tokenizer_refused(tmp_path, name):
    source, candidate = fixture(tmp_path)
    (candidate / name).write_text("different")
    with pytest.raises(ValueError, match="sidecar differs"):
        verify_hf_candidate(candidate, source)


def test_wrong_base_weights_refused_even_with_valid_export_hash(tmp_path):
    source, candidate = fixture(tmp_path)
    (source / "model.safetensors").write_bytes(b"a different core")
    with pytest.raises(ValueError, match="different source"):
        verify_hf_candidate(candidate, source)


def test_changed_candidate_weights_refused(tmp_path):
    source, candidate = fixture(tmp_path)
    (candidate / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="weight bytes differ"):
        verify_hf_candidate(candidate, source)


def test_unqualified_layout_refused(tmp_path):
    source, candidate = fixture(tmp_path)
    path = candidate / "MATERIALIZATION.json"
    receipt = json.loads(path.read_text()); receipt["layout"] = "unknown"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="explicit HF"):
        verify_hf_candidate(candidate, source)


def test_direct_hf_and_research_export_have_identical_weights_logits_and_cache(tmp_path, monkeypatch):
    """Tiny random CPU model only: the storage alternative changes no tensor/math."""
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from experiments.architecture_benchmark_001 import run
    from experiments.architecture_benchmark_001.budget import Config
    from experiments.rab001_exploration.hf_checkpoint import load_hf_candidate

    torch.set_num_threads(1)
    torch.manual_seed(51)
    cfg = Qwen3Config(vocab_size=67, hidden_size=32, intermediate_size=96,
                     num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=4,
                     head_dim=8, tie_word_embeddings=True, max_position_embeddings=256)
    native = Qwen3ForCausalLM(cfg).eval()
    source, candidate = fixture(tmp_path)
    state = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in native.state_dict().items()
             if k != "lm_head.weight"}
    for folder in (source, candidate):
        save_file(state, str(folder / "model.safetensors"))
        (folder / "config.json").write_text(json.dumps(cfg.to_dict()))
    receipt_path = candidate / "MATERIALIZATION.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update(source_weight_sha256=file_hash(source / "model.safetensors"),
                   out_weight_sha256=file_hash(candidate / "model.safetensors"))
    receipt_path.write_text(json.dumps(receipt))
    # Numeric loader comparison needs no tokenizer and makes no language evaluation claim.
    monkeypatch.setattr(run, "_load_tokenizer", lambda _: None)
    tiny = Config(vocab_size=67, hidden_size=32, intermediate_size=96,
                  num_attention_heads=8, num_key_value_heads=4, head_dim=8,
                  encoder_layers=4, decoder_layers=0, architecture="plain")
    # The production source validator remains official-shape-only. Substitute only
    # this fixture's shape to exercise storage numerics without allocating 596M weights.
    monkeypatch.setattr(run, "source_config", lambda _: tiny)
    direct, _ = load_hf_candidate(candidate, source, "cpu")
    research = tmp_path / "research"
    research.mkdir()
    run._save(direct, research)
    restored = run._restore(research, "cpu")
    for name, tensor in direct.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name]), name
    ids = torch.tensor([[3, 5, 7, 11]])
    with torch.inference_mode():
        left = direct(ids, use_cache=True)
        right = restored(ids, use_cache=True)
        assert torch.equal(left.logits, right.logits)
        next_id = torch.tensor([[13]])
        assert torch.equal(direct(next_id, past=left.cache, use_cache=True).logits,
                           restored(next_id, past=right.cache, use_cache=True).logits)
