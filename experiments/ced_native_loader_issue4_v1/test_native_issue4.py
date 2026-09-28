"""Additional real-loader regressions. Tiny random fixtures only; no native skips.

The inherited build_source writes a real tokenizer and safetensors checkpoint.
Only run.source_config is patched inside tests to accept the declared tiny shape;
native construction, assign=True, buffer reconstruction and .to() remain real.
"""
from importlib import metadata
import shutil
import pytest
import torch
from safetensors.torch import load_file, save_file

assert metadata.version('transformers') == '4.51.3', 'pinned native runtime is mandatory'
from transformers import Qwen3ForCausalLM
from experiments.architecture_benchmark_001 import run, source
from experiments.architecture_benchmark_001.gates import GateClosed
from experiments.architecture_benchmark_001.tests.test_native_loader import (
    TINY, ATOL, RTOL, build_source, digests, raw_config,
)
from experiments.architecture_benchmark_001.transfer import native_key_map


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    folder = tmp_path / 'explicit_random_fixture'
    folder.mkdir()
    build_source(folder, torch.float32)
    monkeypatch.setattr(run, 'source_config', lambda raw: TINY)
    return folder


def assert_materialized(native):
    assert type(native) is Qwen3ForCausalLM
    tensors = list(native.named_parameters()) + list(native.named_buffers())
    assert tensors
    assert all(not t.is_meta and t.device == torch.device('cpu') for _, t in tensors)
    rotary = native.model.rotary_emb
    assert not rotary.original_inv_freq.is_meta
    assert rotary.original_inv_freq is rotary.inv_freq
    assert rotary.inv_freq.dtype == torch.float32
    assert 'inv_freq' in rotary._non_persistent_buffers_set
    assert 'model.rotary_emb.inv_freq' not in native.state_dict()
    assert native.lm_head.weight is native.model.embed_tokens.weight


def test_original_meta_failure_is_reproduced_through_actual_load(tiny, monkeypatch):
    before = digests(tiny)
    with monkeypatch.context() as broken:
        # Remove only the repair, retaining the exact real construction/assignment/transfer.
        broken.setattr(run, '_materialize_native_buffers', lambda model, device: [])
        with pytest.raises(NotImplementedError, match='Cannot copy out of meta tensor'):
            run._load(tiny, 'cpu', native=True)
    _, _, native = run._load(tiny, 'cpu', native=True)
    assert_materialized(native)
    assert digests(tiny) == before


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16], ids=['float32', 'bfloat16'])
def test_save_reload_uses_actual_loader(tmp_path, monkeypatch, dtype):
    folder = tmp_path / 'fixture_source'
    folder.mkdir()
    build_source(folder, dtype)
    before = digests(folder)
    monkeypatch.setattr(run, 'source_config', lambda raw: TINY)
    reference, tokenizer, native = run._load(folder, 'cpu', native=True)
    saved = tmp_path / 'roundtrip'
    saved.mkdir()
    tokenizer.save_pretrained(str(saved))
    shutil.copyfile(folder / 'config.json', saved / 'config.json')
    state = native.state_dict()
    save_file({k: state[k].detach().clone().contiguous() for k in native_key_map(TINY)},
              str(saved / 'model.safetensors'))
    saved_before = digests(saved)
    rereference, retokenizer, renative = run._load(saved, 'cpu', native=True)
    assert_materialized(native)
    assert_materialized(renative)
    assert torch.equal(native.model.rotary_emb.inv_freq, renative.model.rotary_emb.inv_freq)
    for key, value in native.state_dict().items():
        assert torch.equal(value, renative.state_dict()[key]), key
    assert tokenizer.get_vocab() == retokenizer.get_vocab()
    assert tokenizer.all_special_ids == retokenizer.all_special_ids
    assert tokenizer('t5 t6 t7')['input_ids'] == retokenizer('t5 t6 t7')['input_ids']
    ids = torch.tensor([[5, 6, 7, 8, 9]])
    with torch.inference_mode():
        expected = native(ids, use_cache=False).logits
        for actual in (renative(ids, use_cache=False).logits, reference(ids).logits,
                       rereference(ids).logits):
            torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    assert digests(folder) == before
    assert digests(saved) == saved_before


@pytest.mark.parametrize(('side', 'chunk'), [('left', 1), ('left', 2), ('right', 1), ('right', 2)],
                         ids=['left-one', 'left-two', 'right-one', 'right-two'])
def test_padded_batch_cache_and_chunk_parity(tiny, side, chunk):
    reference, _, native = run._load(tiny, 'cpu', native=True)
    prefix = ([[0, 0, 5, 6, 7], [0, 8, 9, 10, 11]] if side == 'left'
              else [[5, 6, 7, 0, 0], [8, 9, 10, 11, 0]])
    ids = torch.cat((torch.tensor(prefix), torch.tensor([[12, 13, 14], [15, 16, 17]])), dim=1)
    mask = ids.ne(0).long()
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    with torch.inference_mode():
        full = native(ids, attention_mask=mask, position_ids=pos, use_cache=False).logits
        ref = reference(ids, attention_mask=mask, position_ids=pos).logits
        torch.testing.assert_close(full[mask.bool()], ref[mask.bool()], atol=ATOL, rtol=RTOL)
        first = native(ids[:, :5], attention_mask=mask[:, :5], position_ids=pos[:, :5], use_cache=True)
        past = first.past_key_values
        for start in range(5, ids.shape[1], chunk):
            end = min(start + chunk, ids.shape[1])
            step = native(ids[:, start:end], attention_mask=mask[:, :end],
                          position_ids=pos[:, start:end], past_key_values=past, use_cache=True)
            torch.testing.assert_close(step.logits, full[:, start:end], atol=ATOL, rtol=RTOL)
            past = step.past_key_values
    assert_materialized(native)


@pytest.mark.parametrize('damage', ['missing', 'extra', 'shape'])
def test_corrupt_checkpoint_fails_closed(tiny, damage):
    path = tiny / 'model.safetensors'
    state = load_file(str(path))
    key = 'model.layers.0.self_attn.q_proj.weight'
    if damage == 'missing':
        del state[key]
    elif damage == 'extra':
        state['unexpected.weight'] = torch.ones(1)
    else:
        state[key] = state[key][:-1].clone()
    # This is corruption of a test fixture, never a published checkpoint.
    save_file(state, str(path))
    before = digests(tiny)
    with pytest.raises((GateClosed, ValueError, KeyError, RuntimeError)):
        run._load(tiny, 'cpu', native=True)
    assert digests(tiny) == before


def test_official_shape_validator_remains_closed_to_tiny_source():
    with pytest.raises((ValueError, GateClosed)):
        source.source_config(raw_config())


def test_wrong_transformers_version_blocks_actual_load(tiny, monkeypatch):
    original = metadata.version
    monkeypatch.setattr(metadata, 'version', lambda name: '4.51.4' if name == 'transformers' else original(name))
    with pytest.raises(GateClosed, match='pinned native Transformers'):
        run._load(tiny, 'cpu', native=True)


def test_loading_and_forward_do_not_mutate_assigned_source_tensors(tiny, monkeypatch):
    captured = {}
    real_load_file = run.load_file

    def observe(*args, **kwargs):
        state = real_load_file(*args, **kwargs)
        captured['state'] = state
        captured['before'] = {k: v.clone() for k, v in state.items()}
        return state

    monkeypatch.setattr(run, 'load_file', observe)
    reference, _, native = run._load(tiny, 'cpu', native=True)
    ids = torch.tensor([[5, 6, 7]])
    with torch.inference_mode():
        native(ids, use_cache=True)
        reference(ids)
    assert captured['state'].keys() == captured['before'].keys()
    for key, value in captured['state'].items():
        assert torch.equal(value, captured['before'][key]), key
