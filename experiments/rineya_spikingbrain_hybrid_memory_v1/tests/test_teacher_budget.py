"""Cache admission counts serialized files and refuses overflow before writing."""
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from sbhm import pipeline, protocol


class CaptureSite(nn.Module):
    def forward(self, *, hidden_states, position_embeddings):
        return hidden_states, None


class CaptureHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(layers=[SimpleNamespace(self_attn=CaptureSite())])

    def forward(self, input_ids, **kwargs):
        hidden = torch.zeros(1, input_ids.shape[1], 4)
        rope = torch.zeros(1, input_ids.shape[1], 2)
        self.model.layers[0].self_attn(hidden_states=hidden, position_embeddings=(rope, rope))


def setup_capture(monkeypatch):
    row = {'id': 'fixture', 'family_id': 'fixture', 'prompt': 'fixture', 'target': 'fixture'}
    monkeypatch.setattr(pipeline, 'corpus', lambda _: ({'train': [row]}, 'fixture-hash'))
    monkeypatch.setattr(pipeline, 'tokens', lambda *_: (torch.ones(1, 3, dtype=torch.long), None))
    monkeypatch.setattr(pipeline, 'resources', lambda *a, **k: {})
    return CaptureHost(), {'layer': 0, 'seed': 17, 'device': 'cpu', 'limits': {'steps': 1}}


def test_teacher_receipt_counts_actual_safetensors_file_bytes(tmp_path, monkeypatch):
    model, config = setup_capture(monkeypatch)
    receipt = pipeline.capture(model, None, config, {}, tmp_path)
    path = tmp_path / receipt['entries'][0]['path']
    assert receipt['teacher_cache_file_bytes'] == path.stat().st_size
    assert receipt['teacher_cache_file_bytes'] > (4+4+2+2)*3*4


def test_teacher_cache_overflow_rejected_before_file_write(tmp_path, monkeypatch):
    model, config = setup_capture(monkeypatch)
    class OversizedSerialization:
        def __len__(self):
            return 512*2**20 + 1
    monkeypatch.setattr(pipeline, 'save', lambda _: OversizedSerialization())
    with pytest.raises(protocol.Blocked, match='512 MiB'):
        pipeline.capture(model, None, config, {}, tmp_path)
    assert not list(Path(tmp_path).glob('target-*.safetensors'))
    assert not model.model.layers[0].self_attn._forward_hooks
