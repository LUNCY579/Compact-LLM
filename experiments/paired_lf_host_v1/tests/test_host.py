from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from torch import nn
from .. import host
from experiments.paired_lf_v1.contracts import ContractError


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8,8)
    def forward(self, x):
        return x + self.linear(x), None


class Fixture(nn.Module):
    """Independent shape fixture, NOT a Qwen approximation or CED quality model."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=20)
        self.embed_tokens = nn.Embedding(20,8)
        self.encoder = nn.ModuleList([Block(), Block()])
    def forward(self, ids, **kwargs):
        assert kwargs == dict(use_cache=False, compute_logits=False, return_hidden=True)
        x = self.embed_tokens(ids)
        for block in self.encoder: x, _ = block(x)
        return SimpleNamespace(hidden=x)


def fixture():
    return Fixture().eval().requires_grad_(False)


def test_capture_matches_host_output_and_candidate_projection():
    m = fixture()
    ids = torch.tensor([[1,2,3]])
    before = host.frozen_identity(m)
    f = host.capture_features(m, ids, tuple(range(7)), 1)
    with torch.inference_mode():
        out = m(ids, use_cache=False, compute_logits=False, return_hidden=True)
        torch.testing.assert_close(f.query, out.hidden[0,-1], rtol=0, atol=0)
        torch.testing.assert_close(f.prior, out.hidden[0,-1] @ m.embed_tokens.weight[:7].T, rtol=0, atol=0)
    assert f.memory.shape == (3,8) and before == host.frozen_identity(m)
    assert not m.encoder[0]._forward_hooks


def test_hook_removed_on_failure():
    m = fixture()
    with patch.object(m, 'forward', side_effect=ValueError('injected')):
        with pytest.raises(ValueError): host.capture_features(m, torch.tensor([[1]]), tuple(range(7)), 1)
    assert not m.encoder[0]._forward_hooks


@pytest.mark.parametrize("ids", [torch.tensor([[1.0]]), torch.tensor([1]), torch.ones(2,3,dtype=torch.long), torch.ones(1,513,dtype=torch.long)])
def test_capture_rejects_invalid_inputs(ids):
    with pytest.raises(ContractError): host.capture_features(fixture(), ids, tuple(range(7)), 1)


@pytest.mark.parametrize("candidates", [(0,)*7, tuple(range(6)), (0,1,2,3,4,5,30)])
def test_capture_rejects_bad_answer_mapping(candidates):
    with pytest.raises(ContractError): host.capture_features(fixture(), torch.tensor([[1]]), candidates, 1)


def test_no_unfrozen_backbone():
    m = fixture().requires_grad_(True)
    with pytest.raises(ContractError): host.capture_features(m, torch.tensor([[1]]), tuple(range(7)), 1)


def test_preflight_does_not_load_a_model(tmp_path):
    with patch.object(host, 'verify_parent', side_effect=ValueError('parent absent')):
        r = host.preflight(tmp_path / 'missing')
    assert r['status'] == 'BLOCKED' and r['native_weights_loaded'] is False and r['optimizer_steps'] == 0
    assert any('source asset missing' in b for b in r['blockers'])


def test_tokenizer_receives_only_prompt():
    seen = []
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen.append(messages)
            return torch.tensor([[1,2]])
    row = {'prompt':'Question', 'answer':'SECRET'}
    host._ids(Tokenizer(), row)
    assert seen == [[{'role':'user','content':'Question'}]]


def test_hash_detects_weight_changes():
    m = fixture()
    before = host.frozen_identity(m)
    with torch.no_grad(): m.embed_tokens.weight.add_(0.1)
    assert host.frozen_identity(m) != before


def test_capture_worker_requires_permit(tmp_path):
    from ..__main__ import main
    with patch('sys.argv', ['host','_worker','--source',str(tmp_path/'missing'),'--out',str(tmp_path)]):
        assert main() == 2
    assert not (tmp_path/'bank').exists()


def test_unapproved_permit_cannot_open_capture(tmp_path):
    from ..__main__ import approve
    p = tmp_path/'permit.json'
    p.write_text('{"approved": false}')
    with pytest.raises(ContractError): approve(p,tmp_path/'source')
