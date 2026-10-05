"""Structural cache/hook tests. This test double is NOT a Qwen3 model."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from experiments.rineya_triadic_working_memory_v1.core import MemoryConfig
from experiments.rineya_triadic_working_memory_v1.sidecar import MemorySidecar
from experiments.rineya_triadic_working_memory_v1.native import FrozenQwenSidecar, greedy_decode


class ToyCache:
    def __init__(self):
        self.layers = {}

    def get_seq_length(self):
        return self.layers[0][0].shape[1] if 0 in self.layers else 0


class ToyLayer(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
        self.qkv = nn.Linear(16, 48, bias=False)
        self.out = nn.Linear(16, 16, bias=False)

    def forward(self, hidden, cache=None, mask=None):
        q, k, v = self.qkv(hidden).chunk(3, -1)
        prefix = 0
        if cache is not None:
            if self.index in cache.layers:
                old_k, old_v = cache.layers[self.index]
                prefix = old_k.shape[1]
                k, v = torch.cat((old_k, k), 1), torch.cat((old_v, v), 1)
            cache.layers[self.index] = (k, v)
        scores = q @ k.transpose(-1, -2) / 4
        visible = torch.arange(k.shape[1])[None, :] <= torch.arange(prefix, prefix+q.shape[1])[:, None]
        scores = scores.masked_fill(~visible, -1e9)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None].bool(), -1e9)
        return hidden + self.out(scores.softmax(-1) @ v)


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3")  # Interface stub only.
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(32, 16)
        self.model.layers = nn.ModuleList([ToyLayer(i) for i in range(3)])
        self.lm_head = nn.Linear(16, 32, bias=False)
        self.is_gradient_checkpointing = False

    def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None,
                use_cache=False, return_dict=True):
        hidden = self.model.embed_tokens(input_ids)
        cache = (past_key_values or ToyCache()) if use_cache else None
        for layer in self.model.layers:
            hidden = layer(hidden, cache=cache, mask=attention_mask)
        return SimpleNamespace(logits=self.lm_head(hidden), loss=None, past_key_values=cache)


def setup(active=True):
    torch.manual_seed(12)
    base = ToyBackbone()
    sidecar = MemorySidecar(MemoryConfig(d_model=16, heads=2, key_dim=4, value_dim=4, slices=4)) if active else None
    wrapper = FrozenQwenSidecar(base, sidecar, layer=1).eval()
    return wrapper, torch.tensor([[2, 5, 8, 1, 4]])


def test_exact_neutral_and_disabled_identity():
    model, ids = setup()
    with torch.no_grad():
        before = model.backbone(ids).logits
        assert torch.equal(model(ids).logits, before)
        model.sidecar.gate.fill_(0.2)
        assert torch.equal(model(ids, enabled=False).logits, before)
    model.close()


@pytest.mark.parametrize("active", [False, True])
def test_active_full_cached_logits_and_greedy_tokens(active):
    model, ids = setup(active)
    if active:
        model.sidecar.gate.data.fill_(0.2)
    with torch.no_grad():
        full = model(ids).logits
        state, pieces = None, []
        for i in range(ids.shape[1]):
            result = model(ids[:, i:i+1], cache=state, use_cache=True)
            state = result.cache
            pieces.append(result.logits)
        torch.testing.assert_close(torch.cat(pieces, 1), full, atol=2e-6, rtol=2e-6)
        assert greedy_decode(model, ids, 8, cached=True) == greedy_decode(model, ids, 8, cached=False)
    model.close()


def test_request_isolation_and_cache_ownership():
    model, ids = setup()
    other, _ = setup()
    with torch.no_grad():
        model.sidecar.gate.fill_(0.2)
        first = model(ids).logits
        model(ids.flip(1))
        assert torch.equal(first, model(ids).logits)
        cache = model(ids, use_cache=True).cache
        with pytest.raises(ValueError):
            other(ids[:, :1], cache=cache, use_cache=True)
        with pytest.raises(ValueError):
            model(ids[:, :1], cache=cache, use_cache=True, enabled=False)
        result = model(ids[:, :1], cache=cache, use_cache=True)
        assert result.cache.length == ids.shape[1]+1
        with pytest.raises(ValueError):
            model(ids[:, :1], cache=cache, use_cache=True)
    model.close()
    other.close()


def test_cache_training_padding_and_checkpointing_rejected():
    model, ids = setup()
    with pytest.raises(ValueError):
        model(ids, use_cache=True)
    with torch.no_grad():
        mask = torch.ones_like(ids)
        mask[:, -1] = 0
        with pytest.raises(ValueError):
            model(ids, attention_mask=mask, use_cache=True)
    model.backbone.is_gradient_checkpointing = True
    with pytest.raises(ValueError):
        model(ids)
    model.close()


def test_optimizer_changes_sidecar_not_backbone():
    model, ids = setup()
    model.train()
    assert not model.backbone.training and model.sidecar.training
    initial = {n: p.detach().clone() for n, p in model.backbone.named_parameters()}
    opt = torch.optim.AdamW(model.sidecar.parameters(), lr=0.01)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        model(ids).logits.square().mean().backward()
        model.assert_frozen()
        opt.step()
    assert model.sidecar.gate.abs() > 0
    for n, p in model.backbone.named_parameters():
        assert not p.requires_grad and p.grad is None
        assert torch.equal(initial[n], p)
    model.close()


def test_close_double_wrapping_and_reentrant_failure_cleanup():
    model, ids = setup()
    with pytest.raises(ValueError):
        FrozenQwenSidecar(model.backbone, None, 1)
    # Unsupported memory ablation raises in the hook; call context must be cleared.
    with pytest.raises(ValueError):
        model(ids, ablation="unsupported")
    assert model._context is None
    model(ids)
    model.close()
    with pytest.raises(RuntimeError):
        model(ids)
    replacement = FrozenQwenSidecar(model.backbone, None, 1).eval()
    replacement(ids)
    replacement.close()


def test_decode_profile_counts_prefill_and_continuations():
    model, ids = setup()
    tokens, stop, timing = greedy_decode(model, ids, 3, return_profile=True)
    assert len(tokens) == 3 and stop == "length"
    assert timing["continuation_forwards"] == 2
    assert timing["prefill_seconds"] >= 0 and timing["decode_seconds"] >= 0
    assert timing["prompt_tokens"] == ids.shape[1]
    model.close()
