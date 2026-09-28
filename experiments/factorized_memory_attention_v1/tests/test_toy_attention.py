import torch

from experiments.factorized_memory_attention_v1.fma.config import FMAConfig
from experiments.factorized_memory_attention_v1.fma.toy import ToyLM, tiny_shape


def make(gate):
    torch.manual_seed(20260925)
    return ToyLM(tiny_shape(), FMAConfig(rank=4, layers=(0,), gate=gate)).eval()


def test_causal_prefix_is_unchanged_by_future_tokens():
    model = make(1)
    prefix = torch.tensor([[1, 2, 3]])
    extended = torch.tensor([[1, 2, 3, 9, 10]])
    with torch.inference_mode():
        prefix_logits, _ = model(prefix)
        extended_logits, _ = model(extended)
    torch.testing.assert_close(prefix_logits, extended_logits[:, :3])


def _assert_cached_matches_full(model):
    ids = torch.tensor([[1, 2, 3]])
    past = None
    current = ids
    full = ids
    with torch.inference_mode():
        for _ in range(4):
            step, past = model(current, past=past, use_cache=True)
            direct, _ = model(full, use_cache=False)
            torch.testing.assert_close(step[:, -1], direct[:, -1], atol=1e-6, rtol=1e-6)
            current = step[:, -1].argmax(-1, keepdim=True)
            full = torch.cat((full, current), dim=1)


def test_cache_semantics_match_full_reforward_at_g0():
    _assert_cached_matches_full(make(0))


def test_cache_semantics_match_full_reforward_at_g1():
    _assert_cached_matches_full(make(1))


def test_micro_generation_smoke_is_finite_and_paths_differ():
    ids = torch.tensor([[1, 2, 3]])
    standard = make(0).generate(ids, 4)
    hard_memory = make(1).generate(ids, 4)
    assert standard.shape == hard_memory.shape == (1, 7)
    assert not torch.equal(standard, hard_memory)
