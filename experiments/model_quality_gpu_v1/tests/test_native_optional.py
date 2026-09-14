"""Actual random native-Qwen checks, skipped honestly without the pinned runtime.

These are NOT official-source qualification and cannot open training gates.
"""
import pytest
import torch
from torch.nn import functional as F
from experiments.model_quality_gpu_v1.losses import chunked_sum
from experiments.model_quality_gpu_v1.native import hidden_states, select_tail


@pytest.mark.parametrize("checkpointed", [False, True])
def test_native_random_qwen(checkpointed):
    hf = pytest.importorskip("transformers")
    if hf.__version__ != "4.57.1":
        # Skip, not fail. This module's own docstring promises to be "skipped honestly
        # without the pinned runtime", and a wrong version IS the pinned runtime being
        # unavailable. Failing here made the outcome depend on HOW the runtime was wrong --
        # absent skipped, present-but-mismatched failed -- so the suite was green only on
        # hosts with no Transformers at all, and red on every host that had a different
        # one. A red suite that means "your environment differs" trains an operator to
        # ignore red. The skip reason names both versions, so it is still visible.
        pytest.skip(f"native fixture needs pinned transformers 4.57.1, found "
                    f"{hf.__version__}; this is an environment mismatch, not a defect")
    torch.manual_seed(17); torch.set_num_threads(1)
    c = hf.Qwen3Config(vocab_size=41, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                      head_dim=8, tie_word_embeddings=True, attention_dropout=0.0)
    c._attn_implementation = "eager"
    m = hf.Qwen3ForCausalLM(c)
    select_tail(m, 2)
    if checkpointed:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ids = torch.tensor([[2, 5, 8, 3, 7, 6]])
    labels = torch.tensor([[-100, -100, -100, 3, 7, 6]])
    h = hidden_states(m, ids)
    loss, n = chunked_sum(h, m.lm_head.weight, labels, 2)
    reference = F.cross_entropy(F.linear(h[:, :-1], m.lm_head.weight).reshape(-1, 41),
                                labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
    torch.testing.assert_close(loss, reference)
    loss.backward()
    assert n == 3
    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
