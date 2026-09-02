"""tail_loss must equal the model's own label loss, at a fraction of the memory.

Training passed ``labels`` to the model, which computes logits at every position:
1152 x 151936 values (~350 MB fp16, plus an fp32 copy in the loss and a gradient
of the same size) to supervise a single answer token. tail_loss asks only for the
trailing positions that cover the supervised span. This checks the shortcut is
exact rather than merely cheaper.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from rineya_fwa.train import tail_loss

VOCAB = 37
HID = 8


class _TinyLM(nn.Module):
    """Stands in for a causal LM: honours logits_to_keep the way HF does."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB, HID)
        self.head = nn.Linear(HID, VOCAB, bias=False)

    def forward(self, input_ids=None, attention_mask=None, labels=None, logits_to_keep=0, **kw):
        h = self.emb(input_ids)
        # causal-ish mixing so positions are not independent
        h = torch.cumsum(h, dim=1) / torch.arange(1, h.shape[1] + 1, device=h.device).view(1, -1, 1)
        logits = self.head(h)
        if logits_to_keep:
            logits = logits[:, -int(logits_to_keep) :, :]
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape(-1, VOCAB)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = nn.functional.cross_entropy(shift_logits.float(), shift_labels, ignore_index=-100)
        return type("Out", (), {"logits": logits, "loss": loss})()


@pytest.mark.parametrize("n_sup", [1, 3])
def test_tail_loss_matches_full_label_loss(n_sup):
    torch.manual_seed(0)
    model = _TinyLM()
    T = 24
    ids = torch.randint(0, VOCAB, (1, T))
    attn = torch.ones_like(ids)
    labels = torch.full((1, T), -100)
    labels[0, T - n_sup :] = ids[0, T - n_sup :]

    reference = model(input_ids=ids, attention_mask=attn, labels=labels).loss
    got = tail_loss(model, ids, attn, labels)
    assert got is not None
    torch.testing.assert_close(got, reference, rtol=1e-6, atol=1e-6)


def test_tail_loss_requests_only_the_needed_positions():
    torch.manual_seed(0)
    model = _TinyLM()
    T = 64
    ids = torch.randint(0, VOCAB, (1, T))
    labels = torch.full((1, T), -100)
    labels[0, -1] = ids[0, -1]

    seen = {}
    inner = model.forward

    def spy(**kw):
        seen["logits_to_keep"] = kw.get("logits_to_keep")
        return inner(**kw)

    model.forward = spy
    tail_loss(model, ids, torch.ones_like(ids), labels)
    assert seen["logits_to_keep"] == 2, f"asked for {seen['logits_to_keep']} of {T} positions"


def test_tail_loss_returns_none_when_nothing_is_supervised():
    model = _TinyLM()
    ids = torch.randint(0, VOCAB, (1, 8))
    labels = torch.full((1, 8), -100)
    assert tail_loss(model, ids, torch.ones_like(ids), labels) is None
