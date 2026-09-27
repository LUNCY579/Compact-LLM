"""A small positive and negative control for native state comparison."""
from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from experiments.maxrl_comp002_pretraining_v1.q1_native_loader import state_fingerprint


class TiedTiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(3, 4)
        self.lm_head = torch.nn.Linear(4, 3, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def test_full_tied_state_matches(tmp_path):
    model = TiedTiny()
    source = tmp_path / "tiny.safetensors"
    save_file({"model.embed_tokens.weight": model.model.embed_tokens.weight.detach().clone()}, str(source))
    result = state_fingerprint(model, source, torch)
    assert result["stored_tensors_matched"] == 1
    assert result["tied_head_alias"] is True


def test_one_element_change_is_detected(tmp_path):
    model = TiedTiny()
    source = tmp_path / "tiny.safetensors"
    save_file({"model.embed_tokens.weight": model.model.embed_tokens.weight.detach().clone()}, str(source))
    with torch.no_grad():
        model.model.embed_tokens.weight[0, 0] += 1
    with pytest.raises(RuntimeError, match="loaded tensor differs"):
        state_fingerprint(model, source, torch)
