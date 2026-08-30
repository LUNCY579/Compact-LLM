from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from rineya_tgsa.mtp_adapter import (
    CombinedRetrofitConfig,
    RineyaTGSAWithMTP,
    adapter_only_state_dict,
    apply_combined_stage,
    attach_mtp,
    audit_main_logit_neutrality,
)


class FakeAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.carrier_init = nn.Linear(dim, dim, bias=False)
        self.conditioning = nn.Linear(dim, dim, bias=False)


class FakeLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_attn = FakeAttention(dim)
        self.mlp = nn.Linear(dim, dim, bias=False)


class FakeBackbone(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(dim), FakeLayer(dim)])


class FakeCausalLM(nn.Module):
    def __init__(self, vocab: int = 19, dim: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab, hidden_size=dim)
        self.embed = nn.Embedding(vocab, dim)
        self.model = FakeBackbone(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.lm_head.weight = self.embed.weight

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **_: object,
    ) -> SimpleNamespace:
        hidden = self.embed(input_ids)
        logits = self.lm_head(hidden)
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden,) if output_hidden_states else None,
            past_key_values=None,
        )


def test_attachment_is_bitwise_main_logit_neutral() -> None:
    torch.manual_seed(3)
    base = FakeCausalLM()
    wrapper, manifest = attach_mtp(
        base,
        CombinedRetrofitConfig(max_offset=4, mtp_rank=4, initialization_seed=99),
    )
    input_ids = torch.tensor([[1, 4, 7, 2]])
    report = audit_main_logit_neutrality(base, wrapper, input_ids)
    assert report["bitwise_identical"] is True
    assert report["max_absolute_difference"] == 0.0
    assert manifest["main_logits_modified"] is False
    output = wrapper(input_ids=input_ids)
    assert set(output.mtp_logits) == {2, 3, 4}
    assert all(value.shape == output.logits.shape for value in output.mtp_logits.values())


def test_joint_loss_connects_mtp_and_backbone_gradients() -> None:
    torch.manual_seed(5)
    base = FakeCausalLM()
    wrapper, _ = attach_mtp(
        base,
        CombinedRetrofitConfig(max_offset=3, mtp_rank=4, initialization_seed=17),
    )
    input_ids = torch.tensor([[1, 4, 7, 8, 9, 2]])
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    target = torch.tensor([[False, False, True, True, True, True]])
    output = wrapper(
        input_ids=input_ids,
        attention_mask=mask,
        target_mask=target,
        compute_joint_loss=True,
    )
    assert output.loss is not None
    output.loss.backward()
    assert wrapper.mtp_heads.heads["2"].up.weight.grad is not None
    assert float(wrapper.mtp_heads.heads["2"].up.weight.grad.abs().sum()) > 0.0
    assert base.embed.weight.grad is not None
    assert float(base.embed.weight.grad.abs().sum()) > 0.0


def test_adapter_state_excludes_native_pretrained_tensors() -> None:
    base = FakeCausalLM()
    wrapper, _ = attach_mtp(base, CombinedRetrofitConfig(max_offset=3, mtp_rank=4))
    state = adapter_only_state_dict(wrapper)
    assert any(name.startswith("mtp_heads.") for name in state)
    assert any("carrier_init" in name for name in state)
    assert not any("q_proj" in name for name in state)
    assert not any("lm_head" in name for name in state)


def test_combined_warm_start_releases_expected_groups() -> None:
    base = FakeCausalLM()
    wrapper, _ = attach_mtp(base, CombinedRetrofitConfig(max_offset=3, mtp_rank=4))
    report = apply_combined_stage(wrapper, "adapters", converted_layers=[0])
    assert report["trainable_parameters"] > 0
    named = dict(wrapper.named_parameters())
    assert named["mtp_heads.heads.2.up.weight"].requires_grad
    assert named["base_model.model.layers.0.self_attn.carrier_init.weight"].requires_grad
    assert not named["base_model.model.layers.0.self_attn.q_proj.weight"].requires_grad
    assert not named["base_model.model.layers.1.self_attn.carrier_init.weight"].requires_grad

    apply_combined_stage(wrapper, "attention", converted_layers=[0])
    assert named["base_model.model.layers.0.self_attn.q_proj.weight"].requires_grad
    assert not named["base_model.model.layers.0.mlp.weight"].requires_grad

    apply_combined_stage(wrapper, "converted_layers", converted_layers=[0])
    assert named["base_model.model.layers.0.mlp.weight"].requires_grad
    assert not named["base_model.model.layers.1.mlp.weight"].requires_grad
