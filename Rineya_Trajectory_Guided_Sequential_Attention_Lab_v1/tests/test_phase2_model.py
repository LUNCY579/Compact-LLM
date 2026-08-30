from __future__ import annotations

import torch

from tgsa.mtp import MTPConfig
from tgsa.phase1_model import TinyLMConfig, state_dict_sha256
from tgsa.phase2_model import build_factorial_models


def test_factorial_models_are_exactly_paired_with_identical_mtp_heads() -> None:
    base = TinyLMConfig(
        vocab_size=31,
        max_sequence_length=16,
        embed_dim=24,
        num_heads=4,
        num_layers=1,
        ffn_multiplier=2.0,
        dropout=0.0,
        arm="full",
    )
    mtp = MTPConfig(hidden_size=24, vocab_size=31, max_offset=4, rank=6)
    bundle = build_factorial_models(base, mtp, initialization_seed=17, tgsa_arm="full")
    assert bundle.initialization_audit["checks"]["parallel_ar_mtp_exact_state"] is True
    assert bundle.initialization_audit["checks"]["tgsa_ar_mtp_exact_state"] is True
    assert bundle.initialization_audit["checks"]["mtp_heads_identical_across_all_cells"] is True
    assert state_dict_sha256(bundle.models["parallel_ar"]) == state_dict_sha256(
        bundle.models["parallel_mtp"]
    )
    assert state_dict_sha256(bundle.models["tgsa_ar"]) == state_dict_sha256(
        bundle.models["tgsa_mtp"]
    )
    assert not any(parameter.requires_grad for parameter in bundle.models["parallel_ar"].mtp_heads.parameters())
    assert all(parameter.requires_grad for parameter in bundle.models["parallel_mtp"].mtp_heads.parameters())


def test_mtp_attachment_does_not_change_main_logits_at_initialization() -> None:
    base = TinyLMConfig(
        vocab_size=29,
        max_sequence_length=12,
        embed_dim=24,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        arm="full",
    )
    mtp = MTPConfig(hidden_size=24, vocab_size=29, max_offset=3, rank=6)
    bundle = build_factorial_models(base, mtp, initialization_seed=23, tgsa_arm="full")
    tokens = torch.randint(0, 29, (2, 8))
    for left, right in (("parallel_ar", "parallel_mtp"), ("tgsa_ar", "tgsa_mtp")):
        bundle.models[left].eval()
        bundle.models[right].eval()
        with torch.no_grad():
            left_logits = bundle.models[left](tokens).logits
            right_logits = bundle.models[right](tokens).logits
        assert torch.equal(left_logits, right_logits)
