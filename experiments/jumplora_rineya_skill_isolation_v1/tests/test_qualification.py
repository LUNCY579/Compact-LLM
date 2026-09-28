from __future__ import annotations

from dataclasses import asdict

import torch

from experiments.jumplora_rineya_skill_isolation_v1.jump_lora import (
    JumpLoRALinear,
    JumpSchedule,
    deterministic_topk_mask,
    threshold_for_mask,
)
from experiments.jumplora_rineya_skill_isolation_v1.integration import inject_arm
from experiments.jumplora_rineya_skill_isolation_v1.qualification import (
    check_baseline_load_and_disabled_path,
    check_counts_and_finiteness,
    check_dense_state_equivalence,
    check_deterministic_masks,
    check_matched_density,
    check_merge_unmerge_and_serialization,
    check_paired_ids_and_sealed_guard,
    check_reproducibility,
    check_threshold_ste,
    run_qualification,
)


def test_jlr_qual_001_receipt_passes(tmp_path):
    receipt = run_qualification(tmp_path / "receipt.json")
    assert receipt["qualification_id"] == "JLR-QUAL-001"
    assert receipt["status"] == "PASS"
    assert receipt["result_classification"] == "MECHANICAL_ONLY"
    assert receipt["sealed_access"] is False
    assert receipt["model_loaded"] is False


def test_required_mechanical_checks():
    for check in (
        check_dense_state_equivalence,
        check_deterministic_masks,
        check_threshold_ste,
        check_matched_density,
        check_merge_unmerge_and_serialization,
        check_baseline_load_and_disabled_path,
        check_counts_and_finiteness,
        check_reproducibility,
        check_paired_ids_and_sealed_guard,
    ):
        assert isinstance(check(), dict)


def test_threshold_selection_is_exact_and_deterministic():
    values = torch.tensor([[-5.0, 4.0], [2.0, -1.0]])
    mask = deterministic_topk_mask(values, 2)
    tau = threshold_for_mask(values, 2)
    assert int(mask.sum()) == 2
    assert int(values.abs().gt(tau).sum()) == 2
    assert torch.equal(mask, deterministic_topk_mask(values, 2))


def test_jump_gamma_reaches_sparse_endpoint():
    adapter = JumpLoRALinear(torch.nn.Linear(4, 4, bias=False), rank=2,
                             alpha=4.0, schedule=JumpSchedule(2, 4), threshold=0.1)
    adapter.set_step(0)
    assert adapter.gamma == 0.0
    adapter.set_step(4)
    assert adapter.gamma == 1.0


def test_qv_injection_is_scoped_and_restorable():
    class TinyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.v_proj = torch.nn.Linear(4, 4, bias=False)
            self.other = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return self.q_proj(x) + self.v_proj(x) + self.other(x)

    model = TinyAttention()
    x = torch.randn(3, 4)
    before = model(x).detach()
    handle = inject_arm(model, "J3", rank=2, alpha=4.0,
                        schedule=JumpSchedule(1, 3))
    assert len(handle.bindings) == 2
    torch.testing.assert_close(model(x), before, rtol=0.0, atol=0.0)
    assert handle.parameter_counts()["adapter_parameters"] == 2 * (2 * (4 + 4) + 1)
    handle.restore()
    torch.testing.assert_close(model(x), before, rtol=0.0, atol=0.0)
