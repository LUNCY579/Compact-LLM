from dataclasses import replace
import io
import pytest
import torch
from experiments.rineya_triadic_working_memory_v1.core import MemoryConfig, MemoryState
from experiments.rineya_triadic_working_memory_v1.sidecar import MemorySidecar, build_arm, parameter_count
from experiments.rineya_triadic_working_memory_v1.analysis import memory_diagnostics


def config():
    return MemoryConfig(d_model=16, heads=2, key_dim=4, value_dim=3, slices=4, conv_width=4)


def fixture():
    torch.manual_seed(7)
    module = MemorySidecar(config()).double()
    module.gate.data.fill_(0.2)
    return module, torch.randn(2, 9, 16, dtype=torch.float64)


@pytest.mark.parametrize("arm", ["stateless", "gdn_e1", "triadic_e4", "matrix_state_matched"])
def test_identity_initialization_and_live_gate_gradient(arm):
    torch.manual_seed(42)
    module = build_arm(arm, config())
    hidden = torch.randn(2, 7, 16)
    output, _ = module(hidden)
    assert torch.equal(output, hidden)
    (output-torch.randn_like(output)).square().mean().backward()
    assert module.gate.grad.abs() > 1e-9


@pytest.mark.parametrize("chunks", [[1]*9, [2, 3, 4], [4, 5]])
def test_full_vs_incremental_with_convolution_history(chunks):
    module, hidden = fixture()
    full, full_state = module(hidden)
    cursor, state, outputs = 0, None, []
    for length in chunks:
        out, state = module(hidden[:, cursor:cursor+length], state)
        cursor += length
        outputs.append(out)
    torch.testing.assert_close(torch.cat(outputs, 1), full, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(state.tensor, full_state.tensor, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(state.history, full_state.history, atol=1e-12, rtol=1e-12)


def test_document_reset_clears_both_state_and_history():
    module, hidden = fixture()
    reset = torch.zeros(2, 9, dtype=torch.bool)
    reset[:, 4] = True
    together, state = module(hidden, reset=reset)
    separate, expected_state = module(hidden[:, 4:])
    torch.testing.assert_close(together[:, 4:], separate, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(state.tensor, expected_state.tensor, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(state.history, expected_state.history, atol=1e-12, rtol=1e-12)


def test_padding_does_not_write_decay_or_advance_convolution():
    module, hidden = fixture()
    valid = torch.ones(2, 9, dtype=torch.bool)
    valid[:, [2, 5, 8]] = False
    actual, state = module(hidden, valid=valid)
    expected, expected_state = module(hidden[:, [0, 1, 3, 4, 6, 7]])
    assert torch.equal(actual[:, [2, 5, 8]], hidden[:, [2, 5, 8]])
    torch.testing.assert_close(actual[:, [0, 1, 3, 4, 6, 7]], expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(state.tensor, expected_state.tensor, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(state.history, expected_state.history, atol=1e-12, rtol=1e-12)


def test_future_tokens_do_not_affect_past_outputs():
    module, hidden = fixture()
    altered = hidden.clone()
    altered[:, 5:] *= 100
    a, _ = module(hidden)
    b, _ = module(altered)
    torch.testing.assert_close(a[:, :5], b[:, :5], atol=0, rtol=0)


def test_read_only_preserves_evidence_state_but_advances_query_history():
    module, hidden = fixture()
    _, state = module(hidden[:, :4])
    original = state.clone()
    _, new = module(hidden[:, 4:], state, ablation="read_only")
    assert torch.equal(state.tensor, original.tensor)
    assert torch.equal(state.history, original.history)
    assert torch.equal(new.tensor, original.tensor)
    assert not torch.equal(new.history, original.history)


def test_erase_reads_and_second_key_collapse_ablations():
    module, hidden = fixture()
    actual, _ = module(hidden)
    erased, state = module(hidden, ablation="erase_reads")
    constant, _ = module(hidden, ablation="constant_second_keys")
    assert torch.equal(erased, hidden)
    assert state.tensor.norm() > 0
    assert not torch.allclose(actual, constant)


def test_active_sidecar_parameters_receive_gradients():
    module, hidden = fixture()
    output, _ = module(hidden)
    output.square().mean().backward()
    for name, p in module.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.norm() > 0, name


def test_streaming_preserves_autograd_until_explicit_detach():
    module, hidden = fixture()
    hidden.requires_grad_()
    _, state = module(hidden[:, :4])
    output, _ = module(hidden[:, 4:], state)
    output.sum().backward()
    assert hidden.grad[:, :4].norm() > 0
    detached = state.detach()
    assert not detached.tensor.requires_grad and not detached.history.requires_grad


def test_safe_state_dict_round_trip():
    module, hidden = fixture()
    stream = io.BytesIO()
    torch.save(module.state_dict(), stream)
    stream.seek(0)
    other = MemorySidecar(config()).double()
    other.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(module(hidden)[0], other(hidden)[0], atol=0, rtol=0)


def test_state_budget_rejects_before_allocation():
    module = MemorySidecar(replace(config(), max_state_bytes=1))
    with pytest.raises(MemoryError):
        module.initial_state(1)


def test_bad_cache_and_mask_rejected():
    module, hidden = fixture()
    state = module.initial_state(1)
    with pytest.raises(ValueError):
        module(hidden, state)
    with pytest.raises(ValueError):
        module(hidden, valid=torch.ones(2, 9))
    valid = torch.ones(2, 9, dtype=torch.bool)
    valid[:, 0] = False
    with pytest.raises(ValueError):
        module(hidden, valid=valid, reset=~valid)
    with pytest.raises(ValueError):
        module(hidden[:, :0])


def test_mixed_backbone_bf16_keeps_reference_state_fp32():
    module = MemorySidecar(config())
    output, state = module(torch.randn(2, 4, 16).bfloat16())
    assert output.dtype == torch.bfloat16
    assert state.tensor.dtype == state.history.dtype == torch.float32
    with pytest.raises(ValueError):
        module.bfloat16()(torch.randn(2, 4, 16).bfloat16())


def test_controls_match_declared_budgets_not_dummy_parameters():
    c = MemoryConfig()
    triadic = build_arm("triadic_e4", c)
    stateless = build_arm("stateless", c)
    matrix = build_arm("matrix_state_matched", c)
    assert abs(parameter_count(triadic)-parameter_count(stateless))/parameter_count(triadic) < .01
    a, b = triadic.initial_state(1), matrix.initial_state(1)
    assert a.tensor.numel() == b.tensor.numel()
    assert a.history.numel() != b.history.numel()  # Explicitly NOT a total-buffer match.
    assert parameter_count(matrix) > parameter_count(triadic)


def test_memory_bytes_and_address_rank_diagnostics():
    module, hidden = fixture()
    _, state = module(hidden)
    report = memory_diagnostics(state, torch.eye(4))
    assert report["total_buffer_bytes"] == config().state_bytes(2, 8)
    assert report["second_key_effective_rank"] == pytest.approx(4)
    assert memory_diagnostics(state, torch.ones(8, 4))["second_key_effective_rank"] == pytest.approx(1)
