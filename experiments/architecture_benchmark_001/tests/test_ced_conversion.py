"""Registered CED conversion routes and the properties `run.convert_ced` asserts.

Synthetic mechanics only. The end-to-end `convert-ced` stage cannot be exercised with a
fixture by design: `validate_receipt` refuses any context whose `weights_sha256` is not
the registered official one, so no fixture can forge the parent receipt. Real-weight
integration evidence therefore lives in `evidence/OFFICIAL_GATE_LADDER.md`, and what is
testable here is the route table, the budget contract, and the conversion invariants the
stage checks.
"""
import json
import pytest
import torch
from experiments.architecture_benchmark_001 import run
from experiments.architecture_benchmark_001.budget import Config, budget, presets
from experiments.architecture_benchmark_001.model import DenseReference
from experiments.architecture_benchmark_001.transfer import state_digest, to_ced

SAME = ('vocab_size', 'hidden_size', 'head_dim', 'num_attention_heads',
        'num_key_value_heads', 'rope_theta', 'rms_norm_eps', 'max_position_embeddings')


def tiny_plain(layers=4):
    return Config(vocab_size=37, hidden_size=24, intermediate_size=48, num_attention_heads=6,
                  num_key_value_heads=3, head_dim=8, encoder_layers=layers, decoder_layers=0,
                  architecture='plain')


def tiny_ced(intermediate=64):
    return Config(vocab_size=37, hidden_size=24, intermediate_size=intermediate,
                  num_attention_heads=6, num_key_value_heads=3, head_dim=8,
                  encoder_layers=2, decoder_layers=2, architecture='ced')


@pytest.mark.parametrize('arm', sorted(run.CED_ROUTES))
def test_registered_route_satisfies_every_to_ced_precondition(arm):
    donor_name, target_name, parent_stage = run.CED_ROUTES[arm]
    donor, target = presets()[donor_name], presets()[target_name]
    assert donor.architecture == 'plain' and target.architecture == 'ced'
    assert donor.total_layers == target.total_layers
    assert all(getattr(donor, k) == getattr(target, k) for k in SAME)
    assert target.intermediate_size >= donor.intermediate_size, 'FFN may not shrink'
    # to_ced requires target.encoder_layers <= donor_layer < donor.total_layers, and the
    # stage always passes donor_layer = target.encoder_layers, so that must be in range.
    assert 0 < target.encoder_layers < donor.total_layers
    assert parent_stage in ('G0', 'G1_WIDTH')


@pytest.mark.parametrize('arm', sorted(run.CED_ROUTES))
def test_ced_arms_stay_budget_matched_and_cut_analytic_kv(arm):
    protocol_tolerance = run.protocol()['parameter_relative_tolerance']
    base = budget(presets()['B28_1024_official_shape'])
    target = budget(presets()[run.CED_ROUTES[arm][1]])
    assert abs(target['parameters']/base['parameters'] - 1) <= protocol_tolerance
    assert target['kv_bytes_per_token_batch1'] < base['kv_bytes_per_token_batch1']
    assert target['kv_banks'] == presets()[run.CED_ROUTES[arm][1]].encoder_layers + 1


def test_donor_policy_is_the_first_decoder_layer_and_lower_donors_are_rejected():
    assert run.protocol()['G1']['memory_donor_policy'] == 'first_decoder_layer_of_plain_donor'
    donor, target = DenseReference(tiny_plain(4)).eval(), tiny_ced()
    # The stage always passes donor_layer = target.encoder_layers; anything below the
    # encoder/decoder boundary, or at/beyond the donor depth, must fail closed.
    _, mapping = to_ced(donor, target, donor_layer=target.encoder_layers)
    assert mapping['donor_layer'] == target.encoder_layers == 2
    for bad in (target.encoder_layers - 1, -1, donor.config.total_layers):
        with pytest.raises(ValueError):
            to_ced(donor, target, donor_layer=bad)


@pytest.mark.parametrize('arm', sorted(run.CED_ROUTES))
def test_convert_ced_requires_its_declared_parent_evidence(arm, tmp_path):
    code = run.main(['convert-ced', '--arm', arm, '--source', str(tmp_path/'src'),
                     '--out', str(tmp_path/'out')])
    assert code == 2
    assert not (tmp_path/'out').exists(), 'a rejected invocation must not create an output tree'


def test_conversion_invariants_the_stage_checks_hold_on_a_tiny_pair():
    """Mirrors convert_ced's checks: bank present, no private decoder KV, donor untouched."""
    torch.manual_seed(17)
    donor = DenseReference(tiny_plain(4)).eval()
    target = tiny_ced()
    before = state_digest(donor)
    converted, mapping = to_ced(donor, target, donor_layer=2)
    assert converted.config == target
    assert sum(p.numel() for p in converted.parameters()) == budget(target)['parameters']
    assert converted.memory is not None
    assert [n for n, _ in converted.decoder.named_parameters() if '.k_proj.' in n or '.v_proj.' in n] == []
    assert mapping['decoder_local_attention'] is False
    assert mapping['function_preserving'] is False and mapping['requires_matched_plain_continuation_control']
    assert state_digest(donor) == before, 'conversion must not mutate its donor'


def test_tail_only_and_chunked_cache_parity_on_a_converted_arm():
    """The two CED-specific behaviours convert_ced measures against the G0 tolerance."""
    torch.manual_seed(17)
    converted, _ = to_ced(DenseReference(tiny_plain(4)).eval(), tiny_ced(), donor_layer=2)
    atol = run.protocol()['G0']['native_reference_atol']
    rtol = run.protocol()['G0']['native_reference_rtol']
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    cut = ids.shape[1]//2
    with torch.inference_mode():
        full = converted(ids, use_cache=True)
        tail = converted(ids, use_cache=True, last_position_only=True).logits
        pre = converted(ids[:, :cut], use_cache=True)
        cont = converted(ids[:, cut:], past=pre.cache, use_cache=True).logits
    assert torch.isfinite(full.logits).all()
    torch.testing.assert_close(full.logits[:, -1:], tail, atol=atol, rtol=rtol)
    torch.testing.assert_close(full.logits[:, cut:], cont, atol=atol, rtol=rtol)
    # The stage gates on the decoded choice, which is what a tolerance cannot be tuned into.
    assert bool((full.logits[:, -1:].argmax(-1) == tail.argmax(-1)).all())
    assert bool((full.logits[:, cut:].argmax(-1) == cont.argmax(-1)).all())


def test_ffn_growth_copies_cyclic_donors_and_zeroes_new_down_columns():
    torch.manual_seed(17)
    donor = DenseReference(tiny_plain(4)).eval()
    grown = tiny_ced(intermediate=64)          # 48 -> 64, so 16 inserted neurons
    converted, mapping = to_ced(donor, grown, donor_layer=2)
    assert mapping['ffn_growth'] == 'cyclic_gate_up_donors_zero_down_columns'
    extra = grown.intermediate_size - donor.config.intermediate_size
    block = converted.encoder[0].mlp
    torch.testing.assert_close(block.gate_proj.weight[-extra:], block.gate_proj.weight[:extra])
    torch.testing.assert_close(block.up_proj.weight[-extra:], block.up_proj.weight[:extra])
    assert torch.equal(block.down_proj.weight[:, -extra:],
                       torch.zeros_like(block.down_proj.weight[:, -extra:]))


def test_converted_arm_serializes_and_reloads_exactly(tmp_path):
    torch.manual_seed(17)
    converted, _ = to_ced(DenseReference(tiny_plain(4)).eval(), tiny_ced(), donor_layer=2)
    run._save(converted, tmp_path)
    assert json.loads((tmp_path/'config.json').read_text())['model_type'] == 'rineya_ced_research_v1'
    restored = run._restore(tmp_path, 'cpu')
    assert restored.config == converted.config
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.inference_mode():
        assert torch.equal(converted(ids).logits, restored(ids).logits)
