"""FP32 arm builder and delta bookkeeping (Amendment 1 sections 2, 3, 5, 7). CPU, synthetic only."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct

import pytest
import safetensors.torch
from safetensors.torch import save_file
import torch

from experiments.maxrl_comp002_pretraining_v1 import arm_build as ab
from experiments.maxrl_comp002_pretraining_v1.arm_build import (
    ArmRebuildStop, build_arm, build_arm_streaming, delta_data_digest, extract_delta,
    flush_denormal_enabled, global_norm, placebo_coefficient, placebo_label, read_delta,
    retained_coefficient, state_digest, tensor_digest, twin_coefficient, write_delta)

EMBED, HEAD = ab.TIED_PAIR
UP = 'model.layers.0.mlp.up_proj.weight'
Q = 'model.layers.0.self_attn.q_proj.weight'
NORM = 'model.norm.weight'
INVENTORY = [UP, Q, NORM]


def randn(gen, *shape):
    return torch.randn(*shape, generator=gen, dtype=torch.float32)


@pytest.fixture
def world():
    """BF16 base (stored-checkpoint layout: embed only), three FP32 donor deltas; S skips NORM."""
    gen = torch.Generator().manual_seed(20260926)
    base = {EMBED: randn(gen, 6, 4).to(torch.bfloat16), UP: randn(gen, 5, 4).to(torch.bfloat16),
            Q: randn(gen, 4, 4).to(torch.bfloat16), NORM: randn(gen, 4).to(torch.bfloat16)}
    deltas = {}
    for name in ab.ORDER:
        final = {k: base[k].float() + 0.01 * randn(gen, *base[k].shape) for k in INVENTORY}
        deltas[name] = extract_delta(final, base, INVENTORY)
    deltas['S'] = {k: v for k, v in deltas['S'].items() if k != NORM}
    return base, deltas


def test_tensor_digest_preimage_is_dtype_shape_and_little_endian_bytes():
    t = torch.tensor([[1.0, -2.0]])
    header = b'{"dtype":"torch.float32","shape":[1,2]}\n'
    assert tensor_digest(t) == hashlib.sha256(header + struct.pack('<2f', 1.0, -2.0)).hexdigest()
    bf = torch.tensor([1.0], dtype=torch.bfloat16)
    assert tensor_digest(bf) == hashlib.sha256(
        b'{"dtype":"torch.bfloat16","shape":[1]}\n' + b'\x80\x3f').hexdigest()
    assert tensor_digest(torch.tensor(3.0)) == hashlib.sha256(
        b'{"dtype":"torch.float32","shape":[]}\n' + struct.pack('<f', 3.0)).hexdigest()
    assert tensor_digest(torch.zeros(0)) == hashlib.sha256(
        b'{"dtype":"torch.float32","shape":[0]}\n').hexdigest()
    assert tensor_digest(t.reshape(2)) != tensor_digest(t)
    assert tensor_digest(t.to(torch.bfloat16)) != tensor_digest(t)
    strided = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    assert not strided.is_contiguous()
    assert tensor_digest(strided) == tensor_digest(strided.contiguous())


def test_state_digest_binds_keys_in_sorted_order():
    a, b = torch.ones(2), torch.zeros(3)
    expected = hashlib.sha256(b'a\0' + tensor_digest(a).encode() + b'\n'
                              + b'b\0' + tensor_digest(b).encode() + b'\n').hexdigest()
    assert state_digest({'b': b, 'a': a}) == state_digest({'a': a, 'b': b}) == expected
    assert state_digest({'a': b, 'b': a}) != expected
    for bad in ({}, {'a\0': a}, {1: a}):
        with pytest.raises(ValueError):
            state_digest(bad)


def test_extract_delta_is_final_minus_upcast_base_and_refuses_bad_inputs(world):
    base, _ = world
    final = {k: base[k].float() + 0.5 for k in INVENTORY}
    delta = extract_delta(final, base, INVENTORY)
    assert set(delta) == set(INVENTORY)
    for key in INVENTORY:
        assert delta[key].dtype == torch.float32
        assert torch.equal(delta[key], final[key] - base[key].float())
    cases = [
        ({**final, EMBED: base[EMBED].float()}, base, INVENTORY, 'outside inventory'),
        ({k: final[k] for k in (UP, Q)}, base, INVENTORY, 'missing inventory key in final'),
        (final, {k: base[k] for k in (EMBED, UP, Q)}, INVENTORY, 'missing inventory key in base'),
        ({**final, UP: torch.zeros(4, 5)}, base, INVENTORY, 'shape mismatch'),
        ({**final, UP: final[UP].to(torch.bfloat16)}, base, INVENTORY, 'final not FP32'),
        ({**final, UP: final[UP].double()}, base, INVENTORY, 'final not FP32'),
        ({**final, NORM: torch.full((4,), math.inf)}, base, INVENTORY, 'nonfinite'),
        ({**final, NORM: torch.full((4,), math.nan)}, base, INVENTORY, 'nonfinite'),
        (final, {**base, UP: base[UP].double()}, INVENTORY, 'upcastable'),
        (final, base, [UP, Q, NORM, UP], 'invalid inventory'),
        (final, base, [], 'invalid inventory'),
    ]
    for final_state, base_state, inventory, message in cases:
        with pytest.raises(ValueError, match=message):
            extract_delta(final_state, base_state, inventory)


def test_digest_stable_across_builds_and_input_list_order_is_irrelevant(world):
    base, d = world
    components = [('R', d['R'], 0.5), ('S', d['S'], 0.25), ('P', d['P'], 0.375)]
    state, receipt = build_arm(base, components)
    _, again = build_arm(base, components)
    _, shuffled = build_arm(base, [components[2], components[0], components[1]])
    assert receipt['arm_sha256'] == state_digest(state)
    assert receipt == again == shuffled
    assert receipt['applied_order'] == ['R', 'S', 'P']
    assert set(receipt['cross_terms']) == {'R|S', 'R|P', 'S|P'}
    json.dumps(receipt, allow_nan=False)


def test_build_never_mutates_an_fp32_base(world):
    base, d = world
    base32 = {k: v.float() for k, v in base.items()}
    snapshot = {k: v.clone() for k, v in base32.items()}
    state, _ = build_arm(base32, [('R', d['R'], 1.0)])
    assert all(torch.equal(base32[k], snapshot[k]) for k in base32)
    assert all(state[k] is not base32[k] for k in base32)


def test_fp32_sums_are_order_dependent_so_the_order_is_registered():
    # FP32 addition is not associative: (-1e8 + 1e8) + 1 = 1, but (-1e8 + 1) + 1e8 = 0,
    # because 1e8 - 1 rounds back to 1e8 (FP32 spacing at 1e8 is 8). An arm built in a
    # different order is a different arm, which is why section 2 fixes (R, S, P).
    assert torch.tensor([-1e8]).item() == -1e8
    base = {'w': torch.tensor([-1e8])}
    d_r, d_s = {'w': torch.tensor([1e8])}, {'w': torch.tensor([1.0])}
    components = [('R', d_r, 1.0), ('S', d_s, 1.0)]
    registered, receipt = build_arm(base, components)
    other, other_receipt = build_arm(base, components, order=('S', 'R', 'P'))
    assert registered['w'].item() == 1.0
    assert other['w'].item() == 0.0
    assert receipt['arm_sha256'] != other_receipt['arm_sha256']


def test_rs_a100_is_base_plus_dr_plus_ds_and_never_bf16(world):
    base, d = world
    state, receipt = build_arm(base, [('S', d['S'], 1), ('R', d['R'], 1.0)])
    for key in base:
        expected = base[key].float()
        if key in d['R']:
            expected = expected + d['R'][key]
        if key in d['S']:
            expected = expected + d['S'][key]
        assert torch.equal(state[key], expected)
    assert all(t.dtype == torch.float32 for t in state.values())
    assert receipt['base_dtypes'] == ['torch.bfloat16']
    assert receipt['dtype'] == 'torch.float32' and receipt['all_finite'] is True
    assert [c['alpha_fp32'] for c in receipt['components']] == [1.0, 1.0]
    assert receipt['tensors'] == 4 and receipt['displaced_tensors'] == 3


def test_alpha_is_an_fp32_multiply_followed_by_a_separate_add(world):
    base, d = world
    state, receipt = build_arm(base, [('R', d['R'], 0.1)])
    alpha32 = torch.tensor(0.1, dtype=torch.float32).item()
    assert receipt['components'][0]['alpha'] == 0.1
    assert receipt['components'][0]['alpha_fp32'] == alpha32 != 0.1
    for key in d['R']:
        assert torch.equal(state[key], base[key].float() + d['R'][key] * alpha32)
    assert torch.equal(state[EMBED], base[EMBED].float())


def test_receipt_norms_and_cross_terms_match_manual_float64(world):
    base, d = world
    a_r, a_s = 0.5, 0.25
    state, receipt = build_arm(base, [('R', d['R'], a_r), ('S', d['S'], a_s)])
    manual_cross = sum(float((a_r * d['R'][k].double() * a_s * d['S'][k].double()).sum())
                       for k in d['S'])
    assert receipt['cross_terms'] == {'R|S': pytest.approx(manual_cross, rel=1e-12)}
    norms = {c['name']: c for c in receipt['components']}
    for name, alpha in (('R', a_r), ('S', a_s)):
        raw = math.sqrt(sum(float((t.double() ** 2).sum()) for t in d[name].values()))
        assert norms[name]['delta_norm'] == pytest.approx(raw, rel=1e-12)
        assert norms[name]['scaled_norm'] == pytest.approx(alpha * raw, rel=1e-12)
        assert norms[name]['tensors'] == len(d[name])
    per_tensor = {}
    for key in INVENTORY:
        displacement = state[key].double() - base[key].double()
        per_tensor[key] = math.sqrt(float((displacement ** 2).sum()))
        assert receipt['per_tensor_displacement_norm'][key] == pytest.approx(per_tensor[key],
                                                                             rel=1e-12)
    assert set(receipt['per_tensor_displacement_norm']) == set(INVENTORY)
    total = math.sqrt(sum(v ** 2 for v in per_tensor.values()))
    assert receipt['total_displacement_norm'] == pytest.approx(total, rel=1e-12)
    identity = math.sqrt(norms['R']['scaled_norm'] ** 2 + norms['S']['scaled_norm'] ** 2
                         + 2 * manual_cross)
    assert receipt['total_norm_from_components'] == pytest.approx(identity, rel=1e-12)
    # realized FP32 displacement agrees with the exact component algebra up to FP32 rounding
    assert receipt['total_displacement_norm'] == pytest.approx(identity, rel=1e-4)


def test_digest_is_independent_of_the_norm_chunk_size(world, monkeypatch):
    base, d = world
    components = [('R', d['R'], 0.5), ('S', d['S'], 0.25)]
    _, reference = build_arm(base, components)
    monkeypatch.setattr(ab, 'CHUNK', 3)
    _, chunked = build_arm(base, components)
    assert chunked['arm_sha256'] == reference['arm_sha256']
    assert chunked['total_displacement_norm'] == pytest.approx(
        reference['total_displacement_norm'], rel=1e-12)
    assert chunked['cross_terms']['R|S'] == pytest.approx(reference['cross_terms']['R|S'],
                                                          rel=1e-12)


def test_empty_component_list_is_the_fp32_source_arm(world):
    base, _ = world
    state, receipt = build_arm(base, [])
    assert all(torch.equal(state[k], base[k].float()) for k in base)
    assert receipt['total_displacement_norm'] == 0.0
    assert receipt['displaced_tensors'] == 0 and receipt['cross_terms'] == {}
    assert receipt['tied_alias'] == {'pair_present': False, 'tied_in_base': False,
                                     'touched_by': [], 'equal_in_arm': None}


def test_build_arm_refusals(world):
    base, d = world
    cases = [
        ([('X', d['R'], 1.0)], {}, 'unknown component'),
        ([('R', d['R'], 1.0), ('R', d['S'], 0.5)], {}, 'duplicate component'),
        ([('R', d['R'], 0)], {}, 'alpha'),
        ([('R', d['R'], -0.5)], {}, 'alpha'),
        ([('R', d['R'], math.nan)], {}, 'alpha'),
        ([('R', d['R'], math.inf)], {}, 'alpha'),
        ([('R', d['R'], True)], {}, 'alpha'),
        ([('R', d['R'], '0.5')], {}, 'alpha'),
        ([('R', d['R'], 1e-50)], {}, 'representable'),
        ([('R', {**d['R'], 'model.extra': torch.zeros(2)}, 1.0)], {}, 'not in base'),
        ([('R', {**d['R'], UP: torch.zeros(4, 5)}, 1.0)], {}, 'shape mismatch'),
        ([('R', {**d['R'], UP: d['R'][UP].to(torch.bfloat16)}, 1.0)], {}, 'not FP32'),
        ([('R', {**d['R'], UP: d['R'][UP].double()}, 1.0)], {}, 'not FP32'),
        ([('R', {**d['R'], NORM: torch.full((4,), math.nan)}, 1.0)], {}, 'nonfinite delta'),
        ([('R', {}, 1.0)], {}, 'empty/invalid delta'),
        ([('R', d['R'])], {}, 'component must be'),
        ([('R', d['R'], 1.0)], {'order': ('R', 'R')}, 'registered order'),
        ([('R', d['R'], 1.0)], {'order': ()}, 'registered order'),
        ([('R', {NORM: torch.full((4,), 3e38)}, 1.0), ('S', {NORM: torch.full((4,), 3e38)}, 1.0)],
         {}, 'nonfinite arm'),
    ]
    for components, kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            build_arm(base, components, **kwargs)
    with pytest.raises(ValueError, match='upcastable'):
        build_arm({**base, UP: base[UP].double()}, [('R', d['R'], 1.0)])
    with pytest.raises(ValueError, match='upcastable'):
        build_arm({**base, 'model.ids': torch.arange(3)}, [('R', d['R'], 1.0)])
    with pytest.raises(ValueError, match='base state'):
        build_arm({}, [])


def test_tied_alias_must_stay_equal_or_stay_untouched():
    gen = torch.Generator().manual_seed(7)
    weight = randn(gen, 6, 4).to(torch.bfloat16)
    up = {UP: randn(gen, 5, 4)}
    base = {EMBED: weight, HEAD: weight, UP: randn(gen, 5, 4).to(torch.bfloat16)}
    _, receipt = build_arm(base, [('R', up, 1.0)])
    assert receipt['tied_alias'] == {'pair_present': True, 'tied_in_base': True,
                                     'touched_by': [], 'equal_in_arm': True}
    step = randn(gen, 6, 4)
    state, receipt = build_arm(base, [('R', {EMBED: step, HEAD: step.clone()}, 0.5)])
    assert receipt['tied_alias']['touched_by'] == ['R']
    assert receipt['tied_alias']['equal_in_arm'] is True
    assert state[EMBED] is not state[HEAD]
    with pytest.raises(ValueError, match='tied'):
        build_arm(base, [('R', up, 1.0), ('S', {EMBED: step}, 0.5)])
    equal_copies = {**base, HEAD: weight.clone()}
    with pytest.raises(ValueError, match='tied'):
        build_arm(equal_copies, [('R', {HEAD: step}, 0.5)])
    untied = {**base, HEAD: (weight.float() + 1).to(torch.bfloat16)}
    _, receipt = build_arm(untied, [('R', {EMBED: step}, 0.5)])
    assert receipt['tied_alias'] == {'pair_present': True, 'tied_in_base': False,
                                     'touched_by': ['R'], 'equal_in_arm': False}


def _save(tmp_path: Path, name: str, state: dict) -> Path:
    path = tmp_path / (name + '.safetensors')
    save_file({k: v.contiguous() for k, v in state.items()}, str(path))
    return path


def test_streaming_matches_in_memory_and_never_loads_a_whole_delta(world, tmp_path, monkeypatch):
    base, d = world
    base_path = _save(tmp_path, 'base', base)
    paths = {name: _save(tmp_path, 'delta_' + name, d[name]) for name in ab.ORDER}
    alphas = {'R': 0.5, 'S': 0.25, 'P': 0.375}
    memory_state, memory_receipt = build_arm(base, [(n, d[n], alphas[n]) for n in ('P', 'S', 'R')])
    calls = []
    real_open = ab.safe_open

    class Spy:
        def __init__(self, path, **kwargs):
            self.name = Path(path).name
            self.inner = real_open(path, **kwargs)

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self.inner.__exit__(*exc)

        def keys(self):
            return self.inner.keys()

        def get_tensor(self, key):
            calls.append((self.name, key))
            return self.inner.get_tensor(key)

    def whole_file_load(*args, **kwargs):
        raise AssertionError('whole-file load')

    monkeypatch.setattr(ab, 'safe_open', Spy)
    monkeypatch.setattr(safetensors.torch, 'load_file', whole_file_load)
    monkeypatch.setattr(safetensors.torch, 'load', whole_file_load)
    components = [(n, paths[n], alphas[n]) for n in ('S', 'P', 'R')]
    stream_state, stream_receipt = build_arm_streaming(base_path, components)
    assert stream_receipt == memory_receipt
    assert stream_receipt['arm_sha256'] == state_digest(stream_state)
    assert set(stream_state) == set(memory_state)
    assert all(torch.equal(stream_state[k], memory_state[k]) for k in memory_state)
    assert all(t.dtype == torch.float32 for t in stream_state.values())
    fetched = Counter(calls)
    assert set(fetched.values()) == {1}
    for name in ab.ORDER:
        assert {k for f, k in fetched if f == paths[name].name} == set(d[name])
    assert {k for f, k in fetched if f == base_path.name} == set(base)
    _, again = build_arm_streaming(str(base_path), [(n, str(p), alphas[n])
                                                    for n, p in paths.items()])
    assert again['arm_sha256'] == memory_receipt['arm_sha256']


def test_streaming_refusals(world, tmp_path):
    base, d = world
    base_path = _save(tmp_path, 'base', base)
    good = _save(tmp_path, 'delta_R', d['R'])
    stray = _save(tmp_path, 'stray', {**d['R'], 'model.extra': torch.zeros(2)})
    half = _save(tmp_path, 'half', {k: v.to(torch.bfloat16) for k, v in d['R'].items()})
    cases = [
        (base_path, [('X', good, 1.0)], 'unknown component'),
        (base_path, [('R', good, 1.0), ('R', good, 1.0)], 'duplicate component'),
        (base_path, [('R', tmp_path / 'absent.safetensors', 1.0)], 'missing delta file'),
        (tmp_path / 'absent.safetensors', [('R', good, 1.0)], 'missing base file'),
        (base_path, [('R', stray, 1.0)], 'not in base'),
        (base_path, [('R', half, 1.0)], 'not FP32'),
    ]
    for path, components, message in cases:
        with pytest.raises(ValueError, match=message):
            build_arm_streaming(path, components)
    gen = torch.Generator().manual_seed(11)
    weight = randn(gen, 6, 4).to(torch.bfloat16)
    tied_path = _save(tmp_path, 'tied', {EMBED: weight, HEAD: weight.clone(), UP: base[UP]})
    one_sided = _save(tmp_path, 'one_sided', {EMBED: randn(gen, 6, 4)})
    with pytest.raises(ValueError, match='tied'):
        build_arm_streaming(tied_path, [('S', one_sided, 0.5)])


def test_twin_coefficient_matches_norm_and_never_extrapolates(world):
    delta = {'w': torch.tensor([3.0, 4.0])}
    assert twin_coefficient(delta, 2.5) == 0.5
    assert twin_coefficient(delta, 5.0) == 1.0
    assert twin_coefficient(delta, 5.000001) is None
    for bad in (0, -1.0, math.nan, math.inf, True):
        with pytest.raises(ValueError, match='target norm'):
            twin_coefficient(delta, bad)
    with pytest.raises(ValueError, match='zero-norm'):
        twin_coefficient({'w': torch.zeros(2)}, 1.0)
    base, d = world
    _, rs = build_arm(base, [('R', d['R'], 0.5), ('S', d['S'], 0.25)])
    target = rs['total_displacement_norm']
    for name in ('R', 'S'):
        c = twin_coefficient(d[name], target)
        assert c is not None and c == pytest.approx(target / global_norm(d[name]), rel=1e-15)
        _, twin = build_arm(base, [(name, d[name], c)])
        assert twin['total_displacement_norm'] == pytest.approx(target, rel=1e-4)
    assert twin_coefficient(d['R'], 2 * global_norm(d['R'])) is None


def test_placebo_is_capped_at_one_with_a_positive_shortfall():
    delta_p = {'w': torch.tensor([3.0, 4.0])}
    assert placebo_coefficient(delta_p, 2.5) == (0.5, 0.0)
    assert placebo_coefficient(delta_p, 5.0) == (1.0, 0.0)
    p, shortfall = placebo_coefficient(delta_p, 10.0)
    assert p == 1.0 and shortfall == 0.5
    p, shortfall = placebo_coefficient(delta_p, 5.0001)
    assert p == 1.0 and 0 < shortfall < 1e-4
    assert placebo_label(0.5) == 'PLACEBO_UNDERSIZED(50.00%)'
    assert placebo_label(0.0) is None
    with pytest.raises(ValueError, match='zero-norm'):
        placebo_coefficient({'w': torch.zeros(3)}, 1.0)
    with pytest.raises(ValueError, match='shortfall'):
        placebo_label(1.5)


def test_global_norm_accumulates_in_float64_over_chunks(monkeypatch):
    gen = torch.Generator().manual_seed(3)
    delta = {'a': randn(gen, 10), 'b': randn(gen, 3, 7), 'c': torch.zeros(0)}
    expected = math.sqrt(sum(float((t.double() ** 2).sum()) for t in delta.values()))
    assert global_norm(delta) == pytest.approx(expected, rel=1e-13)
    monkeypatch.setattr(ab, 'CHUNK', 3)
    assert global_norm(delta) == pytest.approx(expected, rel=1e-13)
    for bad in ({}, {'a': torch.tensor([math.inf])}, {'a': torch.arange(3)}):
        with pytest.raises(ValueError):
            global_norm(bad)


def test_retained_coefficient_projects_onto_the_first_delta(world):
    delta_first = {'w': torch.tensor([1.0, 2.0, 0.0]), 'v': torch.tensor([[2.0]])}
    displacement = {'w': torch.tensor([0.5, 1.0, 7.0]), 'v': torch.tensor([[1.0]])}
    assert retained_coefficient(displacement, delta_first) == 0.5  # 4.5 / 9
    with pytest.raises(ValueError, match='key mismatch'):
        retained_coefficient({'w': displacement['w']}, delta_first)
    with pytest.raises(ValueError, match='zero-norm'):
        retained_coefficient({'w': torch.ones(2)}, {'w': torch.zeros(2)})
    with pytest.raises(ValueError, match='shape mismatch'):
        retained_coefficient({'w': torch.ones(3)}, {'w': torch.ones(1, 3)})
    # section 7 checkpoint records for R->S: theta_first = theta0 + delta_R, then 0.3 more of R
    base, d = world
    first, _ = build_arm(base, [('R', d['R'], 1.0)])
    theta_t = {k: first[k] + 0.3 * d['R'][k] for k in INVENTORY}
    from_start = extract_delta(theta_t, base, INVENTORY)
    from_first = extract_delta(theta_t, first, INVENTORY)
    assert retained_coefficient(from_start, d['R']) == pytest.approx(1.3, rel=1e-4)
    assert global_norm(from_first) == pytest.approx(0.3 * global_norm(d['R']), rel=1e-4)


# --- section 2: a rebuild must reproduce the digest, or STOP --------------------------------

def _one_ulp_up(state: dict, key: str) -> dict:
    changed = state[key].clone()
    changed.view(-1)[0] = torch.nextafter(changed.view(-1)[0], torch.tensor(math.inf))
    assert not torch.equal(changed, state[key])
    return {**state, key: changed}


def test_a_rebuild_that_reproduces_the_digest_returns_the_arm_and_records_the_check(world,
                                                                                    tmp_path):
    base, d = world
    components = [('R', d['R'], 0.5), ('S', d['S'], 0.25)]
    first_state, first = build_arm(base, components)
    assert first['rebuild_verified_against'] is None
    registered = first['arm_sha256']
    state, receipt = build_arm(base, components, expected_sha256=registered)
    assert receipt['rebuild_verified_against'] == registered == receipt['arm_sha256']
    assert all(torch.equal(state[k], first_state[k]) for k in base)
    base_path = _save(tmp_path, 'base', base)
    files = [(n, _save(tmp_path, 'delta_' + n, delta), a) for n, delta, a in components]
    _, streamed = build_arm_streaming(base_path, files, expected_sha256=registered)
    assert streamed == receipt


def test_a_rebuild_that_misses_the_digest_raises_stop_and_returns_nothing(world, tmp_path):
    base, d = world
    state, first = build_arm(base, [('R', d['R'], 0.5)])
    registered = first['arm_sha256']
    # the registered arm differs from this rebuild by one ULP in one element
    off_by_one_ulp = state_digest(_one_ulp_up(state, UP))
    with pytest.raises(ArmRebuildStop, match='STOP') as caught:
        build_arm(base, [('R', d['R'], 0.5)], expected_sha256=off_by_one_ulp)
    stop = caught.value
    assert not isinstance(stop, ValueError)  # a refusal handler cannot absorb a STOP
    assert stop.expected == off_by_one_ulp != stop.actual == registered
    assert stop.actual == stop.receipt['arm_sha256']
    assert stop.receipt['rebuild_verified_against'] is None
    # a one-ULP drift in a donor tensor can round away in the sum, so the arm digest does not
    # bind the delta; a drift that survives the sum does STOP
    assert build_arm(base, [('R', _one_ulp_up(d['R'], UP), 0.5)])[1]['arm_sha256'] == registered
    drifted = {**d['R'], UP: d['R'][UP] + 2.0 ** -8}
    with pytest.raises(ArmRebuildStop):
        build_arm(base, [('R', drifted, 0.5)], expected_sha256=registered)
    _, other_order = build_arm(base, [('R', d['R'], 0.5), ('S', d['S'], 0.25)])
    with pytest.raises(ArmRebuildStop):
        build_arm(base, [('R', d['R'], 0.5), ('S', d['S'], 0.25)], order=('S', 'R', 'P'),
                  expected_sha256=other_order['arm_sha256'])
    base_path = _save(tmp_path, 'base', base)
    drifted_path = _save(tmp_path, 'delta_R_drifted', drifted)
    with pytest.raises(ArmRebuildStop):
        build_arm_streaming(base_path, [('R', drifted_path, 0.5)], expected_sha256=registered)
    for bad in (registered.upper(), registered[:-1], registered + '0', 'g' * 64, 1, b'0' * 64):
        with pytest.raises(ValueError, match='expected arm digest'):
            build_arm(base, [('R', d['R'], 0.5)], expected_sha256=bad)
        with pytest.raises(ValueError, match='expected arm digest'):
            build_arm_streaming(base_path, [('R', drifted_path, 0.5)], expected_sha256=bad)


# --- every build receipt records the CPU arithmetic environment -----------------------------

def test_every_build_receipt_records_the_intra_op_thread_count(world, tmp_path):
    base, d = world
    base_path = _save(tmp_path, 'base', base)
    delta_path = _save(tmp_path, 'delta_R', d['R'])
    original = torch.get_num_threads()
    other = 2 if original == 1 else 1
    try:
        torch.set_num_threads(other)
        _, memory = build_arm(base, [('R', d['R'], 0.5)])
        _, streamed = build_arm_streaming(base_path, [('R', delta_path, 0.5)])
    finally:
        torch.set_num_threads(original)
    expected = {'torch_num_threads': other, 'flush_denormal': False,
                'flush_denormal_probe': ab.DENORMAL_PROBE}
    assert memory['runtime'] == streamed['runtime'] == expected
    _, now = build_arm(base, [])
    assert now['runtime']['torch_num_threads'] == original


def test_flush_denormal_is_probed_and_recorded_because_it_changes_the_arm():
    assert flush_denormal_enabled() is False
    # 0.5 * 2**-126 = 2**-127 is subnormal in FP32: kept normally, zero when flushed
    base = {'w': torch.zeros(4)}
    delta = {'w': torch.full((4,), 2.0 ** -126)}
    kept_state, kept = build_arm(base, [('R', delta, 0.5)])
    assert torch.equal(kept_state['w'].view(torch.int32), torch.full((4,), 1 << 22,
                                                                     dtype=torch.int32))
    if not torch.set_flush_denormal(True):
        pytest.skip('this CPU cannot flush subnormals')
    try:
        assert flush_denormal_enabled() is True
        flushed_state, flushed = build_arm(base, [('R', delta, 0.5)])
        with pytest.raises(ArmRebuildStop):
            build_arm(base, [('R', delta, 0.5)], expected_sha256=kept['arm_sha256'])
    finally:
        torch.set_flush_denormal(False)
    assert flush_denormal_enabled() is False
    assert kept['runtime']['flush_denormal'] is False
    assert flushed['runtime']['flush_denormal'] is True
    assert torch.equal(flushed_state['w'], torch.zeros(4))
    assert flushed['arm_sha256'] != kept['arm_sha256']


def test_flush_denormal_probe_refuses_a_state_that_differs_by_op_size(monkeypatch):
    assert ab._denormal_kept(1) is True and ab._denormal_kept(1 << 16) is True
    monkeypatch.setattr(ab, '_denormal_kept', lambda count: count == 1)
    with pytest.raises(ValueError, match='differs'):
        flush_denormal_enabled()
    with pytest.raises(ValueError, match='differs'):
        build_arm({'w': torch.zeros(2)}, [])


# --- FP32 delta files: create-only writer, data-section digest, verifying reader -------------

def _split(path: Path) -> tuple[bytes, bytes]:
    raw = path.read_bytes()
    (length,) = struct.unpack('<Q', raw[:8])
    return raw[:8 + length], raw[8 + length:]


def _rewrite(path: Path, header: dict, section: bytes, pad: int = 0) -> Path:
    text = json.dumps(header, separators=(', ', ': ')).encode() + b' ' * pad
    path.write_bytes(struct.pack('<Q', len(text)) + text + section)
    return path


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


@pytest.fixture
def stored(world, tmp_path):
    """A delta with -0.0 and a subnormal (both must survive bitwise), written once."""
    _, d = world
    delta = {**d['R'], NORM: torch.tensor([-0.0, 2.0 ** -140, 1.0, -1.5]),
             'model.scalar': torch.tensor(0.25), 'model.empty': torch.zeros(0, 3)}
    path = tmp_path / 'delta_R.safetensors'
    return delta, path, write_delta(path, delta)


def test_write_delta_returns_the_data_section_digest_of_what_is_on_disk(stored, tmp_path):
    delta, path, written = stored
    _, section = _split(path)
    assert written == hashlib.sha256(section).hexdigest() == delta_data_digest(path)
    assert written != hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = safetensors.torch.load_file(str(path))
    assert set(loaded) == set(delta)
    for key, t in delta.items():
        assert loaded[key].dtype == torch.float32 and loaded[key].shape == t.shape
        assert torch.equal(_bits(loaded[key]), _bits(t))
    assert [p.name for p in tmp_path.iterdir()] == ['delta_R.safetensors']  # no partial left


def test_write_delta_is_create_only(stored, tmp_path, monkeypatch):
    delta, path, _ = stored
    before = path.read_bytes()
    calls = []
    real_save = ab.save_file
    monkeypatch.setattr(ab, 'save_file', lambda *a, **k: calls.append(a) or real_save(*a, **k))
    with pytest.raises(ValueError, match='create-only'):
        write_delta(path, {UP: delta[UP] + 1})
    assert calls == []  # refused before anything was written
    assert path.read_bytes() == before
    # a file that appears after the check is still never replaced
    racer = tmp_path / 'delta_S.safetensors'

    def save_then_race(tensors, name):
        real_save(tensors, name)
        racer.write_bytes(b'racer')

    monkeypatch.setattr(ab, 'save_file', save_then_race)
    with pytest.raises(ValueError, match='create-only'):
        write_delta(racer, {UP: delta[UP]})
    assert racer.read_bytes() == b'racer'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['delta_R.safetensors',
                                                          'delta_S.safetensors']


def test_write_delta_verifies_what_it_wrote(world, tmp_path, monkeypatch):
    _, d = world
    real_save = ab.save_file

    def corrupt(tensors, name):
        real_save({k: (v + 1 if k == UP else v) for k, v in tensors.items()}, name)

    monkeypatch.setattr(ab, 'save_file', corrupt)
    with pytest.raises(ValueError, match='disagree with the source'):
        write_delta(tmp_path / 'delta.safetensors', d['R'])

    def rename(tensors, name):
        real_save({('renamed' if k == UP else k): v for k, v in tensors.items()}, name)

    monkeypatch.setattr(ab, 'save_file', rename)
    with pytest.raises(ValueError, match='disagrees with the source'):
        write_delta(tmp_path / 'delta.safetensors', d['R'])
    assert list(tmp_path.iterdir()) == []


def test_write_delta_refusals(world, tmp_path):
    _, d = world
    up = d['R'][UP]
    cases = [
        ({}, 'empty/invalid delta'),
        ({'': up}, 'invalid tensor key'),
        ({'__metadata__': up}, 'reserved'),
        ({UP: up.to(torch.bfloat16)}, 'not FP32'),
        ({UP: up.double()}, 'not FP32'),
        ({UP: torch.arange(3)}, 'not FP32'),
        ({UP: [1.0]}, 'not FP32'),
        ({UP: torch.zeros(2, device='meta')}, 'not on CPU'),
        ({UP: torch.full((2,), math.nan)}, 'nonfinite'),
        ({UP: torch.full((2,), -math.inf)}, 'nonfinite'),
    ]
    for delta, message in cases:
        with pytest.raises(ValueError, match=message):
            write_delta(tmp_path / 'delta.safetensors', delta)
    with pytest.raises(ValueError, match='missing delta directory'):
        write_delta(tmp_path / 'absent' / 'delta.safetensors', {UP: up})
    assert list(tmp_path.iterdir()) == []


def test_read_delta_returns_exactly_the_hashed_bytes(stored):
    delta, path, written = stored
    loaded = read_delta(path, written)
    assert list(loaded) == sorted(delta)
    for key, t in delta.items():
        assert loaded[key].dtype == torch.float32 and loaded[key].shape == t.shape
        assert torch.equal(_bits(loaded[key]), _bits(t))
    assert state_digest(loaded) == state_digest(delta)
    assert read_delta(str(path), written, expected_state_sha256=state_digest(delta)).keys() \
        == loaded.keys()


def test_read_delta_refuses_a_digest_mismatch(stored, tmp_path):
    delta, path, written = stored
    header, section = _split(path)
    flipped = bytearray(section)
    flipped[5] ^= 0x01
    tampered = tmp_path / 'tampered.safetensors'
    tampered.write_bytes(header + bytes(flipped))
    assert delta_data_digest(tampered) != written
    with pytest.raises(ValueError, match='digest mismatch'):
        read_delta(tampered, written)
    with pytest.raises(ValueError, match='digest mismatch'):
        read_delta(path, hashlib.sha256(b'another delta').hexdigest())
    for bad in (written.upper(), written[:63], None, 7):
        with pytest.raises(ValueError, match='expected delta digest'):
            read_delta(path, bad)
    with pytest.raises(ValueError, match='expected delta state digest'):
        read_delta(path, written, expected_state_sha256='x')
    with pytest.raises(ValueError, match='missing delta file'):
        read_delta(tmp_path / 'absent.safetensors', written)


def test_the_data_digest_ignores_header_key_order_and_padding(stored, tmp_path):
    delta, path, written = stored
    header_bytes, section = _split(path)
    header = json.loads(header_bytes[8:])
    reordered = {key: dict(reversed(list(header[key].items()))) for key in reversed(header)}
    assert list(reordered) != list(header)
    other = _rewrite(tmp_path / 'reordered.safetensors', reordered, section, pad=5)
    assert _split(other)[0] != header_bytes
    assert delta_data_digest(other) == written
    loaded = read_delta(other, written, expected_state_sha256=state_digest(delta))
    assert all(torch.equal(_bits(loaded[k]), _bits(delta[k])) for k in delta)


def test_the_data_digest_does_not_bind_names_but_the_state_digest_does(tmp_path):
    delta = {'a': torch.tensor([1.0, 2.0]), 'b': torch.tensor([3.0, 4.0])}
    path = tmp_path / 'pair.safetensors'
    written = write_delta(path, delta)
    header_bytes, section = _split(path)
    header = json.loads(header_bytes[8:])
    header['a'], header['b'] = header['b'], header['a']
    swapped = _rewrite(tmp_path / 'swapped.safetensors', header, section)
    assert delta_data_digest(swapped) == written
    assert torch.equal(read_delta(swapped, written)['a'], delta['b'])  # the documented limit
    with pytest.raises(ValueError, match='state digest mismatch'):
        read_delta(swapped, written, expected_state_sha256=state_digest(delta))


def test_delta_files_with_a_bad_layout_are_refused(stored, tmp_path):
    _, path, written = stored
    raw = path.read_bytes()
    header_bytes, section = _split(path)
    header = json.loads(header_bytes[8:])
    size = {k: v['data_offsets'][1] - v['data_offsets'][0] for k, v in header.items()}
    target = max((k for k in header if header[k]['data_offsets'][0] >= 4), key=size.get)
    start, end = header[target]['data_offsets']
    assert end - start > 4

    def edited(**entry):
        changed = json.loads(json.dumps(header))
        changed[target].update(entry)
        return changed

    duplicate = (header_bytes[8:].rstrip()[:-1] + b',"' + target.encode() + b'":'
                 + json.dumps(header[target]).encode() + b'}')
    files = {
        'truncated': raw[:-1],
        'trailing': raw + b'\0',
        'short': b'\x01\x00',
        'huge_header': struct.pack('<Q', 1 << 40) + raw[8:],
        'not_a_dict': struct.pack('<Q', 2) + b'[]',
        'duplicate_key': struct.pack('<Q', len(duplicate)) + duplicate + section,
    }
    layouts = {
        'bf16': edited(dtype='BF16'),
        'bad_shape': edited(shape=[end - start]),
        'gap': edited(data_offsets=[start + 4, end + 4]),
        'overlap': edited(data_offsets=[start - 4, end - 4]),
        'extra_field': edited(extra=1),
        'one_offset': edited(data_offsets=[start]),
    }
    messages = {'truncated': 'file size', 'trailing': 'file size',
                'short': 'not a safetensors file', 'huge_header': 'header length',
                'not_a_dict': 'invalid safetensors header', 'duplicate_key': 'duplicate JSON key',
                'bf16': 'not FP32', 'bad_shape': 'disagree with shape', 'gap': 'not tiled',
                'overlap': 'not tiled', 'extra_field': 'invalid tensor entry',
                'one_offset': 'invalid data offsets'}
    paths = {}
    for name, content in files.items():
        paths[name] = tmp_path / (name + '.safetensors')
        paths[name].write_bytes(content)
    for name, layout in layouts.items():
        paths[name] = _rewrite(tmp_path / (name + '.safetensors'), layout, section)
    assert set(paths) == set(messages)
    for name, bad in paths.items():
        with pytest.raises(ValueError, match=messages[name]):
            delta_data_digest(bad)
        with pytest.raises(ValueError, match=messages[name]):
            read_delta(bad, written)
