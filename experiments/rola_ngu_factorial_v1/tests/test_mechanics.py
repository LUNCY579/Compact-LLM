"""Tiny random CPU fixtures only. None of these tests loads pretrained assets."""
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest
import torch

from experiments.rola_ngu_factorial_v1.attention import CausalRoLA
from experiments.rola_ngu_factorial_v1.sampling import Completion, Sampler, TokenBudget
from experiments.rola_ngu_factorial_v1.objective import policy_loss
from experiments.rola_ngu_factorial_v1.sparsity import prune_delta
from experiments.rola_ngu_factorial_v1.analysis import Outcome, summarize
from experiments.rola_ngu_factorial_v1.preflight import ROOT, report, verify_manifest

torch.set_num_threads(1)


def fixture(heads=4, kv_heads=2, length=7, rank=4):
    torch.manual_seed(19)
    m = CausalRoLA(heads, 8, rank, torch.tensor([1., .1, .01, .001])).double()
    with torch.no_grad():
        m.alpha.fill_(.2)
    q = torch.randn(2, heads, length, 8, dtype=torch.float64)
    k = torch.randn(2, kv_heads, length, 8, dtype=torch.float64)
    v = torch.randn_like(k)
    dense = torch.randn_like(q)
    p = torch.arange(length).expand(2, -1).clone()
    valid = torch.ones(2, length, dtype=torch.bool)
    return m, q, k, v, dense, p, valid


def call(m, q, k, v, dense, p, valid, **kw):
    return m(q, k, v, dense, p, valid, sequence_id='fixture', **kw)


@pytest.mark.parametrize('length', [1, 2, 7, 16])
@pytest.mark.parametrize('kv_heads', [1, 2, 4])
def test_prefix_matches_independent_quadratic(length, kv_heads):
    m, q, k, v, dense, p, valid = fixture(kv_heads=kv_heads, length=length)
    if length > 2:
        valid[0, 1] = False
    out, _ = call(m, q, k, v, dense, p, valid)
    k = k.repeat_interleave(4 // kv_heads, 1)
    v = v.repeat_interleave(4 // kv_heads, 1)
    qf, kf = m.features(q, k, p)
    scores = torch.einsum('bhir,bhjr->bhij', qf, kf)
    causal = torch.ones(length, length, dtype=torch.bool).tril()
    scores = scores * causal * valid[:, None, None, :]
    raw = torch.einsum('bhij,bhjd->bhid', scores, v)
    ref = raw / torch.sqrt(raw.square().mean(-1, keepdim=True) + 1e-6)
    ref = (dense + m.alpha * torch.sqrt(dense.square().mean(-1, keepdim=True) + 1e-6) * ref)
    ref = ref * valid[:, None, :, None]
    torch.testing.assert_close(out, ref, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize('split', [1, 2, 3, 6])
def test_chunk_cache_parity_and_immutability(split):
    m, q, k, v, d, p, valid = fixture()
    full, _ = call(m, q, k, v, d, p, valid)
    first, state = call(m, q[:, :, :split], k[:, :, :split], v[:, :, :split],
                        d[:, :, :split], p[:, :split], valid[:, :split])
    before = state.memory.clone()
    rest, _ = call(m, q[:, :, split:], k[:, :, split:], v[:, :, split:],
                   d[:, :, split:], p[:, split:], valid[:, split:], state=state)
    torch.testing.assert_close(torch.cat((first, rest), 2), full, atol=1e-11, rtol=1e-11)
    assert torch.equal(state.memory, before)


def test_exact_noop_and_alpha_gradient():
    m, q, k, v, d, p, valid = fixture()
    with torch.no_grad():
        m.alpha.zero_()
    out, _ = call(m, q, k, v, d, p, valid)
    assert torch.equal(out, d)
    out.square().sum().backward()
    assert m.alpha.grad.abs() > 0
    assert torch.equal(m.pq.grad, torch.zeros_like(m.pq))


def test_active_projection_gradients():
    m, q, k, v, d, p, valid = fixture()
    out, _ = call(m, q, k, v, d, p, valid)
    out.square().sum().backward()
    assert m.pq.grad.norm() > 0 and m.pk.grad.norm() > 0


def test_future_perturbation_no_leak():
    m, q, k, v, d, p, valid = fixture()
    original, _ = call(m, q, k, v, d, p, valid)
    q[:, :, 4:] += 123
    k[:, :, 4:] -= 78
    v[:, :, 4:] *= -100
    changed, _ = call(m, q, k, v, d, p, valid)
    assert torch.equal(original[:, :, :4], changed[:, :, :4])


def test_global_position_shift():
    m, q, k, v, d, p, valid = fixture()
    a, _ = call(m, q, k, v, d, p, valid)
    b, _ = call(m, q, k, v, d, p + 100, valid)
    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    assert torch.equal(m.inv_freq, torch.tensor([1., .1]).double())


def test_masked_write_and_all_padding():
    m, q, k, v, d, p, valid = fixture()
    valid[0, 0] = False
    a, _ = call(m, q, k, v, d, p, valid)
    k[0, :, 0] += 200
    v[0, :, 0] -= 100
    b, _ = call(m, q, k, v, d, p, valid)
    assert torch.equal(a, b)
    z, state = call(m, q, k, v, d, p, torch.zeros_like(valid))
    assert torch.count_nonzero(z) == 0 and torch.count_nonzero(state.memory) == 0


@pytest.mark.parametrize('bad', ['sequence', 'weights', 'positions', 'module'])
def test_cache_rejects_invalid_reuse(bad):
    m, q, k, v, d, p, valid = fixture()
    _, state = call(m, q, k, v, d, p, valid)
    if bad == 'sequence':
        state = replace(state, sequence_id='other')
    elif bad == 'weights':
        with torch.no_grad():
            m.alpha.add_(.01)
    elif bad == 'module':
        state = replace(state, owner=-1)
    else:
        p -= 7
    with pytest.raises(ValueError):
        call(m, q, k, v, d, p if bad == 'positions' else p + 7, valid, state=state)


@pytest.mark.parametrize('rank', [0, -2, 3, 10, True])
def test_invalid_rank(rank):
    with pytest.raises(ValueError):
        CausalRoLA(4, 8, rank, torch.ones(4))


@pytest.mark.parametrize('bad', ['nan', 'positions', 'valid', 'dtype', 'gqa'])
def test_invalid_attention_input(bad):
    m, q, k, v, d, p, valid = fixture()
    if bad == 'nan':
        q[0, 0, 0, 0] = float('nan')
    elif bad == 'positions':
        p[:, 2] = 0
    elif bad == 'valid':
        valid = valid.float()
    elif bad == 'dtype':
        q = q.float()
    else:
        k, v = torch.randn(2, 3, 7, 8).double(), torch.randn(2, 3, 7, 8).double()
    with pytest.raises(ValueError):
        call(m, q, k, v, d, p, valid)


def batch(round_number, rewards=(0, 0, 0, 0), version=0, prompt='p'):
    return tuple(Completion(f'{prompt}:{round_number}:{i}', prompt, version, r, (-.5, -.3))
                 for i, r in enumerate(rewards))


def continuing_sampler():
    for seed in range(100):
        s = Sampler(ngu=True, seed=seed)
        if s.ingest('e', batch(0), 0).status == 'RETRY':
            return s
    raise AssertionError('fixture could not find a deterministic continuation')


@pytest.mark.parametrize('ngu', [False, True])
@pytest.mark.parametrize('rewards,status', [((1, 1, 1, 1), 'EASY'), ((0, 1, 0, 0), 'UPDATE')])
def test_sampler_mixed_and_easy(ngu, rewards, status):
    s = Sampler(ngu=ngu, seed=1)
    decision = s.ingest('e', batch(0, rewards), 0)
    assert decision.status == status
    assert abs(sum(decision.advantages)) < 1e-12
    assert not s.pending
    assert s.generated_tokens == 8


def test_ordinary_drops_failures():
    assert Sampler(ngu=False, seed=0).ingest('e', batch(0), 0).status == 'GIVE_UP'


def test_ngu_keeps_counts_anchors_positive_and_filters_age():
    s = continuing_sampler()
    decision = s.ingest('e', batch(1, (1, 0, 0, 0), version=4), 4)
    assert decision.status == 'UPDATE'
    assert decision.total_count == 8 and decision.baseline == 1 / 8
    assert decision.stale_count == 4 and len(decision.samples) == 4
    assert decision.advantages[0] == 7 / 8
    assert abs(sum(decision.advantages)) < 1e-12
    assert decision.advantages[1] == -(7 / 8) / 3


def test_prior_failures_plus_current_all_correct_still_updates():
    s = continuing_sampler()
    decision = s.ingest('e', batch(1, (1, 1, 1, 1), version=1), 1)
    assert decision.status == 'UPDATE' and len(decision.samples) == 8


def test_stale_failures_plus_all_correct_no_fabricated_negative():
    s = continuing_sampler()
    decision = s.ingest('e', batch(1, (1, 1, 1, 1), version=4), 4)
    assert decision.status == 'STALE_NO_SIGNAL' and not decision.advantages


def test_hard_episode_cap():
    s = Sampler(ngu=True, seed=1, max_rounds=1)
    assert s.ingest('e', batch(0), 0).status == 'CAPPED'


def test_easy_prompt_can_return_as_new_episode():
    s = Sampler(ngu=True, seed=1)
    s.ingest('e1', batch(0, (1, 1, 1, 1)), 0)
    assert s.ingest('e2', batch(1, (1, 0, 1, 1)), 0).status == 'UPDATE'


def test_replay_ids_and_closed_episodes_rejected():
    s = Sampler(ngu=False, seed=1)
    s.ingest('e', batch(0), 0)
    with pytest.raises(ValueError):
        s.ingest('e2', batch(0), 0)
    with pytest.raises(ValueError):
        s.ingest('e', batch(1), 0)


@pytest.mark.parametrize('bad', ['reward', 'logps', 'future', 'duplicate', 'size', 'mixed_prompt'])
def test_bad_events_are_transactional(bad):
    s = Sampler(ngu=True, seed=1)
    items = list(batch(0))
    if bad == 'reward':
        items[0] = replace(items[0], reward=.5)
    elif bad == 'logps':
        items[0] = replace(items[0], behavior_logps=(float('nan'),))
    elif bad == 'future':
        items[0] = replace(items[0], policy_version=1)
    elif bad == 'duplicate':
        items[1] = items[0]
    elif bad == 'size':
        items.pop()
    else:
        items[0] = replace(items[0], prompt_id='other')
    with pytest.raises(ValueError):
        s.ingest('e', tuple(items), 0)
    assert not s.seen and not s.pending and s.generated_tokens == 0


def test_changed_prompt_in_pending_episode_rejected():
    s = continuing_sampler()
    with pytest.raises(ValueError):
        s.ingest('e', batch(1, prompt='other'), 0)


def test_continuation_replay_deterministic():
    def replay():
        s = Sampler(ngu=True, seed=12)
        return [s.ingest(f'e{i}', batch(i), 0).status for i in range(10)]
    assert replay() == replay()


@pytest.mark.parametrize('value', [1, -1, float('nan'), float('inf')])
def test_invalid_continuation(value):
    with pytest.raises(ValueError):
        Sampler(ngu=True, seed=0, continuation=value)


def test_budget_reservation_settlement_and_failure():
    b = TokenBudget(100)
    b.reserve('one', 70)
    with pytest.raises(ValueError):
        b.reserve('two', 31)
    b.settle('one', 20)
    b.reserve('two', 80)
    b.settle('two')
    assert b.spent == 100 and not b.pending
    with pytest.raises(ValueError):
        b.reserve('three', 1)
    with pytest.raises(ValueError):
        b.reserve('one', 1)


@pytest.mark.parametrize('actual', [-1, 11, True, .5])
def test_budget_rejects_invalid_settlement(actual):
    b = TokenBudget(10)
    b.reserve('one', 10)
    with pytest.raises(ValueError):
        b.settle('one', actual)
    assert b.spent == 0 and b.pending == {'one': 10}


def loss_fixture():
    logp = torch.tensor([[-1., -2.], [-3., -4.]], requires_grad=True)
    return logp, logp.detach().clone(), logp.detach().clone(), torch.tensor([1., -1.]), torch.ones(2, 2, dtype=torch.bool)


def test_policy_loss_current_behavior_parity_and_detached_baselines():
    cur, old, ref, adv, mask = loss_fixture()
    old.requires_grad_(); ref.requires_grad_(); adv.requires_grad_()
    loss = policy_loss(cur, old, ref, adv, mask)
    assert loss.item() == 0
    loss.backward()
    assert cur.grad.norm() > 0
    assert old.grad is None and ref.grad is None and adv.grad is None


def test_policy_mask_sanitizes_nan():
    cur, old, ref, adv, mask = loss_fixture()
    mask[:, 1] = False
    with torch.no_grad():
        cur[:, 1] = float('nan'); old[:, 1] = float('nan'); ref[:, 1] = float('nan')
    loss = policy_loss(cur, old, ref, adv, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(cur.grad).all() and torch.count_nonzero(cur.grad[:, 1]) == 0


@pytest.mark.parametrize('bad', ['mask', 'nan', 'positive', 'ratio', 'shape'])
def test_policy_loss_refuses_bad_data(bad):
    cur, old, ref, adv, mask = loss_fixture()
    if bad == 'mask':
        mask[0] = False
    elif bad == 'nan':
        old[0, 0] = float('nan')
    elif bad == 'positive':
        old[0, 0] = 1
    elif bad == 'ratio':
        old[0, 0] = -100
    else:
        adv = adv[:1]
    with pytest.raises((ValueError, FloatingPointError)):
        policy_loss(cur, old, ref, adv, mask)


def test_delta_mask_matches_titan_tie_break_and_floor():
    base = {'w': torch.zeros(10), 'frozen': torch.ones(2)}
    trained = {'w': torch.tensor([1., -1., 1., -1., 1., -1., 1., -1., 1., -1.]),
               'frozen': torch.ones(2), 'rola.pq': torch.randn(2)}
    before = copy.deepcopy(trained)
    out, receipt = prune_delta(base, trained, ('w',))
    assert torch.equal(out['w'], torch.tensor([1., -1., 0, 0, 0, 0, 0, 0, 0, 0]))
    assert receipt['tensors']['w']['selected'] == 2
    assert not receipt['new_parameters_pruned']
    for name in trained:
        assert torch.equal(trained[name], before[name])
    assert torch.equal(out['rola.pq'], trained['rola.pq'])
    assert out['rola.pq'].data_ptr() != trained['rola.pq'].data_ptr()
    _, second = prune_delta(base, trained, ('w',))
    assert receipt == second


@pytest.mark.parametrize('keep', [0., .25, .5, 1.])
def test_delta_explicit_formula(keep):
    b = {'w': torch.arange(16).float()}
    t = {'w': b['w'] + torch.arange(16).float() / 100}
    out, _ = prune_delta(b, t, ('w',), keep)
    delta = t['w'] - b['w']
    order = torch.argsort(delta.abs(), descending=True, stable=True)
    mask = torch.zeros(16, dtype=torch.bool); mask[order[:int(16 * keep)]] = True
    assert torch.equal(out['w'], b['w'] + torch.where(mask, delta, 0))


@pytest.mark.parametrize('bad', ['dtype', 'nan', 'allowlist', 'unlisted_change', 'shape', 'keep'])
def test_delta_refuses_confounds(bad):
    b, t, names, keep = {'w': torch.zeros(4), 'x': torch.zeros(2)}, {'w': torch.ones(4), 'x': torch.zeros(2)}, ('w',), .25
    if bad == 'dtype':
        t['w'] = t['w'].bfloat16()
    elif bad == 'nan':
        t['w'][0] = float('nan')
    elif bad == 'allowlist':
        names = ('rola.pq',)
    elif bad == 'unlisted_change':
        t['x'] += 1
    elif bad == 'shape':
        t['w'] = torch.ones(3)
    else:
        keep = float('nan')
    with pytest.raises(ValueError):
        prune_delta(b, t, names, keep)


def outcomes(sparse=False):
    rows = []
    for draw in (101, 211, 307):
        for arm in 'ABCD':
            for k in ([False, True] if sparse else [False]):
                for i in range(8):
                    # Synergy: A=1/8, B=C=2/8, D=4/8 => interaction=1/8.
                    count = {'A': 1, 'B': 2, 'C': 2, 'D': 4}[arm] + int(k)
                    rows.append(Outcome(draw, arm, k, str(i), 'even' if i % 2 == 0 else 'odd', i == 0, i < count))
    return rows


def test_factorial_primary_not_contaminated_by_pruning():
    plain, paired = summarize(outcomes()), summarize(outcomes(True))
    assert plain['aggregate'] == paired['aggregate']
    assert plain['aggregate']['interaction']['mean'] == .125
    assert plain['aggregate']['interaction']['sample_sd'] == 0
    assert paired['by_draw'][0]['paired_pruning_effects'] == {a: .125 for a in 'ABCD'}
    assert paired['p_values'] is None and not paired['release_permitted']


@pytest.mark.parametrize('bad', ['missing_cell', 'missing_row', 'duplicate', 'source', 'draws', 'partial_sparse'])
def test_analysis_refuses_incomplete_or_changed_blocks(bad):
    rows = outcomes()
    if bad == 'missing_cell':
        rows = [r for r in rows if not (r.draw == 101 and r.arm == 'D')]
    elif bad == 'missing_row':
        rows.pop()
    elif bad == 'duplicate':
        rows.append(rows[0])
    elif bad == 'source':
        rows[0] = replace(rows[0], source_correct=False)
    elif bad == 'draws':
        rows = [r for r in rows if r.draw != 307]
    else:
        rows.append(replace(rows[0], sparse=True))
    with pytest.raises(ValueError):
        summarize(rows)


def test_registration_separates_factors_and_keeps_native_closed():
    r = json.loads((ROOT / 'registration.json').read_text())
    assert all(a['delta_keep'] == 1 for a in r['primary_arms'].values())
    assert not r['rola_port']['token_sparsity']
    assert len(set(r['draw_seeds'])) == 3
    assert r['data']['confirmation_manifest_sha256'] is None
    assert r['source']['complete_asset_runtime_and_tokenizer_pins'] is None
    assert r['shared_training_candidate']['async_native_trainer'] == 'NOT_IMPLEMENTED_IN_THIS_PACKAGE'


def test_preflight_never_opens_native_from_fixture_pass():
    result = report()
    assert result['status'] == 'BLOCKED_NATIVE_AND_TRAINING' and result['model_calls'] == 0
    assert len(result['blockers']) == 6


@pytest.mark.parametrize('tamper', ['edit', 'extra', 'missing'])
def test_integrity_tampering(tmp_path, tamper):
    target = tmp_path / 'package'
    shutil.copytree(ROOT, target, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    if tamper == 'edit':
        with (target / 'attention.py').open('a') as f:
            f.write('# changed\n')
    elif tamper == 'extra':
        (target / 'unregistered.py').write_text('x = 1\n')
    else:
        (target / 'objective.py').unlink()
    with pytest.raises(ValueError):
        verify_manifest(target)


def test_ngustaleness_boundary_is_exclusive():
    s = continuing_sampler()
    decision = s.ingest('e', batch(1, (1, 0, 0, 0), version=3), 3)
    assert len(decision.samples) == 8 and decision.stale_count == 0


def test_pending_cap_is_checked_before_acceptance():
    s = continuing_sampler()
    s.max_pending = 1
    before = s.generated_tokens
    with pytest.raises(ValueError):
        s.ingest('other', batch(1), 0)
    assert s.generated_tokens == before and len(s.pending) == 1


def test_policy_clip_signs_and_padding_normalization():
    current = torch.tensor([[-1.], [-1.]], requires_grad=True)
    behavior = torch.tensor([[-2.], [-2.]])
    adv = torch.tensor([1., -1.])
    actual = policy_loss(current, behavior, current.detach(), adv,
                         torch.ones(2, 1, dtype=torch.bool), beta=0)
    expected = (torch.exp(torch.tensor(1.)) - 1.28) / 2
    torch.testing.assert_close(actual, expected)


def test_prefix_gradient_matches_chunk_gradient():
    m, q, k, v, d, p, valid = fixture()
    whole, _ = call(m, q, k, v, d, p, valid)
    whole.square().sum().backward()
    grad = m.pq.grad.clone()
    m.zero_grad(set_to_none=True)
    first, state = call(m, q[:, :, :3], k[:, :, :3], v[:, :, :3],
                        d[:, :, :3], p[:, :3], valid[:, :3])
    second, _ = call(m, q[:, :, 3:], k[:, :, 3:], v[:, :, 3:],
                     d[:, :, 3:], p[:, 3:], valid[:, 3:], state=state)
    torch.cat((first, second), 2).square().sum().backward()
    torch.testing.assert_close(m.pq.grad, grad, atol=1e-10, rtol=1e-10)
