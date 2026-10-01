from __future__ import annotations

import json

import pytest

from experiments.maxrl_comp002_pretraining_v1 import s0_competence, s0_freeze, s0_runtime as s0, scoped_scorer
from experiments.maxrl_comp002_pretraining_v1.preflight import digest, sha

PREIMAGE = {'schema': 'COMP-RS-001-S0-DECODING-1', 'device': 'cuda:0', 'dtype': 'float32'}


def q1_receipt(**over):
    value = {'stage': 'S0_Q1_FP32', 'test_mode': False, 'registered_run': True, 'passed': True, 'failed_checks': [],
             'decoding_preimage': PREIMAGE, 'decoding_sha256': s0.decoding_sha256(PREIMAGE),
             'host_floor': {'peak_worker_rss_gib': 2.75}}
    value.update(over)
    return value


def pilot_receipt(inventory='A', outcome=None, **over):
    value = {'stage': 'S0_PILOT_' + inventory, 'test_mode': False, 'registered_run': True,
             'decision': {'outcome': outcome or 'ADOPT_' + inventory, 'inventory': inventory},
             'decoding_sha256': s0.decoding_sha256(PREIMAGE)}
    value.update(over)
    return value


def dump(tmp_path, name, value):
    path = tmp_path / name
    path.write_bytes(json.dumps(value).encode('utf-8'))
    return path


def test_freeze_binds_the_q1_preimage_floor_live_bundle_and_adopted_inventory(tmp_path):
    q1, pilot = dump(tmp_path, 'q1.json', q1_receipt()), dump(tmp_path, 'p.json', pilot_receipt())
    value = s0_freeze.build(q1, pilot)
    assert value['stage'] == 'S0_RUNTIME_FREEZE' and value['registered_run'] is True
    assert value['decoding_sha256'] == s0.decoding_sha256(PREIMAGE) and value['decoding_preimage'] == PREIMAGE
    assert value['host_floor_gib'] == pytest.approx(3.75)
    assert value['scoring_code_bundle_sha256'] == digest(scoped_scorer.code_sha256())
    assert value['inventory']['name'] == 'A' and value['inventory']['blocks'] == [8, 27]
    assert value['inventory']['registered_numel'] == 314_618_880
    out = tmp_path / 'freeze.json'
    out.write_bytes((json.dumps(value) + '\n').encode('ascii'))
    read = s0_competence.read_freeze(out, registered=True)   # the reader accepts what the writer emits
    assert read['host_floor_gib'] == pytest.approx(3.75) and read['decoding_sha256'] == value['decoding_sha256']


def test_inventory_b_needs_the_a_receipt_that_ended_rerun_with_b(tmp_path):
    q1 = dump(tmp_path, 'q1.json', q1_receipt())
    pilot_b = dump(tmp_path, 'b.json', pilot_receipt('B'))
    with pytest.raises(ValueError, match='needs the A pilot receipt'):
        s0_freeze.build(q1, pilot_b)
    bad_a = dump(tmp_path, 'a_bad.json', pilot_receipt('A', 'ADOPT_A'))
    with pytest.raises(ValueError, match='RERUN_WITH_B'):
        s0_freeze.build(q1, pilot_b, a_path=bad_a)
    a = dump(tmp_path, 'a.json', pilot_receipt('A', 'RERUN_WITH_B', decision={'outcome': 'RERUN_WITH_B',
                                                                                'inventory': 'A'}))
    value = s0_freeze.build(q1, pilot_b, a_path=a)
    assert value['inventory']['name'] == 'B' and value['inventory']['blocks'] == [14, 27]
    assert value['inventory']['registered_numel'] == 220_233_216
    with pytest.raises(ValueError, match='only given when B'):
        s0_freeze.build(q1, dump(tmp_path, 'pa.json', pilot_receipt('A')), a_path=a)


@pytest.mark.parametrize('q1_over, message', [
    ({'passed': False, 'failed_checks': ['T3']}, 'did not pass'),
    ({'stage': 'S0_Q1_BF16'}, 'not an S0_Q1_FP32'),
    ({'test_mode': True, 'registered_run': False}, 'not a registered run'),
    ({'decoding_sha256': '0' * 64}, 'not the digest of its preimage'),
    ({'decoding_preimage': dict(PREIMAGE, device='cpu'),
      'decoding_sha256': s0.decoding_sha256(dict(PREIMAGE, device='cpu'))}, 'not for cuda:0'),
    ({'host_floor': {'peak_worker_rss_gib': None}}, 'peak worker RSS missing'),
])
def test_q1_refusals(tmp_path, q1_over, message):
    q1, pilot = dump(tmp_path, 'q1.json', q1_receipt(**q1_over)), dump(tmp_path, 'p.json', pilot_receipt())
    with pytest.raises(ValueError, match=message):
        s0_freeze.build(q1, pilot)


@pytest.mark.parametrize('pilot_over, message', [
    ({'decision': {'outcome': 'RERUN_WITH_B', 'inventory': 'A'}}, 'did not adopt'),
    ({'decoding_sha256': '1' * 64}, 'different decoding_sha256'),
    ({'test_mode': True, 'registered_run': False}, 'not a registered run'),
    ({'stage': 'S0_PILOT_C'}, 'not an S0 pilot'),
])
def test_pilot_refusals(tmp_path, pilot_over, message):
    q1, pilot = dump(tmp_path, 'q1.json', q1_receipt()), dump(tmp_path, 'p.json', pilot_receipt(**pilot_over))
    with pytest.raises(ValueError, match=message):
        s0_freeze.build(q1, pilot)


def test_test_mode_accepts_synthetic_receipts_and_is_recorded(tmp_path):
    q1 = dump(tmp_path, 'q1.json', q1_receipt(test_mode=True, registered_run=False,
                                              decoding_preimage=dict(PREIMAGE, device='cpu'),
                                              decoding_sha256=s0.decoding_sha256(dict(PREIMAGE, device='cpu'))))
    pilot = dump(tmp_path, 'p.json', pilot_receipt(test_mode=True, registered_run=False,
                                                   decoding_sha256=s0.decoding_sha256(dict(PREIMAGE, device='cpu'))))
    value = s0_freeze.build(q1, pilot, test_mode=True)
    assert value['test_mode'] is True and value['registered_run'] is False
    out = tmp_path / 'f.json'
    out.write_bytes(json.dumps(value).encode('ascii'))
    with pytest.raises(ValueError, match='not a registered run'):
        s0_competence.read_freeze(out, registered=True)


def test_cli_is_create_only_and_refuses_the_checkout(tmp_path):
    q1, pilot = dump(tmp_path, 'q1.json', q1_receipt()), dump(tmp_path, 'p.json', pilot_receipt())
    out = tmp_path / 'records' / 'S0_RUNTIME_FREEZE.json'
    args = ['--q1-receipt', str(q1), '--pilot-receipt', str(pilot), '--out', str(out), '--test-mode']
    assert s0_freeze.main(args) == 0 and out.exists()
    with pytest.raises(FileExistsError):
        s0_freeze.main(args)
    inside = s0.HERE / 'S0_RUNTIME_FREEZE_should_not_exist.json'
    with pytest.raises(ValueError):
        s0_freeze.main(['--q1-receipt', str(q1), '--pilot-receipt', str(pilot), '--out', str(inside), '--test-mode'])
    assert not inside.exists()


def test_c12_unmeasured_pilot_freezes_with_inventory_pending(tmp_path):
    q1 = dump(tmp_path, 'q1.json', q1_receipt())
    pilot = dump(tmp_path, 'p.json', pilot_receipt('A', 'UNMEASURED_NO_UPDATE'))
    value = s0_freeze.build(q1, pilot)
    assert value['inventory']['name'] == 'PENDING' and value['inventory']['blocks'] is None
    b = dump(tmp_path, 'b.json', pilot_receipt('B', 'UNMEASURED_NO_UPDATE'))
    a = dump(tmp_path, 'a.json', pilot_receipt('A', 'RERUN_WITH_B'))
    with pytest.raises(ValueError, match='did not adopt'):
        s0_freeze.build(q1, b, a_path=a)


# ------------------------------------------------------------------------------------------ theta0' (section 18)
REAL_FREEZE = s0.RECORDS / 'S0_RUNTIME_FREEZE.json'


def pinned_weights_sha():
    return json.loads(s0.ASSET_RECEIPT.read_bytes())['assets']['model.safetensors']['sha256']


@pytest.fixture
def delta_file(tmp_path):
    import torch
    from experiments.maxrl_comp002_pretraining_v1 import arm_build
    path = tmp_path / 'delta' / 'delta_warm.safetensors'
    path.parent.mkdir()
    delta_sha = arm_build.write_delta(path, {'model.norm.weight': torch.full((4,), 0.25)})
    return path, delta_sha


def warm_receipt(delta, base=None, **over):
    path, delta_sha = delta
    value = {'stage': 'S0_WARMSTART', 'test_mode': False, 'registered_run': True,
             'completed': True, 'outcome': 'WARMSTART_COMPLETE', 'plan': {'rows': 600, 'steps': 75},
             'training': {'steps': 75}, 'steps': [{'index': index} for index in range(75)],
             'anchor': {'base_weights_sha256': base or pinned_weights_sha(), 'delta_path': str(path),
                        'delta_sha256': delta_sha, 'arm_sha256': 'a' * 64}}
    value.update(over)
    return value


def cpu_q1(**over):
    cpu = dict(PREIMAGE, device='cpu')
    return q1_receipt(test_mode=True, registered_run=False, decoding_preimage=cpu,
                      decoding_sha256=s0.decoding_sha256(cpu), **over)


def synthetic_asset(tmp_path):
    return dump(tmp_path, 'asset.json', {'stage': 'Q1_ASSET_CANDIDATE',
                                         'assets': {'model.safetensors': {'sha256': 'e' * 64}}})


def test_warm_freeze_carries_the_q1_runtime_and_adds_the_anchor(tmp_path, delta_file):
    q1 = dump(tmp_path, 'q1.json', q1_receipt())
    warm = dump(tmp_path, 'S0_WARMSTART.json', warm_receipt(delta_file))
    value = s0_freeze.build_warm(q1, warm)
    assert value['stage'] == 'S0_RUNTIME_FREEZE' and value['schema'] == s0_freeze.SCHEMA_WARM
    assert value['registered_run'] is True and value['test_mode'] is False
    assert value['decoding_preimage'] == PREIMAGE and value['decoding_sha256'] == s0.decoding_sha256(PREIMAGE)
    assert value['host_floor_gib'] == pytest.approx(3.75)
    assert value['inventory']['name'] == 'PENDING' and value['inventory']['blocks'] is None
    assert 'C12' in value['inventory']['rule'] and value['inputs']['pilot'] is None
    assert value['anchor'] == {'base_weights_sha256': pinned_weights_sha(), 'delta_path': str(delta_file[0]),
                               'delta_sha256': delta_file[1], 'arm_sha256': 'a' * 64,
                               'warmstart_receipt_sha256': sha(warm)}
    assert value['inputs']['warmstart'] == {'path': str(warm.resolve()), 'sha256': sha(warm)}
    assert value['anchor_digest_rule'] == s0.ANCHOR_DIGEST_RULE
    assert value['scoring_code_bundle_sha256'] == digest(scoped_scorer.code_sha256())
    assert 'arm_build.py' in value['code_sha256']
    out = tmp_path / 'S0_RUNTIME_FREEZE_WARM.json'
    out.write_bytes((json.dumps(value) + '\n').encode('ascii'))
    read = s0_competence.read_freeze(out, registered=True)   # the reader returns the anchor block
    assert read['anchor'] == value['anchor'] and read['inventory']['name'] == 'PENDING'
    supa = dump(tmp_path, 'supa.json', s0_freeze.build(q1, dump(tmp_path, 'p.json', pilot_receipt())))
    assert json.loads(supa.read_bytes())['anchor'] is None
    assert s0_competence.read_freeze(supa, registered=True)['anchor'] is None


def test_the_real_supa_freeze_reads_with_no_anchor():
    if not REAL_FREEZE.is_file():
        pytest.skip('records absent')
    assert 'anchor' not in json.loads(REAL_FREEZE.read_bytes())
    assert s0_competence.read_freeze(REAL_FREEZE, registered=True)['anchor'] is None


@pytest.mark.parametrize('edit, message', [
    ({'test_mode': True, 'registered_run': False}, 'not a registered run'),
    ({'registered_run': False}, 'not a registered run'),
    ({'stage': 'S0_PILOT_A'}, 'not an S0_WARMSTART'),
    ({'anchor': None}, 'has no .anchor. object'),
    ('drop:arm_sha256', 'lacks: arm_sha256'),
    ('set:delta_sha256=ABC', 'not a sha256 hex digest'),
    ('set:base_weights_sha256=' + 'e' * 64, 'other base weights'),
    ('set:delta_sha256=' + '0' * 64, 'does not reproduce'),
    ('set:delta_path=delta_warm.safetensors', 'must be absolute'),
    ('missing', 'delta file is missing'),
    ('inside', 'inside the bound checkout')])
def test_warm_freeze_refusals(tmp_path, delta_file, edit, message):
    value = warm_receipt(delta_file)
    if isinstance(edit, dict):
        value.update(edit)
    elif edit.startswith('drop:'):
        del value['anchor'][edit[5:]]
    elif edit.startswith('set:'):
        key, new = edit[4:].split('=')
        value['anchor'][key] = new
    elif edit == 'missing':
        value['anchor']['delta_path'] = str(tmp_path / 'absent.safetensors')
    else:
        value['anchor']['delta_path'] = str(s0.HERE / 'delta_should_not_exist.safetensors')
    q1 = dump(tmp_path, 'q1.json', q1_receipt())
    with pytest.raises(ValueError, match=message):
        s0_freeze.build_warm(q1, dump(tmp_path, 'warm.json', value))


def test_warm_freeze_carries_the_q1_refusals(tmp_path, delta_file):
    warm = dump(tmp_path, 'warm.json', warm_receipt(delta_file))
    for over, message in (({'passed': False, 'failed_checks': ['T3']}, 'did not pass'),
                          ({'decoding_sha256': '0' * 64}, 'not the digest of its preimage')):
        with pytest.raises(ValueError, match=message):
            s0_freeze.build_warm(dump(tmp_path, 'q1_%d.json' % len(over), q1_receipt(**over)), warm)


def test_warm_freeze_test_mode_and_the_pinned_asset_receipt(tmp_path, delta_file):
    """Test mode admits a synthetic warm-start and asset receipt; a registered build binds the pinned asset."""
    asset = synthetic_asset(tmp_path)
    warm = dump(tmp_path, 'warm.json', warm_receipt(delta_file, base='e' * 64, test_mode=True,
                                                    registered_run=False))
    value = s0_freeze.build_warm(dump(tmp_path, 'q1.json', cpu_q1()), warm, test_mode=True, asset_receipt=asset)
    assert value['test_mode'] is True and value['anchor']['base_weights_sha256'] == 'e' * 64
    with pytest.raises(ValueError, match='pinned asset receipt'):
        s0_freeze.build_warm(dump(tmp_path, 'q1r.json', q1_receipt()),
                             dump(tmp_path, 'warm_r.json', warm_receipt(delta_file, base='e' * 64)),
                             asset_receipt=asset)
    out = tmp_path / 'f.json'
    out.write_bytes(json.dumps(value).encode('ascii'))
    with pytest.raises(ValueError, match='not a registered run'):
        s0_competence.read_freeze(out, registered=True)
    assert s0_competence.read_freeze(out, registered=False)['anchor'] == value['anchor']


def test_read_freeze_validates_the_anchor_block(tmp_path, delta_file):
    value = s0_freeze.build_warm(dump(tmp_path, 'q1.json', q1_receipt()),
                                 dump(tmp_path, 'warm.json', warm_receipt(delta_file)))
    for name, anchor, message in (
            ('extra', dict(value['anchor'], note='x'), 'exactly'),
            ('short', {k: v for k, v in value['anchor'].items() if k != 'warmstart_receipt_sha256'}, 'exactly'),
            ('hex', dict(value['anchor'], arm_sha256='A' * 64), 'digests'),
            ('path', dict(value['anchor'], delta_path=''), 'delta_path'),
            ('type', 'anchor', 'exactly')):
        path = dump(tmp_path, name + '.json', dict(value, anchor=anchor))
        with pytest.raises(ValueError, match=message):
            s0_competence.read_freeze(path, registered=True)


def test_cli_default_names_and_mode_refusals(tmp_path, delta_file, monkeypatch):
    records = tmp_path / 'records'
    monkeypatch.setattr(s0, 'RECORDS', records)
    q1 = dump(tmp_path, 'q1.json', cpu_q1())
    cpu = s0.decoding_sha256(dict(PREIMAGE, device='cpu'))
    pilot = dump(tmp_path, 'p.json', pilot_receipt(test_mode=True, registered_run=False, decoding_sha256=cpu))
    asset = synthetic_asset(tmp_path)
    warm = dump(tmp_path, 'warm.json', warm_receipt(delta_file, base='e' * 64, test_mode=True, registered_run=False))
    warm_argv = ['--q1-receipt', str(q1), '--anchor-receipt', str(warm), '--asset-receipt', str(asset), '--test-mode']
    assert s0_freeze.main(warm_argv) == 0
    assert s0_freeze.main(['--q1-receipt', str(q1), '--pilot-receipt', str(pilot), '--test-mode']) == 0
    assert sorted(p.name for p in records.iterdir()) == ['S0_RUNTIME_FREEZE.json', 'S0_RUNTIME_FREEZE_WARM.json']
    written = json.loads((records / 'S0_RUNTIME_FREEZE_WARM.json').read_bytes())
    assert written['anchor']['delta_sha256'] == delta_file[1] and written['inventory']['name'] == 'PENDING'
    assert json.loads((records / 'S0_RUNTIME_FREEZE.json').read_bytes())['anchor'] is None
    with pytest.raises(FileExistsError):
        s0_freeze.main(warm_argv)
    for argv, message in (
            (['--q1-receipt', str(q1), '--test-mode'], 'exactly one'),
            (['--q1-receipt', str(q1), '--pilot-receipt', str(pilot), '--anchor-receipt', str(warm)], 'exactly one'),
            (['--q1-receipt', str(q1), '--anchor-receipt', str(warm), '--a-receipt', str(pilot)], 'pilot freeze'),
            (['--q1-receipt', str(q1), '--anchor-receipt', str(warm), '--asset-receipt', str(asset)],
             'test-mode only')):
        with pytest.raises(SystemExit, match=message):
            s0_freeze.main(argv)


@pytest.mark.parametrize('over, message', [
    ({'completed': False}, 'did not complete successfully'),
    ({'outcome': 'HOST_GUARD_STOP'}, 'did not complete successfully'),
    ({'plan': None}, 'counts are missing'),
    ({'training': {'steps': 74}}, 'counts disagree'),
    ({'steps': [{}] * 74}, 'counts disagree'),
    ({'plan': {'rows': 592, 'steps': 75}}, 'counts disagree'),
    ({'plan': {'rows': 8, 'steps': 1}, 'training': {'steps': 1}, 'steps': [{}]}, 'not 600 rows / 75 steps'),
    ({'plan': {'rows': 600, 'steps': True}}, 'counts disagree')])
def test_warm_freeze_requires_completed_registered_counts(tmp_path, delta_file, over, message):
    q1 = dump(tmp_path, 'q1.json', q1_receipt())
    incomplete = dump(tmp_path, 'warm.json', warm_receipt(delta_file, **over))
    with pytest.raises(ValueError, match=message):
        s0_freeze.build_warm(q1, incomplete)


def test_warm_test_mode_accepts_a_coherent_small_completed_plan(tmp_path, delta_file):
    asset = synthetic_asset(tmp_path)
    warm = dump(tmp_path, 'warm.json', warm_receipt(delta_file, base='e' * 64, test_mode=True,
            registered_run=False, plan={'rows': 8, 'steps': 1}, training={'steps': 1}, steps=[{}]))
    value = s0_freeze.build_warm(dump(tmp_path, 'q1.json', cpu_q1()), warm,
                                 test_mode=True, asset_receipt=asset)
    assert value['test_mode'] is True and value['registered_run'] is False


def test_supa_freeze_rejects_warm_pilot_and_warm_a_fallback(tmp_path):
    q1 = dump(tmp_path, 'q1.json', q1_receipt())
    anchor = {'arm_sha256': 'a' * 64}
    warm_pilot = dump(tmp_path, 'p.json', pilot_receipt(anchor=anchor))
    with pytest.raises(ValueError, match='pilot anchor differs'):
        s0_freeze.build(q1, warm_pilot)
    plain_b = dump(tmp_path, 'b.json', pilot_receipt('B'))
    warm_a = dump(tmp_path, 'a.json', pilot_receipt('A', 'RERUN_WITH_B', anchor=anchor))
    with pytest.raises(ValueError, match='A pilot anchor differs'):
        s0_freeze.build(q1, plain_b, a_path=warm_a)


def test_b_adoption_rejects_a_fallback_with_other_decoding(tmp_path):
    q1 = dump(tmp_path, 'q1.json', q1_receipt())
    b = dump(tmp_path, 'b.json', pilot_receipt('B'))
    a = dump(tmp_path, 'a.json', pilot_receipt('A', 'RERUN_WITH_B', decoding_sha256='0' * 64))
    with pytest.raises(ValueError, match='A pilot ran under a different decoding'):
        s0_freeze.build(q1, b, a_path=a)


def warm_successor_inputs(tmp_path, delta_file, *, test_mode=False, inventory='A', a_peak=6.0, pilot_peak=4.0):
    q1 = dump(tmp_path, 'q1.json', cpu_q1() if test_mode else q1_receipt())
    asset = synthetic_asset(tmp_path) if test_mode else None
    warm = dump(tmp_path, 'warm.json', warm_receipt(delta_file,
            base='e' * 64 if test_mode else None, test_mode=test_mode, registered_run=not test_mode))
    initial = s0_freeze.build_warm(q1, warm, test_mode=test_mode, asset_receipt=asset)
    initial_path = dump(tmp_path, 'S0_RUNTIME_FREEZE_WARM.json', initial)
    frozen = s0_competence.read_freeze(initial_path, registered=not test_mode)
    bound = {key: initial['anchor'][key] for key in s0.ANCHOR_KEYS}
    competence = {'stage': 'S0_COMPETENCE', 'completed': True, 'outcome': 'SIGNAL', 'test_mode': test_mode,
                  'registered_run': not test_mode, 'anchor': bound, 'runtime_freeze': frozen,
                  'decoding_sha256': initial['decoding_sha256']}
    competence_path = dump(tmp_path, 'S0_COMPETENCE_WARM.json', competence)
    pilot = pilot_receipt(inventory, completed=True, error=None, oom=None, abort=None,
            test_mode=test_mode, registered_run=not test_mode, anchor=initial['anchor'], runtime_freeze=frozen,
            decoding_sha256=initial['decoding_sha256'],
            decision={'outcome': 'ADOPT_' + inventory, 'inventory': inventory, 'peak_worker_rss_gib': pilot_peak})
    pilot_path = dump(tmp_path, 'S0_PILOT_' + inventory + '_WARM.json', pilot)
    a_path = None
    if inventory == 'B':
        a = pilot_receipt('A', 'RERUN_WITH_B', test_mode=test_mode, registered_run=not test_mode,
                anchor=initial['anchor'], runtime_freeze=frozen, decoding_sha256=initial['decoding_sha256'],
                decision={'outcome': 'RERUN_WITH_B', 'inventory': 'A', 'peak_worker_rss_gib': a_peak})
        a_path = dump(tmp_path, 'S0_PILOT_A_WARM.json', a)
        pilot['after_a_pilot'] = {'path': str(a_path.resolve()), 'sha256': sha(a_path),
                                 'outcome': 'RERUN_WITH_B', 'registered_run': not test_mode,
                                 'anchor': initial['anchor']}
        dump(tmp_path, pilot_path.name, pilot)
    return initial_path, pilot_path, competence_path, a_path


@pytest.mark.parametrize('inventory, expected_floor', [('A', 5.0), ('B', 7.0)])
def test_postpilot_warm_freeze_preserves_anchor_and_adopts_inventory(tmp_path, delta_file, inventory, expected_floor):
    initial, pilot, competence, a = warm_successor_inputs(tmp_path, delta_file, inventory=inventory)
    before = initial.read_bytes()
    value = s0_freeze.build_warm_postpilot(initial, pilot, competence, a_path=a)
    original = json.loads(before)
    assert initial.read_bytes() == before and original['inventory']['name'] == 'PENDING'
    assert value['schema'] == s0_freeze.SCHEMA_WARM_PILOT
    assert value['registered_run'] is True and value['test_mode'] is False
    assert value['anchor'] == original['anchor'] and value['anchor_digest_rule'] == original['anchor_digest_rule']
    assert value['decoding_preimage'] == original['decoding_preimage']
    assert value['decoding_sha256'] == original['decoding_sha256']
    assert value['inventory']['name'] == inventory
    assert value['inventory']['blocks'] == list(s0_freeze.INVENTORY_BLOCKS[inventory])
    assert value['host_floor_gib'] == expected_floor
    assert len(value['host_floor']['pilot_requirements']) == (2 if inventory == 'B' else 1)
    assert value['inputs']['initial_warm_freeze']['sha256'] == sha(initial)
    assert value['inputs']['competence']['sha256'] == sha(competence)
    output = dump(tmp_path, 'successor.json', value)
    parsed = s0_competence.read_freeze(output, registered=True)
    assert parsed['anchor'] == original['anchor'] and parsed['inventory']['name'] == inventory


def test_postpilot_host_floor_does_not_lower_initial_q1_floor(tmp_path, delta_file):
    initial, pilot, competence, _ = warm_successor_inputs(tmp_path, delta_file, pilot_peak=1.0)
    value = s0_freeze.build_warm_postpilot(initial, pilot, competence)
    assert value['host_floor_gib'] == json.loads(initial.read_bytes())['host_floor_gib'] == 3.75


@pytest.mark.parametrize('target, edit, message', [
    ('competence', {'outcome': 'STOP_NO_SIGNAL'}, 'did not complete with SIGNAL'),
    ('competence', {'completed': False}, 'did not complete with SIGNAL'),
    ('competence', {'anchor': None}, 'another anchor/decoding'),
    ('competence', {'decoding_sha256': '0' * 64}, 'another anchor/decoding'),
    ('competence', {'runtime_freeze': {'sha256': '0' * 64}}, 'initial warm freeze/anchor'),
    ('pilot', {'anchor': None}, 'pilot anchor differs'),
    ('pilot', {'decoding_sha256': '0' * 64}, 'different decoding_sha256'),
    ('pilot', {'runtime_freeze': {'sha256': '0' * 64}}, 'initial warm freeze/anchor'),
    ('pilot', {'completed': False}, 'did not complete successfully'),
    ('pilot', {'decision': {'outcome': 'UNMEASURED_NO_UPDATE', 'inventory': 'A'}}, 'did not adopt'),
    ('pilot', {'decision': {'outcome': 'ADOPT_A', 'inventory': 'A', 'peak_worker_rss_gib': None}}, 'no measured peak'),
    ('pilot', {'registered_run': False}, 'not a registered run')])
def test_postpilot_warm_freeze_refuses_wrong_phase_or_binding(tmp_path, delta_file, target, edit, message):
    initial, pilot, competence, _ = warm_successor_inputs(tmp_path, delta_file)
    path = competence if target == 'competence' else pilot
    value = json.loads(path.read_bytes())
    value.update(edit)
    dump(path.parent, path.name, value)
    with pytest.raises(ValueError, match=message):
        s0_freeze.build_warm_postpilot(initial, pilot, competence)


@pytest.mark.parametrize('edit, message', [
    ({'anchor': None}, 'A pilot anchor differs'),
    ({'decoding_sha256': '0' * 64}, 'A pilot ran under a different decoding'),
    ({'runtime_freeze': {'sha256': '0' * 64}}, 'initial warm freeze/anchor')])
def test_postpilot_b_requires_a_on_the_same_anchor_and_freeze(tmp_path, delta_file, edit, message):
    initial, pilot, competence, a = warm_successor_inputs(tmp_path, delta_file, inventory='B')
    value = json.loads(a.read_bytes())
    value.update(edit)
    dump(a.parent, a.name, value)
    # Keep the ancestry hash coherent so this test isolates the anchor/runtime checks.
    b_value = json.loads(pilot.read_bytes())
    b_value['after_a_pilot']['sha256'] = sha(a)
    dump(pilot.parent, pilot.name, b_value)
    with pytest.raises(ValueError, match=message):
        s0_freeze.build_warm_postpilot(initial, pilot, competence, a_path=a)


@pytest.mark.parametrize('linkage', [None, {'sha256': '0' * 64}])
def test_postpilot_b_refuses_missing_or_wrong_a_ancestry(tmp_path, delta_file, linkage):
    initial, pilot, competence, a = warm_successor_inputs(tmp_path, delta_file, inventory='B')
    value = json.loads(pilot.read_bytes())
    if linkage is None:
        del value['after_a_pilot']
    else:
        value['after_a_pilot'].update(linkage)
    dump(pilot.parent, pilot.name, value)
    with pytest.raises(ValueError, match='did not bind the supplied A fallback receipt'):
        s0_freeze.build_warm_postpilot(initial, pilot, competence, a_path=a)


def test_postpilot_b_refuses_substitution_of_a_different_same_anchor_fallback(tmp_path, delta_file):
    initial, pilot, competence, a = warm_successor_inputs(tmp_path, delta_file, inventory='B')
    other = json.loads(a.read_bytes())
    other['decision']['peak_worker_rss_gib'] = 2.0
    substitute = dump(tmp_path, 'different_A_same_anchor.json', other)
    assert other['anchor'] == json.loads(a.read_bytes())['anchor']
    with pytest.raises(ValueError, match='did not bind the supplied A fallback receipt'):
        s0_freeze.build_warm_postpilot(initial, pilot, competence, a_path=substitute)


def test_postpilot_cli_creates_distinct_successor_without_mutating_initial(tmp_path, delta_file, monkeypatch):
    records = tmp_path / 'records'
    monkeypatch.setattr(s0, 'RECORDS', records)
    initial, pilot, competence, _ = warm_successor_inputs(tmp_path, delta_file, test_mode=True)
    before = initial.read_bytes()
    argv = ['--initial-warm-freeze', str(initial), '--pilot-receipt', str(pilot),
            '--competence-receipt', str(competence), '--test-mode']
    assert s0_freeze.main(argv) == 0
    assert initial.read_bytes() == before
    target = records / s0_freeze.DEFAULT_WARM_PILOT_OUT
    value = json.loads(target.read_bytes())
    assert value['test_mode'] is True and value['inventory']['name'] == 'A'
    assert value['anchor'] == json.loads(before)['anchor']
    with pytest.raises(FileExistsError):
        s0_freeze.main(argv)
    with pytest.raises(SystemExit, match='distinct successor'):
        s0_freeze.main(argv + ['--out', str(initial)])
    with pytest.raises(SystemExit, match='no Q1, anchor or asset overrides'):
        s0_freeze.main(argv + ['--q1-receipt', str(initial)])
    with pytest.raises(SystemExit, match='needs --pilot-receipt and --competence-receipt'):
        s0_freeze.main(['--initial-warm-freeze', str(initial), '--test-mode'])
    with pytest.raises(SystemExit, match='belongs to post-pilot mode'):
        s0_freeze.main(['--competence-receipt', str(competence), '--test-mode'])
