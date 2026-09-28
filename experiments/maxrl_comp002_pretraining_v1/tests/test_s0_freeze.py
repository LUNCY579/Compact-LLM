from __future__ import annotations

import json

import pytest

from experiments.maxrl_comp002_pretraining_v1 import s0_competence, s0_freeze, s0_runtime as s0, scoped_scorer
from experiments.maxrl_comp002_pretraining_v1.preflight import digest

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
