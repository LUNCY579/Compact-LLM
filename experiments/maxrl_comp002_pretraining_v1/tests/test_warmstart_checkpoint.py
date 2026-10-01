"""CPU durability acceptance: exact boundary resume, corruption and no repeated dose."""
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.maxrl_comp002_pretraining_v1 import warmstart_checkpoint as cp


def usage(path):
    return SimpleNamespace(free=64 * cp.GIB)


def seed():
    random.seed(171)
    np.random.seed(172)
    torch.manual_seed(173)


def model_optimizer():
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(.2), torch.nn.Linear(5, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0, foreach=False)
    return model, optimizer


def update(model, optimizer, store=None):
    emit = store.operation if store is not None else lambda *args, **kwargs: None
    optimizer.zero_grad(set_to_none=True)
    for _ in range(8):
        x = torch.randn(2, 3) + random.random() + float(np.random.random())
        emit('forward', 'attempted', input_positions=2, supervised_positions=2)
        loss = model(x).square().mean() / 8
        emit('forward', 'returned', input_positions=2, supervised_positions=2)
        emit('backward', 'attempted')
        loss.backward()
        emit('backward', 'returned')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1, foreach=False)
    emit('optimizer_step', 'attempted')
    optimizer.step()
    emit('optimizer_step', 'returned')
    optimizer.zero_grad(set_to_none=True)


def snapshots(model, optimizer):
    state = {name: p.detach().clone() for name, p in model.named_parameters()}
    moments = [{key: value.clone() for key, value in optimizer.state[p].items()} for p in model.parameters()]
    rng, tensors = cp.rng_state()
    return state, moments, json.loads(cp.canonical(rng)), {k: v.clone() for k, v in tensors.items()}


def test_interrupted_boundary_resume_is_bitwise_with_all_rng_and_adam(tmp_path):
    bindings = {'code': 'source-A', 'data': 'data-A', 'recipe': {'rows': 24}, 'runtime': 'cpu-fp32'}
    seed()
    model, optimizer = model_optimizer()
    numel = sum(p.numel() for p in model.parameters())
    uninterrupted = cp.Store(tmp_path / 'whole', numel, bindings, usage=usage)
    uninterrupted.save(model, optimizer, 0, {'steps': []})
    for index in range(3):
        uninterrupted.append('step_begin', cursor=index + 1)
        update(model, optimizer)
        uninterrupted.save(model, optimizer, index + 1, {'steps': [{'index': i} for i in range(index + 1)]})
    expected = snapshots(model, optimizer)
    uninterrupted.close()
    assert len(list((tmp_path / 'whole' / cp.DIRECTORY).glob('*.json'))) == 2

    seed()
    model, optimizer = model_optimizer()
    interrupted = cp.Store(tmp_path / 'split', numel, bindings, usage=usage)
    interrupted.save(model, optimizer, 0, {'steps': []})
    interrupted.append('step_begin', cursor=1)
    update(model, optimizer)
    interrupted.save(model, optimizer, 1, {'steps': [{'index': 0}]})
    interrupted.close()
    # A fresh process would have different initialization/RNG. Restore must recover all of it.
    model, optimizer = model_optimizer()
    random.random(), np.random.random(), torch.randn(7)
    resumed = cp.Store(tmp_path / 'split', numel, bindings, resume=True, reason='documented CPU restart', usage=usage)
    boundary = resumed.restore(model, optimizer)
    assert boundary['cursor'] == 1
    for index in range(boundary['cursor'], 3):
        resumed.append('step_begin', cursor=index + 1)
        update(model, optimizer)
        resumed.save(model, optimizer, index + 1, {'steps': [{'index': i} for i in range(index + 1)]})
    actual = snapshots(model, optimizer)
    resumed.close()
    assert actual[2] == expected[2]
    assert all(torch.equal(actual[0][key], expected[0][key]) for key in expected[0])
    assert all(torch.equal(a[key], b[key]) for a, b in zip(actual[1], expected[1]) for key in b)
    assert all(torch.equal(actual[3][key], expected[3][key]) for key in expected[3])
    events = cp.Store.__new__(cp.Store)
    events.root = tmp_path / 'split'
    assert [e['cursor'] for e in events.read_journal() if e['kind'] == 'step_begin'] == [1, 2, 3]
    assert not torch.cuda.is_initialized()


def test_corrupted_latest_checkpoint_refuses_without_falling_back(tmp_path):
    model, optimizer = model_optimizer()
    store = cp.Store(tmp_path, 42, {'binding': 1}, usage=usage)
    store.save(model, optimizer, 0, {'steps': []})
    store.append('step_begin', cursor=1)
    update(model, optimizer)
    store.save(model, optimizer, 1, {'steps': [{'index': 0}]})
    store.close()
    path = tmp_path / cp.DIRECTORY / 'checkpoint-000001.safetensors'
    with path.open('r+b') as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 1]))
    resumed = cp.Store(tmp_path, 42, {'binding': 1}, resume=True, reason='restart', usage=usage)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    try:
        with pytest.raises(ValueError, match='bytes drift'):
            resumed.restore(model, optimizer)
        assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
    finally:
        resumed.close()


def test_claim_binding_and_duplicate_attempt_are_rejected(tmp_path):
    store = cp.Store(tmp_path, 42, {'recipe': 1}, usage=usage)
    with pytest.raises(FileExistsError):
        cp.Store(tmp_path, 42, {'recipe': 1}, usage=usage)
    with pytest.raises(ValueError, match='binding drift'):
        cp.Store(tmp_path, 42, {'recipe': 2}, resume=True, reason='restart', usage=usage)
    with pytest.raises(ValueError, match='another warmstart invocation is active'):
        cp.Store(tmp_path, 42, {'recipe': 1}, resume=True, reason='restart', usage=usage)
    store.close()


def test_uncheckpointed_update_rolls_back_and_repeated_compute_stays_charged(tmp_path):
    """Failure after step37 mutated Adam, before publication: final75 matches uninterrupted75."""
    bindings = {'code': 'bound-source', 'data': 'bound-data', 'recipe': {'steps': 75, 'rows': 600}, 'runtime': 'cpu'}
    seed()
    model, optimizer = model_optimizer()
    for _ in range(75):
        update(model, optimizer)
    expected = snapshots(model, optimizer)
    seed()
    model, optimizer = model_optimizer()
    numel = sum(p.numel() for p in model.parameters())
    store = cp.Store(tmp_path, numel, bindings, usage=usage)
    store.save(model, optimizer, 0, {'steps': []})
    for index in range(37):
        store.append('step_begin', cursor=index + 1)
        update(model, optimizer, store)
        if index < 36:
            store.save(model, optimizer, index + 1, {'steps': [{'index': i} for i in range(index + 1)]})
    store.close()
    model, optimizer = model_optimizer()
    resumed = cp.Store(tmp_path, numel, bindings, resume=True, reason='failure after optimizer step37 before publish', usage=usage)
    try:
        boundary = resumed.restore(model, optimizer)
        assert boundary['cursor'] == 36
        for index in range(boundary['cursor'], 75):
            resumed.append('step_begin', cursor=index + 1)
            update(model, optimizer, resumed)
            resumed.save(model, optimizer, index + 1, {'steps': [{'index': i} for i in range(index + 1)]})
        actual = snapshots(model, optimizer)
        assert all(torch.equal(actual[0][key], expected[0][key]) for key in expected[0])
        assert all(torch.equal(a[key], b[key]) for a, b in zip(actual[1], expected[1]) for key in b)
        assert actual[2] == expected[2] and all(torch.equal(actual[3][key], expected[3][key]) for key in expected[3])
        assert resumed.cost()['optimizer_step_returned'] == 76
        assert resumed.cost()['optimizer_step_attempted'] == 76
        assert resumed.cost()['forward_returned'] == 76 * 8
        assert len(resumed.verify(resumed._manifests()[-1])[0]['progress']['steps']) == 75
        assert any(e['kind'] == 'rollback' and e['restored_cursor'] == 36 for e in resumed.events)
    finally:
        resumed.close()


def test_disk_floor_and_full_ledger_are_checked_again_before_writes(tmp_path):
    expected = 14 * cp.GIB + 28 * 42 + cp.METADATA_RESERVE
    assert cp.disk_gate(tmp_path, 42, usage=lambda p: SimpleNamespace(free=expected))['required_bytes'] == expected
    with pytest.raises(ValueError, match='disk floor/reserve'):
        cp.disk_gate(tmp_path, 42, usage=lambda p: SimpleNamespace(free=expected - 1))
    values = [expected, expected, expected - 1]
    store = cp.Store(tmp_path, 42, {'binding': 1}, usage=lambda p: SimpleNamespace(free=values.pop(0)))
    try:
        with pytest.raises(ValueError, match='disk floor/reserve'):
            store.append('step_begin', cursor=1)
        assert not any(e['kind'] == 'step_begin' for e in store.events)
    finally:
        store.close()


def test_journal_torn_tail_refuses_resume(tmp_path):
    store = cp.Store(tmp_path, 42, {'binding': 1}, usage=usage)
    store.close()
    with (tmp_path / cp.JOURNAL).open('ab') as stream:
        stream.write(b'{"partial":')
    with pytest.raises(ValueError, match='journal truncated'):
        cp.Store(tmp_path, 42, {'binding': 1}, resume=True, reason='restart', usage=usage)
    assert not (tmp_path / cp.ACTIVE).exists()


def test_no_step_claim_recovers_verified_base_and_retains_load_wall_costs(tmp_path, monkeypatch):
    model, optimizer = model_optimizer()
    bindings = {'code': 'same', 'data': 'same', 'runtime': 'cpu'}
    store = cp.Store(tmp_path, 42, bindings, usage=usage)
    store.operation('model_load', 'attempted', purpose='base')
    store.operation('model_load', 'returned', purpose='base')
    original = cp.exclusive_json
    def fail_manifest(path, value):
        if path.name == 'checkpoint-000000.json':
            raise RuntimeError('killed before initial checkpoint manifest')
        return original(path, value)
    monkeypatch.setattr(cp, 'exclusive_json', fail_manifest)
    with pytest.raises(RuntimeError, match='initial checkpoint'):
        store.save(model, optimizer, 0, {'steps': []})
    orphan = tmp_path / cp.DIRECTORY / 'checkpoint-000000.safetensors'
    assert orphan.is_file()
    orphan_sha = cp.sha(orphan)
    store.close()
    monkeypatch.setattr(cp, 'exclusive_json', original)
    resumed = cp.Store(tmp_path, 42, bindings, resume=True, reason='zero-step model-load interruption', usage=usage)
    try:
        resumed.operation('model_load', 'attempted', purpose='base')
        resumed.operation('model_load', 'returned', purpose='base')
        progress = {'steps': [], 'fingerprints_before': 'verified original base'}
        value = resumed.restore(model, optimizer, initial_progress=progress)
        assert value['base_restart'] and value['cursor'] == 0 and value['progress'] == progress
        assert not orphan.exists()
        assert any(e['kind'] == 'orphan_removed' and e['file_sha256'] == orphan_sha for e in resumed.events)
        resumed.save(model, optimizer, 0, progress)
        assert resumed.cost()['model_load_attempted'] == resumed.cost()['model_load_returned'] == 2
        invocations = resumed.invocations()
        assert len(invocations) == 2 and all(row['wall_known_s'] > 0 for row in invocations)
        assert invocations[0]['wall_upper_s'] >= invocations[0]['wall_known_s']
    finally:
        resumed.close()


@pytest.mark.parametrize('next_boot', [2000.0, None])
def test_wall_upper_bound_is_unknown_across_changed_or_unavailable_boot(tmp_path, monkeypatch, next_boot):
    import psutil
    monkeypatch.setattr(psutil, 'boot_time', lambda: 1000.0)
    store = cp.Store(tmp_path, 42, {'bound': True}, usage=usage)
    store.operation('model_load', 'attempted')  # a killed load is not silently counted as returned
    lower = store.invocations()[0]['wall_known_s']
    store.close()
    monkeypatch.setattr(psutil, 'boot_time', lambda: next_boot)
    resumed = cp.Store(tmp_path, 42, {'bound': True}, resume=True, reason='new OS boot', usage=usage)
    try:
        prior = resumed.invocations()[0]
        assert prior['wall_known_s'] == lower and prior['wall_upper_s'] is None
        assert prior['model_load_attempted'] == 1 and prior['model_load_returned'] == 0 and prior['model_load_pending'] == 1
        assert 'boot' in prior['uncertainty_rule']
    finally:
        resumed.close()
