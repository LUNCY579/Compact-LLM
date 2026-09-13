"""Synthetic prospective contract checks; no artifact/model/GPU execution."""
import copy

import pytest

from cross_scale.generation_stage.v1.src import protocol


def synthetic_config(device='cpu'):
    root = 'experiments/supa_adaptive_compute_v2/cross_scale/generation_stage/v1/evidence/'
    return {**protocol._fixed(device), 'dataset_manifest_sha256': 'a' * 64,
        'upstream_export': {'manifest_path': root + '00_upstream_exports/toy/manifest.json',
            'manifest_sha256': 'b' * 64, 'tensor_path': root + '00_upstream_exports/toy/bridges.pt',
            'tensor_sha256': 'c' * 64, 'source_objectives': {'XG': 'B3', 'XS': 'BS', 'XR': 'BR'},
            'tensor_hashes': {arm: {key: 'd' * 64 for key in protocol.upstream.SHAPES}
                              for arm in ('XG', 'XS', 'XR')}},
        'matched_control': {**protocol._control_fixed(), 'alpha': .75,
            'state_sha256': 'e' * 64, 'artifact_path': root + '00_control_calibration/toy/report.json',
            'artifact_sha256': 'f' * 64}}


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_fixed_matrix_and_explicit_runtime(device):
    config = synthetic_config(device)
    checked = protocol.validate_config(config, verify_artifacts=False)
    assert checked == config and checked is not config
    assert len(checked['cells']) == 17
    assert checked['min_available_gib'] == (3.5 if device == 'cpu' else .75)
    assert checked['qualification']['capture_layers'] == [2, 19, 27]
    assert checked['native_comparator']['available'] is False
    assert checked['confirmatory_generation_licensed'] is False


@pytest.mark.parametrize('mutator', [
    lambda c: c['cells'].pop('XM_g0.03'),
    lambda c: c['cells'].update({'XN': {'arm': 'XN', 'gate': .1}}),
    lambda c: c['cells']['XG_g0.03'].update(gate=.05),
    lambda c: c.update(device='cuda:1'),
    lambda c: c.update(dtype='bfloat16'),
    lambda c: c.update(use_cache=True),
    lambda c: c['screen_policy'].update(minimum_paired_wins_vs_X0=1),
    lambda c: c['screen_policy'].update(maximum_additional_preservation_errors=2),
    lambda c: c['data'].update(allowed_roles=['confirmatory_dev']),
    lambda c: c.update(confirmatory_generation_licensed=True),
    lambda c: c['matched_control'].update(seed=17),
    lambda c: c['matched_control'].update(alpha=float('nan')),
    lambda c: c['matched_control'].update(alpha=True),
    lambda c: c['matched_control'].update(alpha=0),
    lambda c: c['matched_control'].update(artifact_path='../outside/report.json'),
    lambda c: c['upstream_export']['source_objectives'].update(XS='B0'),
    lambda c: c['upstream_export']['tensor_hashes']['XR'].pop('down_bias'),
    lambda c: c['upstream_export'].update(tensor_path=c['upstream_export']['manifest_path']),
    lambda c: c['qualification'].update(max_abs_logit_error=.003),
    lambda c: c['qualification'].update(require_gpu_repeat_identity=False),
])
def test_refuses_policy_geometry_scope_and_artifact_drift(mutator):
    config = synthetic_config()
    mutator(config)
    with pytest.raises((ValueError, KeyError)):
        protocol.validate_config(config, verify_artifacts=False)


def test_builder_derives_scalar_from_verifier_without_writing(monkeypatch, tmp_path):
    config = synthetic_config()
    monkeypatch.setattr(protocol, '_verify_export', lambda path, **kwargs: copy.deepcopy(config['upstream_export']))
    def verify(path, export):
        assert export == config['upstream_export']
        return copy.deepcopy(config['matched_control'])
    monkeypatch.setattr(protocol, '_verify_calibration', verify)
    monkeypatch.setattr(protocol.upstream, '_sha', lambda path: config['dataset_manifest_sha256'])
    calls = []
    monkeypatch.setattr(protocol, 'validate_config', lambda result: calls.append(result))
    before = list(tmp_path.iterdir())
    result = protocol.build_config(export_manifest_path=tmp_path/'manifest.json',
        calibration_artifact_path=tmp_path/'report.json')
    assert result == config and calls == [result]
    assert list(tmp_path.iterdir()) == before


def test_builder_propagates_failed_calibration_without_default_alpha(monkeypatch, tmp_path):
    config = synthetic_config()
    monkeypatch.setattr(protocol, '_verify_export', lambda path, **kwargs: config['upstream_export'])
    def refuse(*args):
        raise ValueError('uncommitted or incomplete calibration')
    monkeypatch.setattr(protocol, '_verify_calibration', refuse)
    with pytest.raises(ValueError, match='uncommitted'):
        protocol.build_config(export_manifest_path=tmp_path/'manifest.json',
            calibration_artifact_path=tmp_path/'report.json')
    assert not list(tmp_path.iterdir())
