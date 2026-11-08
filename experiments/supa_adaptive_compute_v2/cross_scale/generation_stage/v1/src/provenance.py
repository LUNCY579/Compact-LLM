"""Stage-owned source, export and registration bindings; no model loading."""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import upstream

STAGE = Path(__file__).resolve().parents[1]
REPO, PROGRAMME = upstream.REPO, upstream.PROGRAMME
REGISTRATIONS = STAGE / 'evidence/00_preregistration'
SCHEMA = 'cross_scale_generation_registration_v1'


def require_committed(path):
    from shared.manifests import _require_committed
    _require_committed(Path(path).resolve())


def stage_sources():
    """Bind all actual transitive code; configs/artifacts are bound separately."""
    from shared.hashing import source_files
    values = source_files('cross_scale')
    values.update(upstream._stage_sources())
    for name in ('operators.py', 'protocol.py'):
        native = PROGRAMME / 'native_armr/src' / name
        values[native.relative_to(REPO).as_posix()] = upstream._sha(native)
    return dict(sorted(values.items()))


def _check(condition, message):
    if not condition:
        raise RuntimeError(message)


def _path(value, root=REPO):
    path = Path(value)
    if not path.is_absolute():
        path = REPO / path
    return upstream._within(path, root)


def _hash_bound(path, expected, *, committed):
    _check(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'invalid file digest')
    _check(upstream._sha(path) == expected, 'bound file changed: ' + str(path))
    if committed:
        require_committed(path)


def verify_export(manifest_path, expected_sha256=None, *, require_committed=True,
                  verify_upstream_now=False):
    """Validate immutable exported execution inputs without loading tensors.

    Full Stage 1 cache/bridge revalidation is performed at export, calibration
    freeze and generation freeze. Ordinary admission rechecks its committed
    provenance records, source bytes and the exact exported tensor file. It
    never refits a bridge or silently substitutes a missing bulk artifact.
    """
    path = _path(manifest_path, upstream.EXPORT_ROOT)
    _check(path.name == 'manifest.json', 'export manifest.json required')
    digest = upstream._sha(path)
    if expected_sha256 is not None:
        _check(digest == expected_sha256, 'export manifest hash mismatch')
    if require_committed:
        globals()['require_committed'](path)
    value = upstream._read(path)
    fixed = {'schema_version': 'cross_scale_generation_bridge_export_v1', 'status': 'COMPLETE',
             'source_objectives': {'XG': 'B3', 'XS': 'BS', 'XR': 'BR'},
             'subset_byte_identical_on_reload': True, 'dtype': 'float32', 'device': 'cpu',
             'frozen': True, 'trainable_parameters': 0, 'gradient_tensors': 0,
             'refit_performed': False, 'weights_changed': False, 'generation_licensed': False,
             'tensor_file': 'bridges.pt'}
    _check(all(type(value.get(k)) is type(v) and value[k] == v for k, v in fixed.items()),
           'invalid unchanged bridge-export contract')
    context, old = value['context'], value['upstream']
    _check(old['context'] == context and old['status'] == 'VERIFIED_MECHANISM_GO',
           'export upstream context mismatch')
    _check((context['supa_layer'], context['donor_layer'], context['selected_objective'],
            context['supa_hidden_size'], context['donor_hidden_size']) == (19, 24, 'B3', 1024, 2560),
           'export differs from the frozen selected pair')
    _check(context['generation_stage_source_files'] == upstream._stage_sources(),
           'export stage implementation changed')
    from shared.hashing import source_files
    _check(context['stage1_sources'] == source_files('cross_scale'), 'Stage 1 source changed')
    for relative, expected in {**context['stage1_sources'], **context['generation_stage_source_files']}.items():
        candidate = _path(relative)
        _hash_bound(candidate, expected, committed=require_committed and candidate.is_relative_to(PROGRAMME))
    capture_dir = _path(old['capture_directory'], PROGRAMME / 'cross_scale/evidence/01_alignment_matrix')
    aggregate = _path(old['aggregate_directory'], capture_dir / 'aggregate')
    audit = _path(old['audit_directory'], PROGRAMME / 'cross_scale/evidence/07_independent_audit')
    metadata = {
        _path(old['registration_path'], PROGRAMME / 'cross_scale/evidence/00_preregistration'): old['registration_file_sha256'],
        capture_dir / 'capture.json': old['capture_sha256'],
        aggregate / 'matrix.json': old['matrix_sha256'],
        aggregate / 'selection.json': old['selection_sha256'],
        aggregate / 'run_manifest.json': old['aggregate_manifest_sha256'],
        audit / 'audit_receipt.json': old['audit_receipt_sha256'],
        audit / 'input_hashes.json': old['audit_input_hashes_sha256'],
        audit / 'attempt_inventory.json': old['audit_attempt_inventory_sha256'],
    }
    for candidate, expected in metadata.items():
        _hash_bound(candidate, expected, committed=require_committed)
    selection = upstream._read(aggregate / 'selection.json')
    _check(selection['status'] == 'MECHANISM_GO' and selection['selected'] == [old['selected']],
           'export selection differs from committed complete matrix')
    _check(value['original_artifact_sha256_before'] == value['original_artifact_sha256_after']
           == old['original_bridges']['sha256'], 'export source tensor binding differs')
    tensors = _path(path.parent / 'bridges.pt', path.parent)
    _hash_bound(tensors, value['tensor_file_sha256'], committed=False)
    _check(set(value['tensor_sha256']) == {'XG', 'XS', 'XR'}, 'export tensor roles differ')
    for hashes in value['tensor_sha256'].values():
        _check(set(hashes) == set(upstream.SHAPES) and all(re.fullmatch('[0-9a-f]{64}', h) for h in hashes.values()),
               'incomplete tensor-level export hashes')
    if verify_upstream_now:
        fresh = upstream.verify_upstream(old['run_id'], old['aggregate_attempt'], audit_directory=audit)
        _check(fresh == old, 'full upstream provenance changed after export')
    capture = upstream._read(capture_dir / 'capture.json')
    activation = _path(PROGRAMME / 'cross_scale/local' / old['run_id'] / 'activations.pt',
                       PROGRAMME / 'cross_scale/local')
    _check(capture['activation_sha256'] == old['upstream_input_hashes'][str(activation)],
           'activation cache binding missing from export provenance')
    return {**value, 'export_manifest_path': str(path), 'export_manifest_sha256': digest,
            'tensor_path': str(tensors), 'activation_path': str(activation),
            'activation_sha256': capture['activation_sha256']}


def _contract(config_path, *, full_upstream=False):
    from .protocol import validate_config
    path = _path(config_path, STAGE / 'configs')
    config = validate_config(upstream._read(path), verify_artifacts=True)
    dataset = _path(config['dataset_manifest'], PROGRAMME / 'data/generated/v2/cross_scale')
    _check(upstream._sha(dataset) == config['dataset_manifest_sha256'], 'stage dataset manifest changed')
    if full_upstream:
        export = config['upstream_export']
        verify_export(export['manifest_path'], export['manifest_sha256'],
                      require_committed=True, verify_upstream_now=True)
    return {'schema_version': SCHEMA, 'track': 'cross_scale',
            'config_path': path.relative_to(REPO).as_posix(), 'config_sha256': upstream._sha(path),
            'config': config, 'sources': stage_sources(),
            'dataset_manifest_path': dataset.relative_to(REPO).as_posix(),
            'dataset_manifest_sha256': upstream._sha(dataset)}


def freeze_registration(config_path):
    from shared.hashing import write_json
    from shared.manifests import git_identity, utc_now
    contract = _contract(config_path, full_upstream=True)
    digest = upstream._canonical(contract)
    path = REGISTRATIONS / digest / 'registration.json'
    write_json(path, {**contract, 'status': 'FROZEN', 'registration_sha256': digest,
                      'registered_against': git_identity(), 'registered_utc': utc_now()})
    return path


def verify_registration(config_path):
    contract = _contract(config_path)
    digest = upstream._canonical(contract)
    path = REGISTRATIONS / digest / 'registration.json'
    value = upstream._read(path)
    _check(value.get('status') == 'FROZEN' and value.get('registration_sha256') == digest,
           'stage registration is not frozen')
    _check(all(value.get(k) == v for k, v in contract.items()), 'stage registration contract changed')
    for relative, expected in contract['sources'].items():
        candidate = _path(relative)
        _hash_bound(candidate, expected, committed=candidate.is_relative_to(PROGRAMME))
    for candidate in (path, _path(contract['config_path']), _path(contract['dataset_manifest_path'])):
        require_committed(candidate)
    return {**value, 'registration_path': str(path.resolve())}
