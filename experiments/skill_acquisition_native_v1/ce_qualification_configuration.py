"""Fixed source adoption of bounded qualification metadata, never model authority.

The adopted declaration contains paths and finite intake bounds, not request,
scope or source-bundle hashes. Those files are generated after code freeze, so
no source/request/configuration hash cycle is required. Source adoption remains
closed by default. The actual controller still independently admits every raw
scope/source/operator reference before any model work.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import time
from types import MappingProxyType
import uuid

from . import ce_native_authority as authority, ce_qualification_artifacts as raw

DECLARATION = 'CE8-QUALIFICATION-METADATA-ADOPTION-1'
RECORD = 'CE8-QUALIFICATION-METADATA-INTAKE-1'
MAX_CONFIGURATION = 8 * 1024 ** 2
_SOURCE_ADOPTION = None
_LOADED = _EVIDENCE = None
require, canonical, digest = authority.require, authority.canonical, authority.digest


def validate_declaration(value):
    require(type(value) is dict and set(value) == {'schema', 'configuration_path', 'owned_directory',
            'source_checkout', 'output_root', 'choice_sha256', 'static_recipe_sha256', 'bounds'}
            and value['schema'] == DECLARATION, 'exact fixed metadata adoption declaration')
    for key in ('configuration_path', 'owned_directory', 'source_checkout', 'output_root'):
        require(type(value[key]) is str and Path(value[key]).is_absolute(), 'absolute source adoption path: ' + key)
    require(Path(value['owned_directory']).is_relative_to(Path(value['output_root']))
            and not Path(value['output_root']).is_relative_to(Path(value['source_checkout'])),
            'separate source-adopted output/intake owner')
    require(authority._hex(value['choice_sha256']) and authority._hex(value['static_recipe_sha256']),
            'selected canonical operator/recipe binding')
    bounds = value['bounds']
    require(type(bounds) is dict and set(bounds) == {'read_bytes', 'write_bytes', 'hash_attempts', 'hash_bytes',
            'metadata_bytes', 'operation_attempts', 'wall_seconds'}
            and all(type(bounds[k]) is int and bounds[k] > 0 for k in bounds if k != 'wall_seconds')
            and type(bounds['wall_seconds']) in (int, float) and 0 < bounds['wall_seconds'] <= 120,
            'finite source-owned metadata intake bounds')
    require(bounds['read_bytes'] >= MAX_CONFIGURATION + 1 and bounds['write_bytes'] >= 65536
            and bounds['hash_attempts'] >= 16 and bounds['hash_bytes'] >= 4 * MAX_CONFIGURATION
            and bounds['metadata_bytes'] >= 65536 and bounds['operation_attempts'] >= 8,
            'whole read/parse/digest/final metadata envelope before work')
    return copy.deepcopy(value)


def _append(path, declaration_sha256, previous, kind, payload, bounds, counts, origin):
    body = {'schema': RECORD, 'declaration_sha256': declaration_sha256, 'previous': previous,
            'kind': kind, 'payload': payload, 'monotonic': time.monotonic(), 'original_origin': origin}
    preimage = canonical(body); hashed = hashlib.sha256(preimage).hexdigest()
    blob = canonical(dict(body, sha256=hashed)) + b'\n'
    charges = {'write_bytes': len(blob), 'metadata_bytes': len(blob), 'hash_attempts': 1,
               'hash_bytes': len(preimage), 'operation_attempts': 1}
    require(all(counts[k] + v <= bounds[k] for k, v in charges.items()), 'metadata intake self-publication cap')
    require(time.monotonic() - origin <= bounds['wall_seconds'], 'metadata intake deadline')
    for key, value in charges.items(): counts[key] += value
    with path.open('ab') as stream:
        require(stream.write(blob) == len(blob), 'short intake metadata append')
        stream.flush(); os.fsync(stream.fileno())
    return hashed


def reserve_metadata_intake(declaration, slot_name):
    """Accounting primitive only; source callers first admit their fixed policy.

    A fixed slot is claimed with exclusive creation. Existing slots are never
    reused or removed, including failed intakes. The returned dict is evidence
    and accounting state; it is not an execution capability.
    """
    require(type(declaration) is dict and type(declaration.get('bounds')) is dict
            and type(declaration.get('owned_directory')) is str
            and type(slot_name) is str and slot_name in ('parent', 'child-0', 'child-1', 'child-2'),
            'fixed concrete metadata intake slot')
    bounds = copy.deepcopy(declaration['bounds'])
    require(set(bounds) == {'read_bytes', 'write_bytes', 'hash_attempts', 'hash_bytes', 'metadata_bytes',
            'operation_attempts', 'wall_seconds'} and bounds['write_bytes'] >= 65536
            and bounds['metadata_bytes'] >= 65536 and bounds['hash_attempts'] >= 16
            and bounds['hash_bytes'] >= 4 * MAX_CONFIGURATION and bounds['operation_attempts'] >= 8,
            'complete metadata slot self-reservation envelope')
    origin = time.monotonic(); counts = dict.fromkeys(bounds, 0)
    declaration_sha = digest(declaration); counts['hash_attempts'] += 1; counts['hash_bytes'] += len(canonical(declaration))
    directory = Path(declaration['owned_directory'])
    require(directory.is_absolute() and not directory.is_symlink(), 'regular absolute source intake directory')
    directory.mkdir(parents=True, exist_ok=True)
    journal = directory / (slot_name + '.jsonl')
    with journal.open('xb') as stream: stream.flush(); os.fsync(stream.fileno())
    head = _append(journal, declaration_sha, None, 'PERMANENT_RESERVATION', {
        'bounds': bounds, 'slot': slot_name, 'metadata_only': True, 'model_authorized': False,
        'production_training_authorized': False, 'evaluation_authorized': False}, bounds, counts, origin)
    return {'journal': journal, 'origin': origin, 'counts': counts, 'head': head,
            'declaration_sha256': declaration_sha, 'bounds': bounds}


def _read_configuration(declaration, journal, origin):
    bounds = declaration['bounds']; counts = dict.fromkeys(bounds, 0)
    declaration_sha = digest(declaration); counts['hash_attempts'] += 1
    counts['hash_bytes'] += len(canonical(declaration))
    # The permanent full upper reservation precedes input path inspection/open.
    head = _append(journal, declaration_sha, None, 'PERMANENT_RESERVATION', {
        'bounds': bounds, 'metadata_only': True, 'model_authorized': False,
        'production_training_authorized': False, 'evaluation_authorized': False}, bounds, counts, origin)
    try:
        path = Path(declaration['configuration_path'])
        require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= MAX_CONFIGURATION,
                'regular bounded adopted configuration')
        with path.open('rb') as stream: blob = stream.read(MAX_CONFIGURATION + 1)
        counts['read_bytes'] += len(blob)
        require(0 < len(blob) <= MAX_CONFIGURATION and counts['read_bytes'] <= bounds['read_bytes'], 'intake read ceiling')
        file_sha = hashlib.sha256(blob).hexdigest(); counts['hash_attempts'] += 1; counts['hash_bytes'] += len(blob)
        value = raw._json(blob)
        require(type(value) is dict and value.get('schema') == 'CE8-QUALIFICATION-SOURCE-CONFIGURATION-1',
                'adopted fixed qualification configuration schema')
        preimage = {key: item for key, item in value.items() if key != 'configuration_sha256'}
        hashed_bytes = canonical(preimage)
        counts['hash_attempts'] += 1; counts['hash_bytes'] += len(hashed_bytes)
        require(hashlib.sha256(hashed_bytes).hexdigest() == value.get('configuration_sha256'), 'adopted configuration digest drift')
        request, spec = value.get('request'), value.get('service_spec')
        require(type(request) is dict and type(spec) is dict and request.get('phase') == 'qualification'
                and request.get('arm') == 'canonical' and request.get('training_authorized') is False,
                'intake cannot adopt a training or noncanonical request')
        require(request['selectedoperatorchoice_ref']['sha256'] == declaration['choice_sha256']
                and spec['scope_adapters']['static_recipe_sha256'] == declaration['static_recipe_sha256']
                and spec['scope_adapters']['worker']['source_checkout'] == declaration['source_checkout']
                and spec['resource']['resource_probe']['output_root'] == declaration['output_root'],
                'source-adopted exact choice/recipe/checkout/output roots')
        for owned in (spec['bootstrap']['owned_directory'], spec['constructor_prepayment']['owned_directory']):
            require(Path(owned).is_relative_to(Path(declaration['output_root'])), 'adopted service metadata owner escaped output root')
        require(counts['hash_attempts'] <= bounds['hash_attempts'] and counts['hash_bytes'] <= bounds['hash_bytes'],
                'intake digest envelope exhausted')
        reference = {'path': str(path), 'sha256': file_sha, 'bytes': len(blob)}
        head = _append(journal, declaration_sha, head, 'RETURNED', {'configuration_ref': reference,
            'configuration_sha256': value['configuration_sha256'], 'actual_tracked_counts': counts,
            'metadata_only': True, 'native_qualified': False}, bounds, counts, origin)
        return value, {'schema': RECORD, 'journal_path': str(journal), 'ledger_head': head,
            'declaration_sha256': declaration_sha, 'configuration_ref': reference,
            'permanent_bounds': copy.deepcopy(bounds), 'actual_tracked_counts': counts,
            'metadata_only': True, 'native_qualified': False, 'execution_verified': False}
    except BaseException as primary:
        try: _append(journal, declaration_sha, head, 'FAILED', {'type': type(primary).__name__,
            'message': str(primary)[:1000], 'charges_refunded': False}, bounds, counts, origin)
        except BaseException as failure: raw._accounting(primary, failure); raise primary from failure
        raise


def load_source_configuration():
    """No caller parameters, env or argv selects this source-adopted intake."""
    global _LOADED, _EVIDENCE
    require(type(_SOURCE_ADOPTION) is MappingProxyType, 'fixed qualification metadata adoption remains closed')
    declaration = validate_declaration(copy.deepcopy(dict(_SOURCE_ADOPTION)))
    if _LOADED is not None:
        require(_EVIDENCE['declaration_sha256'] == digest(declaration), 'source adoption changed after intake')
        return copy.deepcopy(_LOADED), copy.deepcopy(_EVIDENCE)
    origin = time.monotonic()
    directory = Path(declaration['owned_directory'])
    # mkdir and this reservation's own append are admitted from the fixed
    # self-write envelope above, before any configuration input is inspected.
    require(not directory.is_symlink(), 'regular source intake owner')
    directory.mkdir(parents=True, exist_ok=True)
    journal = directory / ('CONFIGURATION_INTAKE.' + uuid.uuid4().hex + '.jsonl')
    require(not os.path.lexists(journal), 'create-only source intake history')
    with journal.open('xb') as stream: stream.flush(); os.fsync(stream.fileno())
    value, evidence = _read_configuration(declaration, journal, origin)
    _LOADED, _EVIDENCE = copy.deepcopy(value), copy.deepcopy(evidence)
    return value, evidence
