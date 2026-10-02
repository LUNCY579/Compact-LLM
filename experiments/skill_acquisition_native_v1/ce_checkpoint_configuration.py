"""Late child configuration transport with source-paid intake; closed by default.

The immutable branch request contains no configuration hash. The parent writes
configuration after that request, then freezes its ref in NativeLaunchPermit
environment. Configuration contains the base launch spec without its own marker;
the receiver reconstructs the final spec after paid intake. No hash cycle or
JSON-selected callback/authority is needed.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import time
from types import MappingProxyType

from . import ce_native_authority as authority, ce_qualification_artifacts as raw
from . import ce_checkpoint_entry_provider as provider, ce_qualification_configuration as intake
from . import ce_native_services as services, ce_process_launch as launch

SCHEMA = 'CE8-CHECKPOINT-CONFIGURATION-TRANSPORT-1'
DECLARATION = 'CE8-CHILD-CONFIGURATION-ADOPTION-1'
MARKER = 'CE8_CHILD_CONFIGURATION_REF'
_SOURCE_ADOPTION = None
_LOADED = _EVIDENCE = None
require, canonical, digest = authority.require, authority.canonical, authority.digest


def validate_declaration(value):
    require(type(value) is dict and set(value) == {'schema', 'configuration_root', 'owned_directory', 'source_checkout',
            'output_root', 'choice_sha256', 'static_recipe_sha256', 'bounds'} and value['schema'] == DECLARATION,
            'fixed child source metadata adoption declaration')
    parent = dict(value, schema=intake.DECLARATION, configuration_path=str(Path(value['configuration_root']) / 'unused-fixed.json'))
    parent.pop('configuration_root')
    intake.validate_declaration(parent)
    require(Path(value['configuration_root']).is_absolute() and Path(value['configuration_root']).is_relative_to(Path(value['output_root'])),
            'child configuration root inside source-adopted output')
    return copy.deepcopy(value)


def prepare_child_configuration(processes, request, request_ref, base_launch_spec, parent):
    """Actual parent paid create-only publication; called before launch permit."""
    from . import ce_checkpoint_processes as process, ce_entry_factories as rows, ce_prefix_handle_transfer as transfer
    require(type(processes) is process.CheckpointProcesses, 'exact original checkpoint process publication owner')
    processes._admit(processes.context, processes.permit, native=True)
    service = processes.backend.operation.service_owner
    require(type(service) is services.NativeControllerServices and service.context is processes.context,
            'actual parent source service owns child configuration')
    require(MARKER not in base_launch_spec['environment'] and request_ref == base_launch_spec['request_ref'],
            'base launch spec excludes its own configuration reference')
    branch = request['branch']; directory = Path(processes.context['bindings']['scope']['checkpoint_qualification']['owned_directory']) / 'children' / branch
    spec = copy.deepcopy(service.spec)
    spec['bootstrap']['owned_directory'] = spec['constructor_prepayment']['owned_directory'] = str(directory / 'service-bootstrap')
    spec['scope_adapters']['owned_directory'] = str(directory / 'service-bootstrap' / 'capsules')
    module = lambda fn: {'module': fn.__module__, 'symbol': fn.__name__,
        'source_sha256': base_launch_spec['approved_code_sha256'][fn.__module__.replace('.', '/') + '.py']}
    configuration = {'schema': provider.SPEC, 'launch_spec': copy.deepcopy(base_launch_spec),
        'controller_request': copy.deepcopy(service.request), 'service_spec': spec,
        'common_context_ref': copy.deepcopy(parent['common_context_ref']),
        'observation_bounds': copy.deepcopy(base_launch_spec['observation_bounds']),
        'entry_factories': {'prepared_rows': module(rows.prepare_checkpoint_entry_rows),
                            'prefix_terminal_handle': module(transfer.prefix_terminal_handle)}}
    body = {'schema': SCHEMA, 'branch': branch, 'request_ref': copy.deepcopy(request_ref),
        'choice_sha256': service.request['selectedoperatorchoice_ref']['sha256'],
        'static_recipe_sha256': processes.context['static_recipe_sha256'], 'configuration': configuration}
    body['configuration_sha256'] = digest(body)
    reference = processes.write_json(directory / 'CHILD_CONFIGURATION.json', body)
    final_spec = reconstruct_launch_spec(body, reference)
    return final_spec, reference


def reconstruct_launch_spec(body, reference):
    require(type(body) is dict and body.get('schema') == SCHEMA and body.get('branch') in ('uninterrupted', 'prefix', 'resumed')
            and body.get('configuration_sha256') == digest({key: value for key, value in body.items() if key != 'configuration_sha256'}),
            'exact child configuration self-digest')
    configuration = copy.deepcopy(body['configuration']); base = configuration['launch_spec']
    require(body['request_ref'] == base['request_ref'] and MARKER not in base['environment'], 'no child request/configuration hash cycle')
    require(type(reference) is dict and set(reference) == {'path', 'sha256', 'bytes'} and raw._hex(reference['sha256'])
            and type(reference['bytes']) is int and 0 < reference['bytes'] <= intake.MAX_CONFIGURATION
            and Path(reference['path']).is_absolute(), 'exact independent configuration transport reference')
    base['environment'][MARKER] = canonical(reference).decode('ascii')
    launch.validate_spec(base)
    return base


def _reserve(declaration):
    """Finite one-shot slots; reservation precedes environment/argv/config/OS."""
    origin = time.monotonic(); bounds = declaration['bounds']; directory = Path(declaration['owned_directory'])
    require(not directory.is_symlink(), 'regular child intake owner')
    directory.mkdir(parents=True, exist_ok=True)
    counts = dict.fromkeys(bounds, 0); declaration_sha = digest(declaration)
    counts['hash_attempts'] = 1; counts['hash_bytes'] = len(canonical(declaration))
    journal = None
    for index in range(3):
        counts['operation_attempts'] += 1
        require(counts['operation_attempts'] <= bounds['operation_attempts'], 'child intake slot attempts exhausted')
        candidate = directory / ('child-' + str(index) + '.jsonl')
        try:
            with candidate.open('xb') as stream: stream.flush(); os.fsync(stream.fileno())
            journal = candidate; break
        except FileExistsError: continue
    require(journal is not None, 'three immutable child metadata intake slots already retained')
    head = intake._append(journal, declaration_sha, None, 'PERMANENT_RESERVATION',
        {'bounds': bounds, 'metadata_only': True, 'model_authorized': False, 'charges_refunded': False}, bounds, counts, origin)
    return {'journal': journal, 'origin': origin, 'counts': counts, 'head': head,
            'declaration_sha256': declaration_sha, 'bounds': bounds}


def _read_after_reservation(declaration, payment, environment):
    head = payment['head']; counts = payment['counts']; bounds = payment['bounds']
    try:
        encoded = environment.get(MARKER)
        require(type(encoded) is str and 0 < len(encoded) <= 4096, 'bounded source launch configuration marker')
        reference = raw._json(encoded)
        require(type(reference) is dict and set(reference) == {'path', 'sha256', 'bytes'} and raw._hex(reference['sha256'])
                and type(reference['bytes']) is int and 0 < reference['bytes'] <= intake.MAX_CONFIGURATION,
                'exact bounded launched child configuration ref')
        path = Path(reference['path']); root = Path(declaration['configuration_root'])
        require(path.is_absolute() and path.parent.name in ('uninterrupted', 'prefix', 'resumed')
                and path.parent.parent.parent == root and path.name == 'CHILD_CONFIGURATION.json'
                and not root.is_symlink() and not path.is_symlink() and path.is_file()
                and path.stat().st_size == reference['bytes'], 'exact source-adopted child configuration path/size')
        with path.open('rb') as stream: blob = stream.read(reference['bytes'] + 1)
        counts['read_bytes'] += len(blob)
        require(len(blob) == reference['bytes'] and counts['read_bytes'] <= bounds['read_bytes']
                and hashlib.sha256(blob).hexdigest() == reference['sha256'], 'late child configuration raw hash/bytes drift')
        counts['hash_attempts'] += 1; counts['hash_bytes'] += len(blob)
        body = raw._json(blob); encoded_body = canonical({key: value for key, value in body.items() if key != 'configuration_sha256'})
        counts['hash_attempts'] += 1; counts['hash_bytes'] += len(encoded_body)
        require(body.get('configuration_sha256') == hashlib.sha256(encoded_body).hexdigest()
                and body['choice_sha256'] == declaration['choice_sha256']
                and body['static_recipe_sha256'] == declaration['static_recipe_sha256'], 'selected child choice/recipe/selfdigest')
        configuration = copy.deepcopy(body['configuration'])
        configuration['launch_spec'] = reconstruct_launch_spec(body, reference)
        provider.validate_configuration(configuration)
        spec = configuration['launch_spec']
        require(spec['source_checkout'] == declaration['source_checkout']
                and configuration['service_spec']['resource']['resource_probe']['output_root'] == declaration['output_root']
                and configuration['service_spec']['bootstrap']['owned_directory'] == str(path.parent / 'service-bootstrap')
                and Path(spec['request_ref']['path']).parent == path.parent,
                'actual launched child checkout/output/request/metadata owner')
        source_root = Path(declaration['source_checkout'])
        references = spec['source_files']
        needed = sum(ref['bytes'] + 1 for ref in references.values())
        require(counts['read_bytes'] + needed <= bounds['read_bytes']
                and counts['hash_attempts'] + len(references) + 2 <= bounds['hash_attempts']
                and counts['hash_bytes'] + needed <= bounds['hash_bytes'], 'complete current source verification prepaid before file IO')
        for name, ref in references.items():
            source = Path(ref['path'])
            require(source.resolve() == (source_root / name).resolve() and source.resolve().is_relative_to(source_root.resolve())
                    and not source.is_symlink() and source.is_file() and source.stat().st_size == ref['bytes'],
                    'current exact child source origin/size')
            with source.open('rb') as stream: code = stream.read(ref['bytes'] + 1)
            counts['read_bytes'] += len(code); counts['hash_attempts'] += 1; counts['hash_bytes'] += len(code)
            require(len(code) == ref['bytes'] and hashlib.sha256(code).hexdigest() == ref['sha256']
                    == spec['approved_code_sha256'][name], 'current frozen source file raw hash drift')
        require(counts['hash_attempts'] + 1 <= bounds['hash_attempts'] and counts['hash_bytes'] <= bounds['hash_bytes'], 'child intake digest cap')
        head = intake._append(payment['journal'], payment['declaration_sha256'], head, 'RETURNED',
            {'configuration_ref': reference, 'configuration_sha256': body['configuration_sha256'],
             'native_qualified': False, 'actual_tracked_counts': counts}, bounds, counts, payment['origin'])
        return configuration, {'journal_path': str(payment['journal']), 'ledger_head': head,
            'configuration_ref': reference, 'permanent_bounds': copy.deepcopy(bounds),
            'actual_tracked_counts': counts, 'configuration_sha256': body['configuration_sha256'],
            'source_files_verified': True, 'native_qualified': False, 'execution_verified': False}
    except BaseException as primary:
        try: intake._append(payment['journal'], payment['declaration_sha256'], head, 'FAILED',
            {'type': type(primary).__name__, 'message': str(primary)[:1000], 'charges_refunded': False}, bounds, counts, payment['origin'])
        except BaseException as failure: raw._accounting(primary, failure); raise primary from failure
        raise


def install_child_configuration():
    global _LOADED, _EVIDENCE
    require(type(_SOURCE_ADOPTION) is MappingProxyType, 'fixed child configuration source adoption remains closed')
    declaration = validate_declaration(copy.deepcopy(dict(_SOURCE_ADOPTION)))
    if _LOADED is not None: return copy.deepcopy(_LOADED), copy.deepcopy(_EVIDENCE)
    payment = _reserve(declaration)
    configuration, evidence = _read_after_reservation(declaration, payment, os.environ)
    from . import ce_qualification_entry as entry
    _LOADED, _EVIDENCE = copy.deepcopy(configuration), copy.deepcopy(evidence)
    try:
        entry.install_child_source_registries(_LOADED)
        provider._SOURCE_CONFIGURATION = copy.deepcopy(configuration)
    except BaseException as primary:
        _LOADED = _EVIDENCE = None
        try: intake._append(payment['journal'], payment['declaration_sha256'], evidence['ledger_head'], 'FAILED_INSTALL',
            {'type': type(primary).__name__, 'charges_refunded': False}, payment['bounds'], payment['counts'], payment['origin'])
        except BaseException as failure: raw._accounting(primary, failure); raise primary from failure
        raise
    return configuration, evidence


def fixture_read_configuration(declaration, environment):
    """Real small-file accounting fixture, never installs native source roles."""
    declaration = validate_declaration(declaration)
    return _read_after_reservation(declaration, _reserve(declaration), environment)
