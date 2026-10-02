"""Create-only DEV R/S/P and guard metadata scopes, before model output.

Only the existing proposal's DEV/guard_dev files are read. Scoring declarations
and receipt bindings use the unchanged scoped scorer. No evaluation grant,
training coupling, model output, model/runtime choice or activation is produced.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import os
from pathlib import Path

from experiments.maxrl_comp002_pretraining_v1 import scoped_scorer as scorer
from . import ce_registration as registration, ce_qualification_artifacts as raw

SCHEMA = 'CE8-DEV-EVALUATION-SCOPE-1'
RUN_PLAN = 'CE8-DEV-FIXED-READ-PLAN-1'
BANKS = ('dev_core_R', 'dev_core_S', 'dev_core_P', 'guard_dev_legacy')
MANIFEST_SHA = '86ac8f0d851f2178752357f62249eab5a23fd3de8fa032e7373614891df4cd44'
DEV_SHA = '147ff62bf516913f9e207e7659c0ad1090c9fa6ff8a7dd67cabb3c975e68ee45'
GUARD_SHA = 'b66a22cc420d65c3619af1d72967310aaadfc5906a8a049c13be7b71f227e7d6'
CORE_MANIFEST_SHA = '15368ddca39cdc1449d7ef67a28f2df3b25d2936f6c0d726a523b0276b705a53'
GUARD_MANIFEST_SHA = 'e7c5c62c467535bb7a9b1e848d9af9fb1e4c12649c0a205062b11cd2d92fa596'
MAX_BANK, MAX_METADATA = 32 * raw.CHUNK, 4 * raw.CHUNK
require, canonical, digest = raw.require, raw.canonical, raw.digest


def _read(reference, expected_sha, maximum):
    require(type(reference) is dict and type(reference.get('path')) is str and Path(reference['path']).is_absolute()
            and reference.get('sha256') == expected_sha and type(reference.get('bytes')) is int
            and 0 < reference['bytes'] <= maximum, 'literal bounded DEV metadata reference')
    path = Path(reference['path'])
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == reference['bytes'], 'regular pinned DEV metadata')
    with path.open('rb') as stream: blob = stream.read(reference['bytes'] + 1)
    require(len(blob) == reference['bytes'] and hashlib.sha256(blob).hexdigest() == expected_sha, 'DEV metadata bytes/hash changed')
    return blob


def _rows(blob, role):
    """Discard prompt bodies and solutions; retain only scope/scorer identities."""
    projected = []
    for line in blob.splitlines():
        require(0 < len(line) <= scorer.MAX_LINE, 'bounded DEV identity row')
        row = raw._json(line)
        require(row.get('role') == role and all(key in row for key in
                ('id', 'parent_id', 'tag', 'skills', 'kind', 'prompt_hash')), 'DEV-only identity fields')
        projected.append({key: row[key] for key in ('id', 'parent_id', 'tag', 'skills', 'role', 'kind', 'prompt_hash')} |
                         {'family': scorer.row_family(row)})
    require(len({row['id'] for row in projected}) == len(projected), 'unique DEV bank IDs')
    return projected


def load_metadata(proposal_ref):
    """Read only literal proposal-selected DEV metadata and manifests."""
    proposal = raw._json(_read(proposal_ref, registration.PROPOSAL_SHA, MAX_METADATA))
    require(set(proposal['banks']) == set(BANKS), 'proposal contains only fixed DEV R/S/P and guard banks')
    refs = proposal['banks']; manifest_ref = proposal['input_refs']['dataset_manifest']
    manifest = raw._json(_read(manifest_ref, MANIFEST_SHA, MAX_METADATA))
    canonical = refs['dev_core_R']['canonical_file']; guard = refs['guard_dev_legacy']['canonical_file']
    require(all(refs[key]['canonical_file'] == canonical for key in BANKS[:3]), 'one shared DEV canonical bank')
    _read(refs['dev_core_R']['core_manifest_ref'], CORE_MANIFEST_SHA, MAX_METADATA)
    _read(refs['guard_dev_legacy']['bank_manifest_ref'], GUARD_MANIFEST_SHA, MAX_METADATA)
    dev_rows = _rows(_read(canonical, DEV_SHA, MAX_BANK), 'dev')
    guard_rows = _rows(_read(guard, GUARD_SHA, MAX_BANK), 'guard_dev')
    require(len(dev_rows) == manifest['counts']['dev'] == 4150 and len(guard_rows) == manifest['counts']['guard_dev'] == 1200
            and manifest['files']['canonical/dev.jsonl']['sha256'] == DEV_SHA
            and manifest['files']['canonical/guard_dev.jsonl']['sha256'] == GUARD_SHA, 'literal complete DEV bank manifests/counts')
    data_root = Path(manifest_ref['path']).parent
    policy_ref = {'path': str(data_root / 'policy' / 'dev.jsonl'), 'sha256': manifest['files']['policy/dev.jsonl']['sha256'],
                  'bytes': (data_root / 'policy' / 'dev.jsonl').stat().st_size}
    policy_ids = [raw._json(line)['id'] for line in _read(policy_ref, policy_ref['sha256'], MAX_BANK).splitlines()]
    require(len(policy_ids) == len(set(policy_ids)) == len(dev_rows)
            and set(policy_ids) == {row['id'] for row in dev_rows}, 'policy/canonical DEV identity coverage')
    return proposal, {'dev': scorer.Bank(dev_rows, role='dev', bank_sha256=DEV_SHA, manifest_sha256=MANIFEST_SHA),
        'guard_dev': scorer.Bank(guard_rows, role='guard_dev', bank_sha256=GUARD_SHA, manifest_sha256=MANIFEST_SHA)}, policy_ref


def build_documents(proposal, banks, *, policy_ref):
    """Pure declarations; model/runtime/donor dependencies remain unresolved."""
    require(type(banks) is dict and set(banks) == {'dev', 'guard_dev'} and all(type(bank) is scorer.Bank for bank in banks.values()),
            'only existing DEV and guard_dev metadata banks')
    require(banks['dev'].role == 'dev' and banks['guard_dev'].role == 'guard_dev'
            and banks['dev'].bank_sha256 == DEV_SHA and banks['guard_dev'].bank_sha256 == GUARD_SHA
            and all(bank.manifest_sha256 == MANIFEST_SHA for bank in banks.values()), 'literal canonical DEV bank identities')
    declarations, entries, identities = {}, {}, {}
    for name in BANKS:
        expected = proposal['banks'][name]; bank = banks[expected['role']]
        guard = name == 'guard_dev_legacy'
        scope = scorer.declare_scope(bank, ids=expected['ids'], tag='preservation' if guard else 'core',
                                     signatures=None if guard else [expected['signature']])
        require(scope['n'] == expected['rows'] == (1200 if guard else 250)
                and scope['parents'] == expected['parents'] == (1200 if guard else 250)
                and scope['ids_sha256'] == expected['ids_sha256'] and scope['parents_sha256'] == expected['parents_sha256'],
                'exact proposal DEV rows/parent population: ' + name)
        selected = {row['id']: row for row in bank if row['id'] in set(scope['ids'])}
        parent_by_id = {rid: selected[rid]['parent_id'] for rid in scope['ids']}
        require(sorted(set(parent_by_id.values())) == sorted(expected['parent_ids']), 'exact proposal parent IDs: ' + name)
        if guard:
            require(scope['ids'] == sorted(row['id'] for row in bank)
                    and dict(Counter(row['family'] for row in bank)) == expected['families'], 'entire guard bank and family counts')
        else:
            entries[expected['signature']] = {key: scope[value] for key, value in
                (('rows', 'n'), ('parents', 'parents'), ('ids_sha256', 'ids_sha256'), ('parents_sha256', 'parents_sha256'))}
        declarations[name] = scope
        identities[name] = {'role': expected['role'], 'bank_sha256': bank.bank_sha256, 'ids': scope['ids'],
            'parent_by_id': parent_by_id, 'prompt_hash_by_id': {rid: selected[rid]['prompt_hash'] for rid in scope['ids']},
            'ids_sha256': scope['ids_sha256'], 'parents_sha256': scope['parents_sha256'], 'scope_receipt_sha256': None,
            'manifest_sha256': MANIFEST_SHA, 'scoring_scope': copy.deepcopy(scope)}
        if guard: identities[name]['family_by_id'] = {rid: selected[rid]['family'] for rid in scope['ids']}
    require(len({digest(sorted(set(identities[name]['parent_by_id'].values()))) for name in BANKS[:3]}) == 1,
            'aligned250 parent worlds across DEV R/S/P')
    receipt = {'experiment': 'COMP-RS-001', 'stage': 'CE8_EXPOSED_DEV_R_S_P_SCOPE', 'role': 'dev',
        'dataset_manifest_sha256': MANIFEST_SHA, 'canonical_dev_sha256': DEV_SHA,
        'policy_dev_sha256': policy_ref['sha256'], 'signatures': entries, 'shared_parent_worlds_R_S_P': 250,
        'test_challenge_read': False, 'model_outputs_read': False, 'evaluation_authorized': False}
    receipt_sha = digest(receipt); bindings = {}
    for name, declaration in declarations.items():
        guard = name == 'guard_dev_legacy'
        bindings[name] = scorer.bind_scope_receipt(banks['guard_dev' if guard else 'dev'], declaration,
            None if guard else receipt, expected_sha256=None if guard else receipt_sha)
        if not guard: identities[name]['scope_receipt_sha256'] = receipt_sha
    code = scorer.code_sha256(); source = {'scorer_code_sha256': code, 'scorer_bundle_sha256': digest(code),
        'source_lock_sha256': scorer.SOURCE_LOCK_SHA256,
        'scope_builder_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    require(code == scorer._IMPORT_CODE and all(code[name] == locked for name, locked in scorer.SOURCE_LOCKED.items()
            if name in code), 'unchanged imported/source-locked scoring kernels')
    model_fields = {'model_sha256': None, 'tokenizer_sha256': None, 'decoding_sha256': None,
                    'scorer_bundle_sha256': source['scorer_bundle_sha256']}
    greedy = {key: copy.deepcopy(value) for key, value in proposal['evaluation']['decoding'].items() if key != 'greedy'}
    require(proposal['evaluation']['decoding']['greedy'] is True, 'fixed proposal greedy read policy')
    greedy['use_model_defaults'] = False
    scope = {'schema': SCHEMA, 'approved': False, 'evaluation_authorized': False, 'production_training_authorized': False,
        'choice': 'canonical_only', 'manifest_sha256': MANIFEST_SHA, 'greedy': greedy,
        'model_outputs_read': False, 'sealed_banks_read': False, 'banks': identities,
        'scoring_declarations': declarations, 'scope_bindings': bindings, 'dev_scope_receipt_sha256': receipt_sha,
        'model_bindings': {name: copy.deepcopy(model_fields) for name in ('Supa', 'warm_anchor', 'CE375_donor')},
        'decoding': copy.deepcopy(greedy), 'source': source,
        'metadata_refs': {'dataset_manifest': copy.deepcopy(proposal['input_refs']['dataset_manifest']),
            'canonical_dev': copy.deepcopy(proposal['banks']['dev_core_R']['canonical_file']),
            'canonical_guard_dev': copy.deepcopy(proposal['banks']['guard_dev_legacy']['canonical_file']),
            'dev_core_manifest': copy.deepcopy(proposal['banks']['dev_core_R']['core_manifest_ref']),
            'guard_dev_manifest': copy.deepcopy(proposal['banks']['guard_dev_legacy']['bank_manifest_ref']),
            'policy_dev': copy.deepcopy(policy_ref)},
        'interpretation': 'fixed exposed DEV mechanics/acquisition and diagnostic source retention; no composition or natural-language claim'}
    reads = [{'read_id': name + '-read-' + str(index), 'model_name': name, 'read_index': index,
              'fresh_process_required': True, 'bank_names': list(BANKS), 'rows': 1950,
              'runtime_ref': None, 'model_ref': None, 'donor_ref': None, 'process_identity': None}
             for name, count in (('Supa', 2), ('warm_anchor', 2), ('CE375_donor', 1)) for index in range(1, count + 1)]
    plan = {'schema': RUN_PLAN, 'approved': False, 'evaluation_authorized': False, 'production_dispatch_authorized': False,
        'scope_sha256': digest(scope), 'dev_scope_receipt_sha256': receipt_sha, 'reads': reads,
        'rows_per_read': 1950, 'total_greedy_completions': 9750, 'max_new_tokens': 128,
        'generation_token_ceiling': 9750 * 128, 'baseline_enrollment_rows': 7800,
        'baseline_second_reads': 'zero generated-token-ID churn validation only; first read freezes source-correct membership',
        'refill_allowed': False, 'checkpoint_selection_allowed': False, 'allowed_roles': ['dev', 'guard_dev'],
        'unresolved_dependencies': ['actual source/runtime/model/tokenizer bindings', 'source-completed CE375 donor',
                                    'independent finite evaluation accounting and explicit evaluation authorization']}
    return {'DEV_RSP_SCOPE_RECEIPT.json': receipt, 'DEV_EVALUATION_SCOPE.json': scope, 'DEV_RUN_PLAN.json': plan}


def write_draft(documents, directory):
    require(set(documents) == {'DEV_RSP_SCOPE_RECEIPT.json', 'DEV_EVALUATION_SCOPE.json', 'DEV_RUN_PLAN.json'}
            and documents['DEV_EVALUATION_SCOPE.json']['evaluation_authorized'] is False
            and documents['DEV_RUN_PLAN.json']['evaluation_authorized'] is False, 'unapproved DEV metadata drafts only')
    directory = Path(directory)
    require(directory.is_absolute() and not directory.is_symlink() and not any(os.path.lexists(directory / name) for name in documents),
            'create-only regular DEV draft owner')
    directory.mkdir(parents=True, exist_ok=True); references = {}
    for name, body in documents.items():
        blob = canonical(body) + b'\n'; require(len(blob) <= MAX_METADATA, 'bounded DEV draft JSON')
        with (directory / name).open('xb') as stream:
            require(stream.write(blob) == len(blob), 'short DEV draft write'); stream.flush(); os.fsync(stream.fileno())
        references[name] = {'path': str(directory / name), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}
    return references
