"""Create-only metadata for the approved exception; no launch or framework API."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
PACKAGE = ROOT / 'experiments/rineya_centralized_reasoning_system_v1'
OUTPUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


preserved = {
    'native.py': '8453128981280019702dff80ebd5083031f7f2dcd8a7c6c9094587d9a2e377a3',
    'operations.py': 'ba27ec8a31437459b956a5642e6ece4838f0584602db2c93ff7bc61437806e30',
    'contracts/NATIVE_DIAGNOSTIC_V1.json': 'd79f3b6a51e939d78a99c5c287ef6f33ef5979cfa50e76ba3d930ed3724f826a',
    'contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_V1.json': 'a6bcdfdffead4ee6ce08cc2f80628379678c3afcfb99923d702fd6234f8001ed',
    'contracts/READER_QUALIFICATION_V1.json': 'b50d61609f8bb933045abac1b769dc7979c3b8f6f2163eb256c2fe3e77376bb0',
}
for name, pin in preserved.items():
    assert sha(PACKAGE / name) == pin, name
changed = ['reader.py', 'reader_execution.py', 'owned_runtime_identity_probe.py',
           'tests/test_reader.py', 'tests/test_owned_runtime_identity_probe.py']
for name in changed:
    ast.parse((PACKAGE / name).read_bytes(), filename=name)
authority = 'authorization/OWNER_EXCEPTION_001_GRANTED_2026-10-04.json'
policy = {
    'policy_id': 'CRS-OWNER-EXCEPTION-001-REVIEWER-POLICY-V2',
    'authority_receipt': authority,
    'authority_receipt_sha256': sha(PACKAGE / authority),
    'required_independent_probe_roles': {'science': '/root/exception_science', 'operations': '/root/exception_ops'},
    'reader_single_review_interface_preserved': True,
    'reader_eligible_reviewers': ['/root/exception_science', '/root/exception_ops'],
    'fresh_science_and_operations_source_audits_required_by_root': True,
    'identities_are_not_authentication': True,
    'normalized_local_namespaces': ['bare leaf', '/root/leaf'],
    'excluded_builders_and_prior_roles': ['/root', '/root/runtime_audit', '/root/scientific_audit',
                                         '/root/data_audit', '/root/exception_builder'],
    'other_identities_rejected': True,
    'shared_model_workspace_ancestry_disclosed': True,
    'builder_self_approval_allowed': False,
    'broader_permission_granted': False,
}
old_probe = json.loads((PACKAGE / 'contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_V1.json').read_bytes())
old_reader = json.loads((PACKAGE / 'contracts/READER_QUALIFICATION_V1.json').read_bytes())
probe = json.loads(json.dumps(old_probe))
probe.update(contract_id='CRS-OWNED-RUNTIME-IDENTITY-DIAGNOSTIC-002', client_date='2026-10-04',
             status='OWNER_EXCEPTION_001_SOURCE_FROZEN_PENDING_ONE_CPU_TEST_AND_TWO_FRESH_INDEPENDENT_REVIEWS',
             scientific_reviewer='/root/exception_science', builder='/root/exception_builder',
             implementation_sha256=sha(PACKAGE / 'owned_runtime_identity_probe.py'),
             independent_reviewers_required=policy['required_independent_probe_roles'], reviewer_identity_policy=policy)
probe['source_sha256'].update({(PACKAGE / name).relative_to(ROOT).as_posix(): sha(PACKAGE / name)
                              for name in ('owned_runtime_identity_probe.py', 'tests/test_owned_runtime_identity_probe.py')})
probe['owner_exception_001'] = {
    'authorization_id': 'CRS-OWNER-RESUMPTION-2026-10-04-001',
    'authority_receipt_sha256': policy['authority_receipt_sha256'],
    'parent_contract_path': 'contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_V1.json',
    'parent_contract_sha256': preserved['contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_V1.json'],
    'prior_key': 'CRS-OWNED-IDENTITY-001', 'new_key': 'CRS-OWNED-IDENTITY-002',
    'prior_terminal_failure_preserved': True, 'additional_targeted_repair_review_rounds': 1,
    'maximum_new_runs': 1, 'maximum_new_wall_seconds': 60, 'automatic_retry': False,
    'changes': ['exact reviewed Path-to-str compatibility patch',
                'new KEY002 immutable registration and fixed independent reviewer policy',
                'required actual pinned API CPU fixture'],
    'prior_builders': ['/root', '/root/scientific_audit', '/root/runtime_audit'],
    'all_other_gates_unchanged': True, 'native_environment_qualification': False,
    'no_permission_granted_by_contract': True,
}
probe['cpu_validation'] = {
    'status': 'PENDING_SINGLE_CPU_INTEGRATION_RUN', 'tests_in_current_file': 20,
    'scope': 'CPU fixtures only; actual psutil API and six contained stdlib subprocesses; no actual import probe or framework/model API',
    'result_path': 'outputs/owner_exception_integration_001/TEST_RESULT.json',
    'native_qualification': False, 'canonical_host_database_access': False,
}
reader = json.loads(json.dumps(old_reader))
reader.update(contract_id='CRS-READER-QUALIFICATION-OWNER-EXCEPTION-001-V2',
              status='OWNER_EXCEPTION_001_SOURCE_FROZEN_PENDING_ONE_CPU_TEST_AND_TWO_FRESH_INDEPENDENT_REVIEWS_NATIVE_QUARANTINED',
              builder='/root/exception_builder', reader_operational_revision='OWNER_EXCEPTION_001_V2',
              reviewer_identity_policy=policy, tests_sha256=sha(PACKAGE / 'tests/test_reader.py'))
reader['source_sha256'].update({(PACKAGE / name).relative_to(ROOT).as_posix(): sha(PACKAGE / name)
                               for name in ('reader.py', 'reader_execution.py', 'tests/test_reader.py')})
reader['owner_exception_001'] = {
    'authorization_id': 'CRS-OWNER-RESUMPTION-2026-10-04-001',
    'authority_receipt_sha256': policy['authority_receipt_sha256'],
    'parent_contract_path': 'contracts/READER_QUALIFICATION_V1.json',
    'parent_contract_sha256': preserved['contracts/READER_QUALIFICATION_V1.json'],
    'additional_targeted_repair_review_rounds': 1,
    'changes': ['exact reviewed physical supervisor-byte reference patch',
                'versioned contract and fixed independent reviewer policy',
                'actual native.write_new immutable references and real temporary reconciliation/read-only recovery fixtures'],
    'prior_builder': '/root/runtime_audit', 'all_other_gates_unchanged': True,
    'native_reader_launch_authorized': False, 'inherited_supervision_quarantine_preserved': True,
    'no_permission_granted_by_contract': True,
}
reader['cpu_validation'] = {
    'status': 'PENDING_SINGLE_CPU_INTEGRATION_RUN', 'tests_in_current_file': 28,
    'scope': 'CPU interface fixtures only; actual producer bytes and temporary SQLite semantics, no native/model/reader launch',
    'result_path': 'outputs/owner_exception_integration_001/TEST_RESULT.json',
    'native_qualification': False, 'canonical_host_database_access': False,
}
for field in ('preserved_native', 'runtime', 'expected_executable_registry', 'runtime_file_sha256', 'environment',
              'phases', 'boundary', 'observations', 'limits', 'claim_limits', 'registry_sha256'):
    assert probe[field] == old_probe[field], field
for field in ('model_dir', 'asset_sha256', 'gpu_uuid', 'gpu_process_registry', 'runtime', 'runtime_file_sha256', 'limits',
              'inherited_supervision_prerequisite', 'historical_bulk_capture', 'historical_prompt_ids', 'long_prompt_ids',
              'mechanical_continuation_ids', 'schedule', 'shape_envelope', 'stock_generation_config'):
    assert reader[field] == old_reader[field], field
for cfg in (reader, probe):
    for name, pin in cfg['source_sha256'].items():
        assert (ROOT / name).is_file() and sha(ROOT / name) == pin, name
probe_name = 'contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_002_V2.json'
reader_name = 'contracts/READER_QUALIFICATION_OWNER_EXCEPTION_001_V2.json'
write_new(PACKAGE / probe_name, probe)
write_new(PACKAGE / reader_name, reader)
all_names = changed + [probe_name, reader_name]
write_new(OUTPUT / 'SOURCE_FREEZE.json', {
    'status': 'FROZEN_BEFORE_ONE_BOUNDED_CPU_RUN',
    'preserved_base_commit': 'a6d6673c94141486ce4a8124b595b847731f32ed', 'builder': '/root/exception_builder',
    'preserved_pins': preserved,
    'original_source_pins': {'probe': old_probe['source_sha256'], 'reader': old_reader['source_sha256'],
                             'test_reader.py': old_reader['tests_sha256']},
    'source_sha256': {name: sha(PACKAGE / name) for name in all_names},
    'reviewed_intermediate_patch_result_pins': {
        'probe': '0e85d5cfe818bf706410b78e12b0129ceeb4a95b4eb720d7d733bcef0832157e',
        'reader': '9a6d2806c218f597a500c7a5dd17acee37bd8edd24fbddd3aa357d5704d85ed3',
        'execution': 'c66a1b6453f1bad19e0f51eb7d2cf2b763eb0e6e7a09e124eebcb613ef53f9d3'},
    'prior_registration_metadata_attempt': {'tests_run': 0, 'candidate_edits': 0,
        'result': 'STOPPED_BEFORE_CONTRACT_CREATION_ON_MISTYPED_LOCAL_OLD_CONTRACT_HASH_ASSERTION', 'tool_seconds': 0.3088343},
    'source_and_contract_gates_preserved': True, 'canonical_host_database_access': False,
    'framework_imports': 0, 'models_loaded': 0, 'native_probe_runs': 0, 'independent_approval': False,
})
print(json.dumps({'contracts': all_names[-2:], 'freeze_sha256': sha(OUTPUT / 'SOURCE_FREEZE.json')}))
