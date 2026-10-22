"""Preserve an explicit reviewed metadata/source allowlist; no payload discovery."""
import hashlib
import json
from pathlib import Path

BUSINESS = Path('C:/Users/clclc/BusinessProject')
RDD = BUSINESS / 'ChatBubble-rineya-distillation-dynamics-v1'
SYNTH = BUSINESS / 'ChatBubble-rineya-synth-centralized-reasoning-v1'
PACKAGE = Path('experiments/rineya_distillation_dynamics_v1')
SP = Path('experiments/rineya_synth_centralized_reasoning_v1')
OUT = RDD / PACKAGE / 'evidence/reviewed_preparation_20261009'

GROUPS = {
    'historical_preparation': ('rdd001-records/preparation-20261005', [
        'PREPARATION_RECEIPT.json', 'PROSPECTIVE_NATIVE_PROPOSAL.json', 'PROSPECTIVE_NATIVE_PROPOSAL.md',
        'inventory/asset_inventory.json', 'inventory/host_inventory.json', 'inventory/inventory_metadata.py',
        'inventory/INVENTORY.md', 'verification/finalize_receipt.py', 'verification/run_offline_checks.py',
        'verification/offline-junit-v1.xml', 'verification/offline-junit-v2.xml', 'verification/offline-junit-v3.xml',
        'verification/offline-tests-v1.log', 'verification/offline-tests-v2.log', 'verification/offline-tests-v3.log',
        'verification/offline-verification-v1.json', 'verification/offline-verification-v2.json',
        'verification/offline-verification-v3.json', 'verification/qualification-junit-v1.xml']),
    'historical_readiness': ('synth-quality-readiness-20261005', [
        'READINESS_PACKET_002.json', 'OWNER_HANDOFF_RECEIPT_001.json']),
    'historical_discoveries': ('rineya-quality-pattern-research-20261005', [
        'RESEARCH_PROGRESS_003.json', 'INTERPRETATION_REVISION_002.json', 'MEASUREMENT_SCOPE_CAUTION_001.json',
        'literature/PRIMARY_SOURCE_AUDIT_001.json', 'mq-output-analysis/FINDINGS_001.md',
        'mq-output-analysis/SEMANTIC_FINDINGS_003.md', 'mq-rab-review/REVIEW_NOTES_001.md',
        'resumed-audit-001/EXPOSURE_COMMITMENT_REVIEW_20261009.json',
        'resumed-audit-001/QUALIFICATION_BLOCKERS_20261009.md']),
    'current_verification': ('rdd001-records/review-20261009-001', [
        'run_offline_checks.py', 'preserve_evidence.py',
        'offline-tests-review001.log', 'offline-junit-review001.xml', 'offline-verification-review001.json',
        'offline-tests-review002.log', 'offline-junit-review002.xml', 'offline-verification-review002.json']),
}
REFERENCES = [
    'supplemental_quality_packet_reviewed_20261005/SYNTH_OWNER_ACCEPTANCE_001.json',
    'retention_binding_package_reviewed_cpu_20261009/ROOT_PRESERVATION.json',
    'guard_coverage_binding_reviewed_cpu_20261009/ROOT_PRESERVATION.json',
    'public_train_gold_reviewed_cpu_20261005/ROOT_PRESERVATION.json',
    'INFQ_exposure_projection_cpu_20261009/ROOT_PRESERVATION.json',
]

def info(path):
    content = path.read_bytes()
    return {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}

def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2) + '\n')

def main():
    # No enumeration of datasets, native runs, banks, caches or environments.
    OUT.mkdir(parents=True, exist_ok=False)
    files = {}
    for group, (root, names) in GROUPS.items():
        for name in names:
            source, target = BUSINESS / root / name, OUT / group / name
            before = info(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(source.read_bytes())
            assert info(source) == info(target) == before, str(source)
            files[str(target.relative_to(OUT)).replace('\\', '/')] = {
                'original_path': str(source), **before, 'original_unchanged': True}
    references = {}
    for relative in REFERENCES:
        path = SYNTH / SP / 'evidence' / relative
        references[relative] = {'owner_branch': 'research/rineya-synth-centralized-reasoning-v1',
            'path': str(path), **info(path), 'copied_here': False}
    source_files = {}
    for pattern in ('*.py', 'tests/*.py', 'configs/selected-pilot.json', 'README.md', 'PROTOCOL.md', 'RUNBOOK.md'):
        for path in sorted((RDD / PACKAGE).glob(pattern)):
            source_files[str(path.relative_to(RDD)).replace('\\', '/')] = info(path)
    verification = json.loads((BUSINESS / GROUPS['current_verification'][0] /
                               'offline-verification-review002.json').read_text())
    assert verification['exit_code'] == 0 and not verification['cuda_initialized']
    assert not any(verification['tripwire_attempts'].values())
    payload = {
        'schema': 'rdd-reviewed-public-preservation-v1', 'recorded_local_date': '2026-10-09',
        'source_start_commit': '2d7be4b5c2ba8a610b09d17455d519eecaa421e6',
        'owner_branch': 'research/rineya-distillation-dynamics-v1',
        'files': files, 'preserved_file_count': len(files),
        'preserved_bytes': sum(row['bytes'] for row in files.values()),
        'referenced_owner_evidence': references, 'verified_source_files': source_files,
        'review_findings_resolved': ['Complete forwards require actual causal input IDs and hashes',
                                    'Parity acceptance checks the full current base/tokenizer snapshot'],
        'cpu_tests_passed': 179, 'optimizer_tests_deselected': 11,
        'runtime_torch': verification['torch'], 'tripwire_attempts': verification['tripwire_attempts'],
        'native_model_or_optimizer_action_performed': False, 'native_qualified': False,
        'export_runtime_parity_established': False, 'model_quality_improvement_established': False,
        'historical_bodies_unchanged': True, 'standing_authorization_preserved': True,
        'current_scope': 'Human deferred GPU/reviewer inputs; review, preserve, commit and push only',
        'limitations': ['Historical permission/host/status flags remain dated observations',
                        'CPU fixtures do not establish native or efficacy evidence',
                        'No model weights, native captures, protected banks, environments or caches copied'],
    }
    payload['sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                                ensure_ascii=False).encode()).hexdigest()
    write_new(OUT / 'PRESERVATION_RECEIPT.json', payload)
    print(json.dumps({'files': len(files), 'bytes': payload['preserved_bytes'],
                      'receipt': str(OUT / 'PRESERVATION_RECEIPT.json'), 'body_sha256': payload['sha256']}))

if __name__ == '__main__':
    main()
