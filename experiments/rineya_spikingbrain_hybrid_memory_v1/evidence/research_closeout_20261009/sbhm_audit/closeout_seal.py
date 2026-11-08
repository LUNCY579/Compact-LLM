"""Validate the offline audit package and write a new closeout/manifest, exclusively."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
REPO = Path('C:/Users/clclc/BusinessProject/ChatBubble-rineya-spikingbrain-hybrid-memory-v1')
PACKAGE = REPO / 'experiments/rineya_spikingbrain_hybrid_memory_v1'
RUN = PACKAGE / 'runs/titan_transition_v2_20261004_memory_efficient'
EXPECTED_REPORT = '3bf1ea27337dafb47f044e4f53a3b555d4270da2a67aca66e0c765104da7726e'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))

def write(name, data):
    with (ROOT / name).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')

def main():
    report = ROOT / 'AUDITED_CLOSEOUT.txt'
    assert sha(report) == EXPECTED_REPORT
    audit_scripts = {
        'provenance_receipt_v2.json': 'provenance_audit_v2.py',
        'provenance_log_receipt.json': 'provenance_log_audit.py',
        'pairing_audit_final.json': 'pairing_audit.py',
        'tokenizer_replay.json': 'tokenizer_replay.py',
        'resources_audit_v2.json': 'resources_audit.py',
    }
    audits = {}
    for receipt_name, script_name in audit_scripts.items():
        receipt = read(receipt_name)
        assert receipt['status'].startswith('PASS'), receipt_name
        assert not receipt.get('failures'), receipt_name
        script_hash = receipt.get('script_sha256', receipt.get('audit_script_sha256'))
        assert script_hash == sha(ROOT / script_name), script_name
        audits[receipt_name] = receipt
    review_names = ['provenance_report_review_final.json', 'pairing_report_review_final.json', 'resources_report_review_final.json']
    for name in review_names:
        review = read(name)
        assert review['status'].startswith('PASS') and not review.get('issues') and not review.get('material_findings'), name
        actual = review.get('report_sha256', review.get('reviewed_report_sha256'))
        if actual is None:
            actual = review['reviewed_report']['sha256']
        assert actual == EXPECTED_REPORT, name
    final = read('preservation-final.json')
    before = read('preservation-before.json')
    assert final['files'] == before['files'] and len(final['files']) == 838
    assert not final['changed_or_missing_original_files'] and not final['added_repository_files']
    assert final['branch_and_head_unchanged']
    assert final['branch'] == 'research/rineya-spikingbrain-hybrid-memory-v1'
    assert final['head'] == 'fe9183f5c8e884316f4a6446d3b5472e9d6ad18f'
    assert sha(ROOT / 'git-status-before.txt') == sha(ROOT / 'git-status-after.txt')
    assert sha(ROOT / 'local-changes-before.patch') == sha(ROOT / 'local-changes-after.patch')
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO).decode().strip() == final['branch']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO).decode().strip() == final['head']
    assert subprocess.run(['git', 'diff', '--check'], cwd=REPO, capture_output=True).returncode == 0
    suites = ET.parse(ROOT / 'fixtures-junit.xml').getroot().findall('testsuite')
    counts = {k: sum(int(s.attrib[k]) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    assert counts == {'tests': 136, 'failures': 0, 'errors': 0, 'skipped': 0}, counts
    assert read('fixture-verification.json')['returncode'] == 0
    replay = audits['tokenizer_replay.json']
    assert not replay['torch_imported'] and not replay['weights_loaded'] and not replay['native_model_executed']
    pairing = audits['pairing_audit_final.json']
    for group, expected in [('distant_heldout', {'A': 51, 'B': 172, 'C': 172, 'D': 171}), ('near_control', {'A': 104, 'B': 179, 'C': 172, 'D': 171}), ('retention', {'A': 22, 'B': 29, 'C': 29, 'D': 29})]:
        assert pairing['scores'][group]['counts'] == expected
    assert pairing['scores']['distant_heldout']['D_only'] == []
    for arm in ('C', 'D'):
        assert pairing['scores']['retention']['retention'][arm] == {'damaged': 4, 'repaired': 11, 'retained': 18}
    evidence_names = list(audit_scripts) + list(audit_scripts.values()) + review_names + [
        'AUDITED_CLOSEOUT.txt', 'fixture-verification.json', 'fixtures-output.txt', 'fixtures-junit.xml',
        'preservation-before.json', 'preservation-final.json', 'ownership-decision.json',
        'git-status-before.txt', 'git-status-after.txt', 'local-changes-before.patch', 'local-changes-after.patch',
    ]
    evidence = {name: {'path': str(ROOT / name), 'sha256': sha(ROOT / name), 'bytes': (ROOT / name).stat().st_size} for name in evidence_names}
    native = {f'{arm}-evaluation/receipt.json': {'path': str(RUN / f'{arm}-evaluation/receipt.json'), 'sha256': sha(RUN / f'{arm}-evaluation/receipt.json')} for arm in 'ABCD'}
    native['comparison.json'] = {'path': str(RUN / 'comparison.json'), 'sha256': sha(RUN / 'comparison.json')}
    result = {
        'schema': 'SBHM-001/offline-closeout-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'user_facing_date': '2026-10-05', 'user_timezone': 'America/Chicago',
        'status': 'AUDITED_NATIVE_DEV_CLOSED_JUSTIFIED_HOLD', 'report_sha256': EXPECTED_REPORT,
        'branch': final['branch'], 'head': final['head'], 'native_run_root': str(RUN),
        'native_binding': audits['provenance_receipt_v2.json']['binding'],
        'decision': {'state': 'JUSTIFIED_HOLD', 'exact_successor_native_proposal': None,
                     'reason': 'No added recurrence benefit on paired distant DEV; same source-correct retention damage, failed missing answerability and higher instrumented reference cost.',
                     'next_concrete_decision': 'Retain the negative result and hold further SBHM native spending. Reopen only with a specific new mechanistic hypothesis, minimal discriminating control, frozen analysis/resource bounds, one exact reviewed proposal and separate native authorization.',
                     'automatic_site_context_recovery_or_recurrence_expansion': False},
        'scores': pairing['scores'], 'missing_answerability': pairing['missing_answerability'],
        'source_correct_damage_separate_from_repairs': True,
        'evidence_level': 'one-source/one-site/seed-17 synthetic DEV; dependent world variants; observational stored native artifacts plus fresh offline tokenizer/fixture checks',
        'preservation': {'files': 838, 'bytes': sum(v['bytes'] for v in final['files'].values()),
                         'all_original_bytes_unchanged': True, 'repository_additions': [],
                         'git_status_and_existing_diff_unchanged': True},
        'fixture_counts': counts, 'current_fixture_capability_claim': False,
        'scope': {'new_native_training': False, 'new_native_evaluation': False, 'new_native_model_load': False,
                  'sealed_or_confirmation_banks_read': False, 'published_source_metadata_hash_only': True,
                  'new_native_permit': False, 'model_promotion': False, 'serving_policy_changed': False,
                  'guards_weakened': False, 'new_outputs_outside_tracked_source': True},
        'limitations': audits['provenance_receipt_v2.json']['limitations'] + replay['limitations'] + audits['resources_audit_v2.json']['guard_evidence_limits'],
        'evidence_files': evidence, 'primary_native_receipts': native,
        'report_reviewers': review_names,
        'historical_or_auditor_drafts_preserved': True,
    }
    write('closeout_receipt.json', result)
    # Root-level final and preserved draft artifacts are sealed; temporary fixture files are excluded.
    manifest = {'schema': 'SBHM-001/audit-deliverables-manifest-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'root-level audit artifacts, including retained auditor drafts; temporary fixture/cache directories excluded',
                'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(ROOT.iterdir()) if p.is_file()}}
    write('deliverables_manifest.json', manifest)
    for name, record in manifest['files'].items():
        assert (ROOT / name).stat().st_size == record['bytes'] and sha(ROOT / name) == record['sha256']
    print(json.dumps({'status': result['status'], 'report_sha256': EXPECTED_REPORT, 'preserved_files': 838,
                      'deliverables': len(manifest['files']), 'reviews': 3, 'fixture_counts': counts}))

if __name__ == '__main__':
    main()
