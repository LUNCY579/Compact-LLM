"""Stdlib-only delivery receipt; no registration, tokenizer or model construction."""
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

root = Path('C:/Users/clclc/BusinessProject/ChatBubble-rineya-distillation-dynamics-v1')
out = Path('C:/Users/clclc/BusinessProject/rdd001-records/preparation-20261005')
package = root / 'experiments/rineya_distillation_dynamics_v1'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}

def git(*args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).rstrip('\r\n')

branch, head = git('branch', '--show-current'), git('rev-parse', 'HEAD')
assert branch == 'research/rineya-distillation-dynamics-v1'
assert head == '2d7be4b5c2ba8a610b09d17455d519eecaa421e6'
changes = git('status', '--porcelain', '--untracked-files=all').splitlines()
assert all(line[3:].startswith('experiments/rineya_distillation_dynamics_v1/') for line in changes)
subprocess.run(['git', 'diff', '--check'], cwd=root, check=True)
compiled = []
for path in sorted(package.rglob('*.py')):
    compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    compiled.append(str(path.relative_to(root)))
source_files = {str(p.relative_to(root)).replace('\\', '/'): sha(p)
                for p in sorted(package.rglob('*')) if p.is_file()}

verification_path = out / 'verification/offline-verification-v3.json'
verification = json.loads(verification_path.read_text(encoding='utf-8'))
assert verification['exit_code'] == 0 and not any(verification['tripwire_attempts'].values())
assert not verification['cuda_initialized']
junit = ET.parse(out / 'verification/offline-junit-v3.xml').getroot()[0]
assert junit.attrib['failures'] == junit.attrib['errors'] == '0'
assert len(verification['optimizer_tests_deselected']) == 11
inventory_path = out / 'inventory/asset_inventory.json'
inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
models = {m['name']: m for m in inventory['models']}
assets = {}
for role, name in [('student', 'Rineya-Titan-Experimental-1.0.0-devfreeze'),
                   ('teacher', 'Rineya-Supa-1.7.0-devfreeze')]:
    model = models[name]
    assets[role] = {'path': model['path'], 'inventory_sha256': model['rdd_inventory_snapshot_sha256'],
                   'weight_sha256': model['files']['model.safetensors'],
                   'stored_tensor_elements': model['header']['stored_tensor_elements'],
                   'runtime_qualified': False, 'teacher_competence_qualified': False}
plan = json.loads((package / 'configs/selected-pilot.json').read_text(encoding='utf-8'))
proposal = {'schema': 'rdd001_prospective_preparation_v1', 'status': 'BLOCKED_NOT_REGISTERED_OR_AUTHORIZED',
    'client_date': '2026-10-05', 'design_review': 'ROOT_AND_INDEPENDENT_SELECTED_DESIGN_CODE_REVIEW',
    'execution_grant': False, 'assets': assets, 'plan': plan,
    'learning_rate_engineering_design_reviewed': True, 'native_asset_lr_reviewer_attestation_set': False,
    'next_action': {'mode': 'qualification_only', 'max_invocations': 1,
        'max_disposable_optimizer_updates': 2, 'treatment_updates': 0, 'teacher_dev_distinct_parents': 16,
        'max_generation_attempts': 35, 'max_generated_tokens': 1120,
        'max_backbone_forward_calls_implied_by_plan': 1130, 'confirmation_reads': 0},
    'resource_limits': inventory['unchanged_rdd_limits'],
    'blocked_prerequisites': ['genuine matching RDD E0', 'reviewed exact-native/HF parity and teacher prefix feedback',
        'fresh reviewed catalog and complete approved exclusions', 'actual named LR/source review and future explicit grant',
        'owned available host and unchanged shared-lock/resource admission'],
    'gd_attachment_complete': False, 'native_peft_parity_qualified': False,
    'document': ref(out / 'PROSPECTIVE_NATIVE_PROPOSAL.md'),
    'inventory': ref(inventory_path), 'source_files': source_files}
with (out / 'PROSPECTIVE_NATIVE_PROPOSAL.json').open('x', encoding='utf-8') as stream:
    json.dump(proposal, stream, indent=2)

gd = Path('C:/Users/clclc/BusinessProject/ChatBubble-rineya-generalization-dynamics-v1')
receipt = {'schema': 1, 'experiment': 'RDD-001',
    'status': 'OFFLINE_PREPARATION_VERIFIED_NATIVE_PREREQUISITES_BLOCKED',
    'client_date': '2026-10-05', 'recorded_utc': datetime.now(timezone.utc).isoformat(),
    'source': {'path': str(root), 'branch': branch, 'starting_head': head, 'current_head': head,
        'initial_worktree_clean': True, 'local_changes_preserved': True, 'current_changes': changes,
        'source_file_hashes': source_files, 'native_source_clean': False, 'committed_or_pushed': False},
    'implementation': ['selected A/B/D/SFT only; one seed/LR; complete scope selection',
        'qualification-only worker with exact16 teacher parents, failure and restoration receipts',
        'complete raw/rendered/causal prompt manifests across all consumption roles',
        'per-step standard-PEFT saves and explicit original/derived/disposable lineage',
        'GD gradient source/filter/consume/unused projection and genuine approved post-save hook',
        'future guarded full-logit/greedy PEFT parity helper; never called',
        'reject failed cleanup/supervisor evidence despite an earlier COMPLETE result'],
    'verification': {'passed': int(junit.attrib['tests']), 'failures': 0, 'errors': 0,
        'skipped': int(junit.attrib.get('skipped', 0)), 'optimizer_tests_deselected': 11,
        'tripwire_attempts': verification['tripwire_attempts'], 'cuda_initialized': False,
        'runtime': verification['runtime'], 'torch': verification['torch'],
        'syntax_compiled_in_memory': compiled, 'whitespace_check': 'PASS',
        'receipt': ref(verification_path), 'junit': ref(out / 'verification/offline-junit-v3.xml'),
        'stdout': ref(out / 'verification/offline-tests-v3.log'),
        'evidence_kind': 'MECHANICAL_FIXTURES_ONLY_NO_RINEYA_CAPABILITY_EVIDENCE'},
    'boundaries': {'native_model_loads': 0, 'native_forward_calls': 0, 'optimizer_updates': 0,
        'confirmation_or_sealed_bank_reads': 0, 'native_registration_created': False,
        'worker_launched': False, 'full_campaign_launched': False, 'forward_screen_launched': False,
        'model_files_modified': False, 'serving_policy_modified': False, 'guards_weakened': False,
        'schedules_modified': False, 'other_experiment_permit_imported': False,
        'environment_installed_or_changed': False, 'historical_evidence_overwritten': False,
        'new_outputs_and_caches_under_businessproject_outside_source': True},
    'limitations': ['native qualification, teacher competence and all efficacy unmeasured',
        'scoped historical CE512 E0 not admissible as RDD E0', 'native HF/PEFT output parity still missing',
        'fresh reviewed data/exclusions and actual named enrollment review missing',
        'host readings vary and other owner processes observed; no admission/availability grant',
        'one-seed/two-step design cannot rank replicated methods or claim broad/long-context retention',
        'GD needs genuinely approved original producer trajectory and complete raw/public exposure exclusions'],
    'gd_coordination': {'owner_thread_id': '01a10d65-39a8-77d3-a0d4-6d4f58c5e3a1',
        'owner_thread_title': 'Prepare GD-001 attachment contract', 'gd_files_edited_by_this_chat': False,
        'contract_file_at_review': ref(gd / 'experiments/rineya_generalization_dynamics_v1/attachment.py'),
        'raw_rendered_overlap_gap_reported_to_owner': True, 'attachment_or_native_admission_performed': False},
    'historical_validation_preserved': ref(package / 'VALIDATION.json'),
    'asset_inventory': ref(inventory_path), 'host_inventory': ref(out / 'inventory/host_inventory.json'),
    'proposal': ref(out / 'PROSPECTIVE_NATIVE_PROPOSAL.json'),
    'next_decision': 'Review fixed Titan/Supa and LR1e-5 design; obtain missing E0/export/data/host prerequisites before considering one qualification-only grant.'}
with (out / 'PREPARATION_RECEIPT.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps({'receipt': str(out / 'PREPARATION_RECEIPT.json'), 'sha256': sha(out / 'PREPARATION_RECEIPT.json'),
                  'passed': receipt['verification']['passed'], 'syntax_files': len(compiled),
                  'status': receipt['status']}, indent=2))
