"""Record this owner's delegated review and explicit four-arm run instruction.

This dated registration does not load a model, relax admission guards, or
certify training fit. It is specific to this chat's actual user instructions.
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sbhm.data import load_corpus
from sbhm.protocol import code_hash, digest, read_json, sha, source_lock, write_json

EVIDENCE = ROOT / 'evidence/qualification_20261003'
CORPUS = ROOT / 'corpora/distant_evidence_v2'


def main():
    manifest_path = CORPUS / 'manifest.json'
    proposal = read_json(manifest_path)
    if proposal['reviewed_for_sbhm'] is not False:
        raise ValueError('registration only admits the audited, still-unreviewed proposal once')
    semantic_path = EVIDENCE / 'robust_corpus_semantic_review.json'
    token_path = EVIDENCE / 'corpus_v2_frozen_token_geometry.json'
    semantic, token = read_json(semantic_path), read_json(token_path)
    if (semantic['status'] != 'PASS_INDEPENDENT_SYNTHETIC_CORPUS_AUDIT' or
            token['status'] != 'PASS_TOKEN_GEOMETRY_PAIR_INVARIANCE_TEACHER_BUDGET_ONLY'):
        raise ValueError('independent audits have not passed')
    for name in ('manifest.json', 'design.json', 'train.jsonl', 'dev.jsonl'):
        observed = sha(CORPUS / name)
        if (observed != semantic['corpus_file_sha256'][name] or
                observed != token['corpus_files_sha256'][name]):
            raise ValueError(f'corpus changed after review: {name}')
    design = read_json(CORPUS / 'design.json')
    if design['generator_sha256'] != sha(ROOT / 'scripts/build_robust_corpus.py'):
        raise ValueError('generator changed after review')
    if semantic['audit_script_sha256'] != sha(ROOT / 'scripts/audit_robust_corpus.py'):
        raise ValueError('semantic audit script changed')
    proposal_sha = sha(manifest_path)
    review = {'status': 'ADMITTED_CONTROLLED_SYNTHETIC_DEV_OWNER_DELEGATED_AGENT_REVIEW',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'owner_instruction': 'Please generate a stronger, more suitable, and more robust dataset with your agents to complete the tasks',
              'human_artifact_inspection_claimed': False,
              'review_mode': 'Owner explicitly delegated corpus generation and review to agents',
              'proposal_manifest_before_admission_sha256': proposal_sha,
              'semantic_audit_receipt_sha256': sha(semantic_path),
              'token_audit_receipt_sha256': sha(token_path),
              'semantic_audit_on_proposal': semantic, 'token_audit_on_proposal': token,
              'native_model_loaded': False, 'native_training_fit_qualified': False}
    write_json(CORPUS / 'review.json', review)
    admitted = {**proposal, 'reviewed_for_sbhm': True,
                'purpose': 'Independently audited controlled synthetic DEV pilot; owner delegated agent review',
                'review': {'path': 'review.json', 'sha256': sha(CORPUS / 'review.json')}}
    manifest_path.write_text(json.dumps(admitted, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    data, corpus_sha = load_corpus(manifest_path)
    snapshot = read_json(EVIDENCE / 'native_source_snapshot.json')['source']
    config = read_json(ROOT / 'configs/native.example.json')
    config.update(source=snapshot, corpus_manifest=str(manifest_path),
                  grant=str(ROOT / 'configs/local.titan_transition.grant.json'))
    config['limits']['steps'] = 512
    config_path = ROOT / 'configs/titan_transition.v2.json'
    write_json(config_path, config)
    bindings = {'code_sha256': code_hash(), 'source_sha256': source_lock(config),
                'config_sha256': digest(config), 'corpus_sha256': corpus_sha}
    current = datetime.now(timezone.utc)
    grant = {**bindings,
             'approved_by': 'Human user of Codex chat 01a10030-c8f0-70c3-9b61-9a867962af19; explicit four-arm goal and delegated dataset instruction',
             'issued_utc': current.isoformat(), 'expires_utc': (current+timedelta(days=2)).isoformat(),
             'actions': ['qualify', 'teacher', 'distill', 'recover', 'evaluate'],
             'authority': 'This chat user requested qualification followed by four-arm comparison; training is necessary to the requested recovery controls',
             'resource_guards_unchanged': True, 'native_training_fit_qualified': False}
    write_json(Path(config['grant']), grant)
    registration = {'status': 'REGISTERED_NOT_NATIVE_QUALIFIED_OR_FIT_CERTIFIED',
                    'created_utc': current.isoformat(), 'binding': bindings,
                    'configuration': str(config_path), 'seed': 17, 'steps_per_stage': 512,
                    'train_rows': len(data['train']), 'dev_rows': len(data['dev']),
                    'primary_contrast': 'D minus C on evaluation_groups.distant_heldout',
                    'retention_measure': 'Per-arm A-correct retention on evaluation_groups.retention',
                    'secondary_control': 'Matched near_control rows diagnose task/format/source competence',
                    'uncertainty_unit_if_later_registered': 'world, shared across position/counterfactual rows',
                    'descriptive_dev_only': True, 'promotion': False,
                    'teacher_target_file_bytes_estimate': token['teacher_cache_estimate']['all_target_files_serialized_bytes'],
                    'model_loaded': False, 'training_fit_qualified': False}
    write_json(EVIDENCE / 'registered_v2_pilot.json', registration)
    print(json.dumps({'status': registration['status'], 'configuration': str(config_path),
                      'train_rows': len(data['train']), 'dev_rows': len(data['dev']), **bindings}))


if __name__ == '__main__':
    main()
