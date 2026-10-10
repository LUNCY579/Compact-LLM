"""Freeze the tested equivalent-memory rerun under the existing owner authority."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sbhm.data import load_corpus
from sbhm.protocol import code_hash, digest, now, read_json, sha, source_lock, write_json


def main():
    evidence = Path(__file__).resolve().parent
    config = read_json(ROOT/'configs/titan_transition.v2.json')
    config['grant'] = str(ROOT/'configs/local.titan_transition.memory_efficient.grant.json')
    config_path = ROOT/'configs/titan_transition.v2.memory_efficient.json'
    data, corpus_sha = load_corpus(config['corpus_manifest'])
    binding = {'code_sha256': code_hash(), 'config_sha256': digest(config),
               'source_sha256': source_lock(config), 'corpus_sha256': corpus_sha}
    write_json(config_path, config)
    current = datetime.now(timezone.utc)
    grant = {**binding, 'approved_by': 'Human user of this Codex chat; existing four-arm goal and explicit continue after freeing GPU memory',
             'issued_utc': current.isoformat(), 'expires_utc': (current+timedelta(days=2)).isoformat(),
             'actions': ['qualify', 'teacher', 'distill', 'recover', 'evaluate'],
             'authority': 'Previously authorized native experiment; same source, corpus, architecture, FP32, optimizer, exposure and scoring. Mathematical loss unchanged.',
             'resource_guards_unchanged': True, 'native_training_fit_qualified': False}
    write_json(Path(config['grant']), grant)
    record = {'status': 'REGISTERED_FRESH_FULL_MATRIX_MEMORY_EQUIVALENT_RERUN', 'created_utc': now(),
              'binding': binding, 'configuration': str(config_path),
              'prior_incomplete_run': str(ROOT/'runs/titan_transition_v2_20261004'),
              'prior_terminal_exit_code': 2,
              'prior_failure': 'D recovery violated physical-free-VRAM reserve; guarded probe failed after step 23 with allocator inactive-cache pressure',
              'native_recovery_change': 'Same full-prefix backbone; project vocabulary logits only at predictors of supervised answer/EOS labels; same pinned mean cross-entropy',
              'runtime_memory_change': 'Release only unused PyTorch cache when physical free VRAM is below reserve, then apply unchanged 1GiB reserve guard',
              'completed_test_count': 136, 'skipped_test_count': 0,
              'equivalence_checks': 'B/C/D selected logits, loss, all trainable gradients, clip norm and one AdamW update; ordinary FP32 tolerances, not bitwise CUDA claims',
              'invariant_recipe': {'dtype': config['dtype'], 'seed': config['seed'], 'layer': config['layer'],
                                   'window': config['window'], 'rank': config['rank'], 'limits': config['limits'],
                                   'learning_rate': config['learning_rate'], 'train_rows': len(data['train']), 'dev_rows': len(data['dev'])},
              'primary': 'D minus C strict exact match on distant_heldout; separate A-correct retention on retention',
              'pool_prior_controls': False, 'new_teacher_and_all_qualifications_required': True,
              'native_training_fit_established': False, 'promotion': False}
    write_json(evidence/'memory-efficient-rerun-registration.json', record)
    print(json.dumps({'status': record['status'], 'configuration': str(config_path), **binding}))


if __name__ == '__main__':
    main()
