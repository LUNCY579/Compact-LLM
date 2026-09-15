"""Read-only extraction of committed evidence; no models or weight files loaded."""
import hashlib
import json
import pathlib
import subprocess

REPO = pathlib.Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification')
OUT = pathlib.Path(__file__).parent

def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, encoding='utf-8')

def obj(ref, path):
    raw = git('show', f'{ref}:{path}')
    return json.loads(raw), hashlib.sha256(raw.encode()).hexdigest()

heads = {}
for line in git('ls-remote', '--heads', 'origin').splitlines():
    sha, ref = line.split()
    heads[ref.removeprefix('refs/heads/')] = {
        'head': sha,
        'parents': git('show', '-s', '--format=%P', sha).strip().split(),
        'subject': git('show', '-s', '--format=%s', sha).strip(),
        'tree': git('show', '-s', '--format=%T', sha).strip(),
    }

rab = heads['research/ced-rab001-hotfix001-qualification-v1']['head']
root = 'experiments/architecture_benchmark_001/evidence/retention-design-v1/'
paths = git('ls-tree', '-r', '--name-only', rab, root).splitlines()
arms = []
for path in paths:
    if '/ARM_DOSE_' not in path or not path.endswith('.json'):
        continue
    d, sha = obj(rab, path)
    cps = d.get('checkpoints', [])
    rec = {k: d.get(k) for k in (
        'arm', 'canonical_tag', 'completed', 'aborted_reason', 'parameters', 'trainable_parameters',
        'lr', 'beta2', 'schedule_spec', 'supervised_target_tokens', 'train_rows_consumed',
        'optimizer_steps', 'optimizer', 'optimizer_storage_backend', 'wall_seconds',
        'source_weights_sha256', 'source_assets_sha256', 'train_row_ids_sha256', 'data_sha256',
        'exploration_harness_sha256', 'optimizer_harness_sha256', 'checkpoint',
        'licenses_registered_training', 'licenses_efficacy_claim')}
    rec['artifact_path'] = path
    rec['artifact_commit'] = rab
    rec['artifact_text_sha256'] = sha
    rec['loss_objective_code_audit'] = 'whole sequence next-token CE; misleading historical evidence_kind contains response_only'
    rec['processed_full_sequence_tokens_derived'] = (d.get('supervised_target_tokens', 0) + d.get('train_rows_consumed', 0))
    rec['trainable_optimizer_state_bytes_analytic_two_fp32_moments'] = 8 * d.get('trainable_parameters', 0)
    if cps:
        rec['last_measured_tokens'] = cps[-1].get('tokens')
        rec['last_measured_steps'] = cps[-1].get('steps')
        rec['dev_ce_nats'] = cps[-1].get('dev_ce_nats')
        rec['panel'] = cps[-1].get('retention', {}).get('permissive')
        rec['per_row_correct'] = cps[-1].get('retention', {}).get('per_row')
    resources = d.get('resource_trace', [])
    rec['cuda_peak_allocated_gib_recorded'] = max((x.get('cuda_peak_allocated_gib', 0) for x in resources), default=None)
    rec['resource_trace_available'] = bool(resources)
    rec['resource_host_sample'] = resources[-1].get('host') if resources else None
    rec['lr_min_actual'] = min((x['lr'] for x in d.get('step_trace', [])), default=None)
    rec['lr_max_actual'] = max((x['lr'] for x in d.get('step_trace', [])), default=None)
    rec['step_trace'] = d.get('step_trace')
    arms.append(rec)

longctx = {}
for label in ['supa', 'opallo']:
    path = root + f'LONGCTX_{label}.json'
    d, sha = obj(rab, path)
    predictions = d['predictions']
    u = {k: v for k, v in predictions.items() if k.endswith('-U')}
    a = {k: v for k, v in predictions.items() if not k.endswith('-U')}
    longctx[label] = {
        'artifact_path': path, 'artifact_commit': rab, 'artifact_text_sha256': sha,
        'rows': len(predictions), 'answerable_rows': len(a), 'unanswerable_rows': len(u),
        'strict_rab_first_line_hits': d['all256_hits'],
        'by_stratum': d['by_stratum'],
        'literal_full_response_UNKNOWN_on_unanswerable': sum(v['prediction'].strip() == 'UNKNOWN' for v in u.values()),
        'case_insensitive_substring_unknown_on_unanswerable': sum('unknown' in v['prediction'].lower() for v in u.values()),
        'literal_full_response_UNKNOWN_on_answerable': sum(v['prediction'].strip() == 'UNKNOWN' for v in a.values()),
        'semantic_annotation_status': 'Not blind-annotated. Literal counts mechanically recomputed; raw outputs retained.',
        'unanswerable_predictions': {k: v['prediction'] for k, v in u.items()},
        'mean_generation_tokens': sum(v['generated_tokens'] for v in predictions.values()) / len(predictions),
        'terminated': sum(v['terminated'] for v in predictions.values()),
        'wall_seconds': d['wall_seconds'], 'device': d['device'],
        'compute_dtype_from_loader_code': 'float32', 'storage_dtype': 'bfloat16',
    }

result = {'schema': 1, 'audit_scope': 'ChatBubble remote branches, committed RAB/Opallo artifacts, lightweight raw-output reclassification; no model runs',
          'heads': heads, 'arms': arms, 'long_context_reclassification': longctx,
          'not_measured': ['blind semantic abstention', 'unsupported answer rubric', 'native BF16 compute parity', 'current model throughput or serving latency']}
out = OUT / 'chatbubble-results-audit.json'
with out.open('x', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print('Created', out, 'with', len(heads), 'heads and', len(arms), 'arm artifacts')
for a in arms:
    if a.get('completed') and ('b5200_seed17' in (a.get('canonical_tag') or '') or a.get('arm') == 'Hlast4'):
        print(a['arm'], a['schedule_spec'], a['cuda_peak_allocated_gib_recorded'], a['lr_min_actual'], a['lr_max_actual'], a['resource_host_sample'])
