"""Exact distinct-study authority and anti-replay; imports no model runtime."""
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
CAMPAIGN = PACKAGE.parent
ROOT = CAMPAIGN.parents[1]
AUTHORITY_SHA = '18f9b5079c05138a28d783404b7c9f919d62e829c17fe0c7a7334b789dd2fa3d'
GIB = 1024**3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_probe_scope(admission):
    required_scope = PACKAGE/'ACTIVE_GOAL_SCOPE_001.json'
    required_goal = PACKAGE/'ACTIVE_GOAL_SNAPSHOT_001.json'
    if Path(admission.get('authority_path','')).resolve() != required_scope.resolve():
        raise PermissionError('exact new-goal research scope required')
    if Path(admission.get('goal_snapshot_path','')).resolve() != required_goal.resolve():
        raise PermissionError('exact active-goal evidence required')
    if sha(required_scope) != admission.get('authority_sha256') or sha(required_goal) != admission.get('goal_snapshot_sha256'):
        raise PermissionError('new-goal authority evidence differs')
    if sha(CAMPAIGN/'AUTHORITY.txt') != AUTHORITY_SHA:
        raise PermissionError('initial restrictions changed')
    scope=json.loads(required_scope.read_text(encoding='utf-8'))
    goal=json.loads(required_goal.read_text(encoding='utf-8'))['goal']
    if goal.get('status') != 'active' or scope.get('objective') != goal.get('objective'):
        raise PermissionError('distinct active research goal not bound')
    if scope.get('status') != 'DISTINCT_FROZEN_DEV_RESEARCH_AUTHORIZED_BY_NEW_GOAL':
        raise PermissionError('new research scope absent')
    if any(scope.get(k) is not False for k in ('training','protected_data','cloud_spending','readout_mutation','operator_mutation','retry_available')):
        raise PermissionError('scope expansion or retry not authorized')
    expected={'study_id':'V3-PASSIVE-PROBE-001','cases':192,'task_forwards':384,
        'model_API_cap':451,'full_forward_attempt_cap':449,'completed_forward_target':448,
        'state_adjoints':0,'generation':False,'hard_deadline_seconds':900,
        'internal_reserve_seconds':120,'maximum_prefix_tokens':357}
    if scope.get('probe') != expected or scope.get('extra_budgets') != [0,1]:
        raise PermissionError('passive probe scope/caps differ')
    if admission.get('status') != 'READY_FOR_PASSIVE_PROBE_FP32_DISPATCH' or admission.get('lane') != 'GPU_FP32_PASSIVE_INPUT_FEATURE_PROBE':
        raise PermissionError('passive probe admission not ready')
    if any(admission.get(k) != v for k,v in expected.items()):
        raise PermissionError('admission exceeds exact passive scope')
    floors={'parent_host_available_floor_bytes':7*GIB,'worker_host_available_floor_bytes':6*GIB,
        'worker_rss_ceiling_bytes':6*GIB,'gpu_initial_free_floor_bytes':6*GIB,
        'gpu_allocated_ceiling_bytes':5*GIB,'gpu_global_free_reserve_bytes':GIB,
        'global_finalization_reserve_seconds':1800,'training':False,'native_test':False,
        'device':'cuda:0','dtype':'float32','attn_implementation':'eager','use_cache':False}
    if any(admission.get(k) != v for k,v in floors.items()):
        raise PermissionError('registered host/operator/resource restrictions differ')
    return scope


def validate_probe_unused(admission):
    scope=validate_probe_scope(admission)
    events=[json.loads(line) for line in (CAMPAIGN/'BUDGET_LEDGER.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if any(e.get('study_id')==admission['study_id'] and (e.get('event')=='native_started' or e.get('event')=='native_completed' and e.get('worker_pid') is not None) for e in events):
        raise PermissionError('distinct passive study already attempted; no retry')
    return scope
