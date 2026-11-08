"""Save a final evidence index with immutable metadata snapshots."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

PACKAGE = Path(__file__).resolve().parent


def main(name):
    if Path(name).name != name or (PACKAGE/name).exists():
        raise ValueError('exclusive campaign filename required')
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    number = int(name.removeprefix('REPRODUCIBILITY_INDEX_v').removesuffix('.json'))
    if number < 19:
        raise ValueError('final index starts at v19')
    previous = PACKAGE/f'REPRODUCIBILITY_INDEX_v{number-1}.json'
    index = json.loads(previous.read_text(encoding='utf-8'))
    snapshot = PACKAGE/f'versions/final_generation_handoff_epoch_{number-18:03d}'
    snapshot.mkdir(exist_ok=False)
    mutable = ('STATE.json', 'CONTINUE.md', 'CAMPAIGN.md', 'RESEARCH_REPORT.md',
               'ENVIRONMENT.json', 'MODEL_BINDING.json', 'OWNERSHIP.json',
               'REGISTRY.jsonl', 'CLAIM_LEDGER.jsonl', 'BUDGET_LEDGER.jsonl', 'RESEARCH_POLICY.jsonl')
    for filename in mutable:
        shutil.copyfile(PACKAGE/filename, snapshot/filename)
    for rows in index['groups'].values():
        for row in rows:
            if row['path'] in mutable:
                row['path'] = (snapshot/row['path']).relative_to(PACKAGE).as_posix()
                row['status'] = 'PRESENT_IMMUTABLE_FINAL_HANDOFF_SNAPSHOT'
            path = PACKAGE/row['path']
            row['sha256'] = sha(path) if path.is_file() else None
    sources = {p.name:sha(p) for p in sorted(PACKAGE.glob('*.py'))}
    tests = {p.relative_to(PACKAGE).as_posix():sha(p) for p in sorted((PACKAGE/'tests').glob('*.py'))}
    selected = set(snapshot.rglob('*'))
    for pattern in ('*bulk_restart*', '*BULK_RESTART*', '*FULL_GENERATION_RESTART*',
                    'full_generation_restart_admission_v1.json', 'RESEARCH_POLICY_V11.md',
                    'plot_bulk_restart_001.py', 'HUMAN_FULL_GENERATION_RESTART*'):
        selected.update(PACKAGE.glob(pattern))
    for folder in ('audits', 'reviews', 'tests'):
        for pattern in ('*bulk_restart*', '*full_generation_restart*', '*research_policy_v11*'):
            selected.update((PACKAGE/folder).glob(pattern))
    for pattern in ('runs/native-bulk-restart-001', 'analyses/bulk-restart*',
                    'versions/bulk_restart_before_offline_helper_correction_001',
                    'versions/handoff_before_bulk_finalization_001'):
        for path in PACKAGE.glob(pattern):
            selected.update(path.rglob('*') if path.is_dir() else [path])
    selected = {p for p in selected if p.is_file() and 'mpl_cache' not in p.parts}
    index['groups']['complete_generation_and_final_offline_packet'] = [
        {'path':p.relative_to(PACKAGE).as_posix(), 'sha256':sha(p), 'status':'PRESENT'}
        for p in sorted(selected)]
    index.update(previous_index_path=previous.name, previous_index_sha256=sha(previous),
                 current_source=sources, current_tests=tests,
                 created_utc=datetime.now(timezone.utc).isoformat(),
                 native_worker_seconds=3581.233, native_seconds_remaining=18018.767,
                 live_worker=False, retry_available=False,
                 completion='SUPPORTED_RESTRICTED_NEGATIVE; final whole-mission lifecycle audit pending',
                 current_priority='No model work; final reporting consistency and completion audit',
                 state_binding_note='Immutable final-generation metadata snapshots; later audit/lifecycle refs may be added to live files without changing these epoch bytes.')
    index['latest_full_generation'] = {'run':'runs/native-bulk-restart-001',
        'status':'BULK_MATRIX_COMPLETE_NOT_CONFIRMATORY', 'cases':288, 'parents':144,
        'physical_evaluations':3296, 'actual_API':9200, 'charged_seconds':743.390,
        'report':'analyses/bulk-restart-manifest-reviewed-001/REPORT.json',
        'report_sha256':sha(PACKAGE/'analyses/bulk-restart-manifest-reviewed-001/REPORT.json'),
        'native_admission':'full_generation_restart_admission_v1.json',
        'native_admission_sha256':sha(PACKAGE/'full_generation_restart_admission_v1.json'),
        'post_terminal_reader':'bulk_restart_analysis_manifest_bound.py',
        'reader_not_native_qualified':True, 'grant_consumed':True,
        'source_confirmation':False, 'promotion':False}
    index['retry_grants']['full_generation_restart']='CONSUMED_ONCE_TERMINAL_SUCCESS'
    index['full_restart_approval_blocker_goal_turn']=0
    for field in ('current_completion_audit', 'current_next_decision', 'current_readiness', 'current_epoch_changes'):
        if field in index:
            index['historical_before_final_generation_'+field] = index.pop(field)
    index.pop('native_model_calls_this_continuation', None)
    index['index_builder_native_model_calls'] = 0
    index['current_epoch_changes'] = 'Full288 native complete; independent raw review; exact offline helper correction; policy11; final negative report and artifacts'
    index['current_completion_audit'] = {'path':'audits/completion_after_bulk_restart_001.json',
        'sha256':sha(PACKAGE/'audits/completion_after_bulk_restart_001.json'),
        'status':'DRAFT_AWAITING_FINAL_WHOLE_MISSION_AUDIT'}
    index['current_next_decision'] = {'path':'BULK_RESTART_RESULT_001.md',
        'sha256':sha(PACKAGE/'BULK_RESTART_RESULT_001.md'), 'promotion':False, 'new_native_work':False}
    index['last_prelaunch_readiness'] = 'FULL_GENERATION_RESTART_READINESS_002.json'
    index['current_native_terminal'] = {'path':'runs/native-bulk-restart-001/result/receipt.json',
        'sha256':sha(PACKAGE/'runs/native-bulk-restart-001/result/receipt.json'),
        'status':'BULK_MATRIX_COMPLETE_NOT_CONFIRMATORY'}
    index['current_metadata_snapshot'] = snapshot.relative_to(PACKAGE).as_posix()
    index['groups']['current_final_handoff'] = [{'path':(snapshot/n).relative_to(PACKAGE).as_posix(),
        'sha256':sha(snapshot/n), 'status':'PRESENT_IMMUTABLE_FINAL_HANDOFF_SNAPSHOT'} for n in mutable]
    index['latest_full_generation']['scientific_review'] = {'path':'reviews/bulk_restart_execution_review_001.json',
        'sha256':sha(PACKAGE/'reviews/bulk_restart_execution_review_001.json')}
    state = json.loads((PACKAGE/'STATE.json').read_text(encoding='utf-8'))
    index['historical_before_actual_goal_block_status'] = index.get('goal_status')
    index['historical_goal_transition_receipt'] = index.get('goal_transition_receipt')
    index['goal_status'] = state['persistent_goal_status']
    index['goal_transition_receipt'] = state['goal_tool_receipt']
    index['human_explicit_resume_recorded'] = True
    index['actual_precompletion_goal_snapshot'] = {'path':'GOAL_LIFECYCLE_PRE_COMPLETION_SNAPSHOT_001.json',
        'sha256':sha(PACKAGE/'GOAL_LIFECYCLE_PRE_COMPLETION_SNAPSHOT_001.json')}
    index['goal_scope_note'] = 'Tool blocked status is an earlier lifecycle state; human explicitly resumed and authorized the completed full run. Final goal completion requires the separate whole-mission audit.'
    with (PACKAGE/name).open('x',encoding='utf-8') as handle:
        json.dump(index,handle,indent=2,sort_keys=True);handle.write('\n')
    print(json.dumps({'index':name, 'sha256':sha(PACKAGE/name),
                      'new_epoch_files':len(selected), 'native_model_calls':0}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    main(parser.parse_args().name)
