"""Post-terminal proof routing only; no real outcome report or model execution."""
from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path

import pytest

from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import bulk_restart_analysis as frozen
from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import bulk_restart_analysis_manifest_bound as corrected
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.tests.test_bulk_restart_analysis import fixture as old_fixture, encoded


def actual_metadata():
    root=corrected.PACKAGE/'runs/native-bulk-restart-001'
    admission=json.loads((root/'admission.json').read_text())
    manifest=(root/'source_manifest.json').read_bytes()
    current={name:corrected.sha(corrected.PACKAGE/name) for name in corrected.OFFLINE_HELPER_PINS}
    return admission,manifest,current


def fixture():
    args,context=old_fixture();actual,manifest,current=actual_metadata()
    context['admission']['dispatch_hashes']=actual['dispatch_hashes']
    prefix=corrected.PACKAGE.relative_to(corrected.REPO).as_posix()+'/'
    runner=actual['dispatch_hashes'][prefix+'bulk_restart_native.py']
    args[5]['runner_sha256']=runner
    context['human_scope']['payload']['worker_sha256']=runner
    scope_hash=hashlib.sha256(encoded(context['human_scope']['payload'])).hexdigest()
    context['human_scope']['sha256']=context['admission']['human_extension_sha256']=args[7]['human_extension_sha256']=scope_hash
    error,bridge=corrected.verify_offline_helper_bridge(context['admission'],manifest,current)
    assert not error
    context['offline_helper_bridge']=bridge
    for index,(name,*_) in enumerate(corrected.INTEGRITY_STAGES,1):
        proof=json.loads(context['proof_bytes'][name]);proof['source_sha256']=actual['dispatch_hashes']
        if index>1:
            blob=encoded(args[5]);proof['journal_commitments']['frozen_development.json']={
                'bytes':len(blob),'lines':len(blob.splitlines()),'sha256':hashlib.sha256(blob).hexdigest()}
        context['proof_bytes'][name]=encoded(proof);digest=hashlib.sha256(context['proof_bytes'][name]).hexdigest()
        next(e for e in context['integrity_journal'] if e.get('checkpoint_index')==index and e['event']=='INTEGRITY_CHECK_PASSED')['proof_sha256']=digest
        args[6][('DEV_deterministic_integrity_sha256','DEV_complete_integrity_sha256','FINAL_integrity_sha256')[index-1]]=digest
        if index==1:
            args[5]['DEV_deterministic_integrity_sha256']=digest
            args[5]['freeze_sha256']=corrected.payload_digest({k:v for k,v in args[5].items() if k!='freeze_sha256'})
    blob=encoded(args[5]);context['journal_bytes']['frozen_development.json']=blob
    marker=next(e for e in context['integrity_journal'] if e['event']=='FREEZE_COMMITTED')
    marker.update(freeze_sha256=hashlib.sha256(blob).hexdigest(),freeze_payload_sha256=args[5]['freeze_sha256'])
    return args,context


def test_actual107_and_exact_prelaunch_snapshot_accept_only_four_offline_helpers():
    admission,manifest,current=actual_metadata()
    prefix=corrected.PACKAGE.relative_to(corrected.REPO).as_posix()+'/'
    assert len(admission['dispatch_hashes'])==107
    assert all(prefix+name not in admission['dispatch_hashes'] for name in corrected.OFFLINE_HELPER_PINS)
    errors,bridge=corrected.verify_offline_helper_bridge(admission,manifest,current)
    assert errors==[] and bridge['verified'] and not bridge['native_qualification_claim']
    assert set(bridge['files'])==set(corrected.OFFLINE_HELPER_PINS)


@pytest.mark.parametrize('defect',['missing_snapshot','wrong_snapshot','changed_current','missing_current','extra_helper','wrong_path','changed_registered_reader','changed_native_membership'])
def test_snapshot_current_path_or_native_map_defect_never_falls_back_generally(defect):
    admission,manifest,current=actual_metadata();prefix=corrected.PACKAGE.relative_to(corrected.REPO).as_posix()+'/'
    if defect=='missing_snapshot':manifest=None
    elif defect=='wrong_snapshot':manifest=manifest+b' '
    elif defect=='changed_current':current['analysis.py']='a'*64
    elif defect=='missing_current':current.pop('gpu_analysis.py')
    elif defect=='extra_helper':current['unrelated.py']='a'*64
    elif defect=='wrong_path':current['elsewhere/analysis.py']=current.pop('analysis.py')
    elif defect=='changed_registered_reader':admission['dispatch_hashes'][prefix+'bulk_restart_analysis.py']='a'*64
    else:admission['dispatch_hashes'][prefix+'bulk_analysis.py']=corrected.ORIGINAL_ANALYZER_SHA
    errors,bridge=corrected.verify_offline_helper_bridge(admission,manifest,current)
    assert errors and not bridge['verified']


def test_bridge_repairs_proof_membership_only_and_preserves_primary_science():
    args,context=fixture()
    old=frozen.aggregate(*args,restart=context)
    assert not old['complete'] and any('original scientific analyzer lineage' in e for e in old['errors'])
    report=corrected.aggregate(*args,restart=context)
    baseline=corrected._aggregate_scientific(*args)
    assert report['complete'],report['errors']
    assert report['reports']==baseline['reports'] and report['costs']==baseline['costs']
    assert not report['independent_source_confirmation']


@pytest.mark.parametrize('defect',['missing_bridge','wrong_state','missing_grant','missing_cell','failed_native'])
def test_existing_native_grant_complete_and_null_gates_stay_required(defect):
    args,context=fixture()
    if defect=='missing_bridge':context.pop('offline_helper_bridge')
    elif defect=='wrong_state':args[6]['after_state_sha256']='a'*64
    elif defect=='missing_grant':context['human_scope']={}
    elif defect=='missing_cell':args[0].pop(next(i for i,r in enumerate(args[0]) if r['physical_run'] and r['split']=='OPEN_VALIDATION'))
    else:args[6]['status']='FAIL'
    report=corrected.aggregate(*args,restart=context)
    assert not report['complete'] and report['reports']['OPEN_VALIDATION']['frozen_fixed_primary']['accuracy'] is None


def test_scientific_and_native_proof_helpers_AST_unchanged():
    before={n.name:n for n in ast.parse(Path(frozen.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
    after={n.name:n for n in ast.parse(Path(corrected.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
    changed=[]
    for name,node in before.items():
        if ast.dump(node,include_attributes=False)!=ast.dump(after[name],include_attributes=False):changed.append(name)
    assert changed==['validate_restart_integrity','aggregate','run']
    assert corrected.sha(Path(frozen.__file__))==corrected.REGISTERED_READER_SHA
