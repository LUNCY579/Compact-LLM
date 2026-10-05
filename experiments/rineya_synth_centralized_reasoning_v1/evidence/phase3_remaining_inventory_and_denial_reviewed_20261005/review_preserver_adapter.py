"""Minimal exact-selector adapter source and before-copy CPU refusal controls."""
import hashlib, importlib.util, json, sys, tempfile, time
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    adapter=load('fixed_remaining_adapter_review',HERE/'preserve_fixed_remaining.py')
    before={'adapter':sha(HERE/'preserve_fixed_remaining.py'),'preserved_original':sha(adapter.PRESERVED),'actual_original':sha(adapter.ORIGINAL)}
    assert before['preserved_original']==before['actual_original']==adapter.PIN
    controls=[]
    for ordinal in (5,6):
        module,adapted_sha=adapter.adapted(ordinal)
        assert module.EXPECTED_ORDINAL==ordinal
        source=adapter.PRESERVED.read_text(encoding='utf-8')
        assert source.count(adapter.OLD)==1 and source.replace(adapter.OLD,adapter.NEW,1).count(adapter.NEW)==1
        controls.append({'id':'exact_single_selector_source_%02d'%ordinal,'passed':True,'adapted_source_sha256':adapted_sha})
    config=adapter.PACKAGE/'configs/SYNTH_SHORT_M0_ATOMIC_BATCH05_PHASE3_A1.json'
    inv=adapter.PACKAGE/'artifacts/short_m0_audit_registrations/SYNTH_SHORT_PHASE3_BATCH05_A1_AUDIT_INVENTORY_001.json'
    args={'config_sha':adapter.ANCHORS[5][0],'inventory_sha':adapter.ANCHORS[5][1],'commit':adapter.HEAD,
        'sup_sha':'d8b1edf4ddec266ed4200dee197454af8221b19173a541011d1daa3dd9b8f680','terminal_signal':adapter.SIGNAL}
    for case in ('ordinal4','boolordinal','config_anchor','inventory_anchor','HEAD','SUP','signal','wrong_config_path'):
        bad=dict(args);ordinal=True if case=='boolordinal' else 4 if case=='ordinal4' else 5;path=config
        if case=='config_anchor':bad['config_sha']='0'*64
        if case=='inventory_anchor':bad['inventory_sha']='0'*64
        if case=='HEAD':bad['commit']='a'*40
        if case=='SUP':bad['sup_sha']=None
        if case=='signal':bad['terminal_signal']=None
        if case=='wrong_config_path':path=adapter.PACKAGE/'configs/SYNTH_SHORT_M0_ATOMIC_BATCH06_PHASE3_A1.json'
        with patch.object(adapter,'adapted',side_effect=AssertionError('read/copy before guard')) as source:
            try:adapter.preserve_fixed(ordinal,path,'UNUSED','UNUSED',inv,**bad)
            except ValueError as exc:controls.append({'id':'before_copy_'+case,'passed':True,'error':str(exc)})
            else:raise AssertionError('bad adapter input accepted')
            assert source.call_count==0
    fixture=load('remaining_adapter_denied_fixture_supplier',adapter.ORIGINAL.parent/'test_preserve_terminal.py')
    for ordinal in (5,6):
        module,_=adapter.adapted(ordinal)
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root,package,c,l,s,r,lease,kwargs=fixture.fixture(directory,mode='denied')
            with patch.object(module,'PACKAGE',package),patch.object(module,'ROOT',root):
                try:module.preserve(c,l,s,r,**kwargs)
                except ValueError as exc:
                    assert 'whole64 Phase3 batch04 purpose' in str(exc)
                    controls.append({'id':'actual_wrong_selector_rejects_before_destination_%02d'%ordinal,'passed':True,'error':str(exc)})
                else:raise AssertionError('unchanged ordinal4 fixture accepted by remaining selector')
            assert not s.exists()
    after={'adapter':sha(HERE/'preserve_fixed_remaining.py'),'preserved_original':sha(adapter.PRESERVED),'actual_original':sha(adapter.ORIGINAL)}
    assert before==after and not any(n in sys.modules for n in ('torch','transformers','tokenizers','psutil','native_supervisor','native_process'))
    receipt={'schema_version':1,'kind':'MINIMAL_FIXED05_06_PRESERVER_ADAPTER_SOURCE_BEFORE_COPY_CONTROLS',
        'controls':controls,'source_before':before,'source_after':after,'source_bytes_unchanged':True,
        'only_original_selector_predicate_changed':True,'explicit_ordinal5_or6_and_external_anchors_required':True,
        'admission_denied_uses_unchanged_no_worker_no_death_branch':True,'actual_raw_read_copy_or_probe_calls':0,
        'native_success_audit_grade_or_retry_performed':False}
    receipt['sha256']=fixture.p.digest(receipt)
    target=HERE/('PRESERVER_ADAPTER_REVIEW_%d.json'%time.time_ns())
    target.write_text(fixture.p.canonical(receipt)+'\n',encoding='ascii')
    print(json.dumps({'receipt':str(target),'receipt_sha256':receipt['sha256'],'controls':len(controls),'source_hashes':after,'native_probe_copy_calls':0}))
    return 0
if __name__=='__main__':raise SystemExit(main())
