"""Owned import supervise -> preserve -> terminal audit; never zero-exit PASS."""
from pathlib import Path
from types import ModuleType
import argparse,hashlib,json
HERE=Path(__file__).resolve().parent;PACKAGE=next(p for p in (HERE,*HERE.parents) if p.name=='rineya_synth_centralized_reasoning_v1')
SCOPE='OWNED_TORCH_IMPORT_LIBRARY_EFFECT_OBSERVATION_ONLY'
CONTRACT={'bytes':15490,'sha256':'c669ba3a2897bbfe200fd35807e299be8d75718b428c926edf42a7ffbbf387cf','role':'python_source'}
SUPERVISOR={'bytes':12047,'sha256':'00726a65bd60f85528d7133607e6af0a5400e4006fc9eee313a1e90ed718da76','role':'python_source'}
ANCHORS=('expected_config_file','expected_config_body','registration_path','expected_registration_file','expected_registration_body','runtime_path','expected_runtime_file','expected_runtime_body','expected_head')
def require(v,m):
    if not v:raise ValueError(m)
def _CPU_factory():
    issued=set()
    class Capability:pass
    def mint():v=Capability();issued.add(v);return v
    def check(v):require(type(v) is Capability and v in issued,'privateCPUpostrunfixture')
    return mint,check
_CPU_capability,_check_CPU=_CPU_factory()
def _load(path,descriptor):
    p=Path(path).absolute()
    for item in (p,*p.parents):require(not item.is_symlink() and not(getattr(item.lstat(),'st_file_attributes',0)&0x400),'selectedpostrunsourcephysical')
    before=p.stat();require(before.st_size<=16*1024**2,'boundedselectedpostrunsource')
    with p.open('rb') as f:raw=f.read(before.st_size+1)
    after=p.stat();require(len(raw)==before.st_size and (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino) and descriptor=={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'},'actualsourceboundpostrunmodule')
    m=ModuleType('selected_import_postrun_'+p.stem);m.__file__=str(p);exec(compile(raw,str(p),'exec'),m.__dict__);return m
def _preservation_target(value,target):
    target=Path(target).absolute();require(target.parent==PACKAGE/'evidence' and target.name.startswith('prefix_input_owned_') and not target.exists(),'predeterminedregisteredpreservationleaf')
    exceptions=value['registration']['git_output_exceptions'];require(type(exceptions) is dict and exceptions,'exactfuturepreservationexceptionleavesregistered')
    covered={Path(p) for p in exceptions if Path(p).is_relative_to(target)};require(target/'PRESERVATION.json' in covered,'exactpreservationreceiptleafregistered')
    for text,entry in exceptions.items():
        p=Path(text)
        if not p.is_relative_to(target):continue
        require(p.parent==target or p.parent==target/'source_snapshots','flatknownpreservationleavesonly');require(entry['role'] in ('evidence_output','evidence_source_snapshot'),'exactpreservationexceptionsroles')
    for text in value['config']['source_files']:
        path=target/'source_snapshots'/(hashlib.sha256(text.encode()).hexdigest()+Path(text).suffix);require(path in covered and exceptions[str(path)]=={'role':'evidence_source_snapshot','source_path':text},'everyselectedsourcesnapshotfutureleafregistered')
    return target
def _expected_measurement_identity(value,output,contract,*,fixture_only=False):
    records=[contract.parse(line) for line in contract.read_bytes(Path(output)/'PHASE_OBSERVATIONS.jsonl').splitlines()];ownership=[r['result'] for r in records if r.get('operation')=='ownership_verify' and r.get('event')=='returned'];require(len(ownership)==1 and type(ownership[0]) is dict and type(ownership[0].get('actual_worker_pid')) is int,'oneactualsourceownedmembershipoperation');pid=ownership[0]['actual_worker_pid'];sup=contract.parse(contract.read_bytes(Path(output)/'SUPERVISOR_RECEIPT.json'));resources=contract.parse(contract.read_bytes(Path(output)/'RESOURCE_OBSERVATIONS.json'));matches=[]
    def visit(obj):
        if type(obj) is dict:
            for born in obj.get('owned_process_identities',[]):
                if type(born) is dict and born.get('pid')==pid:matches.append({k:born[k] for k in ('pid','creation_filetime_100ns')})
            for child in obj.values():visit(child)
        elif type(obj) is list:
            for child in obj:visit(child)
    visit(resources)
    if sup['worker_identity']['pid']==pid:matches.append({k:sup['worker_identity'][k] for k in ('pid','creation_filetime_100ns')})
    require(matches and all(m==matches[0] for m in matches),'oneexactexecutingbirthfromoriginalownedrecords');owner=contract.parse(contract.read_bytes(Path(output)/'supervisor_ownership.json'));producer=str(PACKAGE/'measurement_guard.py');require(producer in value['config']['source_files'],'exactselectedmeasurementproducer')
    return {'source_commit':value['expected_head'],'original_scope':SCOPE,'owned_birth':matches[0],'owned_birth_status':'VALIDATED','context_sha256':contract.digest(value),'owner_receipt_sha256':contract.digest(owner),'producer_source':{'path':producer,'descriptor':value['config']['source_files'][producer]},'fixture_only':fixture_only,'native_PASS':False,'native_authority_issued':False}
def _postrun(context,receipt,target,*,contract,supervisor,terminal,expected_binding,gate_references,_CPU_fixture=None):
    fixture_only=_CPU_fixture is not None
    if fixture_only:_check_CPU(_CPU_fixture)
    value=contract.revalidate_context(context);require(value==contract.context_value(context) and value['scope']==SCOPE and value.get('fixture_only') is fixture_only,'separateactualimportcontext');output=Path(receipt['output_dir']).absolute();target=_preservation_target(value,target)
    if receipt.get('worker_resumed') is not True:
        require(receipt.get('partial_outputs_frozen') is True and receipt.get('sampler_threads_terminal_verified') is True and receipt.get('gpu_telemetry_cleanup_verified') is True and receipt.get('unjoined_sampler_threads')==[],'actualfrozendenialreceiptwithoutworkersemantics');require(contract.parse(contract.read_bytes(output/'SUPERVISOR_RECEIPT.json'))==receipt,'actualoriginaldenialreceiptbytes')
        return {'kind':'OWNED_IMPORT_POSTRUN_TERMINAL_RESULT','supervisor_status':receipt['status'],'terminal_status':'IMPORT_OBSERVATION_NOT_RESUMED_FAILURE_RETAINED','terminal_audit':None,'terminal_audit_not_attempted_reason':'NO_RESUMED_WORKER_SEMANTIC_DENOMINATOR','original_output_directory':str(output),'preserved_directory':None,'native_PASS':False,'native_authority_issued':False}
    exceptions=value['registration']['git_output_exceptions']
    for leaf in output.iterdir():require(str(target/leaf.name) in exceptions,'everyactualrawpreservationleafindividuallyregistered')
    for name in ('reservation.json','worker.json','resume.json','terminal.json'):require(str(target/('lease_'+name)) in exceptions,'everyactualleasepreservationleafindividuallyregistered')
    preservation=supervisor.preserve_terminal(output,target,contract=contract,context=context)
    external=contract.parse(contract.read_bytes(output/'OWNED_EXTERNAL_ANCHORS.json'));require(external=={'schema_version':2,**expected_binding,'grant_reference':gate_references['grant'],'owner_receipt_reference':gate_references['owner_receipt']},'exactoriginalV2postrunbinding')
    semantic=all((output/name).exists() for name in ('worker_result.json','PHASE_ACCOUNTING_FINAL.json','IMPORT_LIBRARY_MEASUREMENT.json'))
    identity=_expected_measurement_identity(value,output,contract,fixture_only=fixture_only) if semantic else None
    if identity is None:identity={'context_sha256':contract.digest(value)}
    p=target/'PRESERVATION.json';raw=contract.read_bytes(p);manifest={**value['config'],'runtime_value':value['runtime']}
    result=terminal.audit_preserved(target,original_directory=output,manifest=manifest,anchors=external,expected_external_binding=expected_binding,expected_gate_references=gate_references,expected_preservation_file_sha256=hashlib.sha256(raw).hexdigest(),expected_preservation_body_sha256=preservation['sha256'],measurement_identity=identity,expected_context_sha256=contract.digest(value),expected_producer_source=identity.get('producer_source'),expected_executing_birth=identity.get('owned_birth'))
    require(result['native_PASS'] is result['native_authority_issued'] is False,'terminalcannotgrantnativePASS');return {'kind':'OWNED_IMPORT_POSTRUN_TERMINAL_RESULT','supervisor_status':receipt['status'],'terminal_status':result['status'],'terminal_audit':result,'preserved_directory':str(target),'native_PASS':False,'native_authority_issued':False}
def dispatch(config,output=None,*,preserved_directory,grant_reference=None,owner_receipt_reference=None,anchors):
    require(grant_reference is not None and owner_receipt_reference is not None,'separateownergrantreferencesrequiredbeforeownedlaunch');C=_load(PACKAGE/'import_observation_contract.py',CONTRACT);binding={'config_path':str(Path(config).absolute()),'anchors':anchors};C.enroll_native_gate(grant_reference,owner_receipt_reference,binding);context=C.validate_native_manifest(config,**anchors);value=C.context_value(context);_preservation_target(value,preserved_directory);SUP=_load(PACKAGE/'import_observation_supervisor.py',SUPERVISOR);terminal_path=PACKAGE/'import_terminal_audit.py';terminal=_load(terminal_path,value['config']['source_files'][str(terminal_path)]);receipt=SUP.supervise_registered(config,output,contract=C,grant_reference=grant_reference,owner_receipt_reference=owner_receipt_reference,**anchors);return _postrun(context,receipt,preserved_directory,contract=C,supervisor=SUP,terminal=terminal,expected_binding=binding,gate_references={'grant':grant_reference,'owner_receipt':owner_receipt_reference})
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output-dir',type=Path);parser.add_argument('--preserved-dir',type=Path,required=True)
    for name in ANCHORS:parser.add_argument('--'+name.replace('_','-'),required=True)
    for prefix in ('grant','owner-receipt'):
        parser.add_argument('--'+prefix+'-path',type=Path,required=True);parser.add_argument('--'+prefix+'-file-sha256',required=True);parser.add_argument('--'+prefix+'-body-sha256',required=True)
    args=parser.parse_args();reference=lambda name:{'path':str(getattr(args,name+'_path').absolute()),'file_sha256':getattr(args,name+'_file_sha256'),'body_sha256':getattr(args,name+'_body_sha256')};result=dispatch(args.config,args.output_dir,preserved_directory=args.preserved_dir,grant_reference=reference('grant'),owner_receipt_reference=reference('owner_receipt'),anchors={name:getattr(args,name) for name in ANCHORS});print(json.dumps(result,sort_keys=True));return 0 if result['terminal_status']=='IMPORT_MEASUREMENT_TERMINAL_VALIDATED_UNQUALIFIED' else 2
if __name__=='__main__':raise SystemExit(main())
