"""Independent pure preserved-byte terminal audit for import measurement only."""
from pathlib import Path
from types import ModuleType
import ast,hashlib,json
HERE=Path(__file__).resolve().parent;PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
ORIGINAL=PACKAGE/'evidence/gate_terminal_V2_join_reviewed_cpu_20261007/final_V2terminal_join003/prefix_input_owned_auditor.py'
PIN='c632d57cc5d992506c0577c2f939cf620d84f472ca76cdf6173108325c8299a8'
SCOPE='OWNED_TORCH_IMPORT_LIBRARY_EFFECT_OBSERVATION_ONLY'
OPS=('source_qualification','ownership_verify','resource_before_import','measurement_begin','framework_import','torch_version_read','resource_after_import','measurement_observations','measurement_cleanup','raw_measurement_publication','worker_result_publication')
def require(v,m):
    if not v:raise ValueError(m)
def original():
    raw=ORIGINAL.read_bytes();require(hashlib.sha256(raw).hexdigest()==PIN,'selected original terminal Reader/resource source');m=ModuleType('import_terminal_original_reader');m.__file__=str(ORIGINAL);exec(compile(raw,str(ORIGINAL),'exec'),m.__dict__);m.PACKAGE=PACKAGE;return m,raw
A,SOURCE=original()
def replay(records):
    stack=None;counts={};returns={};order=[];last=None
    for row in records:
        fields={'operation','event','ordinal','monotonic','utc'}|({'result','error'} if row.get('event')!='attempt' else set());A.fields(row,fields,'import phase marker');name=row['operation'];event=row['event'];require(name in OPS and event in ('attempt','returned','failed') and A.integer(row['ordinal'],1) and A.finite(row['monotonic']) and (last is None or row['monotonic']>=last),'exact finite import phase marker');last=row['monotonic'];item=counts.setdefault(name,A.zero_counts())
        if event=='attempt':require(stack is None and name not in order and row['ordinal']==len(order)+1,'one serial original import operation');stack=(name,row['ordinal']);order.append(name);item['attempted']+=1;item['pending']+=1
        else:require(stack==(name,row['ordinal']) and (row['error'] is None)==(event=='returned'),'matched truthful import terminal');stack=None;item[event]+=1;item['pending']-=1;returns[name]=row
    require(stack is None and records,'no omitted phase terminal');return {'counts':counts,'roots':order,'by_name':returns,'first_clock':records[0]['monotonic'],'last_clock':records[-1]['monotonic']}
def _resource_source_blocks(namespace):
    function=next(n for n in ast.parse(SOURCE,str(ORIGINAL)).body if isinstance(n,ast.FunctionDef) and n.name=='_native_preserved_body');body=function.body
    def index(fragment):return next(i for i,n in enumerate(body) if fragment in ast.unparse(n))
    # Exact unchanged original admission/Job lease/resource source, independent
    # of the tensor-specific worker/measurement predicates omitted here.
    admission_start=index('supervisor = data');admission_end=index('container = data')
    resource_start=index('observations = data');resource_end=index('actual interpreter birth present')
    selected=body[admission_start:admission_end]+body[resource_start:resource_end]
    require(admission_start<admission_end<resource_start<resource_end and len(selected)>20,'finite exact original lifecycle/resource statement slices')
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(ORIGINAL),'exec'),namespace)
    return namespace['births']
def audit_preserved(preserved_directory,*,original_directory,manifest,anchors,expected_external_binding,expected_gate_references,expected_preservation_file_sha256,expected_preservation_body_sha256,measurement_identity):
    root=Path(preserved_directory).absolute();rawroot=Path(original_directory).absolute();reader=A.Reader(root);path=root/'PRESERVATION.json';preservation=A.parse(reader.read(path,{'bytes':path.stat().st_size,'sha256':expected_preservation_file_sha256}));A.checked_seal(preservation,expected_preservation_body_sha256)
    require(preservation['kind']=='PRESERVED_OWNED_TORCH_IMPORT_OBSERVATION' and preservation['scope']==SCOPE and preservation['config_body_sha256']==manifest['sha256'] and preservation['native_authority_issued'] is preservation['native_PASS'] is False,'exact separate import preservation provenance')
    raw={};data={};names=set(preservation['raw_files']);require(set(p.name for p in rawroot.iterdir())==names,'original complete raw denominator');required={'SUPERVISOR_RECEIPT.json','RESOURCE_OBSERVATIONS.json','supervisor_ownership.json','OWNED_EXTERNAL_ANCHORS.json','PHASE_OBSERVATIONS.jsonl'};require(required<=names,'required original supervised/phase evidence')
    for name,d in preservation['raw_files'].items():
        require(Path(name).name==name,'flat bounded original rawleaf');b=reader.read(root/name,d,A.MAX_LEDGER if name.endswith('.jsonl') else A.MAX_JSON);require(reader.read(rawroot/name,d,A.MAX_LEDGER if name.endswith('.jsonl') else A.MAX_JSON,owned=False)==b,'raw original/copy byte identity');raw[name]=b
        if name.endswith('.json'):data[name]=A.parse(b)
    require(data['OWNED_EXTERNAL_ANCHORS.json']==anchors,'actual fullV2record');A._check_V2_external_record(anchors,expected_binding=expected_external_binding,expected_gate_references=expected_gate_references,reader=reader)
    records=A.parse_ledger(raw['PHASE_OBSERVATIONS.jsonl']);phases=replay(records);supervisor=data['SUPERVISOR_RECEIPT.json'];success=False;missing=sorted({'worker_result.json','PHASE_ACCOUNTING_FINAL.json','IMPORT_LIBRARY_MEASUREMENT.json'}-names);measurement=None
    if not missing:
        worker=data['worker_result.json'];final=data['PHASE_ACCOUNTING_FINAL.json'];require(worker['scope']==final['scope']==SCOPE and final['operations']==phases['counts'] and final['markers']==len(records) and final['marker_bytes']==len(raw['PHASE_OBSERVATIONS.jsonl']) and final['closed'] is True and final['native_PASS'] is False,'independent ownledger final count/byte/scope join');status=worker['status'];require(status in ('MEASUREMENT_RECORDED_UNQUALIFIED','MEASUREMENT_FAILURE_PRESERVED') and final['status']==status,'source-fixed import outcome');success=status=='MEASUREMENT_RECORDED_UNQUALIFIED'
        require(worker['model_forwards']==worker['tensor_allocations']==0 and worker['user_CUDA_context_requested'] is False and worker['native_PASS'] is worker['native_authority_issued'] is False,'zero prohibited work or authority');require(worker['measurement_file_sha256']==hashlib.sha256(raw['IMPORT_LIBRARY_MEASUREMENT.json']).hexdigest() and worker['measurement_body_sha256']==data['IMPORT_LIBRARY_MEASUREMENT.json']['sha256'],'actual raw measurement report anchors')
        require(worker['import_only_operations']=={k:v for k,v in phases['counts'].items() if k!='worker_result_publication'},'provisional ownoperation snapshot before resultpublication')
        require(type(final['IO']) is dict and all(type(c) is dict and set(c)=={'attempted','returned','failed','pending'} and all(A.integer(n) for n in c.values()) and c['attempted']==c['returned']+c['failed']+c['pending'] for c in final['IO'].values()),'conserved originalwriterIO counters')
        if success:require(phases['roots']==list(OPS) and len(records)==22 and all(c=={'attempted':1,'returned':1,'failed':0,'pending':0} for c in phases['counts'].values()) and worker['errors']==[] and final['publication_errors']==[] and supervisor['returncode']==0 and supervisor['status']=='SUPERVISED_WORKER_EXITED_ZERO','exact successful11dose and originalzeroexit')
        if success:require(final['IO']=={'ledger_create':{'attempted':1,'returned':1,'failed':0,'pending':0},'marker_write':{'attempted':22,'returned':22,'failed':0,'pending':0},'ledger_close':{'attempted':1,'returned':1,'failed':0,'pending':0},'final_summary_write':{'attempted':1,'returned':1,'failed':0,'pending':0}},'exact successful original durableIO dose')
        else:require(any(c['failed'] for c in phases['counts'].values()) or worker['errors'] or final['publication_errors'],'actual partialfailure cause')
        auditor_path=PACKAGE/'import_observation_auditor.py';d=manifest['source_files'][str(auditor_path)];source=reader.read(auditor_path,{'bytes':d['bytes'],'sha256':d['sha256']},owned=False);auditor=ModuleType('selected_own_import_raw_auditor');auditor.__file__=str(auditor_path);exec(compile(source,str(auditor_path),'exec'),auditor.__dict__);runtime=manifest['runtime_value'];measurement=auditor.replay(raw['IMPORT_LIBRARY_MEASUREMENT.json'],policy=runtime['measurement_policy'],grammar=runtime['measurement_grammar'],identity=measurement_identity,files=runtime['measurement_policy']['library_files'],sources=manifest['source_files']);require(not success or measurement['status']=='MEASUREMENT_RECORDED_UNQUALIFIED' and measurement['native_qualification'] is False,'successfulimportrequiresindependentrawconservation')
    else:require(supervisor['returncode']!=0,'missing report/raw/final cannotzeroexitcomplete')
    source_pins=manifest['source_files'];snapshots=preservation['source_snapshot_files'];require(set(snapshots)==set(source_pins),'complete sourcecopy population')
    for text,d in source_pins.items():
        snap=snapshots[text];name=hashlib.sha256(text.encode()).hexdigest()+Path(text).suffix;require(snap['path']=='source_snapshots/'+name and snap['bytes']==d['bytes'] and snap['sha256']==d['sha256'],'source copy hash identity');reader.read(Path(text),{'bytes':d['bytes'],'sha256':d['sha256']},owned=False);reader.read(root/snap['path'],{'bytes':d['bytes'],'sha256':d['sha256']})
    namespace={**A.__dict__,'root':root,'reader':reader,'preservation':preservation,'data':data,'manifest':manifest,'anchors':anchors,'expected_head':anchors['anchors']['expected_head'],'replayed':phases,'raw':raw};births=_resource_source_blocks(namespace)
    require(A.finite(supervisor['worker_resume_monotonic']) and A.finite(supervisor['deadline_monotonic']) and
        supervisor['worker_resume_monotonic']<=phases['first_clock']<=phases['last_clock']<=supervisor['deadline_monotonic'],'ownphaseclocksinsideoriginalresume/deadline')
    if measurement is not None:
        packet=data['IMPORT_LIBRARY_MEASUREMENT.json'];lower,upper=A._library_clock_window_ns(supervisor['worker_resume_monotonic'],supervisor['deadline_monotonic']);require(all(type(row['time_ns']) is int and lower<=row['time_ns']<=upper for row in packet['observations']),'rawmeasurementclocksinsideoriginalownedwindow')
        require(measurement_identity.get('original_scope')==SCOPE and measurement_identity.get('source_commit')==anchors['anchors']['expected_head'] and measurement_identity.get('owner_receipt_sha256')==A.digest(data['supervisor_ownership.json']),'measurement identity binds originalscope/dispatch/ownedreceipt')
        born=measurement_identity.get('owned_birth');require(type(born) is dict and (born.get('pid'),born.get('creation_filetime_100ns')) in births,'measurement executingbirth in actualsampledownedpopulation')
    death=preservation['current_death_observation'];require(A.finite(death['observed_monotonic']) and death['observed_monotonic']>=phases['last_clock'] and death['supervisor_file_sha256']==preservation['raw_files']['SUPERVISOR_RECEIPT.json']['sha256'],'recordeddeath/SUPclockjoin');observed={}
    A.fields(death,{'observed_monotonic','observed_utc','observations','supervisor_file_sha256'},'originalimportpreservationdeath');require(type(death['observed_utc']) is str and len(death['observed_utc'])<=64,'boundedrecordeddeathUTC')
    for item in death['observations']:
        born=A.identity(item['identity']);key=(born['pid'],born['creation_filetime_100ns']);require(key not in observed and item['currently_alive'] is False,'exact unique recordedownedbirthdead');observed[key]=born
    require(observed==births,'complete allbirthdeath population');require(set(p.name for p in root.iterdir())==names|{'PRESERVATION.json','source_snapshots'}|{'lease_'+n for n in preservation['lease_files']},'exact preservedrawlease/source roster');require(set(p.name for p in (root/'source_snapshots').iterdir())=={Path(s['path']).name for s in snapshots.values()},'exact sourcesnapshot roster');reader.verify();require(set(p.name for p in rawroot.iterdir())==names,'original rawroster stable throughaudit');return A.seal({'kind':'INDEPENDENT_IMPORT_MEASUREMENT_TERMINAL_BYTES_AUDIT','scope':SCOPE,'status':'IMPORT_MEASUREMENT_TERMINAL_VALIDATED_UNQUALIFIED' if success else 'IMPORT_MEASUREMENT_PARTIAL_FAILURE_PRESERVED','missing_semantic_files':missing,'actual_phase_counts':phases['counts'],'measurement_replay':measurement,'owned_process_identities':list(births.values()),'native_fit_established':False,'model':False,'training_authorized':False,'scoring_authorized':False,'native_PASS':False,'native_authority_issued':False})
