"""Saved decoded-long worker and post-Job consumer, with native entry closed.

The scientific per-token and final-byte validator is extracted unchanged from
the selected original collector. The full original Journal/resource validators
handle the outer parity, decoder, asset and cleanup boundaries separately.
"""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import ast, hashlib, json, math, os, stat, time

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
EXPECTED=PACKAGE/'long_guard_terminal_audit_v1'
SCOPE='REGISTERED_LONG2304_M0_DECODED_RAW_COLLECTION_V1'
def require(ok,message):
    if not ok:raise ValueError(message)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
def seal(value):return {**value,'sha256':digest(value)}
def checksum(value):require(type(value) is dict and value.get('sha256')==digest({k:v for k,v in value.items() if k!='sha256'}),'canonical complete worker checksum')
def _cap_factory():
    token=object()
    return lambda:token,lambda value:require(value is token,'private authored terminal capability')
_CPU_capability,_check_cap=_cap_factory()

def originals(core,core_ref,auditor_ref,lease_root):
    """Reuse exact raw-prefix math and exact event/resource validator bodies."""
    raw=core._selected_bytes(core_ref,json_value=False)
    require(Path(core.__file__).resolve()==Path(core_ref['path']).resolve(),'same selected executing core source')
    tree=ast.parse(raw);fn=deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_audit_saved_batch'))
    indices=[i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='events_path' for t in n.targets)]
    require(len(indices)==1,'one original raw-prefix/full-Journal boundary')
    # Nothing in the scientific loop is rewritten. The owned runtime adds APIs
    # around that loop; its Journal is validated independently below.
    fn.body=fn.body[:indices[0]]+ast.parse("return {'rows':joined,'envelope':envelope,'attempted_forwards':cumulative_forwards,'input_positions':cumulative_positions}").body
    namespace=dict(core.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),core_ref['path'],'exec'),namespace)
    auditor_raw=core._selected_bytes(auditor_ref,json_value=False);tree=ast.parse(auditor_raw)
    names=('validate_events','verify_resources');defs=[deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)) for name in names]
    changes=0
    for node in ast.walk(defs[1]):
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div) and isinstance(node.left,ast.Name) and node.left.id=='PACKAGE' and isinstance(node.right,ast.Constant) and node.right.value=='artifacts/_native_gpu_leases':
            node.left.id='_external_root';node.right.value='_native_gpu_leases';changes+=1
    require(changes==1,'one original external lease location substitution')
    def info(path):
        value=Path(path);require(value.is_absolute() and value.resolve()==value,'selected exact absolute terminal file')
        for part in (value,*value.parents):
            if part.exists():
                observed=part.lstat();require(not part.is_symlink() and not (getattr(observed,'st_file_attributes',0)&0x400),'terminal path reparse/symlink forbidden before read')
        observed=value.lstat();require(stat.S_ISREG(observed.st_mode) and 0<observed.st_size<=64*1024**2,'bounded regular terminal file before read')
        with value.open('rb') as stream:
            before=os.fstat(stream.fileno());require(stat.S_ISREG(before.st_mode) and before.st_size==observed.st_size,'same opened bounded terminal file')
            raw=stream.read(before.st_size+1);after=os.fstat(stream.fileno())
        fields=lambda row:(row.st_dev,row.st_ino,row.st_size,row.st_mtime_ns)
        require(fields(observed)==fields(before)==fields(after)==fields(value.stat()) and len(raw)==before.st_size,'terminal bytes/path stable during bounded read')
        return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    def read_json(path,limit=16*1024**2):
        require(Path(path).stat().st_size<=limit,'bounded saved terminal JSON');return core._json(core._selected_bytes({'path':str(path),**info(path)},json_value=False))
    def safe_child(path,leaf):
        path=Path(path);require(path.is_absolute() and path.parent==Path(leaf) and path.resolve()==path,'exact original owned evidence leaf');info(path);return path
    ns=dict(Path=Path,datetime=datetime,json=json,math=math,deepcopy=deepcopy,require=require,
        read_json=read_json,file_info=info,safe_child=safe_child,MAX_EVENTS_BYTES=32*1024**2,
        PACKAGE=PACKAGE,_external_root=Path(lease_root).parent,
        contract=SimpleNamespace(native=SimpleNamespace(_pairs=lambda items:_pairs(items),_bad_constant=lambda v:require(False,'nonfinite saved terminal JSON'))))
    exec(compile(ast.fix_missing_locations(ast.Module(body=defs,type_ignores=[])),auditor_ref['path'],'exec'),ns)
    return namespace['_audit_saved_batch'],ns,{'raw_prefix_source_sha256':core_ref['sha256'],'resource_source_sha256':auditor_ref['sha256'],'external_lease_substitutions':changes}
def _pairs(items):
    result={}
    for k,v in items:require(k not in result,'duplicate saved terminal JSON key');result[k]=v
    return result
def births(sup,resources):
    roster=[sup['worker_identity']]
    for row in resources:
        if row['kind']=='CPU':roster.extend(row['owned_process_identities'])
    result={};pids={}
    for row in roster:
        birth={k:row[k] for k in ('pid','creation_filetime_100ns')}
        require(all(type(v) is int and 0<v<2**64 for v in birth.values()),'typed exact owned launcher/interpreter births')
        require(birth['pid'] not in pids or pids[birth['pid']]==birth['creation_filetime_100ns'],'conflicting exact observed birth')
        pids[birth['pid']]=birth['creation_filetime_100ns'];result[tuple(birth.values())]=birth
    require(0<len(result)<=4096,'finite nonempty observed owned birth roster');return list(result.values())

def causal_journal(events,raw_events,saved,worker,checked):
    """Exact fixture lifecycle, including actual decoded boundaries and sinks."""
    require(not events['publications'],'decoded-long never accepts STUDY checkpoint publications')
    accounting=worker['operation_accounting']
    require(accounting=={'scope':'registered_API_boundaries;internal_ATen_operations_not_individually_metered','operations':events['counts'],'last_call':events['API_calls']},'whole enriched physical Journal accounting')
    expected=[];identity=0
    def call(name):
        nonlocal identity
        identity+=1;expected.extend([('attempt',identity,name),('returned',identity,name)])
    def measurement(ref,name):
        require(Path(ref['path']).name==name,'exact original partial/final filename')
        expected.append(('measurement_preserved',name,ref['sha256']))
    for name in ('install_fp32_compute','install_activation_hooks','install_physical_forward_meter','zero_parity_base_forward','install_qv_lora','zero_lora_query_before','zero_parity_adapter_forward'):call(name)
    parity=next(e for e in raw_events if e.get('path')=='stage_long_zero_parity.json')
    expected.append(('measurement_preserved',parity['path'],parity['sha256']))
    for ordinal,(row_ref,token_refs) in enumerate(zip(saved['rows'],saved['tokens']),1):
        for step,ref in enumerate(token_refs,1):call('inference_forward');measurement(ref,'stage_raw_row%04d_token%02d.json'%(ordinal,step))
        call('generated_only_decode');measurement(row_ref,'stage_inference_row_%04d.json'%ordinal)
    measurement(saved['batch'],'stage_long_guard_batch.json')
    for name in ('zero_lora_query_after','verify_frozen_base','generated_decoder_postcheck','owned_hooks_remove'):call(name)
    require(len(raw_events)==len(expected),'complete enriched decoded Journal causal roster')
    for event,want in zip(raw_events,expected):
        if want[0]=='measurement_preserved':
            require(set(event)=={'event','utc','path','sha256','qualification_established'} and event['qualification_established'] is False and (event['event'],event['path'],event['sha256'])==want,'exact saved measurement causal byte join')
        else:
            require(set(event)=={'event','utc','id','operation'} and type(event['id']) is int and (event['event'],event['id'],event['operation'])==want,'exact physical parity/decode/cleanup Journal order')
    require(events['counts']['inference_forward']=={'attempted':checked['attempted_forwards'],'returned':checked['attempted_forwards'],'failed':0,'pending':0},'raw costs join every physical inference API')
    require(events['counts']['generated_only_decode']=={'attempted':64,'returned':64,'failed':0,'pending':0},'one successful generated-only decoder per original raw row')

def consume(packet,core,component,host,*,fixture_only):
    require(fixture_only is True and type(packet) is dict and packet.get('fixture_only') is True,'only explicit authored terminal packet; genuine decoded qualification absent')
    selected={}
    def read(key):
        ref=packet[key];value=core.read_ref(ref);selected[ref['path']]=deepcopy(ref);return value
    config=read('config');source=read('source');plan=read('plan');worker=read('worker');sup=read('supervisor')
    require(type(config) is dict and config.get('fixture_only') is True and config.get('scope')==SCOPE and type(config.get('ordinal')) is int and 1<=config['ordinal']<=36,'exact authored runtime config scope/typed ordinal')
    ordinal=config['ordinal'];core.check_plan(source,plan,component.validate_row);batch=plan['batches'][ordinal-1]
    require(config['dose']=={'maximum_forward_calls':2050,'maximum_forward_input_positions':838+batch['maximum_input_positions'],'context_cap':batch['context_cap'],'maximum_generated_tokens':32,'rows':64,'optimizer_steps':0,'backwards':0},'original physical long64 dose unchanged')
    component.validate_row(config['short_parity_row'],context_cap=512,generation_cap=32);require(config['short_parity_row']['prompt_tokens']==419,'original exact short419 parity dose')
    checksum(worker)
    require(worker.get('status')=='COMPLETE_PENDING_INDEPENDENT_LONG_DECODED_AUDIT' and worker.get('scope')==SCOPE and worker.get('fixture_only') is True and worker.get('error') is None and worker.get('requires_independent_postJob_audit') is True,'complete scoped original worker pending evidence')
    require(all(worker.get(k) is False for k in ('native_PASS','scoring_authorized','retention_certified','native_qualification_established')) and type(worker.get('optimizer_steps')) is type(worker.get('backwards')) is int and worker['optimizer_steps']==worker['backwards']==0,'unchanged M0 no optimizer/backward/scoring/qualification')
    require(worker.get('cleanup_errors')==[] and 'cleanup_boundary_error' not in worker,'successful actual owned hooks cleanup')
    require(worker['zero_before']['all_B_zero'] is worker['zero_after']['all_B_zero'] is True and worker['base_postcheck']['verified_source_tensors']==310 and worker['base_postcheck']['frozen_base_unchanged'] is True,'original zeroB unchanged/frozen310 evidence')
    require(type(worker['activation_observations']) is dict and worker['activation_observations'] and all(type(v) is int and v>0 for v in worker['activation_observations'].values()),'typed positive original activation observations')
    leaf=Path(packet['output_directory']);require(leaf.is_absolute() and leaf.is_dir() and leaf.resolve()==leaf,'selected original terminal leaf')
    require(Path(packet['worker']['path'])==leaf/'worker_result.json' and Path(packet['supervisor']['path'])==leaf/'SUPERVISOR_RECEIPT.json','original terminal filenames')
    require(sup.get('status')=='SUPERVISED_WORKER_EXITED_ZERO' and type(sup.get('returncode')) is int and sup['returncode']==0 and sup.get('partial_outputs_frozen') is True and sup.get('config_sha256')==packet['config']['sha256'],'successful original frozen config-bound supervisor')
    for ref in packet['sources'].values():
        core._selected_bytes(ref,json_value=False);selected[ref['path']]=deepcopy(ref)
    require(set(packet['sources'])==set(config['source_pins']) and all(k==v['path'] and v['sha256']==config['source_pins'][k] for k,v in packet['sources'].items()),'selected current source path/byte closure')
    require(packet['core_source']==packet['sources'].get(packet['core_source']['path']) and packet['auditor_source']==packet['sources'].get(str(PACKAGE/'auditor.py')),'same executing original selected core and resource source')
    raw_validator,original,adaptation=originals(core,packet['core_source'],packet['auditor_source'],packet['lease_root'])
    saved=packet['saved'];require(Path(saved['events']['path'])==leaf/'events.jsonl','original Journal selected path')
    checked=raw_validator(source,plan,ordinal,saved,fixture_only=True)
    require(checked['envelope']==worker['batch'],'original saved batch and terminal worker exact value join')
    dose=worker['forward_dose'];require(dose=={'attempted_calls':checked['attempted_forwards']+2,'input_positions':checked['input_positions']+838,'operations':{'zero_parity_base_forward':{'attempted_calls':1,'input_positions':419},'zero_parity_adapter_forward':{'attempted_calls':1,'input_positions':419},'inference_forward':{'attempted_calls':checked['attempted_forwards'],'input_positions':checked['input_positions']}},'maximum_forward_calls':2050,'maximum_forward_input_positions':838+batch['maximum_input_positions']},'exact physical2parity/capture cost and maximum join')
    events=original['validate_events'](leaf/'events.jsonl',leaf)
    raw_events=core._selected_bytes(saved['events'],json_value=False);require(raw_events.endswith(b'\n'),'complete original Journal final newline')
    causal_journal(events,[core._json(line) for line in raw_events.splitlines()],saved,worker,checked)
    parity=original['read_json'](leaf/'stage_long_zero_parity.json');require(parity=={'equal':True,'maximum_error':0.0,'fixture_only':True,'native_qualification_established':False},'unqualified exact original parity evidence')
    resources=original['verify_resources'](config,sup,leaf)
    raw=packet['raw_files'];listed=sup['partial_outputs'];require(type(raw) is dict and raw and len(raw)<=4096 and len(listed)==len({v['path'] for v in listed}),'finite exact original raw inventory')
    for item in listed:
        path=original['safe_child'](str(leaf/item['path']),leaf);require(original['file_info'](path)=={k:item[k] for k in ('bytes','sha256')}==raw.get(item['path']),'every actual SUP-listed original file byte join')
    require(set(raw)=={v['path'] for v in listed}|{'SUPERVISOR_RECEIPT.json'} and raw['SUPERVISOR_RECEIPT.json']==original['file_info'](leaf/'SUPERVISOR_RECEIPT.json'),'complete frozen raw inventory plus SUP')
    require({p.name for p in leaf.iterdir()}==set(raw),'actual original leaf no omitted/foreign files')
    lease=Path(sup['lease_path']);require({p.name for p in lease.iterdir()}=={'reservation.json','worker.json','resume.json','terminal.json'},'exact original four lease files')
    lease_before={p.name:original['file_info'](p) for p in lease.iterdir()}
    roster=births(sup,original['read_json'](leaf/'RESOURCE_OBSERVATIONS.json'));ownership=original['read_json'](leaf/'stage_worker_ownership_verified.json')
    require(ownership['actual_worker_pid'] in {b['pid'] for b in roster},'actual execution interpreter observed separately from launcher')
    require(type(host) is not dict and callable(getattr(host,'is_alive',None)),'fresh source-bound host interface; no Boolean roster')
    observed=[];start=time.monotonic()
    for birth in roster:
        alive=host.is_alive(deepcopy(birth));require(type(alive) is bool and alive is False,'fresh exact owned birth live/unknown');observed.append({'birth':birth,'alive':alive})
    require(time.monotonic()-start<=30,'bounded fresh post-Job observations')
    for name,value in raw.items():require(original['file_info'](leaf/name)==value,'raw bytes stable after fresh death observations')
    for ref in selected.values():core._selected_bytes(ref,json_value='value_sha256' in ref)
    core._recheck_saved(saved)
    require({p.name:original['file_info'](p) for p in lease.iterdir()}==lease_before and {p.name for p in leaf.iterdir()}==set(raw),'lease/raw bytes and rosters stable after fresh death observations')
    return seal({'kind':'POST_JOB_DECODED_LONG_SAVED_BYTE_AUDIT_PENDING_QUALIFICATION','scope':SCOPE,'ordinal':ordinal,'source_packet_sha256':batch['source_packet_sha256'],'plan_sha256':plan['sha256'],'worker_file_sha256':packet['worker']['sha256'],'supervisor_file_sha256':packet['supervisor']['sha256'],'physical_forward_dose':dose,'resource_summary':resources,'observed_birth_deaths':observed,'original_validator_adaptations':adaptation,'fixture_only':True,'native_PASS':False,'native_qualification_established':False,'scoring_authorized':False,'retention_certified':False})

def _CPU_consume(packet,core,component,host,*,capability):
    _check_cap(capability);return consume(packet,core,component,host,fixture_only=True)
def audit_registered(packet_ref):
    require(HERE==EXPECTED,'staged decoded-long terminal audit has no native authority')
    raise ValueError('genuine independent decoded qualification and owner-approved byte integration absent; native audit closed')
