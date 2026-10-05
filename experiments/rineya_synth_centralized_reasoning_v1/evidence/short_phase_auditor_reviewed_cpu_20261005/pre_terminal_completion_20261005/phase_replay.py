"""Independent Phase3 marker parser; never imports production phase validators."""
from datetime import datetime
import hashlib,json,math
START={'ordinal','operation','event','monotonic','utc','historical_cause_established'}
END=START|{'result','error'}
LIMITS={'maximum_markers':200000,'maximum_marker_bytes':65536,'maximum_ledger_bytes':64*1024**2}
MAIN=('initial_config_preflight','disk_admission','torch_import','Qwen3Config_symbol_import','Qwen3ForCausalLM_symbol_import',
    'framework_version_read','post_import_growth_observation','direct_CUDA_preload_mem_get_info','precision_and_thread_configuration',
    'Python_seed_101','Torch_CPU_seed_101','Torch_CUDA_seed_all_101','CUDA_reset_peak_memory_stats','load_titan',
    'forward_dose_construct','collect_loaded','final_config_preflight','preserve_terminal')
FIXED={'checkpoint_load':1,'cuda_context_init':1,'generated_decoder_construct':1,'install_activation_hooks':1,
    'install_fp32_compute':1,'install_qv_lora':1,'model_construct':1,'tensor_copy':310,'verify_frozen_base':1,
    'zero_lora_query_after':1,'zero_lora_query_before':1,'zero_parity_adapter_forward':1,'zero_parity_base_forward':1}

def require(value,message):
    if not value:raise ValueError(message)
def canonical(value):return json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode('ascii')).hexdigest()
def pairs(items):
    result={}
    for key,value in items:require(key not in result,'duplicate JSON key');result[key]=value
    return result
def parse(raw):
    require(type(raw) is bytes and len(raw)<=LIMITS['maximum_ledger_bytes'],'bounded ledger bytes')
    result=[]
    for line in raw.splitlines(keepends=True):
        require(line.endswith(b'\n') and len(line)<=LIMITS['maximum_marker_bytes'] and len(result)<LIMITS['maximum_markers'],'complete bounded marker line')
        result.append(json.loads(line.decode('ascii'),object_pairs_hook=pairs,
            parse_constant=lambda text:(_ for _ in ()).throw(ValueError('nonfinite JSON literal'))))
    return result
def zero():return {'attempted':0,'returned':0,'failed':0,'pending':0}
def complete(value):
    return (type(value) is dict and set(value)==set(zero()) and all(type(v) is int and v>=0 for v in value.values())
        and value['attempted']==value['returned'] and value['failed']==value['pending']==0)

def tree(markers):
    require(type(markers) is list and 0<len(markers)<=LIMITS['maximum_markers'],'nonempty complete marker population')
    nodes={};roots=[];open_ids=[];cost={};previous_time=previous_utc=None
    for marker in markers:
        require(type(marker) is dict and set(marker)=={'name','phase'},'exact marker schema')
        p=marker['phase'];require(type(p) is dict and set(p)==(START if p.get('event')=='attempt' else END),'strict start/end schema')
        oid=p['ordinal'];op=p['operation'];now=p['monotonic']
        require(type(oid) is int and oid>0 and type(op) is str and 0<len(op)<=1024
            and type(now) in (int,float) and math.isfinite(now) and p['historical_cause_established'] is False,'typed finite phase identity')
        utc=datetime.fromisoformat(p['utc']);require(utc.tzinfo is not None,'aware UTC')
        require((previous_time is None or now>=previous_time) and (previous_utc is None or utc>=previous_utc),'record chronology')
        previous_time,previous_utc=now,utc;counter=cost.setdefault(op,zero())
        if p['event']=='attempt':
            require(oid==len(nodes)+1 and marker['name']=='phase_%04d_start'%oid,'unique contiguous marker start/name')
            node={'start':p,'end':None,'children':[]};nodes[oid]=node
            (nodes[open_ids[-1]]['children'] if open_ids else roots).append(oid)
            open_ids.append(oid);counter['attempted']+=1;counter['pending']+=1
        else:
            require(p['event']=='returned' and p['error'] is None and open_ids and open_ids[-1]==oid
                and marker['name']=='phase_%04d_end'%oid and nodes[oid]['start']['operation']==op,'exact complete nested marker return')
            nodes[oid]['end']=p;open_ids.pop();counter['pending']-=1;counter['returned']+=1
    require(not open_ids and all(complete(c) for c in cost.values()),'no failed/pending/unclosed phase')
    return nodes,roots,cost

def memory(value,floor):
    require(type(value) is list and len(value)==2 and all(type(v) is int and 0<=v<2**64 for v in value)
        and value[0]<=value[1] and value[0]>=floor,'typed returned CUDA free/total and unchanged floor')

def replay(markers,summary,worker,journal,*,fixture=False):
    nodes,roots,cost=tree(markers)
    require(summary['kind']=='SYNTH_SHORT_PHASE_ACCOUNTING_V1' and summary['sha256']==digest({k:v for k,v in summary.items() if k!='sha256'}),'final canonical checksum')
    require(summary['status']=='COMPLETE_PHASE_OBSERVATION' and worker['status']=='COMPLETE_SCOPED_SHORT_M0_RAW_COLLECTION_UNSCORED'
        and worker.get('error') is None and summary['primary_error'] is summary['terminal_error'] is summary['close_error'] is None,'paired raw/final complete status')
    require(summary['identity']['execution_scope']==('CPU_FIXTURE_PHASE_COLLECTION_PROTOCOL' if fixture else 'OWNED_SHORT_M0_PHASE_OBSERVED_RAW_COLLECTION'),
        'CPU fixture cannot produce native audit authority')
    account=summary['phase_accounting'];require(account['operations']==cost and account['last_ordinal']==len(nodes)
        and account['last_phase']==markers[-1]['phase'] and account['publication_errors']==[],'complete independent phase arithmetic')
    io=summary['phase_IO'];require(io['closed'] is True and set(io['operations'])=={'phase_ledger_create','phase_marker_write','phase_ledger_close','phase_final_summary_write'}
        and all(complete(x) for x in io['operations'].values()),'closed exact IO roster')
    require(summary['limits']==LIMITS and io['markers_returned']==len(markers)==io['operations']['phase_marker_write']['returned']
        and all(io['operations'][name]['returned']==1 for name in ('phase_ledger_create','phase_ledger_close','phase_final_summary_write')),'all actual marker/final writes')
    size=sum(len((canonical(v)+'\n').encode('ascii')) for v in markers)
    require(io['marker_bytes_attempted']==io['marker_bytes_returned']==size,'independent encoded byte conservation')
    starts=[nodes[i]['start']['operation'] for i in range(1,len(nodes)+1)]
    main=[x[5:] for x in starts if x.startswith('main_')]
    require(main==list(MAIN[1:] if fixture else MAIN),'literal source main phase order')
    init_queries=memory_queries=guards=0
    for node in nodes.values():
        name=node['start']['operation'];kids=[nodes[x] for x in node['children']]
        names=[x['start']['operation'] for x in kids]
        if name.startswith('guard_') and name not in ('guard_cuda_is_initialized','guard_cuda_mem_get_info'):
            guards+=1
            if name=='guard_pre_import':require(not kids,'preimport holder must be None')
            else:
                require(names in (['guard_cuda_is_initialized'],['guard_cuda_is_initialized','guard_cuda_mem_get_info']),'source guard exact query order')
                initialized=kids[0]['end']['result'];require(type(initialized) is bool,'literal boolean initialization')
                require(len(kids)==(2 if initialized else 1),'false skips / true queries once')
                init_queries+=1
                if initialized:memory(kids[1]['end']['result'],.5*1024**3);memory_queries+=1
        if name.startswith('api_'):
            op=name[4:];require(names==['guard_before_'+op,'primitive_'+op,'guard_after_'+op],'exact API before/primitive/after tree')
    require(cost.get('guard_cuda_is_initialized',zero())['returned']==init_queries
        and cost.get('guard_cuda_mem_get_info',zero())['returned']==memory_queries,'all queries owned by original guard')
    for node in nodes.values():
        name=node['start']['operation']
        if name=='main_direct_CUDA_preload_mem_get_info':memory(node['end']['result'],4*1024**3)
        if name=='terminal_cuda_mem_get_info':memory(node['end']['result'],.5*1024**3)
    api=[x[4:] for x in starts if x.startswith('api_')];primitives=[x[10:] for x in starts if x.startswith('primitive_')]
    attempted=[e for e in journal if e['event']=='attempt'];returned=[e for e in journal if e['event']=='returned']
    require(api==primitives==[e['operation'] for e in attempted] and [e['id'] for e in attempted]==list(range(1,len(attempted)+1)),
        'independent actual Journal/API/primitive sequence')
    terminal={e['id']:e for e in returned};require(len(terminal)==len(returned)==len(attempted)
        and all(terminal[e['id']]['operation']==e['operation'] for e in attempted)
        and not any(e['event']=='failed' for e in journal),'actual Journal terminal conservation')
    api_cost={}
    for e in attempted:
        c=api_cost.setdefault(e['operation'],zero());c['attempted']+=1;c['returned']+=1
    require(worker['operation_accounting']['operations']==api_cost and worker['operation_accounting']['last_call']==len(attempted),'actual worker ledger')
    for op,counter in api_cost.items():require(cost['api_'+op]==cost['primitive_'+op]==counter,'same complete observed primitive '+op)
    if not fixture:
        expected={**FIXED,'inference_forward':worker['collection_costs']['attempted_forwards']}
        require(api_cost=={op:{'attempted':n,'returned':n,'failed':0,'pending':0} for op,n in expected.items()},'native original322 plus capture; no fixture constructors')
    pubs=['stage_'+x[12:]+'.json' for x in starts if x.startswith('publication_')]
    require(pubs==[e['path'] for e in journal if e['event']=='measurement_preserved'],'every raw publication exact sequence')
    final_nodes=[n for n in nodes.values() if n['start']['operation']=='main_preserve_terminal']
    require(len(final_nodes)==1,'single source-defined terminal preservation')
    terminal_children=[nodes[x] for x in final_nodes[0]['children']]
    names=[n['start']['operation'] for n in terminal_children]
    require(names in (['terminal_cuda_is_initialized'],['terminal_cuda_is_initialized','terminal_cuda_max_memory_allocated',
        'terminal_cuda_max_memory_reserved','terminal_cuda_mem_get_info']),'exact original optional terminal queries')
    initialized=terminal_children[0]['end']['result'];require(type(initialized) is bool and len(names)==(4 if initialized else 1),'terminal initialized query branch')
    return {'status':'PASS_CPU_FIXTURE_PHASE_REPLAY_ONLY' if fixture else 'PASS_INDEPENDENT_PHASE_LEDGER_ONLY_PENDING_FULL_NATIVE_RAW_AUDIT',
        'phases':len(nodes),'markers':len(markers),'guard_boundaries':guards,'init_queries':init_queries,'memory_queries':memory_queries,
        'native_raw_audit_complete':False,'scoring_authorized':False,'fixture_only':fixture}

def loader_tree(markers,loader,shapes,*,chunk_bytes=2*1024**2):
    """Independent exact helper-tree replay from pinned sizes/shapes, no math."""
    nodes,roots,cost=tree(markers)
    def operation(oid):return nodes[oid]['start']['operation']
    def direct(oid):return nodes[oid]['children']
    def unique(name):
        found=[oid for oid in nodes if operation(oid)==name];require(len(found)==1,'one source-defined '+name);return found[0]
    def expect_guard(oid,name):require(operation(oid)=='guard_'+name,'source literal guard '+name)
    def chunks(name):
        import math
        return math.ceil(math.prod(shapes[name])*2/chunk_bytes)
    load=unique('main_load_titan');ids=direct(load);cursor=0
    def take():
        nonlocal cursor
        require(cursor<len(ids),'complete loader tree');value=ids[cursor];cursor+=1;return value
    expect_guard(take(),'load_titan_before')
    for name,entry in loader['asset_hashes'].items():
        require(type(entry['bytes']) is int and entry['bytes']>=0,'independently verified asset byte metadata')
        n=(entry['bytes']+chunk_bytes-1)//chunk_bytes+1
        for _ in range(n):expect_guard(take(),'asset_hash_before:'+name);expect_guard(take(),'asset_hash_after:'+name)
    expect_guard(take(),'config_load_before');expect_guard(take(),'config_load_after')
    construct=take();require(operation(construct)=='api_model_construct','source model construct placement')
    primitive=direct(construct)[1];kids=direct(primitive)
    require(len(kids)==3,'model construction exact3 internal guards')
    for oid,name in zip(kids,('model_construct_before','model_to_empty_before','model_construct_after')):expect_guard(oid,name)
    checkpoint=take();require(operation(checkpoint)=='api_checkpoint_load','source checkpoint placement')
    ids2=direct(direct(checkpoint)[1]);require(ids2 and operation(ids2[0])=='guard_checkpoint_load_before','checkpoint start')
    position=1;seen=[]
    for _ in range(310):
        require(position<len(ids2) and operation(ids2[position])=='api_tensor_copy','one tensor API per source tensor')
        tensor=ids2[position];position+=1;inside=direct(direct(tensor)[1]);require(inside,'nonempty actual tensor copy')
        first=operation(inside[0]);require(first.startswith('guard_tensor_copy_before:'),'typed tensor copy name')
        name=first[len('guard_tensor_copy_before:'):];require(name in shapes and name not in seen,'all310 unique source tensor names');seen.append(name)
        count=chunks(name);require(len(inside)==2*count,'every source-defined tensor copy chunk')
        for index in range(count):expect_guard(inside[2*index],'tensor_copy_before:'+name);expect_guard(inside[2*index+1],'tensor_copy_after:'+name)
        for index in range(count):expect_guard(ids2[position],'device_digest_before:'+name);expect_guard(ids2[position+1],'device_digest_after:'+name);position+=2
    require(set(seen)==set(shapes) and position+1==len(ids2),'complete source tensor/copy/digest denominator')
    expect_guard(ids2[position],'checkpoint_load_after');expect_guard(take(),'load_titan_after');require(cursor==len(ids),'no hidden loader helper guard/API')
    verify=unique('primitive_verify_frozen_base');ids3=direct(verify);position=0
    expect_guard(ids3[position],'verify_frozen_base_before');position+=1
    seen2=[]
    while position<len(ids3)-1:
        name=operation(ids3[position])[len('guard_device_digest_before:'):]
        require(name in shapes and name not in seen2,'all310 second device digest names');seen2.append(name)
        for _ in range(chunks(name)):expect_guard(ids3[position],'device_digest_before:'+name);expect_guard(ids3[position+1],'device_digest_after:'+name);position+=2
    require(set(seen2)==set(shapes) and position+1==len(ids3),'complete final310 digests')
    expect_guard(ids3[position],'verify_frozen_base_after')
    return {'all310_copy_first_digest_and_second_digest_guard_trees_verified':True,'loader_phase_not_model_execution':True}
