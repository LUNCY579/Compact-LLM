"""Independent Phase3 marker parser; never imports production phase validators."""
from datetime import datetime
from pathlib import Path
import ast,hashlib,json,math,stat,struct
START={'ordinal','operation','event','monotonic','utc','historical_cause_established'}
END=START|{'result','error'}
LIMITS={'maximum_markers':200000,'maximum_marker_bytes':65536,'maximum_ledger_bytes':64*1024**2}
MAX_JSON=16*1024**2
LOADER_SOURCE_SHA='43229ee3ab59719ddc076ce252fdab029a9b5f43c11c66e9ccc84d092f378174'
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
def bad_number(text):raise ValueError('nonfinite JSON number')
def finite_float(text):
    value=float(text);require(math.isfinite(value),'finite JSON numeric value');return value
def loads(raw):
    try:return json.loads(raw.decode('utf-8'),object_pairs_hook=pairs,parse_float=finite_float,parse_constant=bad_number)
    except (UnicodeError,RecursionError) as error:raise ValueError('invalid bounded JSON') from error
def regular(path):
    path=Path(path).absolute();entry=path.lstat()
    require(stat.S_ISREG(entry.st_mode) and not path.is_symlink()
        and not (getattr(entry,'st_file_attributes',0)&0x400),'regular unaliased evidence file')
    # An ancestor alias can escape a nominal owned directory on Windows.
    for parent in path.parents:
        item=parent.lstat();require(not parent.is_symlink() and not (getattr(item,'st_file_attributes',0)&0x400),
            'unaliased evidence ancestor')
    return path,entry
def read_bytes(path,limit=MAX_JSON):
    require(type(limit) is int and 0<limit<=LIMITS['maximum_ledger_bytes'],'finite reader bound')
    path,before=regular(path);require(before.st_size<=limit,'bounded evidence file exceeded')
    with path.open('rb') as stream:raw=stream.read(limit+1)
    require(len(raw)==before.st_size<=limit and path.stat().st_size==before.st_size,'bounded complete stable evidence read')
    return raw
def read_json(path,limit=MAX_JSON):return loads(read_bytes(path,limit))
def read_lines(path,*,maximum_lines=200000,canonical_lines=False):
    path,before=regular(path);require(before.st_size<=LIMITS['maximum_ledger_bytes'],'bounded JSONL file exceeded')
    result=[];total=0
    with path.open('rb') as stream:
        while True:
            raw=stream.readline(LIMITS['maximum_marker_bytes']+1)
            if not raw:break
            total+=len(raw);require(raw.endswith(b'\n') and len(raw)<=LIMITS['maximum_marker_bytes']
                and len(result)<maximum_lines and total<=LIMITS['maximum_ledger_bytes'],'complete bounded JSONL line/count')
            value=loads(raw)
            require(not canonical_lines or raw==(canonical(value)+'\n').encode('ascii'),'exact canonical phase ledger bytes')
            result.append(value)
    require(total==before.st_size==path.stat().st_size,'stable complete JSONL file');return result
def file_info(path):
    path,before=regular(path);hasher=hashlib.sha256();total=0
    with path.open('rb') as stream:
        while True:
            chunk=stream.read(2*1024**2)
            if not chunk:break
            total+=len(chunk);require(total<=before.st_size,'file grew during bounded streaming hash');hasher.update(chunk)
    require(total==before.st_size==path.stat().st_size,'complete stable streaming byte inventory')
    return {'bytes':total,'sha256':hasher.hexdigest()}
def parse(raw):
    require(type(raw) is bytes and len(raw)<=LIMITS['maximum_ledger_bytes'],'bounded ledger bytes')
    result=[]
    for line in raw.splitlines(keepends=True):
        require(line.endswith(b'\n') and len(line)<=LIMITS['maximum_marker_bytes'] and len(result)<LIMITS['maximum_markers'],'complete bounded marker line')
        result.append(loads(line))
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

def topology(nodes,roots,worker,*,fixture=False):
    """Exact successful caller/collection placement, independently transcribed."""
    def names(ids):return [nodes[i]['start']['operation'] for i in ids]
    mains=['main_'+x for x in MAIN]
    minimal_fixture=fixture and worker.get('loader',{}).get('scope')=='CPU_FIXTURE'
    expected=([] if minimal_fixture else ['publication_worker_ownership_verified',mains[0]])+['guard_pre_import']+mains[1:7]
    expected+=['api_cuda_context_init']+mains[7:14]+['publication_loader','api_generated_decoder_construct']+mains[14:17]
    expected+=['guard_completed',mains[17]]
    require(names(roots)==expected,'literal complete caller root placement/order')
    by_name={name:[i for i in nodes if nodes[i]['start']['operation']==name] for name in names(roots)}
    collect=nodes[by_name['main_collect_loaded'][0]]['children']
    wanted=['api_install_fp32_compute','api_install_activation_hooks','api_zero_parity_base_forward',
        'api_install_qv_lora','api_zero_lora_query_before','api_zero_parity_adapter_forward','publication_zero_adapter_parity']
    outputs=worker['collection']['rows'];require(len(outputs)==64,'whole64 phase collection denominator')
    for row,output in enumerate(outputs,1):
        ids=output['raw_generated_ids'];require(type(ids) is list and 1<=len(ids)<=32,'bounded phase output length')
        for step in range(1,len(ids)+1):wanted+=['guard_before_inference_forward','api_inference_forward',
            'guard_after_inference_forward','publication_raw_row%04d_token%02d'%(row,step)]
        wanted+=['publication_inference_row_%04d'%row]
    wanted+=['api_zero_lora_query_after','api_verify_frozen_base','publication_short_m0_collection_completed']
    require(names(collect)==wanted,'literal component guard/API/token/final-row/publication order')
    for i,node in nodes.items():
        name=node['start']['operation'];children=node['children']
        if name.startswith('publication_') or name in ('guard_cuda_is_initialized','guard_cuda_mem_get_info') \
                or name.startswith('terminal_cuda_'):
            require(not children,'leaf-only publication/query')
        if name.startswith('main_') and name not in ('main_load_titan','main_collect_loaded','main_preserve_terminal'):
            require(not children,'leaf-only caller operation')
        if name.startswith('primitive_') and name not in ('primitive_model_construct','primitive_checkpoint_load','primitive_verify_frozen_base'):
            require(not children,'leaf-only original primitive')
    if fixture and worker.get('loader',{}).get('scope')=='CPU_FIXTURE':
        require(names(nodes[by_name['main_load_titan'][0]]['children'])==['api_model_construct'],'explicit fixture loader topology')
        for name in ('primitive_model_construct','primitive_verify_frozen_base'):
            require(all(not node['children'] for node in nodes.values() if node['start']['operation']==name),'explicit fixture primitive topology')

def replay(markers,summary,worker,journal,*,fixture=False,scope=None,terminal=None):
    nodes,roots,cost=tree(markers)
    require(summary['kind']=='SYNTH_SHORT_PHASE_ACCOUNTING_V1' and summary['sha256']==digest({k:v for k,v in summary.items() if k!='sha256'}),'final canonical checksum')
    require(summary['status']=='COMPLETE_PHASE_OBSERVATION' and worker['status']=='COMPLETE_SCOPED_SHORT_M0_RAW_COLLECTION_UNSCORED'
        and worker.get('error') is None and summary['primary_error'] is summary['terminal_error'] is summary['close_error'] is None,'paired raw/final complete status')
    require(summary['identity']['execution_scope']==('CPU_FIXTURE_PHASE_COLLECTION_PROTOCOL' if fixture else 'OWNED_SHORT_M0_PHASE_OBSERVED_RAW_COLLECTION'),
        'CPU fixture cannot produce native audit authority')
    require(worker['scope']==summary['identity']['purpose'] and worker['config_sha256']==summary['identity']['config_sha256']
        and worker['source_commit']==summary['identity']['source_commit'],'exact worker scope/config/commit phase joins')
    require(all(summary.get(k) is False for k in ('scoring_authorized','retention_certified','historical_cause_established','native_fit_established')),
        'all final phase authority flags closed')
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
    require(main==list(MAIN[1:] if fixture and worker.get('loader',{}).get('scope')=='CPU_FIXTURE' else MAIN),'literal source main phase order')
    topology(nodes,roots,worker,fixture=fixture)
    require(all(type(e) is dict and e.get('event') in ('attempt','returned','measurement_preserved') for e in journal),
        'closed successful Journal event types')
    for node in nodes.values():
        name=node['start']['operation'];result=node['end']['result']
        if name.startswith(('api_','primitive_','publication_')) or (name.startswith('guard_') and name not in ('guard_cuda_is_initialized','guard_cuda_mem_get_info')) \
                or name.startswith('main_') and name not in ('main_disk_admission','main_framework_version_read','main_post_import_growth_observation','main_direct_CUDA_preload_mem_get_info'):
            require(result is None,'exact successful None result schema')
        if name=='main_disk_admission':require(type(result) is int and result>=4*1024**3,'observed disk admission floor')
        if name=='main_framework_version_read':
            expected={k:scope['runtime_versions'][k] for k in ('torch','transformers')} if scope is not None else {'torch':'2.6.0+cu124','transformers':'5.14.1'}
            require(result==expected,'exact observed registered framework versions')
        if name=='main_post_import_growth_observation':
            require(type(result) is dict and set(result)=={'available_bytes','worker_RSS_bytes'}
                and all(type(v) is int and 0<=v<2**64 for v in result.values())
                and result['available_bytes']>=1.25*1024**3+max(0,2*1024**3-result['worker_RSS_bytes']),
                'observed full post-import growth reserve')
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
    journal_terminal={e['id']:e for e in returned};require(len(journal_terminal)==len(returned)==len(attempted)
        and all(journal_terminal[e['id']]['operation']==e['operation'] for e in attempted)
        and not any(e['event']=='failed' for e in journal),'actual Journal terminal conservation')
    expected_events=[];primitive_ids={};next_id=0
    for marker in markers:
        p=marker['phase'];name=p['operation']
        if name.startswith('primitive_'):
            if p['event']=='attempt':next_id+=1;primitive_ids[p['ordinal']]=next_id
            expected_events.append((p['event'] if p['event']=='attempt' else 'returned',name[10:],primitive_ids[p['ordinal']]))
        elif name.startswith('publication_') and p['event']=='attempt':expected_events.append(('measurement_preserved','stage_'+name[12:]+'.json'))
    actual_events=[('measurement_preserved',e['path']) if e['event']=='measurement_preserved' else (e['event'],e['operation'],e['id']) for e in journal]
    require(actual_events==expected_events,'complete exact phase/Journal interleaved event sequence')
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
    if terminal is not None:
        require(terminal.get('CUDA_initialized') is initialized,'exact terminal initialized diagnostic join')
        if initialized:
            results=[n['end']['result'] for n in terminal_children]
            require(all(type(v) is int and 0<=v<2**64 for v in results[1:3])
                and results[1]<=results[2] and terminal.get('peak_allocated_bytes')==results[1]
                and terminal.get('peak_reserved_bytes')==results[2] and terminal.get('free_bytes_final')==results[3][0],
                'exact terminal allocator/free diagnostic joins')
    return {'status':'PASS_CPU_FIXTURE_PHASE_REPLAY_ONLY' if fixture else 'PASS_INDEPENDENT_PHASE_LEDGER_ONLY_PENDING_FULL_NATIVE_RAW_AUDIT',
        'phases':len(nodes),'markers':len(markers),'guard_boundaries':guards,'init_queries':init_queries,'memory_queries':memory_queries,
        'native_raw_audit_complete':False,'scoring_authorized':False,'fixture_only':fixture}

def loader_source(raw,expected_sha=LOADER_SOURCE_SHA):
    """Only execute the pinned pure shape constructor, never import the loader."""
    require(type(raw) is bytes and len(raw)<=128*1024 and hashlib.sha256(raw).hexdigest()==expected_sha==LOADER_SOURCE_SHA,
        'exact independently pinned loader source bytes')
    source=ast.parse(raw.decode('utf-8'));assignments={}
    for node in source.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            assignments[node.targets[0].id]=node.value
    pins=ast.literal_eval(assignments['ASSET_PINS']);require(len(pins)==7,'exact seven source asset pins')
    require(ast.dump(assignments['CHUNK_BYTES'])==ast.dump(ast.parse('2 * 1024 * 1024',mode='eval').body),'exact source2MiB chunks')
    functions={n.name:n for n in source.body if isinstance(n,ast.FunctionDef)}
    require(ast.unparse(functions['_base_inventory'].body[-1])=='return {name: named[name] for name in shapes}',
        'native base inventory returns source shape insertion order')
    require('shapes = _source_shapes()' in ast.unparse(functions['_base_inventory'])
        and 'for name, parameter in base.items():' in ast.unparse(functions['verify_frozen_base'])
        and "for name in sorted(shapes, key=lambda name: current['entries'][name]['data_offsets'][0]):" in ast.unparse(functions['load_titan']),
        'pinned source defines first offset order and final native inventory order')
    namespace={'__builtins__':{'range':range,'list':list}}
    shape_module=ast.Module(body=[functions['_source_shapes']],type_ignores=[])
    exec(compile(shape_module,'<pinned-pure-source-shapes>','exec'),namespace)
    shapes=namespace['_source_shapes']();require(len(shapes)==310,'all310 literal source shapes')
    return {'source_sha256':expected_sha,'asset_pins':pins,'asset_order':list(pins),'shapes':shapes,
        'final_digest_order':list(shapes),'chunk_bytes':2*1024**2}

def validate_header(header,shapes,file_bytes):
    require(type(header) is dict and set(header)=={'entries','metadata','header_bytes','payload_bytes','file_bytes'},'exact independent header evidence')
    require(type(header['metadata']) is dict and all(type(k) is str and type(v) is str for k,v in header['metadata'].items()),'typed safetensor metadata')
    require(all(type(header[k]) is int for k in ('header_bytes','payload_bytes','file_bytes'))
        and 2<=header['header_bytes']<=1024**2 and header['file_bytes']==file_bytes
        and header['payload_bytes']==file_bytes-8-header['header_bytes'],'bounded exact safetensor sizes')
    entries=header['entries'];require(type(entries) is dict and set(entries)==set(shapes) and len(entries)==310,'exact310 independent serialized names')
    intervals=[]
    for name,shape in shapes.items():
        item=entries[name];require(type(item) is dict and set(item)=={'dtype','shape','data_offsets'} and item['dtype']=='BF16'
            and item['shape']==shape and all(type(v) is int and v>0 for v in item['shape']),'exact independent BF16 shape '+name)
        span=item['data_offsets'];require(type(span) is list and len(span)==2 and all(type(v) is int and v>=0 for v in span)
            and span[1]-span[0]==2*math.prod(shape),'exact source tensor byte span '+name)
        intervals.append((span[0],span[1],name))
    position=0
    for start,end,name in sorted(intervals):require(start==position,'no header gap/overlap '+name);position=end
    require(position==header['payload_bytes'],'complete independent safetensor payload denominator')
    return [name for _,_,name in sorted(intervals)]

def read_header(path):
    path,before=regular(path)
    with path.open('rb',buffering=0) as stream:
        prefix=stream.read(8);require(len(prefix)==8,'complete header size prefix');size=struct.unpack('<Q',prefix)[0]
        require(2<=size<=1024**2 and size<=before.st_size-8,'bounded actual safetensor header')
        raw=stream.read(size);require(len(raw)==size,'complete bounded actual safetensor header')
    entries=loads(raw);require(type(entries) is dict,'object safetensor header')
    metadata=entries.pop('__metadata__',{})
    return {'entries':entries,'metadata':metadata,'header_bytes':size,'payload_bytes':before.st_size-8-size,'file_bytes':before.st_size}

def loader_tree(markers,loader,profile,header,verified_assets):
    """Independent exact helper-tree replay from pinned sizes/shapes, no math."""
    nodes,roots,cost=tree(markers);shapes=profile['shapes'];chunk_bytes=profile['chunk_bytes']
    require(type(verified_assets) is dict and set(verified_assets)==set(profile['asset_pins'])
        and all(type(v) is dict and set(v)=={'bytes','sha256'} and type(v['bytes']) is int and v['bytes']>=0
            and v['sha256']==profile['asset_pins'][name] for name,v in verified_assets.items()),'independent exact seven asset sizes/byte hashes')
    require(loader['asset_hashes']==verified_assets,'receipt asset contents independently verified; serialized order is not execution order')
    first_order=validate_header(header,shapes,verified_assets['model.safetensors']['bytes'])
    require(all(loader[k]==header[k] for k in ('header_bytes','payload_bytes','file_bytes')),'loader bounded header joins')
    require(set(loader['source_tensor_sha256'])==set(shapes),'all310 source digest keys')
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
    for name in profile['asset_order']:
        entry=verified_assets[name]
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
    for expected_name in first_order:
        require(position<len(ids2) and operation(ids2[position])=='api_tensor_copy','one tensor API per source tensor')
        tensor=ids2[position];position+=1;inside=direct(direct(tensor)[1]);require(inside,'nonempty actual tensor copy')
        first=operation(inside[0]);require(first.startswith('guard_tensor_copy_before:'),'typed tensor copy name')
        name=first[len('guard_tensor_copy_before:'):];require(name==expected_name and name not in seen,'exact verified offset first-copy order');seen.append(name)
        count=chunks(name);require(len(inside)==2*count,'every source-defined tensor copy chunk')
        for index in range(count):expect_guard(inside[2*index],'tensor_copy_before:'+name);expect_guard(inside[2*index+1],'tensor_copy_after:'+name)
        for index in range(count):
            require(position+1<len(ids2),'complete bounded first digest tree')
            expect_guard(ids2[position],'device_digest_before:'+name);expect_guard(ids2[position+1],'device_digest_after:'+name);position+=2
    require(set(seen)==set(shapes) and position+1==len(ids2),'complete source tensor/copy/digest denominator')
    expect_guard(ids2[position],'checkpoint_load_after');expect_guard(take(),'load_titan_after');require(cursor==len(ids),'no hidden loader helper guard/API')
    verify=unique('primitive_verify_frozen_base');ids3=direct(verify);position=0
    expect_guard(ids3[position],'verify_frozen_base_before');position+=1
    seen2=[]
    for name in profile['final_digest_order']:
        require(position<len(ids3)-1,'complete final native inventory digests');seen2.append(name)
        for _ in range(chunks(name)):
            require(position+1<len(ids3)-1,'complete bounded final digest chunks')
            expect_guard(ids3[position],'device_digest_before:'+name);expect_guard(ids3[position+1],'device_digest_after:'+name);position+=2
    require(set(seen2)==set(shapes) and position+1==len(ids3),'complete final310 digests')
    expect_guard(ids3[position],'verify_frozen_base_after')
    require(loader['payload_chunk_count']==sum(chunks(n) for n in shapes)
        and loader['device_digest_maximum_chunk_bytes']==chunk_bytes,'exact independently derived payload chunks/limits')
    return {'all310_copy_first_digest_and_second_digest_guard_trees_verified':True,'loader_phase_not_model_execution':True,
        'asset_source_order_sha256':digest(profile['asset_order']),'first_copy_verified_header_offset_order_sha256':digest(first_order),
        'final_digest_native_source_inventory_order_sha256':digest(profile['final_digest_order']),
        'loader_source_sha256':profile['source_sha256'],'tensor_chunks':sum(chunks(n) for n in shapes)}
