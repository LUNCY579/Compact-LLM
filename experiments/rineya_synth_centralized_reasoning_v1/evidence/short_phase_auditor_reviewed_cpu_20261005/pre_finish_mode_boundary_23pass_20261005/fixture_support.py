"""Inert native-shaped ledger and filesystem controls. No model/binary reader."""
from copy import deepcopy
from pathlib import Path
import math
import phase_replay as phase
import audit_adapter as adapter

def native_shaped(old):
    oldmarkers,oldsummary,worker,journal=deepcopy(old)
    profile=phase.loader_source(phase.read_bytes(adapter.PACKAGE/'native_loader.py',128*1024))
    shapes=profile['shapes'];chunk=profile['chunk_bytes'];offset=0;entries={}
    # Deliberately reverse shape insertion order: first copy is offset order,
    # final digest is native source-shape order; neither is canonical JSON order.
    for name in reversed(shapes):
        end=offset+2*math.prod(shapes[name]);entries[name]={'dtype':'BF16','shape':shapes[name],'data_offsets':[offset,end]};offset=end
    header={'entries':entries,'metadata':{},'header_bytes':35216,'payload_bytes':offset,'file_bytes':offset+35224}
    sizes={name:1 for name in profile['asset_order']};sizes['model.safetensors']=header['file_bytes']
    assets={name:{'bytes':sizes[name],'sha256':profile['asset_pins'][name]} for name in sorted(sizes)}
    loader={'loader':'synth-titan-stream-meta-2MiB-1','asset_hashes':assets,**{k:header[k] for k in ('header_bytes','payload_bytes','file_bytes')},
        'source_tensor_sha256':{name:'b'*64 for name in shapes},'payload_chunk_count':sum(math.ceil(math.prod(v)*2/chunk) for v in shapes.values()),
        'device_digest_maximum_chunk_bytes':chunk}
    def node(name,children=(),result=None):return [name,list(children),result]
    def guard(name,*,initialized=True):
        if name=='pre_import':return node('guard_'+name)
        kids=[node('guard_cuda_is_initialized',result=initialized)]
        if initialized:kids.append(node('guard_cuda_mem_get_info',result=[8*1024**3,8*1024**3]))
        return node('guard_'+name,kids)
    def api(name,children=()):return node('api_'+name,[guard('before_'+name),node('primitive_'+name,children),guard('after_'+name)])
    def pub(name):return node('publication_'+name)
    def digest_guards(name):return [guard(prefix+name) for _ in range(math.ceil(math.prod(shapes[name])*2/chunk)) for prefix in ('device_digest_before:','device_digest_after:')]
    load=[guard('load_titan_before')]
    for name in profile['asset_order']:
        for _ in range(math.ceil(sizes[name]/chunk)+1):load.extend([guard('asset_hash_before:'+name),guard('asset_hash_after:'+name)])
    load.extend([guard('config_load_before'),guard('config_load_after'),api('model_construct',
        [guard(n) for n in ('model_construct_before','model_to_empty_before','model_construct_after')])])
    checkpoint=[guard('checkpoint_load_before')]
    for name in reversed(shapes):
        inside=[guard(prefix+name) for _ in range(math.ceil(math.prod(shapes[name])*2/chunk)) for prefix in ('tensor_copy_before:','tensor_copy_after:')]
        checkpoint+=[api('tensor_copy',inside)]+digest_guards(name)
    checkpoint.append(guard('checkpoint_load_after'));load.extend([api('checkpoint_load',checkpoint),guard('load_titan_after')])
    collect=[api(n) for n in ('install_fp32_compute','install_activation_hooks','zero_parity_base_forward','install_qv_lora','zero_lora_query_before','zero_parity_adapter_forward')]+[pub('zero_adapter_parity')]
    for row,output in enumerate(worker['collection']['rows'],1):
        for step in range(1,len(output['raw_generated_ids'])+1):collect.extend([guard('before_inference_forward'),api('inference_forward'),guard('after_inference_forward'),pub('raw_row%04d_token%02d'%(row,step))])
        collect.append(pub('inference_row_%04d'%row))
    verification=[guard('verify_frozen_base_before')]
    for name in shapes:verification+=digest_guards(name)
    verification.append(guard('verify_frozen_base_after'))
    collect.extend([api('zero_lora_query_after'),api('verify_frozen_base',verification),pub('short_m0_collection_completed')])
    observed={v['phase']['operation']:v['phase']['result'] for v in oldmarkers if v['phase']['event']=='returned' and v['phase']['operation'].startswith('main_')}
    roots=[pub('worker_ownership_verified'),node('main_initial_config_preflight'),guard('pre_import')]
    for name in phase.MAIN[1:]:
        if name=='direct_CUDA_preload_mem_get_info':roots.append(api('cuda_context_init'))
        children=load if name=='load_titan' else collect if name=='collect_loaded' else []
        if name=='preserve_terminal':
            roots.append(guard('completed'));children=[node('terminal_cuda_is_initialized',result=True),
                node('terminal_cuda_max_memory_allocated',result=0),node('terminal_cuda_max_memory_reserved',result=0),
                node('terminal_cuda_mem_get_info',result=[8*1024**3,8*1024**3])]
        roots.append(node('main_'+name,children,observed.get('main_'+name)))
        if name=='load_titan':roots.extend([pub('loader'),api('generated_decoder_construct')])
    markers=[];journal=[];ordinal=api_id=0
    def emit(n):
        nonlocal ordinal,api_id
        name,kids,result=n;ordinal+=1;oid=ordinal
        def marker(event):
            p={'ordinal':oid,'operation':name,'event':event,'monotonic':float(len(markers)),
                'utc':'2026-10-05T00:00:00+00:00','historical_cause_established':False}
            if event=='returned':p.update(result=result,error=None)
            markers.append({'name':'phase_%04d_%s'%(oid,'start' if event=='attempt' else 'end'),'phase':p})
        marker('attempt');jid=None
        if name.startswith('primitive_'):api_id+=1;jid=api_id;journal.append({'event':'attempt','operation':name[10:],'id':jid})
        if name.startswith('publication_'):journal.append({'event':'measurement_preserved','path':'stage_'+name[12:]+'.json','qualification_established':False})
        for child in kids:emit(child)
        marker('returned')
        if jid is not None:journal.append({'event':'returned','operation':name[10:],'id':jid})
    for n in roots:emit(n)
    _,_,cost=phase.tree(markers);operations={}
    for event in journal:
        if event['event']=='attempt':
            entry=operations.setdefault(event['operation'],phase.zero());entry['attempted']+=1;entry['returned']+=1
    worker['loader']=loader;worker['operation_accounting']['operations']=operations;worker['operation_accounting']['last_call']=api_id
    worker['source_commit']='a'*40;worker['config_sha256']='f'*64;worker['batch_selector']={'source_kind':'atomic_RS_DEV','ordinal':4}
    summary=deepcopy(oldsummary);summary['identity'].update(source_commit=worker['source_commit'],config_sha256=worker['config_sha256'],batch_selector=worker['batch_selector'])
    summary['phase_accounting'].update(operations=cost,last_ordinal=ordinal,last_phase=markers[-1]['phase'],publication_errors=[])
    size=sum(len((phase.canonical(v)+'\n').encode('ascii')) for v in markers)
    summary['phase_IO'].update(markers_returned=len(markers),marker_bytes_attempted=size,marker_bytes_returned=size)
    summary['phase_IO']['operations']['phase_marker_write']={'attempted':len(markers),'returned':len(markers),'failed':0,'pending':0}
    summary['sha256']=phase.digest({k:v for k,v in summary.items() if k!='sha256'})
    return markers,summary,worker,journal,profile,header,assets

def filesystem(directory,values):
    markers,summary,worker,journal,profile,header,assets=deepcopy(values)
    package=Path(directory)/'CPU_INERT_PACKAGE';leaf=package/'artifacts/CPU_INERT_TERMINAL';saved=package/'evidence/CPU_INERT_PRESERVED'
    config=package/'configs/CPU_INERT.json';lease=package/'artifacts/_native_gpu_leases/CPU_INERT_LEASE'
    for p in (leaf,saved,config.parent,lease):p.mkdir(parents=True)
    def write(path,value):path.write_bytes((phase.canonical(value)+'\n').encode('ascii'))
    rows=[{'row_id':r['row_id'],'prompt_sha256':r['prompt_sha256'],'prompt_ids':[5,6],'prompt_tokens':2,'attention_mask':[1,1]} for r in worker['collection']['rows']]
    packet=phase.digest(rows);worker['collection']['source_packet_sha256']=packet
    scope={'sha256':None,'contract':adapter.VERSION,'authorization':deepcopy(adapter.AUTHORIZATION),'profile':'BF16_RESIDENT_FP32_ACTIVATIONS_QV8_V1',
        'batch_selector':worker['batch_selector'],'rows':rows,'source_packet_sha256':packet,'runtime_versions':{'torch':'2.6.0+cu124','transformers':'5.14.1'},
        'short_phase_observation_profile':{'id':adapter.PROFILE_ID,'scope':'CPU_INERT_ONLY'}}
    inventories={}
    for section,count in (('source',1),('runtime',65),('decoder',17),('data',5),('model',7)):
        paths=[]
        for i in range(count):
            path=package/'CPU_INERT_BYTES'/section/('ast.py' if section=='runtime' and i==64 else 'file%02d.txt'%i)
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(('CPU_INERT_%s_%d'%(section,i)).encode('ascii'));paths.append(path)
        inventories[section]={str(p.resolve()):phase.file_info(p)['sha256'] for p in paths}
    extension={p:h for p,h in inventories['runtime'].items() if Path(p).name=='ast.py'}
    scope['short_RAM_monitor_profile']={'id':'SHORT_M0_PHYSICAL_RAM_COHERENT_EXTERNAL_CHECK_V2',
        'all_floors_and_numerics_unchanged':True,'runtime_extension_pins':extension}
    for section,key in (('source','source_pins'),('runtime','runtime_pins'),('decoder','decoder_runtime_pins'),('data','data_pins'),('model','model_pins')):scope[key]=deepcopy(inventories[section])
    scope.update(scoring_authorized=False,training_authorized=False,prior_correct_enrollment_performed=False)
    scope['sha256']=phase.digest({k:v for k,v in scope.items() if k!='sha256'});write(config,scope)
    config_sha=phase.file_info(config)['sha256'];worker['config_sha256']=config_sha;worker['wall_seconds']=1.;worker['backwards']=worker['optimizer_steps']=0
    summary['identity']['config_sha256']=config_sha;summary['sha256']=phase.digest({k:v for k,v in summary.items() if k!='sha256'})
    write(leaf/'worker_result.json',worker);write(leaf/'PHASE_ACCOUNTING_FINAL.json',summary)
    (leaf/'PHASE_OBSERVATIONS.jsonl').write_bytes(b''.join((phase.canonical(v)+'\n').encode('ascii') for v in markers))
    for event in journal:
        if event['event']=='measurement_preserved':
            value=worker['loader'] if event['path']=='stage_loader.json' else {'CPU_INERT_FRAGMENT':event['path']}
            write(leaf/event['path'],value);event['sha256']=phase.file_info(leaf/event['path'])['sha256']
    write(leaf/'supervisor_ownership.json',{'CPU_INERT_OWNERSHIP':True})
    (leaf/'worker.stdout.log').write_bytes(b'');(leaf/'worker.stderr.log').write_bytes(b'')
    (leaf/'events.jsonl').write_bytes(b''.join((phase.canonical(v)+'\n').encode('ascii') for v in journal))
    write(leaf/'worker_GPU_final.json',{'CUDA_initialized':True,'peak_allocated_bytes':0,'peak_reserved_bytes':0,'free_bytes_final':8*1024**3})
    resources=[]
    for now in (1.,1.1):resources.extend([{'kind':'CPU','sample_started_monotonic':now,'sample_completed_monotonic':now+.01,
        'host_available_bytes':4*1024**3,'worker_tree_rss_bytes':1024**3,'disk_free_bytes':4*1024**3},
        {'kind':'GPU','gpu_sample_started_monotonic':now,'gpu_sample_completed_monotonic':now+.01,'gpu_free_bytes':8*1024**3}])
    write(leaf/'RESOURCE_OBSERVATIONS.json',resources)
    (leaf/'supervisor_heartbeat.jsonl').write_bytes(b'{"unix_seconds":1.0}\n{"unix_seconds":1.1}\n')
    sup={'status':'SUPERVISED_WORKER_EXITED_ZERO','returncode':0,'partial_outputs_frozen':True,'preflight':{'source_commit':'a'*40},
        'config_sha256':config_sha,'output_dir':str(leaf),'lease_path':str(lease),'elapsed_seconds':1.,'admission_observations':[],
        'dispatch_observation':{'gpu_free_bytes':8*1024**3,'host_available_bytes':4*1024**3,'disk_free_bytes':4*1024**3}}
    sup['short_RAM_monitor_diagnostic']={'path':'SHORT_RAM_ABORT_DIAGNOSTIC.json','attempted':False,'persisted':False,
        'write_error':None,'original_abort_reason_preserved':None}
    owned={'assigned_before_resume':True,'job_kill_on_close':True,'breakaway_allowed':False,'scope':'CPU_INERT_JOB_METADATA'}
    sup.update(worker_identity=owned,worker_resume_monotonic=1.,job_closed_kill_on_close=True,owned_worker_identity_currently_alive=False,
        sampler_threads_terminal_verified=True,gpu_telemetry_cleanup_verified=True,unjoined_sampler_threads=[])
    sup['partial_outputs']=[{'path':p.name,**phase.file_info(p)} for p in sorted(leaf.iterdir())]
    write(leaf/'SUPERVISOR_RECEIPT.json',sup)
    write(lease/'worker.json',owned);write(lease/'resume.json',{'worker_identity':owned,'external_monitor_ready_before_resume':True,'resumed_monotonic':1.})
    write(lease/'reservation.json',{'config_sha256':config_sha,'output_dir':str(leaf)})
    (lease/'terminal.json').write_bytes((leaf/'SUPERVISOR_RECEIPT.json').read_bytes())
    infos={p.name:phase.file_info(p) for p in leaf.iterdir()}
    for p in leaf.iterdir():(saved/p.name).write_bytes(p.read_bytes())
    (saved/'config_snapshot.json').write_bytes(config.read_bytes())
    pres={'source_commit':'a'*40,'raw_directory':str(leaf),'preserved_directory':str(saved),'raw_files':infos,'config_snapshot':phase.file_info(config)}
    pres['sha256']=phase.digest(pres);write(saved/'preservation_receipt.json',pres)
    core={p:h for p,h in inventories['runtime'].items() if p not in extension}
    registration={'schema_version':1,'id':adapter.REGISTRATION_ID,'contract':adapter.VERSION,'source_commit':'a'*40,
        'config_path':str(config.resolve()),'config_file_sha256':config_sha,'profile_body_sha256':phase.digest(scope['short_phase_observation_profile']),
        'inventories':inventories,'counts':{k:len(v) for k,v in inventories.items()},'qualified_runtime64_sha256':phase.digest(core)}
    registration['sha256']=phase.digest(registration)
    return package,config,leaf,saved,registration,config_sha,pres['sha256'],phase.digest(scope['short_phase_observation_profile']),profile,header,assets
