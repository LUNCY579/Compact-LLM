"""Independent raw measurement replay; derived effects confer no authority."""
from pathlib import Path
from types import ModuleType
import hashlib
PACKAGE=next(p for p in Path(__file__).resolve().parents if p.name=='rineya_synth_centralized_reasoning_v1')
SOURCE=PACKAGE/'evidence/raw_library_pipeline_reviewed_cpu_20261006/prefix_library_evidence.py'
PIN='aebab23452fa53ad71edbd4762620d929b4b8a936d41eb278f3b1f15ff1a57f2'
raw=SOURCE.read_bytes()
if hashlib.sha256(raw).hexdigest()!=PIN:raise ValueError('exact original independent replay helpers')
E=ModuleType('measurement_independent_original_helpers');E.__file__=str(SOURCE);exec(compile(raw,str(SOURCE),'exec'),E.__dict__)
require=E.require;KIND='TORCH_IMPORT_EFFECT_MEASUREMENT_RAW_V1';SCOPE='OWNED_TORCH_IMPORT_LIBRARY_EFFECT_OBSERVATION_ONLY'
def validate_measurement(raw,*,policy,grammar,expected_identity,selected_file_map,selected_source_map,expected_file_sha256):
    require(type(raw) is bytes and len(raw)<=E.MAX_PACKET_BYTES and hashlib.sha256(raw).hexdigest()==expected_file_sha256,'exact selected measurement raw bytes');v=E.parse(raw)
    fields={'schema_version','kind','identity','policy_sha256','library_files','observations','failure_records','terminal','producer_claims','fixture_only','native_PASS','native_authority_issued','measurement_scope','measurement_grammar','measured_boundary_effects','exact_effects_predeclared','sha256'}
    require(type(v) is dict and set(v)==fields and v['sha256']==E.digest({k:x for k,x in v.items() if k!='sha256'}) and type(v['schema_version']) is int and v['schema_version']==2 and v['kind']==KIND and v['measurement_scope']==SCOPE and v['exact_effects_predeclared'] is False,'separate sealed measurement packet')
    require(E.same(v['identity'],expected_identity) and E.same(v['measurement_grammar'],grammar) and E.same(v['library_files'],selected_file_map) and v['policy_sha256']==E.digest(policy),'external measurement selections')
    require(v['fixture_only'] is expected_identity['fixture_only'] and v['native_PASS'] is v['native_authority_issued'] is False,'measurement provenance never authority')
    gf={'schema_version','kind','required_extensions','optional_extensions','maximum_boundaries','maximum_depth','maximum_observations','model'};require(type(grammar) is dict and set(grammar)==gf and type(grammar['schema_version']) is int and grammar['schema_version']==1 and grammar['kind']=='FINITE_IMPORT_EFFECT_MEASUREMENT_GRAMMAR_V1' and grammar['model'] is False,'independent finite measurement grammar')
    req,opt=grammar['required_extensions'],grammar['optional_extensions'];require(type(req) is list and req and type(opt) is list and all(type(p) is str for p in req+opt) and len(req)==len(set(req)) and len(opt)==len(set(opt)) and not set(req)&set(opt) and set(req)|set(opt)==set(policy['extension_transitions']),'independent finite required extension map')
    require(type(grammar['maximum_boundaries']) is int and grammar['maximum_boundaries']==1+2*len(req+opt)<=2048 and type(grammar['maximum_depth']) is int and grammar['maximum_depth']==64 and type(grammar['maximum_observations']) is int and grammar['maximum_observations']==4096,'unchanged finite measurement bounds')
    # Validate original byte/directory/source selection with a documentary scope
    # projection only; the returned identity retains this separate stage scope.
    identity=dict(expected_identity);require(identity['original_scope']==SCOPE,'separate import measurement identity');identity['original_scope']=E.ORIGINAL_SCOPE;E.observed_identity(identity,expected_identity=identity)
    require(policy['framework_transition'] is None and all(type(x) is dict and set(x)=={'create','exec'} and x['create'] is x['exec'] is None for x in policy['extension_transitions'].values()),'measurement effects are unknown None, not guessed expectations');shape=E.parse(E.canonical(policy).encode('ascii'));shape['framework_transition']=[];shape['extension_transitions']={p:{'create':[],'exec':[]} for p in policy['extension_transitions']};E.selection(shape,identity,selected_file_map,selected_source_map,[],_observed=True)
    terminal=v['terminal'];require(type(terminal) is dict and set(terminal)=={'status','recording_complete','guard_failed','guard_closed'} and terminal['status'] in ('complete','failed','incomplete') and all(type(terminal[k]) is bool for k in ('recording_complete','guard_failed','guard_closed')),'typed measurement terminal')
    success=terminal['status']=='complete';require(not success or terminal['recording_complete'] and not terminal['guard_failed'] and terminal['guard_closed'] and expected_identity['owned_birth_status']=='VALIDATED','complete measurement requires actual bound terminal')
    rows=v['observations'];failures=v['failure_records'];require(type(rows) is list and 0<len(rows)<=4096 and type(failures) is list and len(failures)<=6*4096,'bounded measurement rows/failures');require(not success or not failures,'complete measurement has no failures');require(success or failures or terminal['status']=='incomplete','failed measurement has retained errors')
    for j,failure in enumerate(failures):
        require(type(failure) is dict and set(failure)=={'ordinal','stage','boundary_id','observation_ordinal','error'} and type(failure['ordinal']) is int and failure['ordinal']==j+1 and type(failure['stage']) is str and 0<len(failure['stage'])<=256 and (failure['observation_ordinal'] is None or type(failure['observation_ordinal']) is int and 1<=failure['observation_ordinal']<=len(rows)) and (failure['boundary_id'] is None or type(failure['boundary_id']) is int and failure['boundary_id']>0),'bounded original failure links');E.observed_error(failure['error'])
    stack=[];events=[];state=None;directories=None;effects=[];entered=set();created=set();executed=set();framework=False;cleanup_seen=False;precleanup=False;clock=None;first=None;known=True
    for index,row in enumerate(rows):
        rf={'ordinal','time_ns','event','boundary_id','boundary','scope','snapshot','event_trace','directory_identities','cleanup','observation_errors'};require(type(row) is dict and set(row)==rf and type(row['ordinal']) is int and row['ordinal']==index+1,'exact sequential measurement checkpoints')
        require(type(row['time_ns']) is int and 0<=row['time_ns']<2**64 and (clock is None or row['time_ns']>=clock),'chronological measurement clock');clock=row['time_ns'];first=clock if first is None else first;require(clock-first<=600*10**9,'measurement600s bound')
        event=row['event'];require(event in ('initial','enter','exit','boundary_error','precleanup','cleanup') and not cleanup_seen,'known measurement terminal order');errors=row['observation_errors'];require(type(errors) is dict and set(errors)<=E.OBSERVATION_COMPONENTS,'bounded measurement component errors')
        if errors:
            require(not success and all(any(f['observation_ordinal']==row['ordinal'] and E.same(f['error'],error) for f in failures) for error in errors.values()),'component errors link original retained failures');known=False
        boundary=row['boundary'];frame=None
        if event=='initial':require(index==0 and boundary is None and row['boundary_id'] is None,'initial first')
        elif event=='enter':
            require(not precleanup and type(boundary) is dict and set(boundary)=={'id','parent_id','name'} and type(boundary['id']) is int and boundary['id']==len(entered)+1 and row['boundary_id']==boundary['id'] and boundary['parent_id']==(stack[-1]['id'] if stack else None) and len(entered)<grammar['maximum_boundaries'] and len(stack)<64,'finite sequential LIFO measurement entry')
            name=boundary['name']
            if name=='framework_import':require(not framework and not stack and not entered,'one framework root');framework=True
            else:
                require(framework and stack and type(name) is str and name.startswith('extension:'),'extension in framework');p,sep,op=name[len('extension:'):].rpartition(':');require(sep and p in policy['extension_transitions'] and op in ('create','exec') and (p,op) not in entered,'single finite measurement extension operation');require(op!='exec' or p in created,'measurement exec after returned create')
            frame={**boundary,'before':None,'nested':set()}
        elif event in ('exit','boundary_error'):require(stack and boundary=={k:stack[-1][k] for k in ('id','parent_id','name')} and row['boundary_id']==stack[-1]['id'],'balanced measurement return/failure');frame=stack[-1];require(event!='boundary_error' or not success and any(f['boundary_id']==frame['id'] and f['observation_ordinal']==row['ordinal'] for f in failures),'failed boundary retains original error')
        else:require(not stack and boundary is None and row['boundary_id'] is None,'measurement terminal outside boundaries');require(event!='precleanup' or not precleanup,'one precleanup');require(event!='cleanup' or precleanup or not success,'cleanup follows terminal');precleanup=precleanup or event=='precleanup';cleanup_seen=event=='cleanup'
        if known:
            observed=E.snapshot(row['snapshot'],selected_file_map);dirs=E._observed_directories(row['directory_identities'],policy);require(directories is None or E.same(dirs,directories),'directory identity conservation');directories=dirs
            if event=='cleanup':E.cleanup(row['cleanup'],policy)
            else:E.observed_scope(row['scope'],policy,fixture_only=v['fixture_only'])
            if state is None:require(event=='initial' and set(observed)==set(policy['initial_loaded_libraries']),'fixed initial measured baseline');state=dict(observed)
            require(all(observed.get(p)==h for p,h in state.items()),'measurement old handle conservation');gains=set(observed)-set(state);require(not gains or stack or event in ('exit','boundary_error'),'no effects outside measurement intervals')
            events,next_state=E._observed_events(row['event_trace'],events,state,gains,selected_file_map);require(next_state==observed,'exact independently replayed measurement events/snapshot');state=next_state
        if event=='enter':
            frame['before']=dict(state) if known else None;stack.append(frame);entered.add(('framework',None) if frame['name']=='framework_import' else tuple(frame['name'][len('extension:'):].rpartition(':')[::2]))
        elif event in ('exit','boundary_error'):
            if event=='exit' and errors:continue
            stack.pop()
            if known and frame['before'] is not None:
                total=set(state)-set(frame['before']);own=total-frame['nested']
                if event=='exit':effects.append({'boundary':{k:frame[k] for k in ('id','parent_id','name')},'own_new_libraries':sorted(own),'nested_new_libraries':sorted(frame['nested']),'total_new_libraries':sorted(total)})
                if stack:stack[-1]['nested'].update(total)
            if event=='exit' and frame['name']!='framework_import':p,_,op=frame['name'][len('extension:'):].rpartition(':');(created if op=='create' else executed).add(p)
    require(not success or known and cleanup_seen and precleanup and not stack and framework and created==executed and set(req)<=created,'all required measured boundaries and cleanup returned');require(success or not stack or terminal['status']=='incomplete','unclosed failed measurement requires incomplete state')
    require(not success or E.same(effects,v['measured_boundary_effects']),'producer measurement effect claims equal independent raw derivation')
    claims=v['producer_claims'];require(claims['observation_count']==len(rows) and claims['boundary_count']==sum(r['event']=='enter' for r in rows) and (not known or claims['event_count']==len(events)),'measurement raw count claims');require(claims['complete_search_directory_coverage'] is False if policy['kind']=='FINITE_WINDOWS_DLL_BOUNDARY_V2' else type(claims['complete_search_directory_coverage']) is bool,'foreign scope remains unobserved')
    return {'kind':'INDEPENDENT_IMPORT_EFFECT_MEASUREMENT_REPLAY','status':'MEASUREMENT_RECORDED_UNQUALIFIED' if success and known else 'MEASUREMENT_FAILURE_PRESERVED','measurement_scope':SCOPE,'effects':effects,'verified_events':len(events),'final_libraries':state if known else None,'fixture_only':v['fixture_only'],'native_PASS':False,'native_authority_issued':False,'native_qualification':False,'exact_effects_predeclared':False}
