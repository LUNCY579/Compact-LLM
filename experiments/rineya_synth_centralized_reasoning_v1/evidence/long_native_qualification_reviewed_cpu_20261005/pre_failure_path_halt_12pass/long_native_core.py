"""Fresh native-long protocol; injected CPU controls are never native authority."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,math,sys,time
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1] if HERE.name=='long_native_qualification_work' and HERE.parent.name=='artifacts' else HERE
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(HERE))
import long_native_contract as contract
from rs_contract import canonical,digest,require


def seal(value):return {**value,'sha256':digest(value)}
def checksum(value):require(value['sha256']==digest({k:v for k,v in value.items() if k!='sha256'}),'immutable body checksum')
def counts():return dict(attempted=0,returned=0,failed=0,pending=0)


class Collector:
    def __init__(self,call,guard):self.external,self.guard,self.operations=call,guard,{};self.models_closed=False
    def __call__(self,name,function):
        item=self.operations.setdefault(name,counts());item['attempted']+=1;item['pending']+=1;executed=False
        def actual():
            nonlocal executed
            require(not executed,'primitive must execute once');executed=True
            try:
                require(not self.models_closed or name not in PhysicalDose.ALLOWED,'model operation phase closed before publication')
                value=function()
            except BaseException:item['pending']-=1;item['failed']+=1;raise
            item['pending']-=1;item['returned']+=1;return value
        self.guard('before_'+name);value=self.external(name,actual)
        require(executed and item['pending']==0,'actual primitive must return');self.guard('after_'+name)
        return value
    def snapshot(self):return deepcopy(self.operations)


class PhysicalDose:
    """Record actual model prehooks, independent of mutable component budgets."""
    ALLOWED={'long_native_dense_parity_forward','long_native_zeroB_parity_forward','long_native_greedy_forward',
             'long_native_bulk_forward','long_native_declared_prefix_stress_forward'}
    def __init__(self,scope):self.scope=scope;self.stack=[];self.records=[];self.positions=0;self.models_closed=False
    def wrap(self,call):
        def invoke(name,function):
            self.stack.append(name)
            try:return call(name,function)
            finally:self.stack.pop()
        return invoke
    def charge(self,ids):
        require(not self.models_closed,'physical model phase closed before publication')
        require(self.stack and self.stack[-1] in self.ALLOWED and type(ids) is list and ids
            and all(type(t) is int and 0<=t<151936 for t in ids),'registered typed actual model prefix')
        work=self.scope['dose'];require(len(ids)<=work['context_cap'] and len(self.records)<work['maximum_forward_calls']
            and self.positions+len(ids)<=min(work['maximum_forward_input_positions'],134144),'immutable physical call/position ceiling')
        self.records.append({'operation':self.stack[-1],'input_ids_sha256':digest(ids),'input_positions':len(ids)})
        self.positions+=len(ids)
    def snapshot(self):return {'attempted_calls':len(self.records),'input_positions':self.positions,'records':deepcopy(self.records)}
    def verify(self,trace):
        scope=self.scope;role=scope['role'];row=scope['row'];prefix=scope['short_parity_row']['prompt_ids']
        roster=[('long_native_dense_parity_forward',prefix),('long_native_zeroB_parity_forward',prefix)]
        require(type(trace) is list and 1<=len(trace)<=32,'actual trace required')
        if role=='prefix_stress':
            require(len(trace)==32,'full predeclared stress length')
            roster += [('long_native_declared_prefix_stress_forward',row['prompt_ids']+scope['recipe']['prefix_stress_ids'][:i]) for i in range(32)]
        else:
            require(not any(t['token'] in (151643,151645) for t in trace[:-1]) and (len(trace)==32 or trace[-1]['token'] in (151643,151645)),
                'complete EOS-respecting greedy trace')
            tokens=[t['token'] for t in trace]
            roster += [('long_native_bulk_forward',row['prompt_ids']+tokens[:-1])] if role=='bulk' else [
                ('long_native_greedy_forward',row['prompt_ids']+tokens[:i]) for i in range(len(trace))]
        expected=[{'operation':name,'input_ids_sha256':digest(ids),'input_positions':len(ids)} for name,ids in roster]
        require(self.records==expected,'exact physical prefix/operation roster; hidden calls forbidden')
        return self.snapshot()


def publication(value,payload):
    require(type(value) is dict and set(value)=={'path','bytes','sha256','payload_sha256','durable'} and value['durable'] is True
        and value['payload_sha256']==digest(payload),'durable publication descriptor')
    path=Path(value['path']);require(path.is_absolute() and path.is_file() and not path.is_symlink()
        and type(value['bytes']) is int and 0<value['bytes']<=16*1024**2,'bounded regular publication')
    with path.open('rb') as stream:raw=stream.read(value['bytes']+1)
    require(len(raw)==value['bytes'] and hashlib.sha256(raw).hexdigest()==value['sha256'],'immutable publication bytes')
    parsed=json.loads(raw.decode(),object_pairs_hook=contract.native._pairs,parse_constant=contract.native._bad_constant)
    require(canonical(parsed)==canonical(payload),'publication payload changed')
    return deepcopy(value)


def merge(a,b):
    result=deepcopy(a)
    for name,item in b.items():
        require(set(item)==set(counts()) and all(type(v) is int and v>=0 for v in item.values())
            and item['attempted']==item['returned']+item['failed']+item['pending'],'conserved actual primitive costs')
        target=result.setdefault(name,counts())
        for key,n in item.items():target[key]+=n
    return result


def project(value,name):
    value=deepcopy(value)
    for suffix in ('write','verify'):
        item=value['operations'].setdefault(name+'_'+suffix,counts());item['attempted']+=1;item['returned']+=1
    return value


def validate_chain_memory(node,recipe):
    checksum(node);require(node['kind']==contract.KIND and node['recipe_sha256']==recipe['sha256'],'same purpose/recipe capsule chain')
    payload=node['capsule'];checksum(payload)
    require(payload['purpose']==contract.PURPOSE and payload['recipe_sha256']==recipe['sha256'],'qualification-only capsule purpose')
    terminal=node['terminal']
    require(terminal['capsule_sha256']==payload['sha256'] and terminal['costs']==project(project(payload['posted_scope_costs'],'long_capsule_publish'),'long_terminal_publish'),
        'posted→terminal exact costs')
    promotion=node['promotion']
    require(promotion['capsule_sha256']==payload['sha256'] and promotion['terminal_sha256']==digest(terminal)
        and promotion['costs']==node['cumulative_costs']==project(terminal['costs'],'long_capsule_promote'),'terminal→promotion exact costs')
    return deepcopy(payload)


def validate_chain(node,recipe):
    payload=validate_chain_memory(node,recipe)
    publication(node['capsule_publication'],payload)
    publication(node['terminal_publication'],node['terminal'])
    publication(node['promotion_publication'],node['promotion'])
    return payload


def native_references(scope):
    refs=scope['references'];role=scope['role'];wanted=[] if role=='trace1' else ['trace1'] if role=='trace2' else ['trace1','trace2'] if role=='bulk' else ['trace1','trace2','bulk']
    require(len(refs or[])==len(wanted),'every audited prior role required')
    previous=[]
    for reference,want in zip(refs or[],wanted):
        require(set(reference)=={'result_path','result_sha256','supervisor_path','supervisor_sha256','audit_path','audit_sha256'},'explicit independent prior anchors')
        values={}
        for name in ('result','supervisor','audit'):
            path=Path(reference[name+'_path']);require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and contract.native.filehash(path)==reference[name+'_sha256'],'exact prior '+name+' bytes')
            values[name]=contract.native.read_json(path)
        result,sup,audit=(values[k] for k in ('result','supervisor','audit'));checksum(result);checksum(audit)
        leaf=Path(reference['result_path']).resolve().parent
        require(leaf==Path(reference['supervisor_path']).resolve().parent and leaf.parent==(contract.PACKAGE/'artifacts').resolve(),
            'one prior owned output leaf')
        require(result['status']=='COMPLETE_NATIVE_LONG_SCOPE_PENDING_INDEPENDENT_AUDIT' and result['execution_scope']==contract.NATIVE
            and result['role']==want and result['recipe_sha256']==scope['recipe']['sha256'],'only same-recipe actual prior native role')
        require(sup['status']=='SUPERVISED_WORKER_EXITED_ZERO' and sup['returncode']==0 and sup['job_closed_kill_on_close'] is True
            and sup['sampler_threads_terminal_verified'] is True and sup['gpu_telemetry_cleanup_verified'] is True
            and sup['owned_worker_identity_currently_alive'] is False and sup['config_sha256']==result['config_sha256'],'prior complete owned terminal')
        require(audit['kind']==contract.AUDIT_KIND and audit['status']=='PASS_NATIVE_LONG_SCOPE_NUMERICAL_MECHANICS_ONLY'
            and audit['purpose']==contract.PURPOSE and audit['result_sha256']==reference['result_sha256']
            and audit['supervisor_sha256']==reference['supervisor_sha256'] and audit['role']==want
            and audit['recipe_sha256']==scope['recipe']['sha256'] and audit['study_training_authorized'] is False
            and audit['scoring_authorized'] is False,'independently selected prior audit cannot transfer authority')
        payload=validate_chain(result['capsule_chain'],scope['recipe'])
        require(all(Path(result['capsule_chain'][key]['path']).resolve().parent==leaf for key in
            ('capsule_publication','terminal_publication','promotion_publication')),'prior publications belong to exact owned leaf')
        require(sup['preflight']['recipe']==scope['recipe'] and sup['preflight']['role']==want
            and sup['preflight']['purpose']==contract.PURPOSE and audit['config_sha256']==result['config_sha256']
            and audit['source_commit']==result['source_commit']==sup['preflight']['source_commit'],'prior current source/config/profile joins')
        require(payload['CPU_fixture'] is False and payload['role']==want,'CPU fixture capsule never native predecessor')
        if previous:
            require(payload['previous_capsule_sha256']==previous[-1]['capsule_chain']['sha256'],'exact audited cross-role latest lineage')
            old=previous[-1]['capsule_chain']['cumulative_costs'];new=result['capsule_chain']['cumulative_costs']
            require(new['physical_calls']>=old['physical_calls'] and new['input_positions']>=old['input_positions']
                and all(name in new['operations'] and all(new['operations'][name][key]>=count for key,count in item.items())
                    for name,item in old['operations'].items()),'prior cumulative costs cannot roll back or disappear')
        previous.append(result)
    return previous


def run_scope(scope,factory,call,guard,preserve,*,admission=None,previous_fixtures=None,clock=time.monotonic):
    scope=deepcopy(scope)
    require(scope['purpose']==contract.PURPOSE and scope['authorization']==contract.AUTHORIZATION,'exact qualification-only purpose')
    require(scope['dose']==contract.work(scope['row'],scope['role']),'exact immutable role dose')
    checksum(scope['recipe']);role=scope['role'];mode=getattr(factory,'execution_scope',None)
    require(mode in (contract.NATIVE,contract.FAKE),'explicit implementation scope')
    previous=native_references(scope) if mode==contract.NATIVE else deepcopy(previous_fixtures or[])
    if mode==contract.FAKE:
        for item in previous:
            require(item['execution_scope']==contract.FAKE,'CPU fixture lineage only')
            validate_chain(item['capsule_chain'],scope['recipe'])
        require(len(previous)==(0 if role=='trace1' else 1 if role=='trace2' else 2 if role=='bulk' else 3)
            and [item['role'] for item in previous]==list(contract.ROLES[:len(previous)]),'exact CPU fixture role lineage')
    require((role=='trace1' and not previous) or (role!='trace1' and previous),'fresh first or prior lineage')
    old=previous[-1]['capsule_chain']['cumulative_costs'] if previous else {'operations':{},'physical_calls':0,'input_positions':0}
    started=clock();external_guard=guard
    def guard(stage):
        elapsed=clock()-started;require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<=600,'unchanged600s complete scope deadline')
        external_guard(stage)
    dose=PhysicalDose(scope);collector=Collector(dose.wrap(call),guard);session=None;trace=[];cleaned=False
    result={'status':'FAILED_NATIVE_LONG_SCOPE','execution_scope':mode,'role':role,'recipe_sha256':scope['recipe']['sha256'],'capsule_chain':None,
        'native_launch_authorized':False,'native_long_context_qualified':False,'study_training_authorized':False,'guard_enrollment_authorized':False,'scoring_authorized':False}
    def cost():return {'operations':merge(old['operations'],collector.snapshot()),'physical_calls':old['physical_calls']+len(dose.records),
        'input_positions':old['input_positions']+dose.positions}
    def close():
        nonlocal cleaned
        if cleaned:return
        cleaned=True
        if session is not None:session['cleanup']()
        elif callable(getattr(factory,'cleanup',None)):factory.cleanup()
    try:
        if mode==contract.NATIVE:
            require(not any(n.split('.',1)[0] in ('torch','transformers','jax','tensorflow','flax') for n in sys.modules),'framework-free before owned admission')
            require(callable(admission),'fresh owned admission required before factory')
            request={'purpose':contract.PURPOSE,'recipe_sha256':scope['recipe']['sha256'],'role':role,'seed':101}
            proof=admission(deepcopy(request));now=time.monotonic()
            require(proof['binding']==request and proof['owned_supervision_verified'] is True and proof['model_load_and_constructor_in_scope'] is True
                and proof['observed_monotonic']<=now<=proof['expires_monotonic']<=proof['observed_monotonic']+.2,'fresh200ms owned factory admission')
        session=factory(scope,call=collector,guard=guard,dose=dose)
        engine=session['component'];require(engine.context_cap==scope['dose']['context_cap'] and engine.generation_cap==32,'exact long component envelope')
        require((mode==contract.NATIVE and engine.scope=='UNQUALIFIED_NATIVE_IMPLEMENTATION' and engine.device=='cuda:0')
            or (mode==contract.FAKE and engine.scope=='CPU_FAKE_COMPONENT' and engine.device=='cpu'),'native never fixture relabel')
        start_proof=collector('long_start_attestation',session['attest'])
        require(start_proof['kind']=='NATIVE_OBSERVED_LONG_START' if mode==contract.NATIVE else start_proof['kind']=='CPU_FIXTURE_LONG_START','actual versus fixture attestation')
        if previous:
            require(start_proof['identity_common']==previous[0]['capsule_chain']['capsule']['start_proof']['identity_common'],'same independently observed seed/source/model/encoder/startingRNG')
            if mode==contract.NATIVE:require(all(start_proof['owned_scope_identity']['job_name']!=p['capsule_chain']['capsule']['start_proof']['owned_scope_identity']['job_name']
                for p in previous),'fresh independently owned native Job for every role')
        row=scope['row'];comparison=None
        if role in ('trace1','trace2'):
            def step(value):
                payload=seal({'purpose':contract.PURPOSE,'recipe_sha256':scope['recipe']['sha256'],'role':role,'prefix':deepcopy(trace),'costs':cost()})
                name='long_token_%02d'%len(trace);pub=collector(name+'_write',lambda:preserve({'event':name,'payload':payload}))
                collector(name+'_verify',lambda:publication(pub,payload))
            engine._trace(row,'long_native_greedy_forward',trace,step)
        elif role=='bulk':
            first,second=[p['capsule_chain']['capsule']['trace'] for p in previous[:2]]
            require(first==second,'complete cross-process greedy repeats match');trace=deepcopy(first)
            logits=engine._forward(row['prompt_ids']+[t['token'] for t in trace[:-1]],'long_native_bulk_forward');errors=[];different=[]
            for index,observed in enumerate(trace):
                token,lp=engine._selected(logits,row['prompt_tokens']-1+index)
                if token!=observed['token']:
                    different.append(index)
                    lp=float(collector('long_bulk_original_selected_LP',lambda:
                        engine.torch.log_softmax(logits[0,row['prompt_tokens']-1+index],dim=-1)[observed['token']].item()))
                    require(math.isfinite(lp) and lp<=0,'finite originally selected bulk logprob')
                errors.append(abs(lp-observed['logprob']))
            del logits
            comparison={'actual_compared_steps':len(trace),'argmax_differences':different,'maximum_error':max(errors),'mean_error':sum(errors)/len(errors)}
            require(not different and max(errors)<=.03125 and sum(errors)/len(errors)<=.01,'unchanged bulk numerical gates')
        else:
            require(scope['recipe']['prefix_stress_ids'] is not None,'separate stress predeclared')
            for index in range(32):
                logits=engine._forward(row['prompt_ids']+scope['recipe']['prefix_stress_ids'][:index],'long_native_declared_prefix_stress_forward')
                token,lp=engine._selected(logits,-1);del logits;trace.append({'token':token,'logprob':lp,'prefix_length':row['prompt_tokens']+index})
        physical=dose.verify(trace);end_proof=collector('long_final_attestation',session['finalize'])
        require(end_proof['identity_common']==start_proof['identity_common'] and end_proof['RNG_unchanged'] is True
            and end_proof['base310_unchanged'] is True and end_proof['zeroB_unchanged'] is True,'actual state/base/zeroB preserved')
        # The live native prehook is removed by cleanup. Close the component
        # itself first so even an unknown-operation sink call cannot allocate
        # tensors or reach the model after the frozen physical roster.
        dose.verify(trace);engine.halted=True;collector.models_closed=True;dose.models_closed=True
        close();guard('long_before_capsule_publication');dose.verify(trace)
        posted=cost();require(all(v['failed']==v['pending']==0 for v in posted['operations'].values()),'failed/pending work cannot publish success')
        payload=seal({'kind':contract.KIND,'purpose':contract.PURPOSE,'recipe_sha256':scope['recipe']['sha256'],'role':role,
            'previous_capsule_sha256':previous[-1]['capsule_chain']['sha256'] if previous else None,'start_proof':start_proof,'end_proof':end_proof,
            'trace':trace,'comparison':comparison,'physical_dose':physical,'posted_scope_costs':posted,'CPU_fixture':mode==contract.FAKE,
            'greedy_EOS_respected':role!='prefix_stress','full32_greedy_coverage':role!='prefix_stress' and len(trace)==32,
            'stress_is_greedy':False,'native_launch_authorized':False,'native_long_context_qualified':False,'study_training_authorized':False,'scoring_authorized':False})
        def publish(name,value):
            require(dose.snapshot()==physical,'finalized physical roster changed before publication')
            dose.verify(trace)
            descriptor=collector(name+'_write',lambda:preserve({'event':name,'payload':deepcopy(value)}))
            require(dose.snapshot()==physical,'sink changed finalized physical roster')
            dose.verify(trace)
            verified=collector(name+'_verify',lambda:publication(descriptor,value))
            require(dose.snapshot()==physical,'verification changed finalized physical roster')
            dose.verify(trace)
            return verified
        capsule_pub=publish('long_capsule_publish',payload)
        terminal={'capsule_sha256':payload['sha256'],'costs':project(cost(),'long_terminal_publish')}
        terminal_pub=publish('long_terminal_publish',terminal);require(cost()==terminal['costs'],'terminal projected costs reconcile')
        promotion={'capsule_sha256':payload['sha256'],'terminal_sha256':digest(terminal),'costs':project(cost(),'long_capsule_promote')}
        promotion_pub=publish('long_capsule_promote',promotion);require(cost()==promotion['costs'],'promotion projected costs reconcile')
        node=seal({'kind':contract.KIND,'recipe_sha256':scope['recipe']['sha256'],'capsule':payload,'capsule_publication':capsule_pub,
            'terminal':terminal,'terminal_publication':terminal_pub,'promotion':promotion,'promotion_publication':promotion_pub,'cumulative_costs':cost()})
        # Pure in-memory chain reconciliation adds no uncounted file reread.
        validate_chain_memory(node,scope['recipe'])
        require(dose.snapshot()==physical,'finalized roster unchanged through promotion')
        result.update({'status':'COMPLETE_NATIVE_LONG_SCOPE_PENDING_INDEPENDENT_AUDIT' if mode==contract.NATIVE else 'COMPLETE_CPU_FAKE_LONG_PROTOCOL',
            'capsule_chain':node})
    except BaseException as error:result['error']={'type':type(error).__name__,'message':str(error)}
    finally:
        try:close()
        except BaseException as error:result.update({'status':'FAILED_NATIVE_LONG_SCOPE','capsule_chain':None,'cleanup_error':str(error)})
        result['actual_costs']=cost();result['partial_trace']=deepcopy(trace);result['physical_dose']=dose.snapshot()
    result['sha256']=digest(result);return result
