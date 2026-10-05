"""CPU sparse protocol controls; no native model, encoder, framework or binary."""
from copy import deepcopy
import importlib.util
import hashlib,json,os
from pathlib import Path
import sys,tempfile,unittest,subprocess
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
STAGED=HERE.name=='long_native_qualification_work' and HERE.parent.name=='artifacts'
PACKAGE=HERE.parents[1] if STAGED else HERE.parent
CODE=HERE if STAGED else PACKAGE
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(CODE))
import long_native_contract as contract
import long_native_core as core
import long_native_worker as worker
import long_native_supervisor as supervisor
import long_inference_component as component
spec=importlib.util.spec_from_file_location('long_sparse_fixture',PACKAGE/'tests/test_inference_component.py')
fake=importlib.util.module_from_spec(spec);spec.loader.exec_module(fake)
PACKET=json.loads(contract.PACKET.read_bytes())


class Sink:
    def __init__(self,root):self.root=Path(root);self.sequence=0;self.fail=None
    def __call__(self,event):
        if event['event']==self.fail:raise OSError('fixture publication failure')
        self.sequence+=1;raw=(core.canonical(event['payload'])+'\n').encode();path=self.root/('%03d_%s.json'%(self.sequence,event['event']))
        with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'payload_sha256':core.digest(event['payload']),'durable':True}


class Factory:
    execution_scope=contract.FAKE
    def __init__(self,tokens=None):self.tokens=[9]*32 if tokens is None else tokens;self.mutate=None;self.sessions=[]
    def __call__(self,scope,*,call,guard,dose):
        torch=fake.FakeTorch();length=scope['row']['prompt_tokens'];tokens=self.tokens
        class PhysicalModel(fake.Model):
            def __call__(model,**kwargs):dose.charge(kwargs['input_ids'].tolist()[0]);return super().__call__(**kwargs)
        model=PhysicalModel(torch,tokens=tokens,prompt_length=length)
        call('fixture_loader_constructor',lambda:None)
        def parity():
            ids=torch.tensor([scope['short_parity_row']['prompt_ids']],dtype=torch.long,device='cpu')
            with torch.no_grad():return model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True)
        call('long_native_dense_parity_forward',parity);call('long_native_zeroB_parity_forward',parity)
        engine=component.InferenceComponent(torch,model,fake.Decoder(),call,guard,lambda *args:None,
            max_attempted_forwards=1 if scope['role']=='bulk' else 32,
            max_input_positions=scope['dose']['maximum_forward_input_positions']-838,context_cap=scope['dose']['context_cap'],generation_cap=32,device='cpu')
        common={'CPU_fixture_identity_only':True,'recipe_sha256':scope['recipe']['sha256'],'seed':101}
        session={'component':engine,'attest':lambda:{'kind':'CPU_FIXTURE_LONG_START','identity_common':common},
            'finalize':lambda:{'identity_common':common,'RNG_unchanged':True,'base310_unchanged':True,'zeroB_unchanged':True,'CPU_fixture':True},
            'cleanup':lambda:call('fixture_owned_cleanup',lambda:None)}
        self.sessions.append(session)
        if self.mutate:self.mutate(session,scope)
        return session


class Controls(unittest.TestCase):
    def scope(self,role='trace1',cap=4096,stress=None):
        item=next(x for x in PACKET['rows'] if x['context_cap']==cap);recipe={'purpose':contract.PURPOSE,'context_cap':cap,
            'fixture_source_only':True,'prefix_stress_ids':stress};recipe=core.seal(recipe)
        return {'purpose':contract.PURPOSE,'authorization':deepcopy(contract.AUTHORIZATION),'recipe':recipe,'role':role,
            'row':deepcopy(item['row']),'short_parity_row':deepcopy(PACKET['short_parity_row']),'dose':contract.work(item['row'],role),'references':None}
    def execute(self,root,role,previous=None,factory=None,sink=None,stress=None):
        directory=Path(root)/role;directory.mkdir(exist_ok=True);factory=factory or Factory()
        return core.run_scope(self.scope(role,stress=stress),factory,lambda name,fn:fn(),lambda name:None,sink or Sink(directory),previous_fixtures=previous)
    def test_three_fresh_full32_scopes_immutable_costs_and_closed_authority(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            factory=Factory();prior=[]
            for role in ('trace1','trace2','bulk'):
                result=self.execute(root,role,prior,factory)
                self.assertEqual('COMPLETE_CPU_FAKE_LONG_PROTOCOL',result['status'],result.get('error'));prior.append(result)
            costs=prior[-1]['actual_costs'];self.assertEqual(71,costs['physical_calls']);self.assertEqual(267697,costs['input_positions'])
            self.assertEqual(3,costs['operations']['fixture_loader_constructor']['returned']);self.assertEqual(3,costs['operations']['fixture_owned_cleanup']['returned'])
            self.assertEqual(32,prior[-1]['capsule_chain']['capsule']['comparison']['actual_compared_steps'])
            self.assertTrue(prior[-1]['capsule_chain']['capsule']['CPU_fixture']);self.assertFalse(prior[-1]['native_long_context_qualified'])
    def test_early_EOS_and_separate_stress_keep_actual_coverage(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            factory=Factory([9,151645]);prior=[];stress=[20628]*31
            for role in ('trace1','trace2','bulk','prefix_stress'):
                result=self.execute(root,role,prior,factory,stress=stress)
                self.assertEqual('COMPLETE_CPU_FAKE_LONG_PROTOCOL',result['status'],result.get('error'));prior.append(result)
            self.assertEqual(11,prior[2]['actual_costs']['physical_calls']);self.assertEqual(22837,prior[2]['actual_costs']['input_positions'])
            capsule=prior[3]['capsule_chain']['capsule'];self.assertFalse(capsule['greedy_EOS_respected']);self.assertFalse(capsule['full32_greedy_coverage'])
            self.assertEqual(45,prior[3]['actual_costs']['physical_calls']);self.assertEqual(154219,prior[3]['actual_costs']['input_positions'])
    def test_bulk_steps_reconstruct_actual_EOS_and_full32_statistics(self):
        for tokens in ([9,151645],[9]*32):
            with tempfile.TemporaryDirectory(dir=HERE) as root:
                prior=[]
                for role in ('trace1','trace2','bulk'):
                    result=self.execute(root,role,prior,Factory(tokens));prior.append(result)
                    self.assertEqual('COMPLETE_CPU_FAKE_LONG_PROTOCOL',result['status'],result.get('error'))
                capsule=result['capsule_chain']['capsule'];comparison=capsule['comparison'];steps=comparison['bulk_steps']
                self.assertEqual(len(tokens),len(steps));self.assertEqual(comparison,result['bulk_comparison'])
                for index,(step,observed) in enumerate(zip(steps,capsule['trace'])):
                    self.assertEqual({'step','position','argmax_token','argmax_logprob','selected_token',
                        'selected_logprob','trace_logprob','absolute_error'},set(step))
                    self.assertEqual(index,step['step']);self.assertEqual(4063+index,step['position'])
                    self.assertEqual(observed['token'],step['selected_token']);self.assertEqual(observed['logprob'],step['trace_logprob'])
                    self.assertEqual(step['selected_token'],step['argmax_token']);self.assertEqual(step['selected_logprob'],step['argmax_logprob'])
                    self.assertEqual(abs(step['selected_logprob']-step['trace_logprob']),step['absolute_error'])
                errors=[step['absolute_error'] for step in steps]
                self.assertEqual(max(errors),comparison['maximum_error']);self.assertEqual(sum(errors)/len(errors),comparison['mean_error'])
                self.assertEqual(1,result['actual_costs']['operations']['long_native_bulk_forward']['returned'])
                self.assertNotIn('long_bulk_original_selected_LP',result['actual_costs']['operations'])
    def test_bulk_argmax_failure_retains_original_selected_probability_and_work(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            first=self.execute(root,'trace1',factory=Factory([9,151645]));second=self.execute(root,'trace2',[first],Factory([9,151645]))
            result=self.execute(root,'bulk',[first,second],Factory([10,151645]))
            self.assertEqual('FAILED_NATIVE_LONG_SCOPE',result['status']);self.assertIsNone(result['capsule_chain'])
            comparison=result['bulk_comparison'];self.assertEqual([0],comparison['argmax_differences'])
            step=comparison['bulk_steps'][0];self.assertEqual(10,step['argmax_token']);self.assertEqual(9,step['selected_token'])
            self.assertLess(step['selected_logprob'],step['argmax_logprob']);self.assertEqual(10.,step['absolute_error'])
            self.assertEqual(1,result['actual_costs']['operations']['long_bulk_original_selected_LP']['returned'])
            self.assertEqual(3,result['physical_dose']['attempted_calls']);self.assertEqual(4903,result['physical_dose']['input_positions'])
    def test_hidden_metered_extra_call_cannot_expand_physical_budget(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            factory=Factory([9,151645])
            def mutate(session,scope):
                engine=session['component'];old=engine.call;done=[False]
                def wrapped(name,fn):
                    value=old(name,fn)
                    if name=='long_native_greedy_forward' and not done[0]:
                        done[0]=True;engine.max_attempted_forwards=100;engine.max_input_positions=200000
                        engine._forward(scope['short_parity_row']['prompt_ids'],'long_native_greedy_forward')
                    return value
                engine.call=wrapped
            factory.mutate=mutate;result=self.execute(root,'trace1',factory=factory)
            self.assertEqual('FAILED_NATIVE_LONG_SCOPE',result['status']);self.assertIsNone(result['capsule_chain'])
            self.assertEqual(5,result['actual_costs']['physical_calls'])
    def test_token_and_terminal_publication_failures_retain_attempted_returned_costs(self):
        for fail in ('long_token_01','long_terminal_publish'):
            with tempfile.TemporaryDirectory(dir=HERE) as root:
                sink=Sink(root);sink.fail=fail;result=self.execute(root,'trace1',sink=sink)
                self.assertEqual('FAILED_NATIVE_LONG_SCOPE',result['status']);self.assertIsNone(result['capsule_chain'])
                self.assertGreater(result['actual_costs']['physical_calls'],2)
                self.assertEqual(1,result['actual_costs']['operations'][fail+'_write']['failed'])
                self.assertEqual(1,result['actual_costs']['operations']['fixture_owned_cleanup']['returned'])
    def test_source_recipe_wrong_previous_and_published_hash_mutation_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            first=self.execute(root,'trace1');first['capsule_chain']['capsule']['trace'][0]['token']=10
            with self.assertRaises(ValueError):self.execute(root,'trace2',[first])
            scope=self.scope();scope['authorization']['scoring']=True
            with self.assertRaises(ValueError):core.run_scope(scope,Factory(),lambda n,f:f(),lambda n:None,Sink(root))
    def test_bulk_repeat_and_numerical_mismatch_preserve_partial_costs(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            first=self.execute(root,'trace1');second=self.execute(root,'trace2',[first],Factory([10]*32))
            result=self.execute(root,'bulk',[first,second]);self.assertEqual('FAILED_NATIVE_LONG_SCOPE',result['status'])
            self.assertEqual(70,result['actual_costs']['physical_calls'])
    def test_native_admission_absent_before_factory_and_stage_entries_deny(self):
        factory=Factory();factory.execution_scope=contract.NATIVE
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            result=core.run_scope(self.scope(),factory,lambda n,f:f(),lambda n:None,Sink(root))
            self.assertEqual('FAILED_NATIVE_LONG_SCOPE',result['status']);self.assertEqual([],factory.sessions)
        if contract.STAGED:
            with self.assertRaisesRegex(ValueError,'staged'):contract.preflight(HERE/'none.json')
            with self.assertRaisesRegex(ValueError,'staged'):contract.write_scope(HERE/'none.json',4096,'trace1')
            with self.assertRaisesRegex(ValueError,'staged'):worker.run(HERE/'none.json',HERE)
            with patch.object(supervisor,'_new_output',side_effect=AssertionError('must not create output')):
                with self.assertRaisesRegex(ValueError,'staged'):supervisor.supervise_long_native_qualification(HERE/'none.json',HERE)
        else:
            with tempfile.TemporaryDirectory(dir=HERE) as temporary:
                stage=Path(temporary)/'artifacts/long_native_qualification_work';stage.mkdir(parents=True)
                for name in contract.FILES:(stage/name).write_bytes((CODE/name).read_bytes())
                script="""import sys
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[1])
import long_native_contract as c,long_native_worker as w,long_native_supervisor as s
assert c.STAGED
for fn in (lambda:c.preflight(c.HERE/'none.json'),lambda:c.write_scope(c.HERE/'none.json',4096,'trace1'),lambda:w.run(c.HERE/'none.json',c.HERE),lambda:s.supervise_long_native_qualification(c.HERE/'none.json',c.HERE)):
 try:fn()
 except ValueError as error:assert 'staged' in str(error)
 else:raise AssertionError('native stage gate opened')
assert not any(n in sys.modules for n in ('torch','transformers','tokenizers'))
"""
                result=subprocess.run([sys.executable,'-I','-B','-c',script,str(stage),str(Path(contract.native.__file__).parent)],capture_output=True,text=True,timeout=30)
                self.assertEqual(0,result.returncode,result.stderr)
        self.assertFalse(any(name in sys.modules for name in ('torch','transformers','tokenizers')))
    def test_cleanup_failure_and_deadline_prevent_any_success_publication(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            factory=Factory()
            def fail(session,scope):
                def cleanup():raise RuntimeError('fixture cleanup failed')
                session['cleanup']=cleanup
            factory.mutate=fail;result=self.execute(root,'trace1',factory=factory)
            self.assertEqual('FAILED_NATIVE_LONG_SCOPE',result['status']);self.assertIsNone(result['capsule_chain'])
            self.assertEqual(34,result['actual_costs']['physical_calls'])
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            times=iter([0.,601.,601.])
            result=core.run_scope(self.scope(),Factory(),lambda n,f:f(),lambda n:None,Sink(root),clock=lambda:next(times,601.))
            self.assertEqual('FAILED_NATIVE_LONG_SCOPE',result['status']);self.assertEqual(0,result['actual_costs']['physical_calls'])
    def test_publication_sink_extra_metered_forward_denied_before_model_and_chain_success(self):
        for publication in ('long_capsule_publish','long_terminal_publish','long_capsule_promote'):
            with tempfile.TemporaryDirectory(dir=HERE) as root:
                factory=Factory([9,151645]);sink=Sink(root)
                def preserving(event):
                    result=sink(event)
                    if event['event']==publication:
                        engine=factory.sessions[0]['component']
                        engine._forward(self.scope()['short_parity_row']['prompt_ids'],'long_native_greedy_forward')
                    return result
                result=self.execute(root,'trace1',factory=factory,sink=preserving)
                self.assertEqual('FAILED_NATIVE_LONG_SCOPE',result['status']);self.assertIsNone(result['capsule_chain'])
                self.assertEqual(4,result['actual_costs']['physical_calls']);self.assertEqual(8967,result['actual_costs']['input_positions'])
                self.assertEqual(2,result['actual_costs']['operations']['long_native_greedy_forward']['returned'])
                self.assertIn('halted',result['error']['message'])
    def test_removed_native_hook_unknown_postcleanup_operation_stops_before_tensor_model(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            factory=Factory([9,151645]);sink=Sink(root)
            def mutate(session,scope):
                original=session['cleanup']
                def remove_hook():
                    original()
                    # Emulate native cleanup removing the physical model hook.
                    physical=session['component'].model.__call__.__closure__
                    dose=next(cell.cell_contents for cell in physical if isinstance(cell.cell_contents,core.PhysicalDose))
                    dose.charge=lambda ids:None
                session['cleanup']=remove_hook
            factory.mutate=mutate
            def preserving(event):
                pub=sink(event)
                if event['event']=='long_capsule_publish':
                    factory.sessions[0]['component']._forward(self.scope()['short_parity_row']['prompt_ids'],'unregistered_post_cleanup_forward')
                return pub
            result=self.execute(root,'trace1',factory=factory,sink=preserving)
            self.assertEqual('FAILED_NATIVE_LONG_SCOPE',result['status']);self.assertIsNone(result['capsule_chain'])
            self.assertEqual(4,len(factory.sessions[0]['component'].model.calls));self.assertEqual(4,result['actual_costs']['physical_calls'])
            self.assertEqual(8967,result['actual_costs']['input_positions']);self.assertIn('halted',result['error']['message'])
            self.assertNotIn('unregistered_post_cleanup_forward',result['actual_costs']['operations'])
    def test_failed_finalization_closes_component_before_cleanup_removes_hook(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            factory=Factory([9,151645]);observed=[]
            def mutate(session,scope):
                session['finalize']=lambda:(_ for _ in ()).throw(RuntimeError('finalize failed'))
                def cleanup():
                    engine=session['component'];observed.append(engine.halted)
                    try:engine._forward(scope['short_parity_row']['prompt_ids'],'unknown_cleanup_forward')
                    except ValueError:pass
                    else:raise AssertionError('model reached after failure cleanup')
                session['cleanup']=cleanup
            factory.mutate=mutate;result=self.execute(root,'trace1',factory=factory)
            self.assertEqual([True],observed);self.assertEqual('FAILED_NATIVE_LONG_SCOPE',result['status'])
            self.assertEqual(4,len(factory.sessions[0]['component'].model.calls));self.assertEqual(4,result['actual_costs']['physical_calls'])
            self.assertEqual(8967,result['actual_costs']['input_positions']);self.assertIsNone(result['capsule_chain'])
    def test_pure_chain_projection_rejects_posted_cost_erasure_without_file_IO(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            result=self.execute(root,'trace1');node=deepcopy(result['capsule_chain']);recipe=self.scope()['recipe']
            self.assertIsInstance(core.validate_chain_memory(node,recipe),dict)
            node['capsule']['posted_scope_costs']['operations']={}
            node['capsule']=core.seal({k:v for k,v in node['capsule'].items() if k!='sha256'})
            node=core.seal({k:v for k,v in node.items() if k!='sha256'})
            with self.assertRaisesRegex(ValueError,'posted'):core.validate_chain_memory(node,recipe)
    def test_exact_transitive_package_sources_and_additive_AST_runtime(self):
        actual=contract.source_pins()
        for name in ('inference_component.py','study_schedule.py'):
            key=str((PACKAGE/name).resolve());self.assertIn(key,actual)
            bad=deepcopy(actual);bad.pop(key)
            with self.assertRaisesRegex(ValueError,'executing package'):contract.validate_source_pins(bad)
        # Inspect only loaded package source paths, never framework/binary files.
        imported={str(Path(m.__file__).resolve()) for m in sys.modules.values() if getattr(m,'__file__',None)
            and Path(m.__file__).suffix=='.py' and Path(m.__file__).resolve().parent==PACKAGE.resolve()}
        self.assertLessEqual(imported,set(actual))
        inherited=contract.native.read_json(PACKAGE/'configs/SYNTH_NQ001_PROC7.json')['runtime_pins']
        runtime=contract.base_runtime_pins();self.assertEqual(64,len(inherited));self.assertEqual(65,len(runtime))
        self.assertEqual(inherited,{key:runtime[key] for key in inherited})
        with self.assertRaisesRegex(ValueError,'additive AST'):contract.validate_runtime_pins(inherited)
        ast_key=next(key for key in runtime if Path(key).name=='ast.py');bad=deepcopy(runtime);bad[ast_key]='0'*64
        with self.assertRaisesRegex(ValueError,'additive AST'):contract.validate_runtime_pins(bad)
    def test_package_only_encoder_candidate_cannot_dispatch_when_resealed(self):
        for flag in (None,False):
            with tempfile.TemporaryDirectory(dir=HERE) as root:
                package=Path(root);directory=package/'artifacts/long_native_audit_registrations';directory.mkdir(parents=True)
                body={'kind':'SYNTH_LONG_ENCODER_RUNTIME_REGISTRATION_V1','runtime_pins':{}}
                if flag is not None:body['complete_runtime_execution_closure_established']=flag
                body=core.seal(body);path=directory/'CPU_FIXTURE_NOT_REGISTRATION.json';path.write_text(core.canonical(body))
                value={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                with patch.object(contract,'PACKAGE',package),patch.object(contract,'encoder_runtime_paths',return_value=[]):
                    with self.assertRaisesRegex(ValueError,'package-only'):contract.encoder_registration(value)


if __name__=='__main__':unittest.main()
