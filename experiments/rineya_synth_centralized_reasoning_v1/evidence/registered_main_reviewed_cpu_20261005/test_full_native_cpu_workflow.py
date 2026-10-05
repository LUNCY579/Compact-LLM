"""Private actual source composition, real inert files, hardware fully injected."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
import hashlib,importlib.util,sys,tempfile,unittest
from outer_context_fixture import OuterContextFixture,HERE

def compiled(path):
    path=Path(path);raw=path.read_bytes();module=ModuleType('CPU_full_'+path.stem);module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    module.__compiled_source_descriptor__={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}
    return module

class FullNativeCPUWorkflow(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='CPU_composed_native_sources_',dir=HERE);self.base=Path(self.tmp.name).resolve();self.assertEqual(self.base.parent,HERE.resolve());self.addCleanup(self.tmp.cleanup)
    def composed(self,api_faults=None,mutate=None,death=None):
        fixture=OuterContextFixture(self.base);C=fixture.C;context=fixture.context()
        W=compiled(fixture.package/'prefix_input_owned_worker.py');W.C=C
        P=compiled(fixture.package/'prefix_native_pipeline.py')
        S=compiled(fixture.package/'prefix_input_owned_supervisor.py')
        A=compiled(fixture.package/'prefix_input_owned_auditor.py')
        F=compiled(HERE/'fixture_support.py');B=compiled(HERE/'test_worker_bindings.py')
        output=fixture.package/'artifacts/prefix_input_owned_run_CPU';output.mkdir()
        owner={'worker_pid':12345,'root_worker_pid':12345,'supervisor_pid':12344,'job_name':'Local\\SYNTH_NQ_JOB_'+'1'*32,'supervisor_create_time':1000.25,'config_sha256':fixture.anchors['expected_config_file']}
        inert=B.InertSuppliers(fixture,owner)
        fake=F.FakeTorch(F.Recorder(api_faults)) if hasattr(F,'FakeTorch') else F.F.FakeTorch(F.F.Recorder(api_faults))
        fake.__version__=fixture.config['torch_version'];fake.backends=inert.torch.backends
        for name in ('set_float32_matmul_precision','use_deterministic_algorithms','set_num_threads'):setattr(fake,name,getattr(inert.torch,name))
        inert.torch=fake
        host=F.FakeHost(output);host.offset=0;host.started=__import__('time').monotonic();host.assigned=host.resumed=True;host.job_name=owner['job_name'];host.supervisor_pid=owner['supervisor_pid']
        with fixture.activate():
            binding=W._CPU_real_torch_bindings(context,owner,output,_capability=C._mint_CPU_native_context_fixture(),suppliers=inert.mapping())
            execute=P._registered_core_executor(W)
            report=execute(binding['api'],W.original_guard(host,owner,host.started),output,fixture_identity=C.NATIVE_SCOPE,source_pins={p:d['sha256'] for p,d in fixture.config['source_files'].items()},runtime_pins={})
            self.assertEqual(report['status'],'NATIVE_INPUT_CHECKS_RETURNED_PENDING_TERMINAL_AUDIT' if not api_faults else 'NATIVE_INPUT_CHECKS_FAILED',report)
            membership=next(x['topology']['membership'] for x in W.OWNED_TOPOLOGY_OBSERVATIONS.values())
            W.write_owned_return(output,report,{},membership,publisher_descriptor=fixture.config['source_files'][str(fixture.package/'prefix_input_owned_supervisor.py')])
        envelope=compiled(HERE/'full_native_cpu_fixture_support.py')
        frame=envelope.materialize_full_native_CPU_fixture(fixture,output,context=context,topology=binding['topology_observations'][0],ownership=owner)
        target=fixture.package/'evidence/prefix_input_owned_CPU_COMPOSITION'
        def revalidate():
            actual=C._native_manifest_value(fixture.config_path,**fixture.anchors);value=C.context_value(context)
            for key in ('fixture_only','native_PASS','native_authority_issued','status'):value.pop(key,None)
            C.require(actual==value,'unchanged composed private context');return actual
        with fixture.activate():
            preservation=S._CPU_preserve_native_fixture(frame['raw'],target,context,contract=C,_capability=C._mint_CPU_native_context_fixture(),death_verifier=lambda _:False,revalidate=revalidate)
            if mutate is not None:mutate(fixture,frame,target,context)
            raw=(target/'PRESERVATION.json').read_bytes()
            result=A._CPU_integrated_native_audit_fixture(target,original_directory=frame['raw'],contract=C,context=context,_capability=C._mint_CPU_native_context_fixture(),death_verifier=(lambda _:False) if death is None else death,expected_preservation_file_sha256=hashlib.sha256(raw).hexdigest(),expected_preservation_body_sha256=preservation['sha256'])
        self.assertTrue(result['fixture_only']);self.assertFalse(result['native_PASS']);self.assertFalse(result['native_authority_issued'])
        self.assertFalse({'torch','transformers','jax','tensorflow','flax'}&{name.split('.')[0] for name in sys.modules})
        return result
    def test_actual_context_binding_core_writer_preserve_audit_private_CPU_path(self):
        result=self.composed();self.assertEqual(result['status'],'CPU_INTEGRATED_SOURCE_WORKFLOW_COMPLETE');self.assertEqual(result['source_body_status'],'CPU_NATIVE_SCHEMA_COMPLETE')
    def test_real_producer_exception_is_preserved_without_complete_claim(self):
        result=self.composed({'torch.tensor':RuntimeError('CPU original tensor construction failure')});self.assertEqual(result['status'],'CPU_INTEGRATED_SOURCE_FAILURE_PRESERVED');self.assertEqual(result['source_body_status'],'CPU_NATIVE_SCHEMA_FAILURE_PRESERVED')
    def test_selected_source_drift_after_preservation_rejects(self):
        def mutate(fixture,frame,target,context):
            path=fixture.package/'prefix_registered_loader.py';raw=path.read_bytes();path.write_bytes(b'#'+raw[1:])
        with self.assertRaises(ValueError):self.composed(mutate=mutate)
    def test_original_copy_pair_drift_after_preservation_rejects(self):
        def mutate(fixture,frame,target,context):
            path=frame['raw']/'input_check_summary.json';path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaises(ValueError):self.composed(mutate=mutate)
    def test_actual_current_owned_birth_not_dead_rejects(self):
        with self.assertRaises(ValueError):self.composed(death=lambda _:True)
if __name__=='__main__':unittest.main()
