"""Actual worker binding/factory body with CPU context and trapped suppliers."""
from copy import deepcopy
from pathlib import Path
from types import ModuleType,SimpleNamespace
import builtins,hashlib,tempfile,unittest
from unittest.mock import patch
from outer_context_fixture import OuterContextFixture,HERE

def compiled_worker():
    path=HERE/'prefix_input_owned_worker.py';raw=path.read_bytes();module=ModuleType('CPU_actual_worker_binding_source');module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    module.__compiled_source_descriptor__={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}
    return module

class InertSuppliers:
    def __init__(self,fixture,owner,*,membership_fault=None):
        self.events=[];self.fixture=fixture;self.owner=deepcopy(owner);self.membership_fault=membership_fault
        self.torch=SimpleNamespace(__version__=fixture.config['torch_version'],backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),cudnn=SimpleNamespace(allow_tf32=True)),set_float32_matmul_precision=lambda value:self.events.append(('precision',value)),use_deterministic_algorithms=lambda value:self.events.append(('deterministic',value)),set_num_threads=lambda value:self.events.append(('threads',value)))
    def native_worker(self):
        self.events.append('native_worker_supplier');return SimpleNamespace(verify_worker_ownership=self.verify)
    def verify(self,owner,config_path):
        self.events.append('ownership_verify')
        if owner!=self.owner or config_path!=str(self.fixture.config_path):raise ValueError('inert exact ownership supplier arguments')
        result={'actual_worker_pid':owner['worker_pid'],'actual_parent_pid':owner['supervisor_pid'],'launcher_root_pid':owner['worker_pid'],'supervisor_pid':owner['supervisor_pid'],'job_name':owner['job_name'],'owned_job_membership_verified':True,'query_job_handle_closed_before_framework_import':True}
        if self.membership_fault is not None:self.membership_fault(result)
        return result
    def disk(self,output):
        self.events.append('disk_free')
        if Path(output).parent!=self.fixture.package/'artifacts':raise ValueError('inert selected output')
        return 64*1024**3
    def framework(self):self.events.append('framework_supplier');return self.torch
    def growth(self):self.events.append('growth_supplier');return {'available_bytes':8*1024**3,'worker_RSS_bytes':512*1024**2}
    def seed(self,value):self.events.append(('python_seed',value))
    def mapping(self):return {'native_worker':self.native_worker,'disk_free':self.disk,'framework':self.framework,'growth':self.growth,'python_seed':self.seed}

class WorkerBindingControls(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='CPU_actual_worker_bindings_',dir=HERE);self.root=Path(self.temporary.name).resolve()
        self.fixture=OuterContextFixture(self.root);self.C=self.fixture.C;self.context=self.fixture.context();self.worker=compiled_worker();self.worker.C=self.C
        self.output=self.fixture.package/'artifacts/prefix_input_owned_run_CPU';self.output.mkdir()
        self.owner={'worker_pid':12345,'root_worker_pid':12345,'supervisor_pid':12344,'job_name':'Local\\SYNTH_NQ_JOB_'+'1'*32,'supervisor_create_time':1000.25,'config_sha256':self.fixture.anchors['expected_config_file']}
    def tearDown(self):self.temporary.cleanup()
    def bind(self,suppliers=None,owner=None):
        suppliers=InertSuppliers(self.fixture,self.owner) if suppliers is None else suppliers
        with self.fixture.activate():
            proof=self.worker._CPU_real_torch_bindings(self.context,self.owner if owner is None else owner,self.output,_capability=self.C._mint_CPU_native_context_fixture(),suppliers=suppliers.mapping())
        return proof,suppliers
    def test_actual_binding_factory_calls_actual_context_body_and_precise_consumers(self):
        original=self.C.context_value(self.context);proof,suppliers=self.bind();api=proof['api']
        self.assertEqual(proof['kind'],'CPU_ACTUAL_TENSOR_BINDING_FACTORY_SOURCE_FIXTURE');self.assertTrue(proof['fixture_only']);self.assertFalse(proof['native_PASS']);self.assertFalse(proof['native_authority_issued']);self.assertEqual(suppliers.events,[])
        original_checks=len(self.fixture.actual_file_checks)
        with self.fixture.activate():
            initial=api['config_preflight']();ownership=api['ownership_verify']();self.assertEqual(api['disk_free'](),64*1024**3)
            self.assertIs(api['torch_import'](),suppliers.torch);self.assertEqual(api['torch_version_read'](),self.fixture.config['torch_version'])
            self.assertEqual(api['growth_observation'](),{'available_bytes':8*1024**3,'worker_RSS_bytes':512*1024**2})
            api['configure']();api['python_seed'](101);final=api['config_preflight']()
        self.assertEqual(initial,final);self.assertEqual(self.C.context_value(self.context),original)
        self.assertGreater(len(self.fixture.actual_file_checks),original_checks)
        self.assertEqual(suppliers.events,['native_worker_supplier','ownership_verify','disk_free','framework_supplier','growth_supplier',('precision','highest'),('deterministic',True),('threads',1),('python_seed',101)])
        self.assertFalse(suppliers.torch.backends.cuda.matmul.allow_tf32);self.assertFalse(suppliers.torch.backends.cudnn.allow_tf32)
        self.assertEqual(ownership['actual_worker_pid'],self.owner['worker_pid']);self.assertEqual(ownership['actual_parent_pid'],self.owner['supervisor_pid'])
        actual=proof['factory_source']['compiled_source'];self.assertEqual(actual['sha256'],self.worker.__compiled_source_descriptor__['sha256']);self.assertTrue(actual['current_module_execution_observed'])
        self.assertEqual(proof['factory_source']['selected_source_path'],str(self.fixture.package/'prefix_input_owned_worker.py'));self.assertTrue(proof['factory_source']['fixture_rebased_source_path'])
    def test_preflight_ownership_disk_import_order_fails_before_suppliers(self):
        proof,suppliers=self.bind();api=proof['api']
        for key,args in (('ownership_verify',()),('disk_free',()),('torch_import',()),('torch_version_read',()),('growth_observation',()),('configure',()),('python_seed',(101,))):
            with self.assertRaises(ValueError):api[key](*args)
        self.assertEqual(suppliers.events,[])
        with self.fixture.activate():api['config_preflight']()
        with self.assertRaises(ValueError):api['torch_import']()
        self.assertEqual(suppliers.events,[]);api['ownership_verify']()
        with self.assertRaises(ValueError):api['torch_import']()
        self.assertNotIn('framework_supplier',suppliers.events);api['disk_free']();api['torch_import']()
        with self.assertRaises(ValueError):api['torch_import']()
        with self.assertRaises(ValueError):api['python_seed'](True)
        with self.assertRaises(ValueError):api['python_seed'](102)
    def test_exact_native_owner_fields_types_and_selected_config_are_required(self):
        changes=({'worker_pid':True},{'root_worker_pid':3},{'supervisor_pid':12345},{'supervisor_create_time':float('nan')},{'config_sha256':'0'*64},{'job_name':'CPU_not_native_job'},{'extra':1})
        for change in changes:
            bad={**self.owner,**change};suppliers=InertSuppliers(self.fixture,self.owner)
            with self.assertRaises(ValueError):self.bind(suppliers,owner=bad)
            self.assertEqual(suppliers.events,[])
    def test_mutated_original_owner_rejected_before_context_or_supplier_reads(self):
        proof,suppliers=self.bind();self.owner['worker_pid']=999
        with self.fixture.activate(),patch.object(self.C,'_native_manifest_value',side_effect=AssertionError('must not read changed owner context')):
            with self.assertRaisesRegex(ValueError,'immutable exact original ownership'):proof['api']['config_preflight']()
        self.assertEqual(suppliers.events,[])
    def test_typed_actual_membership_dictionary_mismatches_fail_before_framework(self):
        for mutate in (lambda x:x.update(actual_worker_pid=True),lambda x:x.update(actual_parent_pid=99),lambda x:x.update(launcher_root_pid=99),lambda x:x.update(supervisor_pid=99),lambda x:x.update(job_name='changed'),lambda x:x.update(owned_job_membership_verified=1),lambda x:x.update(query_job_handle_closed_before_framework_import=False),lambda x:x.update(extra=True)):
            suppliers=InertSuppliers(self.fixture,self.owner,membership_fault=mutate);proof,_=self.bind(suppliers)
            with self.fixture.activate():proof['api']['config_preflight']()
            with self.assertRaises(ValueError):proof['api']['ownership_verify']()
            with self.assertRaises(ValueError):proof['api']['torch_import']()
            self.assertNotIn('framework_supplier',suppliers.events)
    def test_actual_final_context_hash_detects_selected_source_mutation(self):
        proof,suppliers=self.bind();api=proof['api']
        with self.fixture.activate():api['config_preflight']()
        api['ownership_verify']();api['disk_free']()
        p=self.fixture.package/'prefix_registered_loader.py';raw=p.read_bytes();p.write_bytes(b'#'+raw[1:])
        with self.fixture.activate():
            with self.assertRaisesRegex(ValueError,'registered runtime exact stable bytes'):api['config_preflight']()
        with self.assertRaisesRegex(ValueError,'current preflight'):api['torch_import']()
        self.assertEqual(suppliers.events,['native_worker_supplier','ownership_verify','disk_free'])
    def test_private_capability_and_same_opaque_context_are_required(self):
        suppliers=InertSuppliers(self.fixture,self.owner)
        with self.fixture.activate():
            with self.assertRaisesRegex(ValueError,'private CPU'):self.worker._CPU_real_torch_bindings(self.context,self.owner,self.output,_capability=object(),suppliers=suppliers.mapping())
            with self.assertRaisesRegex(ValueError,'actual integrated'):self.worker._CPU_real_torch_bindings(self.C.context_value(self.context),self.owner,self.output,_capability=self.C._mint_CPU_native_context_fixture(),suppliers=suppliers.mapping())
        self.assertEqual(suppliers.events,[])
    def test_public_stage_and_CPU_context_consumers_refuse_before_imports_and_body(self):
        actual_import=builtins.__import__
        def trapped(name,*args,**kwargs):
            if name.split('.',1)[0] in {'torch','native_worker','psutil','random','shutil'}:raise AssertionError('native binding import forbidden')
            return actual_import(name,*args,**kwargs)
        with patch.object(builtins,'__import__',side_effect=trapped),patch.object(self.worker,'_tensor_binding_body',side_effect=AssertionError('native body must not run')):
            with self.assertRaisesRegex(ValueError,'no native authority'):self.worker.real_torch_bindings(self.context,self.owner,self.output)
            with patch.object(self.C,'stage_native_gate'):
                with self.assertRaisesRegex(ValueError,'cannot create real native bindings'):self.worker.real_torch_bindings(self.context,self.owner,self.output)
    def test_compile_descriptor_mismatch_rejected_before_suppliers(self):
        self.worker.__compiled_source_descriptor__['sha256']='0'*64;suppliers=InertSuppliers(self.fixture,self.owner)
        with self.assertRaisesRegex(ValueError,'compile descriptor retained'):self.bind(suppliers)
        self.assertEqual(suppliers.events,[])

if __name__=='__main__':unittest.main()
