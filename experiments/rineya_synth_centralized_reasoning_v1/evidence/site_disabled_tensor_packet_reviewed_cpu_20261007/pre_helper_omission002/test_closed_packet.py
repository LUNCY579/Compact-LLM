"""One linked prospective source packet; no native entry is authorized."""
from pathlib import Path
import ast,builtins,copy,hashlib,json,sys,tempfile,types,unittest
HERE=Path(__file__).resolve().parent
ORIGINAL=builtins.__import__
def trap(name,*args,**kwargs):
    if name.split('.',1)[0] in {'ctypes','_ctypes','torch','psutil','native_worker','native_supervisor','native_process','native_nvml','pywin32_bootstrap'} or name.startswith('__editable__'):raise AssertionError('closed packet cannot import native/site modules')
    return ORIGINAL(name,*args,**kwargs)
builtins.__import__=trap
def load(name):
    path=HERE/name;module=types.ModuleType('CPU_'+path.stem);module.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__);return module
C=load('prefix_tensor_contract.py');W=load('prefix_input_owned_worker.py');P=load('prefix_native_pipeline.py');S=load('prefix_input_owned_supervisor.py');A=load('prefix_input_owned_auditor.py');L=load('prefix_registered_loader.py')
class Controls(unittest.TestCase):
    def setUp(self):self.packet=json.loads((HERE/'PROSPECTIVE_PACKET_002.json').read_text(encoding='utf-8'))
    def seal(self,p):p.pop('sha256',None);p['sha256']=C.digest(p);return p
    def test_linked_sealed_packet_and_six_compiled_sources_are_closed(self):
        result=C.validate_prospective_packet(self.packet);self.assertFalse(result['native_authority_issued']);self.assertEqual(result['pending_observation_count'],11);self.assertEqual(self.packet['permission']['logical_bytes'],6752);self.assertEqual(self.packet['permission']['allocations'],2)
        self.assertTrue(self.packet['permission']['CUDA_context']);self.assertTrue(self.packet['permission']['torch_import']);self.assertFalse(self.packet['permission']['model']);self.assertEqual(P.CONTRACT_BOOTSTRAP['sha256'],hashlib.sha256((HERE/'prefix_tensor_contract.py').read_bytes()).hexdigest());self.assertIs(C._native_manifest_value,C._amended_manifest_value)
    def test_authority_pending_unknown_fields_and_input_rebounds_refuse(self):
        for key,val in [('native_launch_authorized',True),('owner_approval',{'approved':True}),('extra_approval',True)]:
            bad=copy.deepcopy(self.packet);bad[key]=val;self.seal(bad)
            with self.assertRaises(ValueError):C.validate_prospective_packet(bad)
        bad=copy.deepcopy(self.packet);bad['pending'][next(iter(bad['pending']))]=True;self.seal(bad)
        with self.assertRaises(ValueError):C.validate_prospective_packet(bad)
        bad=copy.deepcopy(self.packet);bad['input']['prefix_ids_sha256']='0'*64;self.seal(bad)
        with self.assertRaises(ValueError):C.validate_prospective_packet(bad)
    def test_source_runtime_and_observer_reference_rebinding_refuse(self):
        for group,key in [('source_candidates','prefix_input_owned_worker.py'),('helper_sources',next(iter(self.packet['helper_sources'])))]:
            bad=copy.deepcopy(self.packet);bad[group][key]['sha256']='0'*64;self.seal(bad)
            with self.assertRaises(ValueError):C.validate_prospective_packet(bad)
        bad=copy.deepcopy(self.packet);bad['runtime_candidate']['files']={};self.seal(bad)
        with self.assertRaises(ValueError):C.validate_prospective_packet(bad)
        bad=copy.deepcopy(self.packet);bad['observed_program_specific_observer006']['artifact_refs'][next(iter(bad['observed_program_specific_observer006']['artifact_refs']))]['sha256']='0'*64;self.seal(bad)
        with self.assertRaises(ValueError):C.validate_prospective_packet(bad)
    def test_native_calls_refuse_before_setup_and_dispatch_follows_definitions(self):
        for call in [lambda:C.native_entry(None),lambda:W._site_disabled_bootstrap_setup(object()),lambda:W._registered_search_roots(object()),lambda:P.supervise_registered(None,None)]:
            with self.assertRaises(ValueError):call()
        for filename,helper in [('prefix_tensor_contract.py','_amended_manifest_value'),('prefix_input_owned_worker.py','_site_disabled_bootstrap_setup')]:
            tree=ast.parse((HERE/filename).read_bytes());dispatch=[n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__'];self.assertEqual(len(dispatch),1);self.assertIs(tree.body[-1],dispatch[0]);self.assertGreater(dispatch[0].lineno,next(n.lineno for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==helper))
    def test_actual_source_argv_adaptation_and_owned_root_restoration(self):
        tree=ast.parse((HERE/'prefix_native_pipeline.py').read_bytes());func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='supervise_registered');classes=[n for n in func.body if isinstance(n,ast.ClassDef) and n.name in ('NewScope','NoSite')];ns={'ast':ast,'C':C,'changes':[],'counts':[]};exec(compile(ast.Module(body=classes,type_ignores=[]),'selected_pipeline_transformers','exec'),ns)
        raw=(C.PACKAGE/'native_supervisor.py').read_bytes();body=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='supervise');body=ns['NewScope']().visit(body);body=ns['NoSite']().visit(body);ast.fix_missing_locations(body);self.assertEqual(len(ns['changes']),3);self.assertEqual(ns['counts'],['flag',3,5])
        argv=next(n for n in body.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='argv');writes=[n for n in ast.walk(body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Subscript) and isinstance(n.targets[0].value,ast.Name) and n.targets[0].value.id=='argv'];self.assertEqual([n.targets[0].slice.value for n in writes],[4,6])
        env={'sys':types.SimpleNamespace(executable='CPU_EXE'),'PACKAGE':C.PACKAGE,'Path':Path,'config_path':Path('CPU_CONFIG'),'output':Path('CPU_OUTPUT'),'config':{'worker_path':'SELECTED_WORKER','config_path':'SELECTED_CONFIG'}};exec(compile(ast.Module(body=[argv,*writes],type_ignores=[]),'actual_selected_argv_statements','exec'),env);self.assertEqual(env['argv'][1:4],['-I','-S','-B']);self.assertEqual(env['argv'][4],'SELECTED_WORKER');self.assertEqual(env['argv'][6],'SELECTED_CONFIG')
        original=sys.path;before=list(original);original.append('CPU_OWNED_ROOT');transition={'original_container':original,'before':tuple(before),'path_after':list(original)};W._restore_registered_roots(transition);self.assertIs(sys.path,original);self.assertEqual(sys.path,before)
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));r={'kind':'CLOSED_SITE_DISABLED_TENSOR_SOURCE_PACKET_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'native_API_or_ctypes_Torch_site_imported':False,'native_authority_issued':False,'source_hashes':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['prefix_tensor_contract.py','prefix_input_owned_worker.py','prefix_native_pipeline.py','prefix_input_owned_supervisor.py','prefix_input_owned_auditor.py','prefix_registered_loader.py','registration_git_join.py','test_closed_packet.py']},'packet_file_sha256':hashlib.sha256((HERE/'PROSPECTIVE_PACKET_002.json').read_bytes()).hexdigest()}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
