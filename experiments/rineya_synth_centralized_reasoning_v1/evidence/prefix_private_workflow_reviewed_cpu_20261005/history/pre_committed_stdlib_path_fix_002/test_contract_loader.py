"""Pure/inert contract/importer negatives; no real runtime or cache byte reads."""
import hashlib
import json
import importlib.machinery
import importlib.util
from pathlib import Path
import stat
import tempfile
import types
from copy import deepcopy
import unittest
from unittest.mock import patch,Mock

HERE=Path(__file__).resolve().parent
def source(name):
    p=HERE/name;module=types.ModuleType('CPU_contract_loader_'+p.stem);module.__file__=str(p)
    exec(compile(p.read_bytes(),str(p),'exec'),module.__dict__);return module
C=source('prefix_tensor_contract.py')
L=source('prefix_registered_loader.py')

def metadata_fixture(package=None):
    """Synthetic native-shaped maps from preserved JSON/source metadata only.

    No runtime path is opened or statted. These maps are not qualified bytes,
    and the actual context route must run with all external/live boundaries
    injected. The original JSON documents are source metadata, never permits.
    """
    package=C.PACKAGE if package is None else Path(package)
    actual_package=HERE.parents[1]
    original=json.loads((actual_package/'configs/SYNTH_CONTEXT_PHASE_NO_MODEL_D001.json').read_bytes())
    stdlib=json.loads((actual_package/'artifacts/stdlib_runtime_closure_work/STDLIB_CANDIDATE_001.json').read_bytes())
    def seal(value):value=dict(value);value['sha256']=C.digest(value);return value
    def descriptor(raw,role='python_source'):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':role}
    files={}
    for path,pin in {**original['runtime_pins'],**stdlib['inventory']['runtime_pins']}.items():
        meta=stdlib['inventory']['files'].get(path);files[path]={'bytes':1 if meta is None else meta['bytes'],'sha256':pin,'role':C.runtime_role(path,original['runtime_pins'])}
    single=str(C.SITE/'typing_extensions.py');files[single]=descriptor(b'# inert external source metadata\n')
    external={single:files[single]}
    sources={};source_raw={}
    for name in ('prefix_tensor_contract.py','prefix_input_owned_worker.py','prefix_input_owned_supervisor.py','prefix_input_owned_auditor.py','prefix_registered_loader.py','prefix_native_pipeline.py'):
        raw=(HERE/name).read_bytes();path=str(package/name);sources[path]=descriptor(raw);source_raw[path]=raw
    for name,pin in original['helper_pins'].items():
        path=str(package/name);sources[path]={'bytes':1,'sha256':pin,'role':'python_source'}
    head='a'*40;config_path=str(package/'configs/CPU_NATIVE_SOURCE_FIXTURE.json');registration_path=str(package/'artifacts/CPU_NATIVE_SOURCE_REGISTRATION.json');runtime_path=str(package/'artifacts/CPU_NATIVE_SOURCE_RUNTIME.json');review_path=str(package/'artifacts/CPU_NATIVE_SOURCE_REVIEW.json');bootstrap_path=str(package/'artifacts/CPU_NATIVE_BOOTSTRAP_ASSUMPTIONS.json')
    preload={'CPU_bootstrap_inert':{'path':single,'source_sha256':files[single]['sha256'],'cache_path':None,'cache_sha256':None}}
    bootstrap=seal({'kind':'ROOT_QUALIFIED_FRESH_BOOTSTRAP_BYTE_SOURCE_ASSUMPTIONS','source_commit':head,'preloaded_bootstrap_modules_sha256':C.digest(preload),'fresh_process_bootstrap_qualified':True,'environment_and_loader_assumptions_qualified':True,'current_module_execution_proven':False})
    def anchor(path,value):return {'path':path,'file_sha256':hashlib.sha256((C.canonical(value)+'\n').encode()).hexdigest(),'body_sha256':value['sha256']}
    names=['stdlib_python_source','stdlib_extension_runtime','interpreter_root_executable_runtime','torch']+['external_package:'+name for name in ('psutil','safetensors','transformers')]
    rosters={}
    for name in names:
        root,suffixes,excluded,recursive=C.roster_policy(name)
        selected={p:{'bytes':d['bytes']} for p,d in files.items() if Path(p).is_relative_to(root) and Path(p).suffix in suffixes and (recursive or Path(p).parent==root)}
        rosters[name]={'root':str(root),'suffixes':suffixes,'excluded_directory_names':excluded,'recursive':recursive,'files':selected}
    cache={'mode':'all_registered_python_source_compiled_without_cache','allow_unregistered':False,'source_loader_sha256':sources[str(package/'prefix_registered_loader.py')]['sha256'],'Torch_root':str(C.SITE/'torch'),'preloaded_module_cache_pins':{},'namespace_roots':{},'preloaded_bootstrap_modules':preload,'fresh_bootstrap_evidence':anchor(bootstrap_path,bootstrap)}
    runtime=seal({'kind':'ROOT_QUALIFIED_TENSOR_RUNTIME_SOURCE_BYTE_CLOSURE','source_commit':head,'current_binary_bytes_qualified':True,'external_dependencies_complete':True,'interpreter_layout_qualified':True,'cache_execution_policy_qualified':True,'files':files,'inherited65_map':original['runtime_pins'],'stdlib779_map':stdlib['inventory']['runtime_pins'],'cache_policy':cache,'external_dependency_files':external,'rosters':rosters,'library_search_policy':{'mode':'qualified_complete_registered_DLL_search_rosters','roster_names':['torch','stdlib_extension_runtime','interpreter_root_executable_runtime'],'unregistered_search_allowed':False},'stdlib_layout':dict(C.QUALIFIED_LAYOUT)})
    config=seal({'schema_version':1,'kind':'REGISTERED_ONE_PREFIX_TENSOR_OWNED_DIAGNOSTIC','scope':C.NATIVE_SCOPE,'permission':C.PERMISSION,'resources':C.RESOURCES,'input_file_sha256':C.INPUT_FILE,'input_body_sha256':C.INPUT_BODY,'prefix_ids_sha256':C.PREFIX_SHA,'source_commit':head,'native_launch_authorized':True,'source_files':sources,'runtime_files':files,'helper_pins':original['helper_pins'],'torch_version':original['runtime_versions']['torch'],'authorization':{'scope':C.NATIVE_SCOPE,'production_training':False,'protected_bank_access':False,'paid_services':False,'model':False,'forward':False,'decoder':False,'scoring':False},'offline_environment':original['offline_environment'],'tolerances':original['tolerances'],'runtime_versions':original['runtime_versions'],'GPU_physical_ordinal':0})
    review=seal({'kind':'ROOT_REVIEWED_TENSOR_RUNTIME_ROSTER_LIBRARY_SOURCE_CACHE_CLOSURE','source_commit':head,'source_inventory_sha256':C.digest(sources),'runtime_closure_sha256':C.digest(files),'rosters_sha256':C.digest(rosters),'cache_policy_sha256':C.digest(cache),'library_search_policy_sha256':C.digest(runtime['library_search_policy']),'complete_current_byte_review':True})
    registration=seal({'kind':'ROOT_REVIEWED_ONE_PREFIX_OWNED_REGISTRATION','source_commit':head,'native_launch_authorized':True,'config_body_sha256':config['sha256'],'runtime_body_sha256':runtime['sha256'],'source_files':sources,'runtime_files':files,'config_path':config_path,'qualification_review_path':review_path,'qualification_review_file_sha256':anchor(review_path,review)['file_sha256'],'qualification_review_body_sha256':review['sha256'],'profile_sha256':C.digest({'permission':C.PERMISSION,'resources':C.RESOURCES,'torch_version':config['torch_version']}),'git_output_exceptions':{}})
    documents={config_path:config,registration_path:registration,runtime_path:runtime,review_path:review,bootstrap_path:bootstrap,str(package/'configs/SYNTH_CONTEXT_PHASE_NO_MODEL_D001.json'):original,str(C.STDLIB_CANDIDATE):stdlib}
    anchors={'expected_config_file':anchor(config_path,config)['file_sha256'],'expected_config_body':config['sha256'],'registration_path':registration_path,'expected_registration_file':anchor(registration_path,registration)['file_sha256'],'expected_registration_body':registration['sha256'],'runtime_path':runtime_path,'expected_runtime_file':anchor(runtime_path,runtime)['file_sha256'],'expected_runtime_body':runtime['sha256'],'expected_head':head}
    return {'config':config,'registration':registration,'runtime':runtime,'review':review,'bootstrap':bootstrap,'original':original,'stdlib':stdlib,'documents':documents,'config_path':config_path,'anchors':anchors,'source_raw':source_raw,'fixture_only':True,'native_PASS':False,'native_authority_issued':False}

class ContractLoaderControls(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='CPU_contract_loader_',dir=HERE)
        self.root=Path(self.temporary.name).resolve()
    def tearDown(self):self.temporary.cleanup()
    def descriptor(self,raw,role='python_source'):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':role}
    def test_reader_empty_source_and_bounded_size(self):
        p=self.root/'empty.py';p.write_bytes(b'');self.assertEqual(C.read_bytes(p,0),b'');C.file_descriptor(p,self.descriptor(b''))
        p.write_bytes(b'x')
        with self.assertRaisesRegex(ValueError,'bounded'):C.read_bytes(p,0)
    def test_reader_rejects_regular_file_mutation(self):
        p=self.root/'source.py';p.write_bytes(b'x=1\n');original=Path.lstat;calls=[0]
        def metadata(path):
            meta=original(path)
            if path==p:
                calls[0]+=1
                if calls[0]>=3:return types.SimpleNamespace(st_file_attributes=0,st_mode=meta.st_mode,st_size=meta.st_size,st_mtime_ns=meta.st_mtime_ns+calls[0],st_ino=meta.st_ino)
            return meta
        with patch.object(Path,'lstat',metadata):
            with self.assertRaisesRegex(ValueError,'stable bounded'):C.read_bytes(p)
    def test_stream_hash_rejects_wrong_bytes_and_closed_deadline(self):
        p=self.root/'source.py';p.write_bytes(b'x=1\n')
        with self.assertRaisesRegex(ValueError,'exact stable bytes'):C.file_descriptor(p,self.descriptor(b'y=1\n'))
        now=[0.];budget=C.QualificationBudget(lambda:now[0]);now[0]=600.1
        with self.assertRaisesRegex(ValueError,'deadline/total'):C.file_descriptor(p,self.descriptor(b'x=1\n'),budget=budget)
    def test_budget_total_and_final_metadata_time_are_bounded(self):
        now=[0.];budget=C.QualificationBudget(lambda:now[0]);budget.check(16*1024**3)
        with self.assertRaisesRegex(ValueError,'deadline/total'):budget.check(1)
        budget=C.QualificationBudget(lambda:now[0]);now[0]=601.
        with self.assertRaisesRegex(ValueError,'deadline/total'):budget.check()
    def alias_patch(self,changed=None):
        logical=Path(C.QUALIFIED_LAYOUT['logical_base_prefix']);physical=Path(C.QUALIFIED_LAYOUT['physical_base_prefix'])
        settings={'attributes':1040,'tag':2684354563,'target':physical,'contained':physical/'Lib/inert.py'}
        settings.update({} if changed is None else changed)
        def resolve(path,*args,**kwargs):
            if path==logical:return settings['target']
            if path==logical/'Lib/inert.py':return settings['contained']
            return path
        original_lstat=Path.lstat
        def metadata(path):
            if path.is_relative_to(self.root):return original_lstat(path)
            return types.SimpleNamespace(st_file_attributes=settings['attributes'] if path==logical else 0,st_reparse_tag=settings['tag'],st_mode=stat.S_IFREG)
        return patch.object(Path,'resolve',resolve),patch.object(Path,'lstat',metadata),patch.object(Path,'is_symlink',return_value=False),logical,physical
    def test_only_exact_reviewed_alias_normalizes_in_reader_and_loader(self):
        a,b,c,logical,physical=self.alias_patch()
        with a,b,c:
            self.assertEqual(C.qualified_path(logical/'Lib/inert.py',C.QUALIFIED_LAYOUT),physical/'Lib/inert.py')
            self.assertEqual(L.physical(logical/'Lib/inert.py',C.QUALIFIED_LAYOUT),str(physical/'Lib/inert.py'))
        for change in ({'attributes':0},{'tag':0},{'target':self.root},{'contained':self.root/'bad.py'}):
            a,b,c,logical,_=self.alias_patch(change)
            with a,b,c:
                for check in (C.qualified_path,L.physical):
                    with self.assertRaises(ValueError):check(logical/'Lib/inert.py',C.QUALIFIED_LAYOUT)
        bad=dict(C.QUALIFIED_LAYOUT,python_version='3.11.14')
        for check in (C.qualified_path,L.physical):
            with self.assertRaises(ValueError):check('must_not_read',bad)
    def test_alias_is_applied_before_bounded_source_and_runtime_read(self):
        a,b,c,logical,physical=self.alias_patch();raw=b'x=1\n';descriptor=self.descriptor(raw)
        original=C.qualified_path;calls=[]
        def normalize(path,layout=None):
            calls.append((str(path),layout));original(path,layout);return self.root/'inert.py'
        (self.root/'inert.py').write_bytes(raw)
        with a,b,c,patch.object(C,'qualified_path',side_effect=normalize):
            self.assertEqual(C.read_bytes(logical/'Lib/inert.py',layout=C.QUALIFIED_LAYOUT),raw)
            C.file_descriptor(logical/'Lib/inert.py',descriptor,layout=C.QUALIFIED_LAYOUT)
        self.assertEqual(len(calls),2)
    def test_fixed_root_and_role_policy_refuses_arbitrary_roster_and_payload(self):
        with self.assertRaises(ValueError):C.roster_policy('external_package:arbitrary')
        with self.assertRaises(ValueError):C.roster_policy('any_host_root')
        self.assertEqual(C.runtime_role(str(C.SITE/'torch/a.py'),{}),'python_source')
        self.assertEqual(C.runtime_role(str(C.SITE/'torch/lib/inert.dll'),{}),'torch_binary')
        with self.assertRaises(ValueError):C.runtime_role(str(self.root/'unrostered.dll'),{})
        with self.assertRaises(ValueError):C.runtime_role(str(C.SITE/'numpy/payload.pt'),{})
    def test_exact_registration_git_leaves_never_mask_source_or_directories(self):
        leaf=str(C.PACKAGE/'artifacts/prefix_input_owned_run_CPU/inert.json');exceptions={leaf:{'role':'run_output'}}
        C._git_exception_policy(exceptions,{},{});relative=Path(leaf).relative_to(C.PACKAGE.parents[1]).as_posix()
        C._git_status(('?? '+relative+'\0').encode(),exceptions,C.PACKAGE.parents[1])
        for raw in ((' M '+relative+'\0').encode(),('?? '+relative.rsplit('/',1)[0]+'/\0').encode(),('?? '+relative.replace('inert.json','other.json')+'\0').encode()):
            with self.assertRaises(ValueError):C._git_status(raw,exceptions,C.PACKAGE.parents[1])
        with self.assertRaises(ValueError):C._git_exception_policy(exceptions,{leaf:{}},{})
        with self.assertRaises(ValueError):C._git_exception_policy({str(C.PACKAGE/'configs/inert.json'):{'role':'run_output'}},{},{})
        with self.assertRaises(ValueError):C._git_exception_policy({leaf.replace('.json','.py'):{'role':'run_output'}},{},{})
        source_path=str(C.PACKAGE/'prefix_tensor_contract.py');snapshot=str(C.PACKAGE/'evidence/prefix_input_owned_CPU/source_snapshots'/(hashlib.sha256(source_path.encode()).hexdigest()+'.py'))
        C._git_exception_policy({snapshot:{'role':'evidence_source_snapshot','source_path':source_path}},{source_path:{}},{})
        with self.assertRaises(ValueError):C._git_exception_policy({snapshot.replace('.py','.json'):{'role':'evidence_source_snapshot','source_path':source_path}},{source_path:{}},{})
        with self.assertRaises(ValueError):C._git_exception_policy({snapshot:{'role':'evidence_source_snapshot','source_path':source_path}},{},{})
    def finder(self,files=None,namespaces=None):return L.RegisteredFinder({} if files is None else files,C,C.QUALIFIED_LAYOUT,namespaces)
    def test_pathfinder_miss_and_unregistered_source_cannot_fallback(self):
        with patch.object(importlib.machinery.PathFinder,'find_spec',return_value=None):
            with self.assertRaisesRegex(ModuleNotFoundError,'fallback refused'):self.finder().find_spec('CPU_missing_inert')
        p=str(self.root/'not_registered.py');spec=importlib.util.spec_from_loader('CPU_missing_inert',importlib.machinery.SourceFileLoader('CPU_missing_inert',p))
        with patch.object(importlib.machinery.PathFinder,'find_spec',return_value=spec),patch.object(C,'file_descriptor',side_effect=AssertionError('must reject before bytes')):
            with self.assertRaisesRegex(ValueError,'unregistered source'):self.finder().find_spec('CPU_missing_inert')
    def test_namespace_requires_selected_exact_root_not_any_file_parent(self):
        root=str(self.root);p=str(self.root/'child.py');spec=importlib.machinery.ModuleSpec('CPU_namespace',None,is_package=True);spec.submodule_search_locations=[root]
        with patch.object(importlib.machinery.PathFinder,'find_spec',return_value=spec):
            with self.assertRaisesRegex(ValueError,'exact qualified'):self.finder({p:self.descriptor(b'')}).find_spec('CPU_namespace')
            self.assertIs(self.finder({p:self.descriptor(b'')},{'CPU_namespace':[root]}).find_spec('CPU_namespace'),spec)
            with self.assertRaises(ValueError):self.finder({p:self.descriptor(b'')},{'CPU_namespace':[str(self.root.parent)]}).find_spec('CPU_namespace')
    def test_source_or_extension_custom_loader_rejects_before_create(self):
        for suffix in ('.py','.pyd'):
            p=str(self.root/('inert'+suffix));actual=Mock();actual.path=p;actual.name='CPU_inert';spec=importlib.machinery.ModuleSpec('CPU_inert',actual,origin=p)
            with patch.object(importlib.machinery.PathFinder,'find_spec',return_value=spec),patch.object(C,'file_descriptor',side_effect=AssertionError('custom loader bytes must not be read')):
                with self.assertRaisesRegex(ValueError,'exact registered PathFinder'):self.finder({p:self.descriptor(b'')}).find_spec('CPU_inert')
            actual.create_module.assert_not_called()
    def test_extension_validation_fails_before_create_or_exec(self):
        actual=Mock(spec=importlib.machinery.ExtensionFileLoader);loader=L.RegisteredLoader(actual,str(self.root/'inert.pyd'),self.descriptor(b'',role='extension_binary'),C,{})
        with patch.object(C,'file_descriptor',side_effect=ValueError('trapped byte validation')):
            with self.assertRaisesRegex(ValueError,'trapped'):loader.create_module(None)
            with self.assertRaisesRegex(ValueError,'trapped'):loader.exec_module(types.ModuleType('CPU_inert'))
        actual.create_module.assert_not_called();actual.exec_module.assert_not_called()
    def test_source_changed_after_hash_rejects_before_create_compile_or_exec(self):
        p=self.root/'inert.py';raw=b'raise AssertionError("must not execute")\n';loader=L.RegisteredLoader(Mock(),str(p),self.descriptor(raw),C,{})
        with patch.object(C,'file_descriptor'),patch.object(C,'read_bytes',return_value=b'changed\n'):
            with self.assertRaisesRegex(ValueError,'actual compile consumes'):loader.create_module(None)
            with self.assertRaisesRegex(ValueError,'actual compile consumes'):loader.exec_module(types.ModuleType('CPU_inert'))
    def test_preloaded_torch_rejected_even_without_cache_or_source_file(self):
        contract=types.SimpleNamespace(stage_native_gate=lambda:None,context_value=lambda _: {'runtime':{'files':{},'stdlib_layout':C.QUALIFIED_LAYOUT,'cache_policy':{'namespace_roots':{}}},'config':{'source_files':{}}},require=C.require,QualificationBudget=C.QualificationBudget)
        for module in (types.SimpleNamespace(__file__='must_not_read',__cached__=None),types.SimpleNamespace()):
            with patch.object(sys_proxy:=L.sys,'modules',{'torch':module}),patch.object(sys_proxy,'meta_path',[importlib.machinery.PathFinder]):
                with self.assertRaisesRegex(ValueError,'Torch must be fresh'):L.install(object(),contract)
    def test_preloaded_source_requires_external_assumptions_not_equivalence(self):
        p=str(self.root/'inert.py');module=types.SimpleNamespace(__file__=p,__cached__=None)
        contract=types.SimpleNamespace(stage_native_gate=lambda:None,context_value=lambda _: {'runtime':{'files':{p:self.descriptor(b'')},'stdlib_layout':C.QUALIFIED_LAYOUT,'cache_policy':{'namespace_roots':{},'Torch_root':str(C.SITE/'torch'),'preloaded_bootstrap_modules':{}}},'config':{'source_files':{}}},require=C.require,QualificationBudget=C.QualificationBudget)
        with patch.object(L.sys,'modules',{'CPU_inert':module}),patch.object(L.sys,'meta_path',[importlib.machinery.PathFinder]):
            with self.assertRaisesRegex(ValueError,'external-qualified'):L.install(object(),contract)
    def test_code_fingerprint_covers_nested_name_line_stack_exception_metadata(self):
        code=compile('def f(x):\n try:\n  return x+1\n except ValueError:\n  return 0\n','CPU_inert.py','exec');fp=L.code_fingerprint(code)
        for change in ({'co_name':'changed'},{'co_qualname':'changed'},{'co_firstlineno':2},{'co_stacksize':code.co_stacksize+1},{'co_linetable':b''},{'co_exceptiontable':b'x'},{'co_filename':'changed.py'}):
            self.assertNotEqual(fp,L.code_fingerprint(code.replace(**change)))
        nested=next(v for v in code.co_consts if type(v) is types.CodeType);constants=tuple(nested.replace(co_name='different') if v is nested else v for v in code.co_consts)
        self.assertNotEqual(fp,L.code_fingerprint(code.replace(co_consts=constants)))
        with self.assertRaisesRegex(ValueError,'unsupported'):L.code_fingerprint(code.replace(co_consts=(object(),)))
        deep=0
        for _ in range(66):deep=(deep,)
        with self.assertRaisesRegex(ValueError,'bounded'):L.code_fingerprint(code.replace(co_consts=(deep,)))
        with self.assertRaisesRegex(ValueError,'bounded'):L.code_fingerprint(code.replace(co_names=tuple('n'+str(i) for i in range(32769))))
    def fixture_boundaries(self,frame):
        def sealed(path,file_pin,body_pin,*,budget=None):
            value=frame['documents'][str(path)];C.require(value['sha256']==body_pin,'fixture exact external body anchor')
            raw=(C.canonical(value)+'\n').encode();C.require(value['sha256']==C.digest({k:v for k,v in value.items() if k!='sha256'}),'fixture immutable body checksum')
            if str(path).endswith('SYNTH_CONTEXT_PHASE_NO_MODEL_D001.json'):expected='53cfe48bef01d531c8c16555edc97eeebd02b76c9e5b995d20513e6b37c1b030'
            elif str(path)==str(C.STDLIB_CANDIDATE):expected='e5e185f980e2f0feae7bf49384089eedd4d47f5b41c10a828c6ef2a30ae777a5'
            else:expected=hashlib.sha256(raw).hexdigest()
            C.require(file_pin==expected,'fixture exact external file anchor')
            if budget is not None:budget.check(len(raw))
            return deepcopy(value)
        def roster(root,suffixes,excluded,*,recursive=True,budget=None):
            if budget is not None:budget.check()
            return deepcopy(next(v['files'] for v in frame['runtime']['rosters'].values() if (v['root'],v['suffixes'],v['excluded_directory_names'],v['recursive'])==(root,suffixes,excluded,recursive)))
        def descriptor(path,expected,*,layout=None,budget=None):
            C.require(str(path) in frame['config']['source_files'] or str(path) in frame['runtime']['files'],'inert selected byte boundary')
            if budget is not None:budget.check(expected['bytes'])
        def git(args,**kwargs):
            C.require(args[0]=='git' and kwargs.get('timeout',0)>0,'only injected bounded Git query')
            if args[1]=='rev-parse':return types.SimpleNamespace(stdout=frame['anchors']['expected_head']+'\n')
            if args[1]=='status':return types.SimpleNamespace(stdout=b'')
            if args[1]=='show':return types.SimpleNamespace(stdout=(C.canonical(frame['config'])+'\n').encode())
            raise AssertionError('unselected live call')
        return sealed,roster,descriptor,git
    def test_actual_metadata_roster_git_factory_source_route_is_CPU_only(self):
        frame=metadata_fixture();sealed,roster,descriptor,git=self.fixture_boundaries(frame)
        import subprocess
        with patch.object(C,'sealed_json',side_effect=sealed),patch.object(C,'physical_roster',side_effect=roster),patch.object(C,'package_source_roster',return_value=set(frame['config']['source_files'])),patch.object(C,'file_descriptor',side_effect=descriptor),patch.object(C,'load_input'),patch.object(C,'read_bytes',return_value=(C.canonical(frame['config'])+'\n').encode()),patch.object(subprocess,'run',side_effect=git):
            context=C._CPU_native_context_fixture(frame['config_path'],_capability=C._mint_CPU_native_context_fixture(),**frame['anchors'])
            value=C.context_value(context);self.assertTrue(value['fixture_only']);self.assertFalse(value['native_PASS']);self.assertFalse(value['native_authority_issued'])
            copy=C.context_value(context);copy['config']['scope']='changed';self.assertEqual(C.context_value(context),value)
            with patch.object(C,'stage_native_gate'):
                with self.assertRaisesRegex(ValueError,'no native consumer authority'):C.revalidate_context(context)
                with self.assertRaisesRegex(ValueError,'no native loader authority'):L.install(context,C)
            bad=dict(frame['anchors'],expected_config_file='0'*64)
            with self.assertRaisesRegex(ValueError,'external file anchor'):C._CPU_native_context_fixture(frame['config_path'],_capability=C._mint_CPU_native_context_fixture(),**bad)
    def test_resealed_metadata_and_bootstrap_negatives_fail_actual_schema(self):
        frame=metadata_fixture();sealed,roster,_,_=self.fixture_boundaries(frame)
        with patch.object(C,'sealed_json',side_effect=sealed):
            for mutate in (lambda x:x['runtime']['files'].update({str(self.root/'arbitrary.json'):{'bytes':1,'sha256':'a'*64,'role':'source_json'}}),lambda x:x['runtime']['files'][next(iter(x['runtime']['files']))].update(role='installation_metadata'),lambda x:x['registration']['git_output_exceptions'].update({str(C.PACKAGE/'configs/CPU_NATIVE_SOURCE_FIXTURE.json'):{'role':'evidence_output'}})):
                bad=deepcopy(frame);mutate(bad)
                bad['config']['runtime_files']=bad['registration']['runtime_files']=bad['runtime']['files']
                with self.assertRaises(ValueError):C.native_metadata(bad['config'],bad['registration'],bad['runtime'],bad['anchors']['expected_head'])
        with patch.object(C,'sealed_json',side_effect=sealed),patch.object(C,'physical_roster',side_effect=roster),patch.object(C,'package_source_roster',return_value=set(frame['config']['source_files'])):
            value=C.native_metadata(frame['config'],frame['registration'],frame['runtime'],frame['anchors']['expected_head']);C.qualification_rosters(value)
            bad=deepcopy(value);bad['runtime']['rosters']['torch']['root']=str(self.root)
            with self.assertRaisesRegex(ValueError,'exact finite'):C.qualification_rosters(bad)
            frame['bootstrap']['current_module_execution_proven']=True
            frame['bootstrap']['sha256']=C.digest({k:v for k,v in frame['bootstrap'].items() if k!='sha256'})
            anchor=value['runtime']['cache_policy']['fresh_bootstrap_evidence'];anchor['body_sha256']=frame['bootstrap']['sha256'];anchor['file_sha256']=hashlib.sha256((C.canonical(frame['bootstrap'])+'\n').encode()).hexdigest()
            with self.assertRaises(ValueError):C.qualification_rosters(value)
    def test_public_native_entries_stay_stage_closed_before_any_reads(self):
        with patch.object(C,'sealed_json',side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError,'no native authority'):C.validate_native_manifest('must_not_read')
            with self.assertRaisesRegex(ValueError,'no native authority'):C.native_entry('must_not_read')
        with self.assertRaisesRegex(ValueError,'private CPU'):C._CPU_native_context_fixture('must_not_read',_capability=object())

if __name__=='__main__':unittest.main()
