"""Real inert document/source/roster checks; historical host bytes injected.

No Git, host runtime, interpreter layout, bootstrap or native qualification is
proved. The private capability and returned context both retain CPU identity.
"""
from contextlib import contextmanager, ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import hashlib, os

HERE = Path(__file__).resolve().parent
ACTUAL_PACKAGE = next(p for p in HERE.parents if p.name == 'rineya_synth_centralized_reasoning_v1')
STDLIB_METADATA = ACTUAL_PACKAGE/'evidence/stdlib_inventory_reviewed_cpu_20261005/STDLIB_CANDIDATE_001.json'
DIRECT = ('prefix_tensor_contract.py','prefix_input_owned_worker.py','prefix_input_owned_supervisor.py',
          'prefix_input_owned_auditor.py','prefix_registered_loader.py','prefix_native_pipeline.py')

def source(name):
    p=HERE/name; m=ModuleType('outer_CPU_'+p.stem); m.__file__=str(p)
    exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__); return m

def descriptor(raw,role='python_source'):
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':role}

def write_new(path,raw):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    if path.read_bytes()!=raw: raise ValueError('fixture exact writer readback')

class OuterContextFixture:
    """Use activate() for downstream CPU consumers; context() runs the factory.

    Historical65/779 descriptors are exact metadata obligations, deliberately
    not read. All selected documents, package sources and finite inert runtime
    files use the production readers, stream hashes and physical roster walk.
    """
    def __init__(self,base):
        self.root=Path(base).resolve()/'CPU_OUTER_CONTEXT_FILESYSTEM'; self.root.mkdir()
        self.package=self.root/'repository/experiments/rineya_synth_centralized_reasoning_v1'
        self.package.mkdir(parents=True); self.C=source('prefix_tensor_contract.py'); C=self.C
        self.original=C.sealed_json(ACTUAL_PACKAGE/'configs/SYNTH_CONTEXT_PHASE_NO_MODEL_D001.json',
            '53cfe48bef01d531c8c16555edc97eeebd02b76c9e5b995d20513e6b37c1b030',
            '944db028d5c7fcc7a0210590ac551200a184e143af4973ef5c08a13faa84cc62')
        self.stdlib=C.sealed_json(STDLIB_METADATA,
            'e5e185f980e2f0feae7bf49384089eedd4d47f5b41c10a828c6ef2a30ae777a5',
            'f6284b28e689b2c5e42a97c12256b3e35a41395a6b9612c0a01ee786eb28b149')
        self.historical_runtime={}
        for p,h in {**self.original['runtime_pins'],**self.stdlib['inventory']['runtime_pins']}.items():
            meta=self.stdlib['inventory']['files'].get(p)
            self.historical_runtime[p]={'bytes':1 if meta is None else meta['bytes'],'sha256':h,
                'role':C.runtime_role(p,self.original['runtime_pins'])}
        self.historical_runtime_checks=[]; self.actual_file_checks=[]; self.git_calls=[]
        self.site=self.root/'inert_runtime/venv/Lib/site-packages'; self.base=self.root/'inert_runtime/base'
        self.layout={**C.QUALIFIED_LAYOUT,'logical_base_prefix':str(self.root/'unused_logical_alias'),
                     'physical_base_prefix':str(self.base)}
        self.head='a'*40; self.config_path=self.package/'configs/CPU_NATIVE_SOURCE_FIXTURE.json'
        self.registration_path=self.package/'artifacts/CPU_NATIVE_SOURCE_REGISTRATION.json'
        self.runtime_path=self.package/'artifacts/CPU_NATIVE_SOURCE_RUNTIME.json'
        self.review_path=self.package/'artifacts/CPU_NATIVE_SOURCE_REVIEW.json'
        self.bootstrap_path=self.package/'artifacts/CPU_NATIVE_BOOTSTRAP_ASSUMPTIONS.json'
        self.stdlib_path=self.package/'evidence/CPU_READONLY_METADATA/STDLIB_CANDIDATE_001.json'
        self.input_path=self.package/'evidence/exception_frame_helper_reviewed_cpu_20261005/root/ROOT_FAILURE_PREFIX_INPUT_001.json'
        self.sources={}
        for n in DIRECT: self._source(self.package/n,(HERE/n).read_bytes())
        for optional in ('prefix_library_policy.py','prefix_library_evidence.py'):
            if (HERE/optional).exists():self._source(self.package/optional,(HERE/optional).read_bytes())
        for n,h in self.original['helper_pins'].items():
            raw=(ACTUAL_PACKAGE/n).read_bytes(); C.require(hashlib.sha256(raw).hexdigest()==h,'exact original helper source')
            self._source(self.package/n,raw)
        for n in C.CORE_PINS:
            self._source(self.package/'evidence/prefix_input_core_reviewed_cpu_20261005'/n,(C.CORE/n).read_bytes())
        for n in ('PILOT_DECISIONS.json','pilot_trainable_inventory.json','NATIVE_QUALIFICATION_PROPOSAL.json','AGENTS.md'):
            self._source(self.package/n,(ACTUAL_PACKAGE/n).read_bytes(),'source_json' if n.endswith('.json') else 'source_markdown')
        for n in ('AGENTS.md','RINEYA_SYNTH_CENTRALIZED_REASONING.md'):
            self._source(self.package.parents[1]/n,(ACTUAL_PACKAGE.parents[1]/n).read_bytes(),'source_markdown')
        n='evidence/exception_frame_helper_reviewed_cpu_20261005/exception_frames.py'
        self._source(self.package/n,(ACTUAL_PACKAGE/n).read_bytes())
        write_new(self.package/'configs/SYNTH_CONTEXT_PHASE_NO_MODEL_D001.json',
                  (ACTUAL_PACKAGE/'configs/SYNTH_CONTEXT_PHASE_NO_MODEL_D001.json').read_bytes())
        write_new(self.stdlib_path,STDLIB_METADATA.read_bytes()); write_new(self.input_path,C.INPUT.read_bytes())
        self.inert_files={}
        inert={'base/Lib/CPU_inert.py':b'# inert stdlib\n','base/DLLs/CPU_inert.pyd':b'inert extension bytes\n',
               'base/python.exe':b'inert interpreter bytes\n','venv/Lib/site-packages/torch/__init__.py':b'# never imported\n',
               'venv/Lib/site-packages/typing_extensions.py':b'# inert bootstrap\n'}
        inert.update({'venv/Lib/site-packages/'+n+'/__init__.py':b'# inert external\n' for n in ('psutil','safetensors','transformers')})
        for relative,raw in inert.items():
            p=self.root/'inert_runtime'/relative; write_new(p,raw)
            with self.activate(): role=C.runtime_role(str(p),{})
            self.inert_files[str(p)]=descriptor(raw,role)
        self._build_documents()

    def _source(self,p,raw,role='python_source'):
        write_new(p,raw); self.sources[str(p)]=descriptor(raw,role)
    def _seal(self,value):
        body={k:v for k,v in value.items() if k!='sha256'}; return {**body,'sha256':self.C.digest(body)}
    def _raw(self,value): return (self.C.canonical(value)+'\n').encode('ascii')
    def anchor(self,p,value):
        return {'path':str(p),'file_sha256':hashlib.sha256(self._raw(value)).hexdigest(),'body_sha256':value['sha256']}

    def _build_documents(self):
        C=self.C; files={**self.historical_runtime,**self.inert_files}; single=str(self.site/'typing_extensions.py')
        external={p:d for p,d in self.inert_files.items() if Path(p).is_relative_to(self.site) and not Path(p).is_relative_to(self.site/'torch')}
        preload={'CPU_bootstrap_inert':{'path':single,'source_sha256':files[single]['sha256'],'cache_path':None,'cache_sha256':None}}
        self.bootstrap=self._seal({'kind':'ROOT_QUALIFIED_FRESH_BOOTSTRAP_BYTE_SOURCE_ASSUMPTIONS','source_commit':self.head,
            'preloaded_bootstrap_modules_sha256':C.digest(preload),'fresh_process_bootstrap_qualified':True,
            'environment_and_loader_assumptions_qualified':True,'current_module_execution_proven':False})
        rosters={}
        with self.activate():
            for name in ['stdlib_python_source','stdlib_extension_runtime','interpreter_root_executable_runtime','torch']+['external_package:'+n for n in ('psutil','safetensors','transformers')]:
                root,suffixes,excluded,recursive=C.roster_policy(name)
                rosters[name]={'root':str(root),'suffixes':suffixes,'excluded_directory_names':excluded,'recursive':recursive,
                    'files':C.physical_roster(root,suffixes,excluded,recursive=recursive)}
        cache={'mode':'all_registered_python_source_compiled_without_cache','allow_unregistered':False,
            'source_loader_sha256':self.sources[str(self.package/'prefix_registered_loader.py')]['sha256'],
            'Torch_root':str(self.site/'torch'),'preloaded_module_cache_pins':{},'namespace_roots':{},
            'preloaded_bootstrap_modules':preload,'fresh_bootstrap_evidence':self.anchor(self.bootstrap_path,self.bootstrap)}
        search={'mode':'qualified_complete_registered_DLL_search_rosters','roster_names':['torch','stdlib_extension_runtime','interpreter_root_executable_runtime'],'unregistered_search_allowed':False}
        self.runtime=self._seal({'kind':'ROOT_QUALIFIED_TENSOR_RUNTIME_SOURCE_BYTE_CLOSURE','source_commit':self.head,
            'current_binary_bytes_qualified':True,'external_dependencies_complete':True,'interpreter_layout_qualified':True,
            'cache_execution_policy_qualified':True,'files':files,'inherited65_map':self.original['runtime_pins'],
            'stdlib779_map':self.stdlib['inventory']['runtime_pins'],'cache_policy':cache,'external_dependency_files':external,
            'rosters':rosters,'library_search_policy':search,'stdlib_layout':self.layout})
        self.config=self._seal({'schema_version':1,'kind':'REGISTERED_ONE_PREFIX_TENSOR_OWNED_DIAGNOSTIC','scope':C.NATIVE_SCOPE,
            'permission':C.PERMISSION,'resources':C.RESOURCES,'input_file_sha256':C.INPUT_FILE,'input_body_sha256':C.INPUT_BODY,
            'prefix_ids_sha256':C.PREFIX_SHA,'source_commit':self.head,'native_launch_authorized':True,'source_files':self.sources,
            'runtime_files':files,'helper_pins':self.original['helper_pins'],'torch_version':self.original['runtime_versions']['torch'],
            'authorization':{'scope':C.NATIVE_SCOPE,'production_training':False,'protected_bank_access':False,'paid_services':False,
                'model':False,'forward':False,'decoder':False,'scoring':False},'offline_environment':self.original['offline_environment'],
            'tolerances':self.original['tolerances'],'runtime_versions':self.original['runtime_versions'],'GPU_physical_ordinal':0})
        self.review=self._seal({'kind':'ROOT_REVIEWED_TENSOR_RUNTIME_ROSTER_LIBRARY_SOURCE_CACHE_CLOSURE','source_commit':self.head,
            'source_inventory_sha256':C.digest(self.sources),'runtime_closure_sha256':C.digest(files),'rosters_sha256':C.digest(rosters),
            'cache_policy_sha256':C.digest(cache),'library_search_policy_sha256':C.digest(search),'complete_current_byte_review':True})
        self.registration=self._seal({'kind':'ROOT_REVIEWED_ONE_PREFIX_OWNED_REGISTRATION','source_commit':self.head,
            'native_launch_authorized':True,'config_body_sha256':self.config['sha256'],'runtime_body_sha256':self.runtime['sha256'],
            'source_files':self.sources,'runtime_files':files,'config_path':str(self.config_path),'qualification_review_path':str(self.review_path),
            'qualification_review_file_sha256':self.anchor(self.review_path,self.review)['file_sha256'],
            'qualification_review_body_sha256':self.review['sha256'],'profile_sha256':C.digest({'permission':C.PERMISSION,
                'resources':C.RESOURCES,'torch_version':self.config['torch_version']}),'git_output_exceptions':{}})
        self.documents={str(self.config_path):self.config,str(self.registration_path):self.registration,str(self.runtime_path):self.runtime,
                        str(self.review_path):self.review,str(self.bootstrap_path):self.bootstrap}
        for p,v in self.documents.items(): write_new(p,self._raw(v))
        self.anchors={'expected_config_file':self.anchor(self.config_path,self.config)['file_sha256'],'expected_config_body':self.config['sha256'],
            'registration_path':str(self.registration_path),'expected_registration_file':self.anchor(self.registration_path,self.registration)['file_sha256'],
            'expected_registration_body':self.registration['sha256'],'runtime_path':str(self.runtime_path),
            'expected_runtime_file':self.anchor(self.runtime_path,self.runtime)['file_sha256'],'expected_runtime_body':self.runtime['sha256'],'expected_head':self.head}

    @contextmanager
    def activate(self):
        C=self.C; real_role=C.runtime_role; real_descriptor=C.file_descriptor
        def role(p,inherited):
            if p in self.historical_runtime:
                C.require(inherited==self.original['runtime_pins'],'fixture historical role map exact'); return self.historical_runtime[p]['role']
            return real_role(p,inherited)
        def check(p,expected,**kw):
            text=str(p)
            if text in self.historical_runtime:
                C.require(expected==self.historical_runtime[text],'fixture exact injected historical descriptor')
                self.historical_runtime_checks.append(text)
                if kw.get('budget') is not None: kw['budget'].check()
                return
            C.require(Path(p).is_relative_to(self.root),'CPU fixture never hashes actual host runtime')
            real_descriptor(p,expected,**kw); self.actual_file_checks.append(text)
        def git(args,**kw):
            self.git_calls.append(list(args)); C.require(args[0]=='git' and Path(kw['cwd'])==self.package.parents[1],'exact fixture Git seam')
            if args==['git','rev-parse','HEAD']: return SimpleNamespace(stdout=self.head+'\n')
            if args==['git','status','--porcelain=v1','-z','--untracked-files=all']: return SimpleNamespace(stdout=b'')
            C.require(args==['git','show',self.head+':'+self.config_path.relative_to(self.package.parents[1]).as_posix()],'exact fixture Git show')
            return SimpleNamespace(stdout=self._raw(self.config))
        with ExitStack() as stack:
            for key,v in {'PACKAGE':self.package,'CORE':self.package/'evidence/prefix_input_core_reviewed_cpu_20261005',
                'INPUT':self.input_path,'STDLIB_CANDIDATE':self.stdlib_path,'SITE':self.site,'QUALIFIED_LAYOUT':self.layout}.items():
                stack.enter_context(patch.object(C,key,v))
            stack.enter_context(patch.object(C,'runtime_role',side_effect=role)); stack.enter_context(patch.object(C,'file_descriptor',side_effect=check))
            stack.enter_context(patch('subprocess.run',side_effect=git)); yield self

    def context(self):
        with self.activate():
            context=self.C._CPU_native_context_fixture(self.config_path,_capability=self.C._mint_CPU_native_context_fixture(),**self.anchors)
        v=self.C.context_value(context)
        self.C.require(v['fixture_only'] is True and v['native_PASS'] is v['native_authority_issued'] is False,'CPU context cannot create native authority')
        return context

    def coverage(self):
        return {'kind':'CPU_ACTUAL_OUTER_DOCUMENT_SOURCE_ROSTER_FIXTURE','fixture_only':True,'native_PASS':False,'native_authority_issued':False,
            'real_descriptor_paths':sorted(set(self.actual_file_checks)),'injected_historical_runtime_paths':sorted(set(self.historical_runtime_checks)),
            'injected_host_boundaries':['historical65/779 descriptor checks','Git outputs','finite SITE/interpreter layout substitution'],
            'native_runtime_byte_closure_proven':False,'current_module_execution_proven':False}
