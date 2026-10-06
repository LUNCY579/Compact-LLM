"""Source-only same-process cloned-code/state limitation; no native authority."""
import ast, builtins, contextlib, hashlib, json, sys, time, types
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[4]
STAGE=ROOT/'experiments/rineya_synth_centralized_reasoning_v1/artifacts/prefix_input_owned_execution_work'
OUT=Path(__file__).resolve().parent
NAMES=('prefix_input_owned_worker.py','prefix_native_pipeline.py','prefix_registered_loader.py','prefix_library_policy.py','prefix_tensor_contract.py')
def raw(path):
    path=Path(path);assert path.parent==STAGE and path.suffix=='.py'
    value=path.read_bytes();assert len(value)<=16*1024**2;return value
def hashes():return {name:hashlib.sha256(raw(STAGE/name)).hexdigest() for name in NAMES}
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('ascii')
def require(condition,message):
    if not condition:raise ValueError(message)
def code_child(parent,name):return next(item for item in parent.co_consts if type(item) is types.CodeType and item.co_name==name)
def cell(value):return (lambda:value).__closure__[0]
before=hashes()
paths={name:str(STAGE/name) for name in NAMES}
pins={paths[name]:{'bytes':len(raw(STAGE/name)),'sha256':before[name],'role':'python_source'} for name in NAMES}
value={'fixture_only':False,'config':{'source_files':pins}}
C=types.SimpleNamespace(PACKAGE=STAGE,require=require,canonical=canonical,read_bytes=raw)
worker_tree=ast.parse(raw(STAGE/NAMES[0]),paths[NAMES[0]])
selected=[node for node in worker_tree.body if isinstance(node,ast.FunctionDef) and node.name in ('_installed_framework_boundary','_installed_framework_boundary_body')]
selector={'C':C,'ast':ast,'hashlib':hashlib}
exec(compile(ast.Module(body=selected,type_ignores=[]),paths[NAMES[0]],'exec'),selector)
loader_code=compile(ast.parse(raw(STAGE/NAMES[2]),paths[NAMES[2]]),paths[NAMES[2]],'exec')
helper_code=compile(ast.parse(raw(STAGE/NAMES[3]),paths[NAMES[3]]),paths[NAMES[3]],'exec')
finder_code=code_child(code_child(loader_code,'RegisteredFinder'),'framework_import')
guard_code=code_child(code_child(helper_code,'LibraryGuard'),'framework_import')
check_code=code_child(code_child(loader_code,'_installation_registry_factory'),'check')

# No genuine selected loader/helper module, installation or trusted registry is
# modified. The cloned executable methods receive caller-created globals/state.
finder_globals={'sys':sys};guard_globals={'__compiled_source_descriptor__':dict(pins[paths[NAMES[3]]])}
Finder=type('RegisteredFinder',(),{'framework_import':types.FunctionType(finder_code,finder_globals)})
Guard=type('LibraryGuard',(),{'framework_import':types.FunctionType(guard_code,guard_globals),
                            '_boundary':lambda self,*args:contextlib.nullcontext()})
finder_globals['RegisteredFinder']=Finder;guard_globals['LibraryGuard']=Guard
guard=Guard();guard.active=True;guard.failed=False;guard.closed=False;guard.policy={'framework_transition':[]}
finder=Finder();finder.contract=C;finder.library_guard=guard;finder.selected_library_guard=guard
finder.fixture_only=False;finder.selected_loader_source=dict(pins[paths[NAMES[2]]]);finder.selected_context_projection=canonical(value)
records={finder:(Finder,Finder.framework_import,guard,Guard,Guard.framework_import,canonical(value),False)}
assert check_code.co_freevars==('records',)
validator=types.FunctionType(check_code,finder_globals,closure=(cell(records),))
validator.__kwdefaults__={'_CPU_capability':None}
finder_globals['_check_installed_record']=validator
attempts=[];original_import=builtins.__import__
def imports(name,*args,**kwargs):
    if name.split('.',1)[0] in {'torch','ctypes','psutil','native_worker','native_process','native_nvml','transformers','safetensors','numpy'}:
        attempts.append(name);raise AssertionError('forbidden native import')
    return original_import(name,*args,**kwargs)
with patch.object(builtins,'__import__',side_effect=imports),patch.object(sys,'meta_path',[finder]):
    actual=selector['_installed_framework_boundary'](value,finder.framework_import)
    assert actual.__self__ is finder
    with actual():delegated=True
after=hashes();assert before==after and not attempts
receipt={'kind':'INDEPENDENT_SAME_PROCESS_CLONED_VALIDATOR_GLOBAL_CLOSURE_AUTHORITY_LIMIT_REPRODUCTION',
 'fixture_only':True,'native_PASS':False,'native_authority_issued':False,'native_stage_opened':False,
 'native_factory_readiness_proven':False,'root_source_edits':False,'genuine_installer_memory_mutated':False,
 'source_before':before,'source_after':after,'sources_unchanged':before==after,'forbidden_import_attempts':attempts,
 'exact_selected_finder_guard_validator_code_with_counterfeit_globals_and_closure_accepted':True,
 'unobserved_noop_boundary_delegate_reached':delegated,
 'injected_assumptions':['plain selector input, no actual native context issuance',
 'same-process caller constructed selected CodeType copies with counterfeit class globals and a counterfeit validator record closure'],
 'finding':'Executable code equality plus class/global consistency does not authenticate the identity/provenance of a caller-created module namespace or installation record closure.',
 'scope':'Known unresolved same-process authority limitation. The finite no-op/object substitution controls remain useful; this does not authorize a generic anti-tamper redesign or any native opening. Native runtime/library observer/host/source qualification remain absent.'}
receipt['sha256']=hashlib.sha256(canonical(receipt)).hexdigest()
path=OUT/('installed_boundary_cloned_state_limit_'+str(time.time_ns())+'.json')
with path.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({'receipt':str(path),'body_sha256':receipt['sha256'],'cloned_state_accepted':True,'source_stable':True,'forbidden_imports':attempts}))
