"""Fixed original containment adaptation for decoded long M0 collection.

No public launch or installed qualification authority is present while staged.
"""
from copy import deepcopy
from pathlib import Path
from types import ModuleType
import ast, hashlib
HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
EXPECTED=PACKAGE/'long_guard_owned_runtime_v1'
SCOPE='REGISTERED_LONG2304_M0_DECODED_RAW_COLLECTION_V1'
def require(ok,text):
    if not ok:raise ValueError(text)
def supervisor_body(namespace, config, output_root):
    """Only purpose/worker substitutions; original guards/lifecycle unchanged."""
    raw=(PACKAGE/'native_supervisor.py').read_bytes();tree=ast.parse(raw)
    function=deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='supervise'))
    changes={'worker':0,'scope':0}
    for node in ast.walk(function):
        if isinstance(node,ast.Constant) and node.value=='native_worker.py':node.value='long_guard_owned_runtime_v1/long_guard_owned_runtime.py';changes['worker']+=1
        elif isinstance(node,ast.Constant) and node.value=='mechanical_native_qualification_only':node.value=SCOPE;changes['scope']+=1
    require(changes=={'worker':2,'scope':1},'original fixed supervisor substitution counts')
    root=Path(output_root);repository=PACKAGE.parents[1]
    require(root.is_absolute() and root.resolve()==root and root.is_dir() and root.is_relative_to(repository.parent)
        and not root.is_relative_to(repository) and '.codex' not in root.parts,'external bound BusinessProject output root')
    require(Path(config['output_directory']).parent==root,'exact selected external output')
    def preflight(path):
        require(Path(path).absolute()==Path(config['config_path']),'same selected config argument')
        return deepcopy(config)
    definitions=[];external_changes={}
    for name,target in (('_new_output','artifacts'),('_acquire_lease','root')):
        fn=deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name));count=0
        for node in ast.walk(fn):
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id==target:
                node.value=ast.Name(id='_external_root',ctx=ast.Load()) if target=='artifacts' else ast.BinOp(left=ast.Name(id='_external_root',ctx=ast.Load()),op=ast.Div(),right=ast.Constant(value='_native_gpu_leases'));count+=1
        require(count==1,'original fixed external output/lease adaptation');external_changes[name]=count;definitions.append(fn)
    adapted={**namespace,'_preflight':preflight,'_external_root':root}
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions+[function],type_ignores=[])),str(PACKAGE/'native_supervisor.py'),'exec'),adapted)
    return adapted['supervise'],{'original_supervisor_sha256':hashlib.sha256(raw).hexdigest(),'purpose_worker_changes':changes,'external_changes':external_changes}
def run_registered(launch_ref):
    require(HERE==EXPECTED,'staged decoded long dispatch has no native authority')
    path=HERE/'long_guard_owned_runtime.py';runtime=ModuleType('bound_long_owned_runtime');runtime.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),runtime.__dict__)
    runtime.preflight(launch_ref)
    raise ValueError('decoded long owner-approved qualifier integration remains closed')
