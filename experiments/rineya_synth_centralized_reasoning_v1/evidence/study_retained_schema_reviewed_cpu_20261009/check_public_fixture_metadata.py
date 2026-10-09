from pathlib import Path
from types import ModuleType,SimpleNamespace
from copy import deepcopy
import ast,hashlib,json
P=Path(__file__).resolve().parents[2];S=P/'tests/test_study_native_chunk.py';tree=ast.parse(S.read_bytes());actor=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QuadraticAdam');method=next(n for n in actor.body if isinstance(n,ast.FunctionDef) and n.name=='metadata')
digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
namespace={'deepcopy':deepcopy,'chunk':SimpleNamespace(BACKEND_KIND='SYNTH_STUDY_TORCH_BOUNDARY_V1'),'schedule':SimpleNamespace(digest=digest)};exec(compile(ast.Module(body=[method],type_ignores=[]),str(S),'exec'),namespace)
obj=SimpleNamespace(execution_scope='CPU_PROTOCOL_FAKE',arm='M1',recipe_sha256='1'*64,cohort_sha256='2'*64,backend_source_sha256='3'*64,_attestation={'sha256':'4'*64});v=namespace['metadata'](obj,{'AUTHORED_BINDING':True})
expected={'schema_version','kind','experiment','scope','arm','seed','recipe_sha256','cohort_sha256','backend_source_sha256','fresh_start_attestation_sha256','adapter_parameter_order','adapter_inventory_sha256','binding','native_launch_authorized','training_authorized','native_resume_qualified','qualification_STATE1_used'};assert set(v)==expected and v['adapter_parameter_order']==['AUTHORED_SCALAR'] and len(v['adapter_inventory_sha256'])==64
receipt={'kind':'ROOT_AST_ONLY_AUTHORED_QUADRATIC_METADATA_COMPATIBILITY','metadata_fields':len(v),'source_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),'original_dataset_setup_executed':False,'model_framework_or_native_called':False};receipt['sha256']=digest(receipt)
(Path(__file__).parent/'ROOT_FIXTURE_METADATA001.json').write_bytes((json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode());print(json.dumps(receipt))
