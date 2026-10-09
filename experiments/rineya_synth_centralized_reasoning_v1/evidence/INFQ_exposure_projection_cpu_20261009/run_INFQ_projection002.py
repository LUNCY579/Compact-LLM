"""Additive legacy-schema repair; preserve the first actual failure."""
from pathlib import Path
from types import ModuleType
import hashlib,json,sys
P=Path(__file__).resolve().parent.parent
W=P/'artifacts/infq_exposure_projection_work/frozen_INFQ_projection004'
SOURCE='53b378ebb4d55c9f369335a858b22b9af9445c8ebdd0c328700df5a1b7213a3f'
SCOPE='67a5d8251d7bedf4f9fb2ff24845cdecf7b471ae6b535c966e32d13bc4581bfd'
path=W/'infq_projection.py';raw=path.read_bytes()
assert hashlib.sha256(raw).hexdigest()==SOURCE
assert hashlib.sha256((W/'READ_SCOPE003.json').read_bytes()).hexdigest()==SCOPE
m=ModuleType('root_pinned_original_INFQ_projection_repaired');m.__file__=str(path)
exec(compile(raw,str(path),'exec'),m.__dict__)
receipt=m.run(W/'READ_SCOPE003.json',SCOPE)
assert hashlib.sha256(path.read_bytes()).hexdigest()==SOURCE
assert not {'torch','transformers','tokenizers','safetensors','numpy','ctypes','psutil'}.intersection(sys.modules)
print(json.dumps({'output':str(m.OUTPUT),'result_reference':receipt['result_reference'],
    'wall_seconds':receipt.get('wall_seconds'),
    'all95_original_byte_links_verified':receipt['all95_original_byte_links_verified'],
    'global_exposure_coverage':False,'model_GPU_or_scoring_performed':False}))
