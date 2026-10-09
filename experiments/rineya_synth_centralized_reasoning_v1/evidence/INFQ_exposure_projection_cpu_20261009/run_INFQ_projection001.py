"""Root-selected existing-record CPU projection under standing authorization."""
from pathlib import Path
from types import ModuleType
import hashlib,json,sys
P=Path(__file__).resolve().parent.parent
W=P/'artifacts/infq_exposure_projection_work/frozen_INFQ_projection002'
SOURCE='ab273d23cf6a3dc24e80de4a3df5ba735100b164bc3dd399f188d2755b17bf5d'
SCOPE='d4c636c3fd23ff2ea14f639cdcc725d99e978672de23e44e0c9a4f393afa2fa7'
path=W/'infq_projection.py';raw=path.read_bytes()
assert hashlib.sha256(raw).hexdigest()==SOURCE
assert hashlib.sha256((W/'READ_SCOPE002.json').read_bytes()).hexdigest()==SCOPE
m=ModuleType('root_pinned_original_INFQ_projection');m.__file__=str(path)
exec(compile(raw,str(path),'exec'),m.__dict__)
receipt=m.run(W/'READ_SCOPE002.json',SCOPE)
assert hashlib.sha256(path.read_bytes()).hexdigest()==SOURCE
assert not {'torch','transformers','tokenizers','safetensors','numpy','ctypes','psutil'}.intersection(sys.modules)
print(json.dumps({'output':str(m.OUTPUT),'result_reference':receipt['result_reference'],
    'wall_seconds':receipt['wall_seconds'],'all95_original_byte_links_verified':receipt['all95_original_byte_links_verified'],
    'global_exposure_coverage':False,'model_GPU_or_scoring_performed':False}))
