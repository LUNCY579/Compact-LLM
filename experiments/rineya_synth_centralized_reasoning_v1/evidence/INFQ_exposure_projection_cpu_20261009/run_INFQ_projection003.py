"""Root-selected original-loader nesting repair; no model execution."""
from pathlib import Path
from types import ModuleType
import hashlib,json,sys
P=Path(__file__).resolve().parent.parent
W=P/'artifacts/infq_exposure_projection_work/frozen_INFQ_projection005'
SOURCE='cb121eac78a61e8634a17a48b61416e103b5ffeb5dd68bb7dab8db3b943d74ce'
SCOPE='c6cabdae832bb844a7ee3bf66e30fb6e7515b8eef87d645571b563fad608c0ed'
path=W/'infq_projection.py';raw=path.read_bytes()
assert hashlib.sha256(raw).hexdigest()==SOURCE
assert hashlib.sha256((W/'READ_SCOPE004.json').read_bytes()).hexdigest()==SCOPE
m=ModuleType('root_pinned_INFQ_projection_loader_schema');m.__file__=str(path)
exec(compile(raw,str(path),'exec'),m.__dict__)
receipt=m.run(W/'READ_SCOPE004.json',SCOPE)
assert hashlib.sha256(path.read_bytes()).hexdigest()==SOURCE
assert not {'torch','transformers','tokenizers','safetensors','numpy','ctypes','psutil'}.intersection(sys.modules)
print(json.dumps({'output':str(m.OUTPUT),'result_reference':receipt['result_reference'],
    'wall_checkpoint_seconds':receipt['wall_seconds_through_final_input_and_output_recheck'],
    'all95_original_byte_links_verified':receipt['all95_original_byte_links_verified'],
    'global_exposure_coverage':False,'model_GPU_or_scoring_performed':False}))
