"""Offline file/hash/header inventory only: never imports tensor/model libraries."""
from pathlib import Path
import datetime
import hashlib
import importlib.metadata
import json
import math
import struct
import subprocess

OUT = Path(__file__).resolve().parent
BASE = Path('C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores')
CENTRAL = Path('C:/Users/clclc/BusinessProject/ChatBubble-post-training-Mimo/experiments/post_training_mimo_centralized_v1')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def header(p):
    with p.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        assert n <= 16 * 1024**2
        raw = f.read(n)
    h = json.loads(raw)
    tensors = {k:v for k,v in h.items() if k != '__metadata__'}
    return {'header_bytes':n, 'header_sha256':hashlib.sha256(raw).hexdigest(),
            'tensor_count':len(tensors), 'stored_tensor_elements':sum(math.prod(v['shape']) for v in tensors.values()),
            'dtype_counts':{dt:sum(v['dtype']==dt for v in tensors.values()) for dt in sorted({v['dtype'] for v in tensors.values()})},
            'keys':sorted(tensors), 'shapes':{k:v['shape'] for k,v in tensors.items()},
            'payload_loaded':False}

models = []
for name in ('Rineya-Titan-Experimental-1.0.0-devfreeze','Rineya-Supa-1.7.0-devfreeze','Rineya-Baya-1.7.0-devfreeze','Rineya-Halo-1.2.0-devfreeze'):
    p = BASE/name
    c=read(p/'config.json')
    files = {f.name:sha(f) for f in sorted(p.iterdir()) if f.is_file() and f.suffix in {'.safetensors','.json','.txt','.model','.jinja'}}
    tc=read(p/'tokenizer_config.json')
    tj=read(p/'tokenizer.json')
    template=(p/'chat_template.jinja').read_text() if (p/'chat_template.jinja').exists() else tc.get('chat_template')
    models.append({'name':name,'path':str(p),'config':c,'files':files,'rdd_inventory_snapshot_sha256':digest(files),
                   'serialized_weight_bytes':(p/'model.safetensors').stat().st_size,
                   'header':header(p/'model.safetensors'),
                   'rdd_static_layout_candidate':c.get('model_type')=='qwen3' and c.get('architectures')==['Qwen3ForCausalLM'] and not c.get('auto_map') and not c.get('quantization_config'),
                   'tokenizer':{'backend_canonical_sha256':digest(tj),'template_sha256':hashlib.sha256(template.encode()).hexdigest() if isinstance(template,str) else None,
                               'template_origin':'chat_template.jinja' if (p/'chat_template.jinja').exists() else 'tokenizer_config.json',
                               'added_tokens_sha256':digest(tj.get('added_tokens')),'vocab_size':len(tj['model']['vocab']),
                               'eos_token':tc.get('eos_token'),'pad_token':tc.get('pad_token'),'bos_token':tc.get('bos_token'),
                               'tokenizer_config_sha256':sha(p/'tokenizer_config.json')},
                   'model_runtime_qualified':False,'teacher_competence_qualified':False})
    print(name, 'hash/header inventory complete', flush=True)

e0=[]
for p in sorted((CENTRAL/'results').glob('*/E0_RESULT.json')):
    r=read(p)
    needed=['exact_native_load','save_reload_resume_bitwise','frozen_base_unchanged','finite_scoped_backward','eos_pad_mask']
    e0.append({'path':str(p),'sha256':sha(p),'status':r.get('status'),'config_sha256':r.get('config_sha256'),
               'checks':r.get('checks',{}),'pretrained_loads':r.get('pretrained_loads'),'optimizer_steps':r.get('optimizer_steps'),
               'error':r.get('error'),'accepted_by_unchanged_rdd_e0_contract':r.get('status')=='PASS_E0_SCOPED_NO_EFFICACY_CLAIM' and all(r.get('checks',{}).get(k) is True for k in needed)})

runtime={}
for package in ('torch','transformers','tokenizers','safetensors','peft','psutil','pytest'):
    try:runtime[package]=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:runtime[package]=None

result={'schema':1,'experiment':'RDD-001','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'evidence_kind':'STATIC_FILE_HASH_AND_SAFETENSORS_HEADER_ONLY','native_model_loads':0,'native_forward_calls':0,'optimizer_updates':0,
        'sealed_or_confirmation_bank_reads':0,'model_files_changed':False,'models':models,'e0_receipts':e0,
        'runtime_distribution_metadata':runtime,'runtime_executable':__import__('sys').executable,
        'unchanged_rdd_limits':{'host_admission_gib':7.4,'host_running_gib':6.0,'gpu_admission_gib':4.0,'gpu_running_gib':0.5,'disk_gib':4.0,'worker_tree_rss_gib':4.0,'wall_seconds':1800},
        'boundaries':['No tensor payload deserialization, tokenizer/model instantiation, optimizer, GPU allocation, conversion, evaluation, or confirmation reads.','Header parameter sums are stored tensor counts, not runtime-loaded parameter qualification.','Equal file/template hashes are static evidence; native rendered-token/output parity remains unqualified.','A currently clear lock or favorable memory snapshot is not resource admission or host authorization.','No other experiment permit was imported.']}
with (OUT/'asset_inventory.json').open('x',encoding='utf-8') as f:
    json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
print('RECEIPT',str(OUT/'asset_inventory.json'),sha(OUT/'asset_inventory.json'))
