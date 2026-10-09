from pathlib import Path
from types import ModuleType
import hashlib,json,sys
P=Path(__file__).resolve().parent.parent
INVENTORY_PATH=P/'artifacts/arithmetic_tokeninventory_work/PASSIVE_CPU_TOKENIZER_INVENTORY003.json'
INVENTORY_REF={'path':str(INVENTORY_PATH),'bytes':859099,'sha256':'0b50e0509199b949c02a83259760d17a8bd2e4a8977058c8e7d6e12d8da3e441'}
def load(path,pin):
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==pin;module=ModuleType('selected_'+path.stem);module.__file__=str(path);exec(compile(raw,str(path),'exec'),module.__dict__);return module
assert sys.flags.isolated==1 and sys.flags.no_site==1 and sys.dont_write_bytecode is True
I=load(P/'artifacts/arithmetic_tokeninventory_work/cpu_token_inventory.py','f817cdbe5f2ba2c3a5c696b5ad31e915581de3ae7ce7f1c840b9974f70ff6ea0')
raw=INVENTORY_PATH.read_bytes();assert len(raw)==INVENTORY_REF['bytes'] and hashlib.sha256(raw).hexdigest()==INVENTORY_REF['sha256'];value=json.loads(raw)
R=load(P/'artifacts/arithmetic_tokenmeasurement_work/measure_arithmetic_cpu.py','97e56045c8db5a43e93ff427e61b44e3151f666933ac1a8dfcc07ae254cb16a5')
original_path=sys.path;before=list(original_path);site=value['selected_site_packages'];assert Path(site).resolve()==Path(site) and Path(site).is_dir()
print(json.dumps({'event':'CPU_ARITHMETIC_TOKEN_MEASUREMENT_STARTED','scope':'512_ROWS_TOKENIZER_ONLY','inventory_body_sha256':value['sha256'],'model_GPU_optimizer_scoring_work_authorized':False}),flush=True)
try:
    # Root-selected checked site is needed for optional Rust/Jinja tokenizer
    # dependencies. No .pth processing or site module is invoked.
    sys.path.append(site)
    result=R.run(population_path=P/'evidence/arithmetic_guard_reviewed_cpu_20261009/current/ARITHMETIC_GUARD_POPULATION001.json',model_dir=value['asset_root'],inventory_check=lambda stage:I.verify_inventory(INVENTORY_REF))
finally:
    sys.path=original_path;original_path[:]=before
print(json.dumps({'status':result['status'],'error':result['error'],'resources':result['resources'],'receipt':str(R.OUTPUT/'RUN_RECEIPT.json'),'Root_import_path_restored':sys.path==before}),flush=True)
raise SystemExit(0 if result['error'] is None else 2)
