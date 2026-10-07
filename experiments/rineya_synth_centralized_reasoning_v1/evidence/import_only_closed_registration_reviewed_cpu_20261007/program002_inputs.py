"""Exact frozen import-program source graph and passive closed inputs."""
from pathlib import Path
from types import ModuleType
import hashlib,json
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1];ROOT=PACKAGE.parents[1]
PROGRAM=PACKAGE/'artifacts/torch_import_observation_work/frozen_program002'
PINS={'import_observation_contract.py':'e8e5fdde7a112567faf2beee7bbbf1524445a39de4efd2628db4ca136d6cf43e','import_observation_worker.py':'7774e6e6f12cab39e2ec333ce42dc28fbf5f879f080d55c69ece51911408f886','import_observation_auditor.py':'27ae100fb74444609f6cc2e5f00e622c73b80b3d12a90dded98c9b1b814cf43e','measurement_guard.py':'55bcdff206a8574b6246e5f2c0c047ded4cac3735e1898dc1ec85c32fd01d8d1','measurement_replay.py':'ce8ea36cd113310d35bd09ee223f279bdbf5703b8c7d93ae143325bd9f95124f'}
def raw_selected(path,pin=None):
    before=path.stat();assert path.resolve()==path and before.st_size<=16*1024**2
    with path.open('rb') as f:raw=f.read(before.st_size+1)
    after=path.stat();assert len(raw)==before.st_size and (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
    assert pin is None or hashlib.sha256(raw).hexdigest()==pin
    return raw
def descriptor(path):
    raw=raw_selected(path);return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}
def module(path,name):
    raw=raw_selected(path,PINS.get(path.name));m=ModuleType(name);m.__file__=str(path);exec(compile(raw,str(path),'exec'),m.__dict__);return m
def inputs():
    sources={str(PROGRAM/name):descriptor(PROGRAM/name) for name in PINS}
    for name,pin in PINS.items():assert sources[str(PROGRAM/name)]['sha256']==pin
    dependencies=json.loads(raw_selected(PROGRAM/'DEPENDENCIES.json'))
    for relative,d in dependencies.items():
        if 'predecessors' in Path(relative).parts:path=PROGRAM/'predecessors'/Path(relative).name
        else:path=(ROOT/relative).resolve()
        assert descriptor(path)==d;sources[str(path)]=d
    E=PACKAGE/'evidence/raw_library_pipeline_reviewed_cpu_20261006/prefix_library_evidence.py';sources[str(E)]=descriptor(E)
    def json_input(relative):
        path=PACKAGE/relative;raw=raw_selected(path);v=json.loads(raw);return v,{'path':str(path),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':v['sha256']}
    runtime,rref=json_input('artifacts/library_expectation_registration_work/ADDITIVE_WINDOWS_RUNTIME_CANDIDATE001.json');rosters,sref=json_input('artifacts/runtime_refresh_work/TENSOR_RUNTIME_BYTE_CANDIDATE001.json');proposal,pref=json_input('artifacts/tensor_two_tensor_proposal_work/TWO_TENSOR_PROSPECTIVE_PROPOSAL_V2_001.json');initial,iref=json_input('artifacts/library_expectation_registration_work/LIBRARY_EXPECTATION_INPUTS001.json')
    observation=proposal['observed009_selection'];observation={**observation,'sha256':hashlib.sha256(json.dumps(observation,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()}
    return {'runtime_candidate':runtime,'roster_candidate':rosters,'observation':observation,'initial35_expectation':initial['expected_initial_loaded_libraries'],'source_files':sources,'main_source':str(PROGRAM/'import_observation_worker.py'),'resources':module(PROGRAM/'import_observation_contract.py','CPU_actual_import_contract').RESOURCES,'metadata_references':{'runtime_candidate':rref,'roster_candidate':sref,'proposal':pref,'initial_expectation':iref}}
