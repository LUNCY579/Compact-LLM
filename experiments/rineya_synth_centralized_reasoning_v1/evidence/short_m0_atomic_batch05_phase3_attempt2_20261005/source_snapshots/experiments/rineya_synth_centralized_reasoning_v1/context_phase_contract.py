"""Source-bound NO_MODEL diagnostic scope; staged entrypoints reject native IO."""
from pathlib import Path
import hashlib
import json
import sys

HERE=Path(__file__).resolve().parent
STAGED=HERE.name=='context_phase_diagnostic_work' and HERE.parent.name=='artifacts'
PACKAGE=HERE.parents[1] if STAGED else HERE
sys.path.insert(0,str(PACKAGE))
import native_contract as native
from rs_contract import digest,require

VERSION='CONTEXT_PHASE_NO_MODEL_DIAGNOSTIC_STAGED_1'
PURPOSE='CUDA_CONTEXT_PHASES_NO_MODEL_ONLY'
AUTHORIZATION={'scope':PURPOSE,'production_training':False,'protected_bank_access':False,'paid_services':False,
               'model_loading':False,'model_construction':False,'forward':False,'scoring':False}
FIELDS={'contract','id','authorization','resources','tolerances','runtime_versions','runtime_pins','source_pins',
        'offline_environment','helper_pins','diagnostic_profile','sha256'}
HELPERS=('native_worker.py','inference_qual_worker.py','native_supervisor.py','native_process.py','native_nvml.py',
         'short_ram_monitor.py','worker_ram_probe.py')
FORBIDDEN_FIELDS={'model_dir','model_pins','rows','checkpoint','model_config','training','grading','empty_cache','reset_device'}
SOURCE_NAMES=('context_phase_contract.py','context_phase_worker.py','context_phase_supervisor.py')


def source_roster():
    """Exact package/transitive/test roster, without model/runtime-byte reads."""
    paths=set(PACKAGE.glob('*.py')) | set((PACKAGE/'tests').glob('*.py'))
    paths.update(PACKAGE/name for name in SOURCE_NAMES)
    paths.update(PACKAGE/name for name in HELPERS)
    paths.update(PACKAGE/name for name in ('native_contract.py','rs_contract.py','inference_contract.py',
        'inference_component.py','native_loader.py','native_precision.py'))
    paths.update([PACKAGE/'PILOT_DECISIONS.json',PACKAGE/'pilot_trainable_inventory.json',
        PACKAGE/'NATIVE_QUALIFICATION_PROPOSAL.json',PACKAGE/'AGENTS.md',PACKAGE.parents[1]/'AGENTS.md',
        PACKAGE.parents[1]/'RINEYA_SYNTH_CENTRALIZED_REASONING.md'])
    return {str(path.resolve()) for path in paths}


def integrated():
    require(not STAGED and HERE==PACKAGE,'staged NO_MODEL diagnostic cannot authorize native dispatch')


def validate_manifest(body):
    require(type(body) is dict and set(body)==FIELDS and not (set(body)&FORBIDDEN_FIELDS),'exact NO_MODEL scope allowlist')
    require(body['sha256']==digest({k:v for k,v in body.items() if k!='sha256'}),'diagnostic manifest checksum')
    require(body['contract']==VERSION and body['authorization']==AUTHORIZATION,'distinct closed NO_MODEL purpose')
    require(body['resources']==native.RESOURCE_BOUNDS and body['tolerances']==native.TOLERANCES,'unchanged diagnostic resource/numerical floors')
    require(body['runtime_versions']==native.RUNTIME and body['offline_environment']==native.OFFLINE_ENV,'exact runtime versions/offline ENV')
    require(set(body['helper_pins'])==set(HELPERS),'complete existing helper/NVML/coherent monitor pins')
    for mapping in (body['source_pins'],body['runtime_pins'],body['helper_pins']):
        require(type(mapping) is dict and mapping and all(type(k) is str and type(v) is str and len(v)==64
                and all(c in '0123456789abcdef' for c in v) for k,v in mapping.items()),'typed complete source/runtime SHA256 inventory')
    require(set(body['source_pins'])==source_roster(),'exact own/transitive/package/test source roster; omissions forbidden')
    require(all(body['source_pins'][str((PACKAGE/name).resolve())]==pin for name,pin in body['helper_pins'].items()),'same helper/transitive source byte pins')
    require(len(body['runtime_pins'])==65,'registered runtime65 including executed AST')
    require(body['diagnostic_profile']=={'id':VERSION,'no_model':True,'no_tensor_allocation':True,
        'guard_query_proxy':'one exact actual call; nonrecursive phase accounting; unchanged result/error',
        'external_NVML_authority_unchanged':True,'Windows_counters':'optional diagnostic only; unsupported explicit',
        'native_fit_or_historical_cause_established':False},'exact non-attributive diagnostic profile')
    return body


def preflight(path):
    integrated()
    path=Path(path).absolute()
    require(path.parent.resolve()==(PACKAGE/'configs').resolve() and path.is_file() and not path.is_symlink(),'owned committed diagnostic config')
    body=validate_manifest(native.read_json(path))
    for name,pin in body['helper_pins'].items():require(native.filehash(PACKAGE/name)==pin,'unchanged helper byte drift: '+name)
    for name,pin in body['source_pins'].items():require(native.filehash(Path(name))==pin,'diagnostic source byte drift')
    core=native.runtime_files()
    import sysconfig
    ast_path=(Path(sysconfig.get_path('stdlib'))/'ast.py').resolve()
    require(body['runtime_pins']=={**core,str(ast_path):native.filehash(ast_path)},'exact current runtime65 closure')
    import subprocess
    require(not subprocess.run(['git','status','--porcelain'],cwd=PACKAGE.parents[1],capture_output=True,text=True,check=True).stdout.strip(),'clean committed diagnostic sources/config')
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=PACKAGE.parents[1],capture_output=True,text=True,check=True).stdout.strip()
    return {**body,'worker_path':str(PACKAGE/'context_phase_worker.py'),'config_path':str(path.resolve()),
            'config_sha256':native.filehash(path),'source_commit':commit,
            'bindings':{'source':body['source_pins'],'runtime':body['runtime_pins'],'ENV':body['offline_environment'],'helpers':body['helper_pins']}}
