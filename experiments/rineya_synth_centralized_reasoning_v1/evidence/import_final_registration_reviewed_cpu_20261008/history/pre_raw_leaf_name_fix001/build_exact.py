"""One closed import recipe on actual implementation source, no native grant."""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
import hashlib,json,subprocess
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1];ROOT=PACKAGE.parents[1]
IMPLEMENTATION='d39d64b1447c8f0baffdebcc4a0a1092f76bbd88'
RUN_ID='prefix_input_owned_import_observation_001'
RAW_LEAVES=('OWNED_EXTERNAL_ANCHORS.json','SUPERVISOR_RECEIPT.json','RESOURCE_OBSERVATIONS.json','supervisor_ownership.json','PHASE_OBSERVATIONS.jsonl','PHASE_ACCOUNTING_FINAL.json','IMPORT_LIBRARY_MEASUREMENT.json','worker_result.json','worker.stdout.log','worker.stderr.log','SUPERVISOR_HEARTBEAT.jsonl','SHORT_RAM_ABORT_DIAGNOSTIC.json','WORKER_LAUNCH_FAILURE.json')
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def seal(v):return {**v,'sha256':digest(v)}
def write(path,value):
    raw=(canonical(value)+'\n').encode()
    with path.open('xb') as f:f.write(raw)
    return {'path':str(path),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':value['sha256']}
def desc(path):
    raw=path.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'source_json' if path.suffix=='.json' else 'source_markdown' if path.suffix=='.md' else 'native_observer_source' if path.suffix in ('.c','.h') else 'python_source'}
def source_inventory():
    require=lambda v,m:None if v else (_ for _ in ()).throw(ValueError(m));require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()==IMPLEMENTATION,'actual selected implementation HEAD')
    path=PACKAGE/'import_runtime_qualification.py';raw=path.read_bytes();Q=ModuleType('selected_public_import_qualification');Q.__file__=str(path);exec(compile(raw,str(path),'exec'),Q.__dict__)
    sourceplan=json.loads((PACKAGE/'evidence/import_public_graph_reviewed_cpu_20261007/frozen_public_graph002/PUBLIC_IMPORT_SOURCE_PLAN002.json').read_bytes());sources={p:desc(Path(p)) for p in Q.B.package_source_roster()};sources.update(sourceplan['durable_dependencies'])
    for row in sourceplan['sources'].values():sources[row['target_path']]=desc(Path(row['target_path']))
    for name in ('import_dispatch_postrun.py','import_terminal_audit.py'):sources[str(PACKAGE/name)]=desc(PACKAGE/name)
    for p,d in sources.items():
        raw=Path(p).read_bytes();require(desc(Path(p))==d,'actual selected source descriptor');blob=subprocess.check_output(['git','cat-file','blob',IMPLEMENTATION+':'+Path(p).relative_to(ROOT).as_posix()],cwd=ROOT);require(blob==raw,'source exact immutable implementation blob')
    return sources,Q
def main():
    sources,Q=source_inventory();old=json.loads((PACKAGE/'artifacts/torch_import_registration_work/public_graph_stage/PUBLIC_IMPORT_CLOSED_PROPOSAL002.json').read_bytes());config=deepcopy(old['config']);runtime=deepcopy(old['runtime']);registration=deepcopy(old['registration'])
    output=PACKAGE/'artifacts'/RUN_ID;preserved=PACKAGE/'evidence'/(RUN_ID+'_preserved');paths={name:str(HERE/('IMPORT_OBSERVATION_'+name.upper()+'_001.json')) for name in ('config','runtime','registration')}
    transitive={p:d for p,d in sources.items() if Path(p).is_relative_to(PACKAGE/'evidence')};recipe=seal({'kind':'REGISTERED_IMPORT_TRANSITIVE_SOURCE_ROSTER','source_files':transitive});recipe_ref=write(HERE/'IMPORT_SOURCE_RECIPE001.json',recipe)
    for v in (config,runtime,registration):v.pop('sha256',None);v['source_commit']=IMPLEMENTATION
    config['source_files']=sources;runtime['metadata_references']['source_roster']=recipe_ref;registration['source_files']=sources;registration['config_path']=paths['config'];exceptions={}
    for leaf in (*RAW_LEAVES,'PRESERVATION.json',*( 'lease_'+name for name in ('reservation.json','worker.json','resume.json','terminal.json'))):exceptions[str(preserved/leaf)]={'role':'evidence_output'}
    for p in sources:exceptions[str(preserved/'source_snapshots'/(hashlib.sha256(p.encode()).hexdigest()+Path(p).suffix))]={'role':'evidence_source_snapshot','source_path':p}
    registration['git_output_exceptions']=exceptions;runtime=seal(runtime);config=seal(config);registration.update(config_body_sha256=config['sha256'],runtime_body_sha256=runtime['sha256']);registration=seal(registration)
    inventory=seal({'kind':'ACTUAL_D39_IMPORT_REGISTRATION_SOURCE_RUNTIME_SELECTION','implementation_commit':IMPLEMENTATION,'source_files':sources,'runtime_files':runtime['files'],'rosters':runtime['rosters'],'source_inventory_sha256':digest(sources),'runtime_inventory_sha256':digest(runtime['files']),'source_count':len(sources),'runtime_count':len(runtime['files']),'current_whole_hash_completed':False,'native_authority_issued':False});invref=write(HERE/'ACTUAL_SOURCE_RUNTIME_SELECTION001.json',inventory)
    refs={name:write(Path(paths[name]),v) for name,v in (('config',config),('runtime',runtime),('registration',registration))}
    manifest=seal({'kind':'CLOSED_EXACT_IMPORT_ONLY_RUN_RECIPE','implementation_commit':IMPLEMENTATION,'external_dispatch_head':None,'run_id':RUN_ID,'output_directory':str(output),'preserved_directory':str(preserved),'packet_references':refs,'source_recipe':recipe_ref,'source_runtime_selection':invref,'source_count':len(sources),'runtime_count':len(runtime['files']),'raw_future_leaf_allowlist':list(RAW_LEAVES),'exact_preservation_exception_count':len(exceptions),'owner_grant_reference':None,'owner_receipt_reference':None,'complete_current_byte_review':None,'future_command':{'executable':runtime['cache_policy']['interpreter_profile']['executable'],'argv_prefix':['-I','-S','-B',str(PACKAGE/'import_dispatch_postrun.py'),'--config',paths['config'],'--output-dir',str(output),'--preserved-dir',str(preserved)],'nine_anchors':{'expected_config_file':refs['config']['file_sha256'],'expected_config_body':refs['config']['body_sha256'],'registration_path':paths['registration'],'expected_registration_file':refs['registration']['file_sha256'],'expected_registration_body':refs['registration']['body_sha256'],'runtime_path':paths['runtime'],'expected_runtime_file':refs['runtime']['file_sha256'],'expected_runtime_body':refs['runtime']['body_sha256'],'expected_head':None},'separate_owner_references_required':True},'scope':Q.SCOPE,'permission':Q.PERMISSION,'resources':config['resources'],'native_launch_authorized':False,'native_runtime_qualified':False,'native_authority_issued':False});write(HERE/'EXACT_IMPORT_RUN_RECIPE001.json',manifest)
    print(canonical({'sources':len(sources),'runtime':len(runtime['files']),'exceptions':len(exceptions),'selection_file':invref['file_sha256'],'recipe_body':manifest['sha256'],'implementation':IMPLEMENTATION,'launch':False}))
if __name__=='__main__':main()
