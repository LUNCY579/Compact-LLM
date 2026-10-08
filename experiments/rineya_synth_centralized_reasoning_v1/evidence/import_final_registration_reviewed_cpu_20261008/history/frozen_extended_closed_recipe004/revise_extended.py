"""Extended closed recipe; new implementation/whole-byte review remain pending."""
from pathlib import Path
from types import ModuleType
import hashlib,json
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
PASSIVE=PACKAGE/'evidence/import_launch_corrections_reviewed_cpu_20261007/frozen_ram_source_fix002/PASSIVE_IMPORT_BOOTSTRAP_ASSUMPTIONS002.json'
PASSIVE_PIN='0fe6d869bbf61a3125841164a0e11d2eb2a8ef46743f89bd7423259b02d62d91'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def seal(v):return {**v,'sha256':digest(v)}
def write(p,v):
    raw=(canonical(v)+'\n').encode()
    with p.open('xb') as f:f.write(raw)
    return {'path':str(p),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':v['sha256']}
def overlay(receipt):
    assert receipt['schema_version']==2 and receipt['kind']=='PASSIVE_IMPORT_BOOTSTRAP_SOURCE_CACHE_PE_ASSUMPTIONS002' and receipt['copy_source_or_cache_executed'] is False
    result={}
    for name,row in receipt['source_modules'].items():
        matches=[(p,d) for p,d in receipt['additional_cache_candidates'].items() if d['source_path']==row['path']];assert len(matches)==1;p,d=matches[0];assert d['code_source_equivalence'] is True and d['cache_executed'] is False
        result[name]={'path':row['path'],'descriptor':row['descriptor'],'cached_attribute_present':True,'cached_value_kind':'str','cache_path':p,'cache_descriptor':d['descriptor'],'source_execution_history_proven':False}
    for name,row in receipt['extension_modules'].items():result[name]={'path':row['path'],'descriptor':row['descriptor'],'cached_attribute_present':False,'cached_value_kind':'missing','cache_path':None,'cache_descriptor':None,'source_execution_history_proven':False}
    assert len(result)==11 and len(receipt['additional_cache_candidates'])==8;return result
def build():
    raw=PASSIVE.read_bytes();assert hashlib.sha256(raw).hexdigest()==PASSIVE_PIN;passive=json.loads(raw);assert passive['sha256']==digest({k:v for k,v in passive.items() if k!='sha256'})
    config=json.loads((HERE/'IMPORT_OBSERVATION_CONFIG_003.json').read_bytes());runtime=json.loads((HERE/'IMPORT_OBSERVATION_RUNTIME_003.json').read_bytes());registration=json.loads((HERE/'IMPORT_OBSERVATION_REGISTRATION_003.json').read_bytes());recipe=json.loads((HERE/'EXACT_IMPORT_RUN_RECIPE003.json').read_bytes());sources={}
    for p,old in config['source_files'].items():
        data=Path(p).read_bytes();sources[p]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'role':old['role']}
    for value in (config,runtime,registration,recipe):value.pop('sha256',None)
    for value in (config,runtime,registration):value['source_commit']=None
    config['source_files']=registration['source_files']=sources;runtime['cache_policy']['additional_preload_sources']=overlay(passive);runtime['metadata_references']['additional_preload']={'path':str(PASSIVE),'file_sha256':PASSIVE_PIN,'body_sha256':passive['sha256']}
    original59=runtime['cache_policy']['preloaded_module_cache_pins'];files=runtime['files']
    for row in runtime['cache_policy']['additional_preload_sources'].values():
        assert row['path'] in files and files[row['path']]==row['descriptor']
        cp=row['cache_path']
        if cp is not None:assert cp not in files or files[cp]==row['cache_descriptor'];files[cp]=row['cache_descriptor']
    assert len(files)==8725 and len(original59)==59 and len(set(row['cache_path'] for row in runtime['cache_policy']['additional_preload_sources'].values() if row['cache_path'])-set(original59))==8
    config['runtime_files']=registration['runtime_files']=files
    transitive={p:d for p,d in sources.items() if Path(p).is_relative_to(PACKAGE/'evidence')};source_recipe=seal({'kind':'REGISTERED_IMPORT_TRANSITIVE_SOURCE_ROSTER','source_files':transitive});source_ref=write(HERE/'IMPORT_SOURCE_RECIPE003.json',source_recipe);runtime['metadata_references']['source_roster']=source_ref;runtime=seal(runtime);config=seal(config);registration.update(config_path=str(HERE/'IMPORT_OBSERVATION_CONFIG_004.json'),config_body_sha256=config['sha256'],runtime_body_sha256=runtime['sha256']);registration=seal(registration);refs={name:write(HERE/('IMPORT_OBSERVATION_'+name.upper()+'_004.json'),value) for name,value in (('config',config),('runtime',runtime),('registration',registration))}
    selection=seal({'kind':'PENDING_NEW_IMPLEMENTATION_IMPORT_SOURCE_RUNTIME_SELECTION','implementation_commit':None,'source_files':sources,'runtime_files':files,'rosters':runtime['rosters'],'source_count':len(sources),'runtime_count':len(files),'source_inventory_sha256':digest(sources),'runtime_inventory_sha256':digest(files),'actual_current_bytes_hashed_in_one_window':False,'native_authority_issued':False});selection_ref=write(HERE/'ACTUAL_SOURCE_RUNTIME_SELECTION003.json',selection)
    recipe.update(implementation_commit=None,packet_references=refs,source_runtime_selection=selection_ref,source_recipe=source_ref,runtime_count=len(files),source_count=len(sources));recipe['future_command']['argv_prefix'][recipe['future_command']['argv_prefix'].index('--config')+1]=refs['config']['path'];recipe['future_command']['nine_anchors'].update(expected_config_file=refs['config']['file_sha256'],expected_config_body=refs['config']['body_sha256'],registration_path=refs['registration']['path'],expected_registration_file=refs['registration']['file_sha256'],expected_registration_body=refs['registration']['body_sha256'],runtime_path=refs['runtime']['path'],expected_runtime_file=refs['runtime']['file_sha256'],expected_runtime_body=refs['runtime']['body_sha256'],expected_head=None);recipe['passive_extended_reference']=runtime['metadata_references']['additional_preload'];recipe['preload_candidate_count']=11;recipe['passive_cache_candidate_total']=67;recipe['current_full_byte_review_pending_new_final_version_commit']=True;recipe=seal(recipe);write(HERE/'EXACT_IMPORT_RUN_RECIPE004.json',recipe)
    print(canonical({'sources':len(sources),'runtime':len(files),'cache_total':67,'overlay':11,'implementation':None,'recipe_body':recipe['sha256']}))
if __name__=='__main__':build()
