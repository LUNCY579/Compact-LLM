"""Source-plan-bound inputs for a pending public import proposal."""
from pathlib import Path
from types import ModuleType
import hashlib,json
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[2];ROOT=PACKAGE.parents[1]
PLAN=PACKAGE/'artifacts/torch_import_observation_work/native_candidate/frozen_public_graph001/SOURCE_PLAN.json'
def module(path):
    m=ModuleType('closed_proposal_constructor');m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
M=module(HERE/'construct_proposal.py')
def read(path):
    raw=path.read_bytes();require=lambda v:None if v else (_ for _ in ()).throw(ValueError('selectedbytes'));require(len(raw)<=16*1024**2);v=json.loads(raw);M.sealed(v);return v,{'path':str(path),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':v['sha256']}
def descriptor(path,role):return M.descriptor(path.read_bytes(),role)
def inputs():
    plan,pref=read(PLAN);sources={}
    for name,row in plan['sources'].items():
        p=PLAN.parent/'sources'/name;M.require(descriptor(p,'python_source')==row['descriptor'],'frozenpublicsourcebytes');sources[row['target_path']]=row['descriptor']
    sources.update(plan['durable_dependencies'])
    # Current public package leaves remain exact; planned targets add only their
    # selected descriptors. Artifact/evidence scope is supplied by the recipe.
    helperpath=PACKAGE/'artifacts/torch_import_observation_work/native_candidate/import_runtime_qualification.py';Q=module(helperpath)
    for text in Q.B.package_source_roster():
        p=Path(text)
        if text in sources:continue
        role='source_json' if p.suffix=='.json' else 'source_markdown' if p.suffix=='.md' else 'python_source';sources[text]=descriptor(p,role)
    observer,oref=read(PACKAGE/'configs/SYNTH_OBSERVER_STARTUP_RUNTIME_009.json');observerreg=observer['library_event_observer'];sources.update(observerreg['native_source_files']);build=observerreg['build_receipt']['path'];sources[build]=descriptor(Path(build),'source_json')
    aliases=PACKAGE/'typing_alias_checks.py'
    if str(aliases) not in sources:
        candidate=PACKAGE/'artifacts/tensor_observed_admission_work/typing_alias_checks.py';sources[str(aliases)]=descriptor(candidate,'python_source')
    # Explicit source recipe supplies exact durable source-only dependencies.
    recipe=M.seal({'kind':'REGISTERED_IMPORT_TRANSITIVE_SOURCE_ROSTER','source_files':{p:d for p,d in sources.items() if p in plan['durable_dependencies'] or not Path(p).is_relative_to(PACKAGE) or 'evidence' in Path(p).relative_to(PACKAGE).parts}})
    recipe_path=HERE/'IMPORT_TRANSITIVE_SOURCE_RECIPE001.json';raw=(M.canonical(recipe)+'\n').encode();recipe_ref={'path':str(recipe_path),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':recipe['sha256']}
    runtime,rref=read(PACKAGE/'artifacts/library_expectation_registration_work/ADDITIVE_WINDOWS_RUNTIME_CANDIDATE001.json');derivative,dref=read(PACKAGE/'artifacts/runtime_refresh_work/TENSOR_RUNTIME_BYTE_CANDIDATE001.json');runtime={**runtime,'rosters':derivative['rosters'],'inherited65_map':derivative['inherited65_map'],'stdlib779_map':derivative['stdlib779_map'],'external_dependency_files':derivative['external_dependency_files'],'stdlib_layout':observer['stdlib_layout']};runtime={k:v for k,v in runtime.items() if k!='sha256'};runtime=M.seal(runtime)
    passive,aref=read(PACKAGE/'evidence/psutil_bootstrap_assumptions_reviewed_cpu_20261007/PASSIVE_PSUTIL_BOOTSTRAP_ASSUMPTIONS001.json');proposal,_=read(PACKAGE/'artifacts/tensor_two_tensor_proposal_work/TWO_TENSOR_PROSPECTIVE_PROPOSAL_V2_001.json');old,oldref=read(PACKAGE/'configs/SYNTH_CONTEXT_PHASE_NO_MODEL_D001.json');expectations,_=read(PACKAGE/'artifacts/library_expectation_registration_work/LIBRARY_EXPECTATION_INPUTS001.json')
    return {'package':str(PACKAGE),'source_plan':plan,'source_files':sources,'runtime_basis':runtime,'passive_preload':passive,'observed_selection':proposal['observed009_selection'],'observer_registration':observerreg,'original_config':old,'initial35':expectations['expected_initial_loaded_libraries'],'source_recipe_reference':recipe_ref,'metadata_references':{'source_plan':pref,'additional_preload':aref,'runtime_basis':rref,'tensor_derivative':dref,'observer_runtime':oref,'original_config':oldref},'packet_paths':{name:str(HERE/('IMPORT_'+name.upper()+'_CLOSED001.json')) for name in ('config','runtime','registration')}},(recipe_path,raw)
def main():
    kwargs,(recipe_path,recipe_raw)=inputs();proposal=M.build(**kwargs);M.validate(proposal,source_plan=kwargs['source_plan'],expected_source_map_sha256=M.digest(kwargs['source_files']),expected_runtime_map_sha256=M.digest(proposal['runtime']['files']))
    with recipe_path.open('xb') as f:f.write(recipe_raw)
    for name in ('config','runtime','registration'):
        with Path(kwargs['packet_paths'][name]).open('x',encoding='utf-8') as f:f.write(M.canonical(proposal[name])+'\n')
    p=HERE/'PUBLIC_IMPORT_CLOSED_PROPOSAL001.json'
    with p.open('x',encoding='utf-8') as f:f.write(M.canonical(proposal)+'\n')
    print(M.canonical({'sources':len(kwargs['source_files']),'runtime':len(proposal['runtime']['files']),'rosters':len(proposal['runtime']['rosters']),'baseline':len(proposal['runtime']['measurement_policy']['initial_loaded_libraries']),'body_sha256':proposal['sha256'],'file_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'native_launch_authorized':False}))
if __name__=='__main__':main()
