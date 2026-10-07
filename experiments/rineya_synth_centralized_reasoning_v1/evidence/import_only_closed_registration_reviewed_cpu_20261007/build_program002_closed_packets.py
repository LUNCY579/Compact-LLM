"""Write closed program002 packets; commits/approval/native binding pending."""
from pathlib import Path
from types import ModuleType
import hashlib,json
HERE=Path(__file__).resolve().parent
def load(path,name):
    m=ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
def main():
    M=load(HERE/'closed_registration.py','CPU_closed_registration_builder');I=load(HERE/'program002_inputs.py','CPU_frozen_program_inputs');inputs=I.inputs();C=I.module(I.PROGRAM/'import_observation_contract.py','CPU_selected_program_contract')
    paths={name:str(HERE/('CLOSED_IMPORT_'+name.upper()+'_PROGRAM002.json')) for name in ('config','runtime','registration')}
    bundle=M.build(**inputs,implementation_commit=None,dispatch_head=None,packet_paths=paths);C.validate_import_metadata(bundle['config'],bundle['runtime'],bundle['registration'],None);M.validate(bundle)
    references={}
    for name,path in paths.items():
        raw=(M.canonical(bundle[name])+'\n').encode();p=Path(path)
        with p.open('xb') as f:f.write(raw)
        assert p.read_bytes()==raw;references[name]={'path':path,'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':bundle[name]['sha256']}
    record={'kind':'CLOSED_PROGRAM002_IMPORT_OBSERVATION_PACKET_CONSTRUCTION','packet_references':references,'external_selection':bundle['selected_inputs'],'operation_registration':bundle['operation_registration'],'current_source_count':len(inputs['source_files']),'runtime_count':len(bundle['runtime']['files']),'roster_count':len(bundle['runtime']['rosters']),'observed_cache_pin_count':59,'actual_contract_descriptor':inputs['source_files'][str(I.PROGRAM/'import_observation_contract.py')],'implementation_commit':None,'dispatch_head':None,'nine_external_dispatch_anchors':None,'owner_approval':None,'native_binding_main_implemented':False,'current_native_runtime_qualified':False,'native_launch_authorized':False,'native_authority_issued':False};record=M.seal(record)
    with (HERE/'CLOSED_PROGRAM002_PACKET_RECEIPT001.json').open('x',encoding='utf-8') as f:f.write(M.canonical(record)+'\n')
    print(M.canonical({'sources':len(inputs['source_files']),'runtime':len(bundle['runtime']['files']),'rosters':len(bundle['runtime']['rosters']),'body_sha256':record['sha256'],'packets':references}))
if __name__=='__main__':main()
