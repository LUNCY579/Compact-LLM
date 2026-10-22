"""Freeze the final tested executable identities before the one real-model launch."""
from pathlib import Path
from argparse import Namespace
import importlib.util
import sys
import time
from prepare import BASE,OUT,OLD,HERE,PILOT,sha,read,write,copy_new

def main():
    spec=importlib.util.spec_from_file_location('doselora_final_train',HERE/'train.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.inputs(Namespace(prepared=str(OUT/'prepared'),plan=str(OUT/'PLAN.json'),grant=str(OUT/'GRANT.json')),retain=False)
    identity=module.execution_identity()
    gate=read(OUT/'BUILD_GATE_LORA.json')
    gate['training_execution_identity']=identity
    gate['storage_amendment_sha256']=sha(OUT/'AMENDMENT_STORAGE.md')
    gate['finalized_unix']=time.time()
    for name,expected in gate['source_assets_sha256'].items():
        if sha(Path(gate['source'])/name)!=expected:
            raise ValueError('Source asset changed')
    if sha(OUT/'PREREG.md')!=read(OUT/'GRANT.json')['registration_sha256']:
        raise ValueError('Preregistration changed')
    code={name:sha(HERE/name) for name in ('train.py','adapters.py','materialize.py','prepare.py','finalize_prelaunch.py')}
    write(OUT/'BUILD_GATE_LORA_FINAL.json',gate)
    write(OUT/'PRELAUNCH_BINDING.json',{'schema':1,'status':'PASS_INPUTS_SOURCE_CODE_BEFORE_EXECUTION',
        'training_execution_identity':identity,'code_files_sha256':code,
        'source_assets_sha256':gate['source_assets_sha256'],
        'plan_sha256':sha(OUT/'PLAN.json'),'grant_sha256':sha(OUT/'GRANT.json'),
        'registration_sha256':sha(OUT/'PREREG.md'),
        'session_envelope_sha256':sha(OUT/'SESSION_ENVELOPE_V2.json'),
        'build_gate_sha256':sha(OUT/'BUILD_GATE_LORA_FINAL.json'),
        'expected_charged_training_and_qualification_tokens':389717,
        'expected_qualification_calls':48,'expected_optimizer_updates':193,
        'model_execution':False,'quality_claim':False})
    archive=OUT/'code_before_training';archive.mkdir()
    for name in code:
        copy_new(HERE/name,archive/name)
    write(archive/'FILES.json',code)
    print('FINAL_PRELAUNCH_BINDING_FROZEN; no model was run')

if __name__=='__main__':
    main()
