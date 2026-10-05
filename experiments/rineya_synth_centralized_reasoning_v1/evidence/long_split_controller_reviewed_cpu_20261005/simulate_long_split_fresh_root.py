"""CPU fresh root layout control; no ignored stage, model or decoder."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

STAGE=Path(__file__).resolve().parent
PACKAGE=STAGE.parents[1]

with tempfile.TemporaryDirectory(dir=STAGE) as directory:
    fresh=Path(directory)/'CPU_FRESH_ROOT';(fresh/'tests').mkdir(parents=True)
    for name in ('long_inference_component.py','long_split_controller.py'):
        (fresh/name).write_bytes((STAGE/name).read_bytes())
    (fresh/'inference_component.py').write_bytes((PACKAGE/'inference_component.py').read_bytes())
    for name in ('test_inference_component.py','test_long_split_controller.py'):
        source=PACKAGE/'tests'/name if name=='test_inference_component.py' else STAGE/name
        (fresh/'tests'/name).write_bytes(source.read_bytes())
    evidence=fresh/'evidence/long_train_prompts_cpu_20261005';evidence.mkdir(parents=True)
    (evidence/'TRAIN_PROMPTS_CPU.json').write_bytes((PACKAGE/'evidence/long_train_prompts_cpu_20261005/TRAIN_PROMPTS_CPU.json').read_bytes())
    ignored=fresh/'artifacts/long_context_qualification_work';assert not ignored.exists()
    sys.path.insert(0,str(fresh))
    for name in ('long_split_controller','long_inference_component','inference_component'):
        sys.modules.pop(name,None)
    test=fresh/'tests/test_long_split_controller.py'
    spec=importlib.util.spec_from_file_location('fresh_root_split_tests',test)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.PACKAGE==fresh.resolve() and module.CODE==fresh.resolve()
    assert module.split.PACKAGE==fresh.resolve() and not module.split.STAGED
    assert Path(module.split.component.__file__).resolve()==fresh/'long_inference_component.py'
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    assert not ignored.exists()
    print(json.dumps({'root_layout_tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'controller_and_test_PACKAGE_fresh_root':True,'exact_tracked_packet_path_used':True,
        'ignored_stage_absent_before_after':True,'derived_component_fresh_root':True,
        'native_model_GPU_decoder_or_grading':False}))
    if not result.wasSuccessful():raise SystemExit(1)
