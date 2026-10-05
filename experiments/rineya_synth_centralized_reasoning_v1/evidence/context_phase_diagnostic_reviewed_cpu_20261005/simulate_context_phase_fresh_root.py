"""CPU fresh-root test layout only; no native diagnostic or ignored stage."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

STAGE=Path(__file__).resolve().parent
PACKAGE=STAGE.parents[1]
sys.path.insert(0,str(PACKAGE))
import native_contract,inference_qual_worker,short_ram_monitor

with tempfile.TemporaryDirectory(dir=STAGE) as directory:
    fresh=Path(directory)/'CPU_FRESH_ROOT';(fresh/'tests').mkdir(parents=True)
    for name in ('context_phase_contract.py','context_phase_worker.py','context_phase_supervisor.py'):
        (fresh/name).write_bytes((STAGE/name).read_bytes())
    test=fresh/'tests/test_context_phase_diagnostic.py';test.write_bytes((STAGE/test.name).read_bytes())
    ignored=fresh/'artifacts/context_phase_diagnostic_work';assert not ignored.exists()
    spec=importlib.util.spec_from_file_location('fresh_context_tests',test)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.PACKAGE==fresh.resolve() and module.CODE==fresh.resolve()
    assert Path(module.worker.__file__).resolve()==fresh/'context_phase_worker.py'
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    assert not ignored.exists()
    print(json.dumps({'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),
        'actual_freshroot_source_targets':True,'ignored_stage_absent_before_after':True,
        'native_model_GPU_or_largebinary_reads':False}))
    if not result.wasSuccessful():raise SystemExit(1)
