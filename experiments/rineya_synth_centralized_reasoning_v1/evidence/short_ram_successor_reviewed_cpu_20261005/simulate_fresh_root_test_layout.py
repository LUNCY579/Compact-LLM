"""CPU fresh-checkout test layout, with ignored stage absent throughout."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest

STAGE=Path(__file__).resolve().parent
PACKAGE=STAGE.parents[1]
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(STAGE))
import native_supervisor,native_contract,worker_ram_probe

with tempfile.TemporaryDirectory(dir=STAGE) as directory:
    fresh=Path(directory)/'CPU_FRESH_CHECKOUT';fresh.mkdir();(fresh/'tests').mkdir()
    for name in ['short_collection_contract.py','short_collection_worker.py','short_collection_supervisor.py','short_ram_monitor.py']:
        (fresh/name).write_bytes((STAGE/name).read_bytes())
    for name in ['native_supervisor.py','native_contract.py','inference_qual_worker.py','native_process.py','worker_ram_probe.py']:
        (fresh/name).write_bytes((PACKAGE/name).read_bytes())
    history=fresh/'evidence/short_ram_successor_reviewed_cpu_20261005/original_source_preserved';history.mkdir(parents=True)
    for path in (STAGE/'original_source_preserved').iterdir():
        if path.is_file():(history/path.name).write_bytes(path.read_bytes())
    proof=fresh/'evidence/worker_ram_probe_cpu_20261005';proof.mkdir(parents=True)
    for name in ['CPU_ram_callback_comparison_20261005.json','preservation_receipt.json','worker_ram_probe.py']:
        (proof/name).write_bytes((PACKAGE/'evidence/worker_ram_probe_cpu_20261005'/name).read_bytes())
    (fresh/'evidence/available_ram_primary_source_review_20261005.json').write_bytes((PACKAGE/'evidence/available_ram_primary_source_review_20261005.json').read_bytes())
    test=fresh/'tests/test_short_ram_successor.py';test.write_bytes((STAGE/'test_short_ram_successor.py').read_bytes())
    ignored=fresh/'artifacts/short_ram_successor_work';assert not ignored.exists()
    spec=importlib.util.spec_from_file_location('fresh_checkout_short_tests',test)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.CODE==fresh.resolve() and module.ORIGINAL==history.resolve() and not ignored.exists()
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    assert not ignored.exists()
    print(json.dumps({'root_layout_tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'ignored_stage_absent_before_after':True,'CODE_explicit_fresh_root':True,'ORIGINAL_tracked_evidence_only':True,
        'native_model_GPU_or_decoder_execution':False}))
    if not result.wasSuccessful():raise SystemExit(1)
