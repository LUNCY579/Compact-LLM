"""Temporary fresh repository layout, actual copied code, no ignored-stage dependency."""
from pathlib import Path
import subprocess,sys,tempfile,json
STAGE=Path(__file__).resolve().parent;PACKAGE=STAGE.parents[1];ROOT=PACKAGE.parents[1]
with tempfile.TemporaryDirectory(dir=STAGE) as directory:
    repo=Path(directory)/'CPU_FRESH_REPO';fresh=repo/'experiments/rineya_synth_centralized_reasoning_v1'
    (fresh/'tests').mkdir(parents=True)
    for path in PACKAGE.glob('*.py'):(fresh/path.name).write_bytes(path.read_bytes())
    for name in ('short_phase_contract.py','short_phase_observation.py','short_phase_worker.py','short_phase_supervisor.py'):
        (fresh/name).write_bytes((STAGE/name).read_bytes())
    for name in ('test_inference_component.py','test_inference_qualification.py','test_short_collection.py'):
        (fresh/'tests'/name).write_bytes((PACKAGE/'tests'/name).read_bytes())
    (fresh/'tests/test_short_phase_collection.py').write_bytes((STAGE/'test_short_phase_collection.py').read_bytes())
    paths=[PACKAGE/name for name in ('PILOT_DECISIONS.json','pilot_trainable_inventory.json','NATIVE_QUALIFICATION_PROPOSAL.json','AGENTS.md',
        'configs/SYNTH_NQ001_PROC7.json','evidence/context_phase_no_model_D001_attempt1_independent_audit_20261005.json',
        'evidence/available_ram_primary_source_review_20261005.json')]
    paths+=list((PACKAGE/'evidence/worker_ram_probe_cpu_20261005').glob('*.json'))
    paths+=[PACKAGE/'evidence/worker_ram_probe_cpu_20261005/worker_ram_probe.py']
    for path in paths:
        target=fresh/path.relative_to(PACKAGE);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    for name in ('AGENTS.md','RINEYA_SYNTH_CENTRALIZED_REASONING.md'):(repo/name).write_bytes((ROOT/name).read_bytes())
    metadata=PACKAGE.parent/'post_training_mimo_centralized_v1/receipts/titan_static_v1/PRIMARY_STUDENT_RECEIPT.json'
    target=fresh.parent/'post_training_mimo_centralized_v1/receipts/titan_static_v1/PRIMARY_STUDENT_RECEIPT.json'
    target.parent.mkdir(parents=True);target.write_bytes(metadata.read_bytes())
    ignored=fresh/'artifacts/short_phase_collection_work';assert not ignored.exists()
    script="""from pathlib import Path
import importlib.util,sys,unittest,json
test=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('fresh_short_phase_controls',test)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert not m.STAGED and m.CODE==test.parent.parent and m.PACKAGE==test.parent.parent
for target in (m.contract,m.obs,m.worker,m.supervisor):assert Path(target.__file__).resolve().parent==m.CODE
assert Path(m.worker.original.__file__).resolve().parent==m.PACKAGE
result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(m))
assert not any(n in sys.modules for n in ('torch','transformers','tokenizers'))
print(json.dumps({'freshroot_tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'actual_freshroot_code_targets':True}))
raise SystemExit(0 if result.wasSuccessful() else 1)
"""
    result=subprocess.run([sys.executable,'-I','-B','-c',script,str(fresh/'tests/test_short_phase_collection.py')],capture_output=True,text=True,timeout=180)
    assert not ignored.exists()
    print(result.stdout,end='');sys.stderr.write(result.stderr)
    print(json.dumps({'ignored_stage_absent_before_after':True,'copied_metadata_only_no_model_runtime_binary_or_encoder_reads':True}))
    raise SystemExit(result.returncode)
