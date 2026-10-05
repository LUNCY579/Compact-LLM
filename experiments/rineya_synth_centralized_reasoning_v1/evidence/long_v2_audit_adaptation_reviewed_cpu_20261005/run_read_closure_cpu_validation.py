"""Write-once staged CPU review capture. No native/device or asset probes."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
HERE=Path(__file__).resolve().parent
NAMES=("long_native_auditor_v2.py","long_terminal_driver_v2.py","test_long_v2_audit_adaptation.py","fixtures/V2_DECLARATION_METADATA_ONLY.json")
def hashes():return {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in NAMES}
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
def main():
    target=HERE/"READ_CLOSURE_CPU_VALIDATION_001.json"
    if target.exists():raise ValueError("preserve prior validation; receipt is write-once")
    before=hashes();started=time.monotonic()
    script="""import pathlib,sys,unittest,json
p=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(p))
s=unittest.defaultTestLoader.discover(str(p),pattern='test_long_v2_audit_adaptation.py')
r=unittest.TextTestRunner(verbosity=2).run(s)
forbidden=sorted(n for n in sys.modules if n.split('.',1)[0] in {'torch','transformers','tensorflow','jax','flax','tokenizers','jinja2','markupsafe'})
print(json.dumps({'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped),'forbidden_modules_loaded':forbidden}))
sys.exit(0 if r.wasSuccessful() and not forbidden else 1)
"""
    argv=[sys.executable,"-I","-B","-c",script,str(HERE)]
    completed=subprocess.run(argv,capture_output=True,text=True,timeout=300)
    after=hashes();receipt={"schema_version":1,"status":"PASS_STAGED_FIRST_READ_COMPLETE_IMMUTABLE_CLOSURE_CPU30" if completed.returncode==0 and before==after else "FAIL_CPU_OR_SOURCE_DRIFT",
        "source_before":before,"source_after":after,"source_unchanged":before==after,"returncode":completed.returncode,
        "elapsed_seconds":time.monotonic()-started,"argv":argv,"stdout":completed.stdout,"stderr":completed.stderr,
        "native_model_encoder_framework_GPU_execution":False,"native_runtime_binary_or_model_asset_reads":False,
        "OS_process_or_resource_probes":False,"native_AUDIT_KIND_PASS_issued":False,"three_arm_learning_goal_complete":False}
    receipt["sha256"]=hashlib.sha256(canonical(receipt).encode()).hexdigest()
    with target.open("xb") as stream:stream.write((canonical(receipt)+"\n").encode())
    print(canonical({key:receipt[key] for key in ("status","returncode","elapsed_seconds","source_unchanged","sha256")}))
    return 0 if receipt["status"].startswith("PASS") else 1
if __name__=="__main__":sys.exit(main())
