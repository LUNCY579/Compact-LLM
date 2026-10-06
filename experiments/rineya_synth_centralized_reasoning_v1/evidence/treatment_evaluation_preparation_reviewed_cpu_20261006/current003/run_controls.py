"""Replay authored CPU treatment controls and preserve exact source pins."""
from pathlib import Path
from types import ModuleType
import hashlib,json,time,unittest

HERE=Path(__file__).resolve().parent
FILES=('treatment_evaluation.py','test_treatment.py','run_controls.py')
def pins():return {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in FILES}
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
before=pins();start=time.monotonic()
module=ModuleType('authored_treatment_controls');module.__file__=str(HERE/'test_treatment.py')
exec(compile((HERE/'test_treatment.py').read_bytes(),module.__file__,'exec'),module.__dict__)
suite=unittest.defaultTestLoader.loadTestsFromModule(module)
result=unittest.TextTestRunner(verbosity=1).run(suite)
after=pins();cls=module.TreatmentControls
expected_probe_reads=len(cls.denied)
value={'kind':'AUTHORED_CPU_TREATMENT_PREPARATION_CONTROLS_V3','tests':result.testsRun,'success':result.wasSuccessful(),
    'failures':[str(x[0]) for x in result.failures],'errors':[str(x[0]) for x in result.errors],
    'source_before':before,'source_after':after,'source_unchanged':before==after,'seconds':time.monotonic()-start,
    'forbidden_import_attempts':cls.imports,'monitored_read_count':len(cls.reads),
    'intentional_read_probe_denials':expected_probe_reads,'expected_intentional_probe_denials':20,
    'unexpected_read_denials':expected_probe_reads!=20,'data_suppliers':'authored384/96 panel,27/540 schedule,finish,guard policy,labels,qualification,archive/publications,captures',
    'original_source_functions_executed':['verify_final_chain with explicit authored registration supplier','recipe','binding','_validate_resume','original inner raw row join','_validate_enrollment with explicit authored guard policy supplier'],
    'native_wiring':'conditional trusted packet->original restore_final->validate384->six64 collect->independent raw-byte audit; not executed',
    'fixture_only':True,'native_PASS':False,'native_launch_authorized':False,'scoring_authorized':False,'quality_gain_established':False,
    'limits':['No actual DEV/TRAIN/public finish data read, original model/tokenizer/captures read, framework/native import, model restore or scoring.',
        'Synthetic qualification receipts test byte/source/schema joins; they do not establish real scorer quality or human permission.',
        'Prior unsafe test reads remain disclosed in PRIOR_PUBLIC_READ_DISCLOSURE_001.json; this repair does not erase that exposure.']}
value['sha256']=digest(value)
target=HERE/'CPU_CONTROLS_003.json'
with target.open('x',encoding='utf-8') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps({'path':str(target),'sha256':value['sha256'],'tests':value['tests'],'success':value['success'],'source_unchanged':value['source_unchanged'],'unexpected_read_denials':value['unexpected_read_denials']}))
raise SystemExit(0 if result.wasSuccessful() and before==after and not value['unexpected_read_denials'] else 1)
