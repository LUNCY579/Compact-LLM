from pathlib import Path
from types import ModuleType
import hashlib,json,os,sys
P=Path(__file__).resolve().parent.parent;W=P/'artifacts/arithmetic_token_projection_fix_work';O=Path('C:/Users/clclc/BusinessProject/synth-arithmetic-tokenmeasurement-20261009-attempt001');D=Path('C:/Users/clclc/BusinessProject/synth-arithmetic-tokenmeasurement-20261009-attempt001_projection_corrected')
assert len(sys.argv)==2;expected_source=sys.argv[1];source=W/'arithmetic_token_projection.py';raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected_source;M=ModuleType('selected_projection_helper');M.__file__=str(source);exec(compile(raw,str(source),'exec'),M.__dict__)
original_receipt_raw=(O/'RUN_RECEIPT.json').read_bytes();receipt=M.parse(original_receipt_raw);assert receipt['error'] is None and receipt['resources']['cpu_budget_satisfied'] is True
payloads={}
for name,ref in receipt['files'].items():
    data=(O/name).read_bytes();assert len(data)==ref['bytes'] and hashlib.sha256(data).hexdigest()==ref['sha256'];payloads[name]=data
metadata=M.parse(payloads['measurement.json']);policy=M.parse(payloads['policy_prompt_tokens.json']);oracle=M.parse(payloads['oracle_CPU_only.json'])
full={**metadata,'policy_rows':policy['rows'],'oracle_rows_CPU_only':oracle['rows']};assert full['sha256']==M.digest({k:v for k,v in full.items() if k!='sha256'})
assert metadata['sha256']!=M.digest({k:v for k,v in metadata.items() if k!='sha256'})
projection=M.project(full,receipt['files']['policy_prompt_tokens.json'],receipt['files']['oracle_CPU_only.json']);assert M.join(projection,payloads['policy_prompt_tokens.json'],payloads['oracle_CPU_only.json'])==full
assert not D.exists();D.mkdir()
def write(path,value):
    data=(M.canonical(value)+'\n').encode()
    with path.open('xb') as stream:
        assert stream.write(data)==len(data);stream.flush();os.fsync(stream.fileno())
    assert path.read_bytes()==data
    return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
projection_ref=write(D/'measurement_projected.json',projection)
record={'kind':'ADDITIVE_ARITHMETIC_TOKEN_METADATA_PROJECTION_CORRECTION001','projection_reference':projection_ref,'projection_helper_reference':{'path':str(source),'bytes':len(raw),'sha256':expected_source},'original_run_receipt_reference':{'path':str(O/'RUN_RECEIPT.json'),'bytes':len(original_receipt_raw),'sha256':hashlib.sha256(original_receipt_raw).hexdigest()},'original_measurement_reference':receipt['files']['measurement.json'],'original_full_result_sha256':full['sha256'],'full_result_rejoin_verified':True,'original_files_unchanged':True,'retokenization_performed':False,'model_GPU_scoring_M0_or_native_qualification':False};record['sha256']=M.digest(record);write(D/'CORRECTION_RECEIPT.json',record)
for name,data in payloads.items():assert (O/name).read_bytes()==data
assert (O/'RUN_RECEIPT.json').read_bytes()==original_receipt_raw
print(json.dumps({'output':str(D),'projection_reference':projection_ref,'rows':len(full['policy_rows']),'overflows':len(full['overflows']),'retokenized':False}))
