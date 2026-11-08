"""Lossless filesystem compression of owned completed artifacts, verified by hash.

Never removes, rewrites, relocates or normalizes logical file contents. Windows
COMPACT changes transparent filesystem storage. Every source must already match
its completed receipt; failures are preserved and stop the process.
"""
from pathlib import Path
from argparse import ArgumentParser
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

BASE=Path('C:/Users/clclc/BusinessProject')
OLD=BASE/'rineya-dataset-retention-records/doseint001-20260919'
OUT=BASE/'rineya-dataset-retention-records/highdose-lora002-20260920/storage'
PILOT=BASE/'ChatBubble-retention-pilot-20260917'

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write(p,value):
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--batch',choices=('initial','recovery'),default='initial')
    args=parser.parse_args()
    global OUT
    if args.batch=='recovery':
        OUT=OUT.with_name('storage-002')
    if os.name!='nt':
        raise SystemExit('This storage operation requires Windows NTFS COMPACT')
    if OUT.exists():
        raise SystemExit('Create-only audit directory already exists')
    sys.path.insert(0,str(PILOT))
    from experiments.model_quality_gpu_v1.resources import lease
    from experiments.model_quality_gpu_v1.integrity import unseal
    import tempfile
    OUT.mkdir(parents=True)
    targets=[]
    if args.batch=='initial':
        for arm in ('a100','a025','a050','a075'):
            folder=OLD/'arms'/arm
            receipt=folder/'ARM.json'
            targets.append((folder/'model.safetensors',read(receipt)['arm_weight_sha256'],receipt))
        checkpoints=[(OLD/'journal-dual899-preloadretry/runs/train-001/microfit/checkpoint-00002',2)]
    else:
        checkpoints=[(OLD/'journal-dual899-recovery/runs/train-001/microfit'/f'checkpoint-{step:05d}',step) for step in (2,193)]
    for checkpoint,step in checkpoints:
        manifest_path=checkpoint/'manifest.json'
        manifest=unseal(read(manifest_path))
        assert manifest['status']=='COMPLETE' and manifest['progress']['updates']==step
        targets.extend((checkpoint/entry['file'],entry['sha256'],manifest_path) for entry in manifest['entries'])
    start=time.time()
    before_free=shutil.disk_usage(OLD).free
    write(OUT/'PLAN.json',{'what':'transparent NTFS file compression, logical bytes unchanged',
        'command':'compact /C /EXE:LZX /Q <verified-file>', 'old_root':str(OLD),
        'target_free_bytes':int(19.5*2**30),'free_before':before_free,
        'targets':[{'path':str(p),'sha256':h,'receipt':str(r),'receipt_sha256':sha(r)} for p,h,r in targets]})
    records=[]
    try:
        with lease(Path(tempfile.gettempdir())/'rineya-gpu-0.lock'):
            for index,(path,expected,receipt) in enumerate(targets):
                if shutil.disk_usage(OLD).free>=19.5*2**30:
                    break
                path=path.resolve()
                assert path.is_relative_to(OLD.resolve()) and path.is_file() and not path.is_symlink()
                actual=sha(path)
                if actual!=expected:
                    raise ValueError(f'Pre-compression receipt mismatch; file untouched: {path}')
                length=path.stat().st_size
                if index%25==0:
                    print(f'COMPRESS {index+1}/{len(targets)} {path.parent.name}/{path.name}',flush=True)
                began=time.monotonic()
                with (OUT/f'{index:04d}.compact.log').open('xb') as log:
                    code=subprocess.run(['compact.exe','/C','/EXE:LZX','/Q',str(path)],
                        stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW).returncode
                after=sha(path)
                record={'path':str(path),'before_sha256':actual,'after_sha256':after,'expected_sha256':expected,
                        'logical_bytes_before':length,'logical_bytes_after':path.stat().st_size,
                        'returncode':code,'wall_seconds':time.monotonic()-began,
                        'free_after':shutil.disk_usage(OLD).free,'receipt_sha256':sha(receipt)}
                write(OUT/f'{index:04d}.json',record)
                records.append(record)
                if code!=0 or after!=expected or path.stat().st_size!=length:
                    raise RuntimeError('Compression did not verify; preserve files and audit')
            free=shutil.disk_usage(OLD).free
            write(OUT/'RESULT.json',{'status':'HASH_VERIFIED_SPACE_READY' if free>=19.5*2**30 else 'HASH_VERIFIED_SPACE_INSUFFICIENT',
                'files_verified':len(records),'logical_bytes_preserved':sum(r['logical_bytes_before'] for r in records),
                'free_before':before_free,'free_after':free,'wall_seconds':time.time()-start,
                'files_removed':0,'model_execution':False,'quality_claim':False})
            print(json.dumps({'files_verified':len(records),'free_GiB':free/2**30}),flush=True)
            return 0 if free>=19.5*2**30 else 2
    except Exception as exc:
        write(OUT/'FAILURE.json',{'type':type(exc).__name__,'message':str(exc),'completed':len(records),'files_removed':0})
        raise

if __name__=='__main__':
    raise SystemExit(main())
