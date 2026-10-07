"""Externally pinned separate import-observation dispatcher; grant absent by default."""
from pathlib import Path
from types import ModuleType
import argparse,hashlib
HERE=Path(__file__).resolve().parent
CONTRACT_DESCRIPTOR={'bytes': 15490, 'sha256': 'c669ba3a2897bbfe200fd35807e299be8d75718b428c926edf42a7ffbbf387cf', 'role': 'python_source'}
SUPERVISOR_DESCRIPTOR={'bytes': 12047, 'sha256': '00726a65bd60f85528d7133607e6af0a5400e4006fc9eee313a1e90ed718da76', 'role': 'python_source'}
ANCHORS=('expected_config_file','expected_config_body','registration_path','expected_registration_file','expected_registration_body','runtime_path','expected_runtime_file','expected_runtime_body','expected_head')
def _load(path,descriptor):
    path=Path(path).absolute()
    for part in (path,*path.parents):
        if part.is_symlink() or getattr(part.lstat(),'st_file_attributes',0)&0x400:raise ValueError('dispatcher exact source ancestors')
    before=path.stat()
    if not 0<=before.st_size<=16*1024**2:raise ValueError('bounded dispatcher source')
    with path.open('rb') as f:raw=f.read(before.st_size+1)
    after=path.stat()
    if len(raw)!=before.st_size or (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino) or descriptor!={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}:raise ValueError('externally pinned stable dispatcher source')
    m=ModuleType('selected_import_dispatch_'+path.stem);m.__file__=str(path);exec(compile(raw,str(path),'exec'),m.__dict__);return m

def dispatch(config,output=None,*,grant_reference=None,owner_receipt_reference=None,anchors):
    C=_load(HERE/'import_observation_contract.py',CONTRACT_DESCRIPTOR)
    C.enroll_native_gate(grant_reference,owner_receipt_reference,{'config_path':str(Path(config).absolute()),'anchors':anchors})
    SUP=_load(HERE/'import_observation_supervisor.py',SUPERVISOR_DESCRIPTOR)
    return SUP.supervise_registered(config,output,contract=C,grant_reference=grant_reference,owner_receipt_reference=owner_receipt_reference,**anchors)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output-dir',type=Path)
    for key in ANCHORS:parser.add_argument('--'+key.replace('_','-'),required=True)
    for prefix in ('grant','owner-receipt'):
        parser.add_argument('--'+prefix+'-path',type=Path,required=True);parser.add_argument('--'+prefix+'-file-sha256',required=True);parser.add_argument('--'+prefix+'-body-sha256',required=True)
    args=parser.parse_args();ref=lambda key:{'path':str(getattr(args,key+'_path').absolute()),'file_sha256':getattr(args,key+'_file_sha256'),'body_sha256':getattr(args,key+'_body_sha256')}
    return dispatch(args.config,args.output_dir,grant_reference=ref('grant'),owner_receipt_reference=ref('owner_receipt'),anchors={k:getattr(args,k) for k in ANCHORS})
if __name__=='__main__':raise SystemExit(main())
