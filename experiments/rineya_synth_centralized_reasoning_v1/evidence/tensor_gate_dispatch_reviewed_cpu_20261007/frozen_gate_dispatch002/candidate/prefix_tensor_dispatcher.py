"""Exact owned two-tensor dispatcher candidate; no default grant or native run."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
HERE=Path(__file__).resolve().parent
CONTRACT_DESCRIPTOR={'bytes': 85254, 'sha256': '80a4650ab7c4b386f560de7c18f08d5c8d317f948fc0212afdf5cb5be4d9a577', 'role': 'python_source'}
PIPELINE_DESCRIPTOR={'bytes': 21637, 'sha256': 'dbc3e7130e219106b4ad53ac86e28a62fd7763c78e4ed502d103eae943d2f734', 'role': 'python_source'}
ANCHORS=('expected_config_file','expected_config_body','registration_path','expected_registration_file','expected_registration_body','runtime_path','expected_runtime_file','expected_runtime_body','expected_head')
def _load(path,descriptor,contract=None):
    path=Path(path).absolute()
    if path.resolve()!=path or path.is_symlink() or type(descriptor) is not dict:raise ValueError('externally reviewed dispatcher source descriptor')
    for part in (path,*path.parents):
        if part.is_symlink() or getattr(part.lstat(),'st_file_attributes',0)&0x400:raise ValueError('dispatcher source ancestor reparse refused')
    before=path.stat()
    if not 0<=before.st_size<=16*1024**2:raise ValueError('bounded dispatcher source')
    with path.open('rb') as f:raw=f.read(before.st_size+1)
    after=path.stat()
    if len(raw)!=before.st_size or (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino) or descriptor!={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}:raise ValueError('exact stable selected dispatcher module')
    spec=importlib.util.spec_from_file_location('owned_dispatch_'+path.stem,path);module=importlib.util.module_from_spec(spec)
    if contract is not None:module.__dict__['__owned_contract__']=contract
    exec(compile(raw,str(path),'exec'),module.__dict__);return module

def dispatch(config_path,output_dir,*,grant_reference=None,owner_receipt_reference=None,anchors):
    contract=_load(HERE/'prefix_tensor_contract.py',CONTRACT_DESCRIPTOR)
    # This enrolls source/owner evidence only. The actual full amended manifest
    # and resource/SUP checks remain in supervise_registered afterward.
    contract.enroll_native_gate(grant_reference,owner_receipt_reference,{'config_path':str(Path(config_path).absolute()),'anchors':anchors})
    pipeline=_load(HERE/'prefix_native_pipeline.py',PIPELINE_DESCRIPTOR,contract)
    return pipeline.supervise_registered(config_path,output_dir,grant_reference=grant_reference,owner_receipt_reference=owner_receipt_reference,**anchors)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output-dir',type=Path)
    for key in ANCHORS:parser.add_argument('--'+key.replace('_','-'),required=True)
    for prefix in ('grant','owner-receipt'):
        parser.add_argument('--'+prefix+'-path',type=Path,required=True);parser.add_argument('--'+prefix+'-file-sha256',required=True);parser.add_argument('--'+prefix+'-body-sha256',required=True)
    args=parser.parse_args();reference=lambda prefix:{'path':str(getattr(args,prefix+'_path').absolute()),'file_sha256':getattr(args,prefix+'_file_sha256'),'body_sha256':getattr(args,prefix+'_body_sha256')}
    return dispatch(args.config,args.output_dir,grant_reference=reference('grant'),owner_receipt_reference=reference('owner_receipt'),anchors={k:getattr(args,k) for k in ANCHORS})
if __name__=='__main__':raise SystemExit(main())
