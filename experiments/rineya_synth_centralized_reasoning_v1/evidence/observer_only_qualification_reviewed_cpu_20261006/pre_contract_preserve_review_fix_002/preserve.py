"""Observer-only exact byte/source/lease preservation after terminal death."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,os,time

def preserve(output,target,*,contract,context,revalidate,fresh_death):
    value=contract.context_value(context);original=contract.canonical(value)
    contract.require(revalidate()==value,'original observer context before preservation')
    output=Path(output).absolute();target=Path(target).absolute()
    contract.require(output!=target and not target.exists() and output.parent==contract.PACKAGE/'artifacts','fresh separate observer original/preserved leaves')
    sup_raw=contract.read_bytes(output/'SUPERVISOR_RECEIPT.json');sup=contract.parse(sup_raw)
    contract.require(value['config']['scope']==contract.SCOPE and ('scope' not in sup or sup['scope']==contract.SCOPE),'observer-only scope retained')
    lease=Path(sup['lease_path']);lease_names=('reservation.json','worker.json','resume.json','terminal.json')
    contract.require(lease.parent==contract.PACKAGE/'artifacts/_native_gpu_leases' and set(p.name for p in lease.iterdir())==set(lease_names),'exact original terminal lease population')
    contract.require(contract.read_bytes(lease/'terminal.json')==sup_raw,'lease terminal byte-identical SUP')
    births={}
    for born in sup['owned_process_identities']:
        birth={k:born[k] for k in ('pid','creation_filetime_100ns')}
        contract.require(all(type(v) is int and v>0 for v in birth.values()),'typed raw owned birth before preservation')
        births[(birth['pid'],birth['creation_filetime_100ns'])]=birth
    contract.require(births and all(fresh_death(dict(b)) is False for b in births.values()),'fresh externally bound every owned birth dead')
    names=sorted(p.name for p in output.iterdir());contract.require(len(names)<=4096 and all((output/n).is_file() for n in names),'flat complete observer raw roster')
    before={n:contract.read_bytes(output/n) for n in names};lease_before={n:contract.read_bytes(lease/n) for n in lease_names}
    contract.require(sum(len(v) for v in before.values())<=128*1024**2,'bounded observer raw byte population')
    partial={Path(d['path']).name:{k:d[k] for k in ('bytes','sha256')} for d in sup['partial_outputs']}
    contract.require(partial=={n:{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()} for n,raw in before.items() if n!='SUPERVISOR_RECEIPT.json'},'exact original SUP partial denominator')
    target.mkdir();(target/'source_snapshots').mkdir()
    def copy(path,raw):
        with path.open('xb') as f:contract.require(f.write(raw)==len(raw),'complete exclusive observer copy');f.flush();os.fsync(f.fileno())
        contract.require(contract.read_bytes(path)==raw,'exact observer preservation readback')
        return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    raw_files={n:copy(target/n,raw) for n,raw in before.items()}
    lease_files={n:{'source_path':str(lease/n),'path':'lease_'+n,**copy(target/('lease_'+n),raw)} for n,raw in lease_before.items()}
    sources=value['config']['source_files'];snapshots={}
    for path,descriptor in sources.items():
        contract.file_descriptor(path,descriptor);raw=contract.read_bytes(path);name=hashlib.sha256(path.encode('utf-8')).hexdigest()+Path(path).suffix
        snapshots[path]={'path':'source_snapshots/'+name,**copy(target/'source_snapshots'/name,raw),'role':descriptor['role']}
    for n,raw in before.items():contract.require(contract.read_bytes(output/n)==contract.read_bytes(target/n)==raw,'final original/copy byte conservation')
    for n,raw in lease_before.items():contract.require(contract.read_bytes(lease/n)==contract.read_bytes(target/('lease_'+n))==raw,'final original/copy lease conservation')
    for path,descriptor in sources.items():contract.file_descriptor(path,descriptor);contract.require(contract.read_bytes(path)==contract.read_bytes(target/snapshots[path]['path']),'final source/snapshot conservation')
    contract.require(sorted(p.name for p in output.iterdir())==names and set(p.name for p in target.iterdir())==set(names)|{'source_snapshots'}|{'lease_'+n for n in lease_names},'final exact whole observer rosters')
    contract.require(contract.canonical(contract.context_value(context))==original and revalidate()==value,'same original observer context after preservation')
    death={'observed_monotonic':time.monotonic(),'observed_utc':datetime.now(timezone.utc).isoformat(),'supervisor_file_sha256':hashlib.sha256(sup_raw).hexdigest(),'config_file_sha256':value['anchors']['expected_config_file'],'observations':[{'identity':b,'currently_alive':False} for b in births.values()],'death_verifier_source_sha256':value['config']['source_files'][str(contract.PACKAGE/'native_supervisor.py')]['sha256']}
    receipt={'kind':'PRESERVED_OBSERVER_ONLY_QUALIFICATION_V1','scope':contract.SCOPE,'manifest_sha256':value['config']['sha256'],'raw_files':raw_files,'source_files':sources,'source_snapshot_files':snapshots,'lease_files':lease_files,'current_death_observation':death,'fixture_only':value['fixture_only'],'native_authority_issued':False}
    receipt=contract.seal(receipt);copy(target/'PRESERVATION.json',(contract.canonical(receipt)+'\n').encode('ascii'))
    return receipt
