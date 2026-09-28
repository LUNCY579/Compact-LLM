"""Preserve existing non-weight artifacts byte-for-byte in a create-only archive."""
import gzip,hashlib,json,tarfile
from pathlib import Path
B=Path('C:/Users/clclc/BusinessProject'); O=B/'model-quality-audit-20260915'
files=[p for name in ('mq-runs','mq-data-001','mq-data-002','mq-data-003') for p in (B/name).rglob('*') if p.is_file()]
out=O/'historical-mq-artifacts.tar.gz'; inventory=[]
with out.open('xb') as raw:
    with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz:
        with tarfile.open(fileobj=gz,mode='w|') as tar:
            for p in sorted(files):
                data=p.read_bytes();rel=p.relative_to(B).as_posix()
                inventory.append({'path':rel,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
                info=tar.gettarinfo(str(p),arcname=rel);info.mtime=0;info.uid=0;info.gid=0;info.uname='';info.gname=''
                with p.open('rb') as f:tar.addfile(info,f)
with tarfile.open(out,'r:gz') as tar:
    actual={m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers()}
assert actual=={r['path']:r['sha256'] for r in inventory}
manifest={'scope':'Historical local non-weight records; no execution in this audit; no absent checkpoints recreated',
    'archive_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'archive_bytes':out.stat().st_size,'members':inventory,'byte_verification':True}
with (O/'historical-mq-archive-manifest.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
print({'files':len(files),'archive_bytes':out.stat().st_size,'verified':True})
