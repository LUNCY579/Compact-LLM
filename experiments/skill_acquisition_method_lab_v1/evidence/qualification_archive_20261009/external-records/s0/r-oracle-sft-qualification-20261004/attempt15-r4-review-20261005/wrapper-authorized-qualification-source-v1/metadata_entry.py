"""Fixed metadata producer entry; no grant creation or target dispatch.

The existing local caller, this entry, verified bootstrap and exact accounting
component are explicit trusted layer 0. One preparation owner precedes all
admission/materializer execution. Its startup is not a Native result.
"""
import hashlib,json,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    if len(sys.argv)!=1:raise RuntimeError('fixed metadata producer accepts no scope arguments')
    with (ROOT/'ENTRY_IDENTITIES_V2.json').open('rb') as stream:raw=stream.read(16385)
    if not 0<len(raw)<=16384:raise RuntimeError('bounded fixed entry identities')
    identities=json.loads(raw)
    item=identities['entry_ref'];path=ROOT/'bootstrap_entry.py'
    if (type(item) is not dict or set(item)!={'path','bytes','sha256'} or item['path']!=str(path)
      or type(item['bytes']) is not int or not 0<item['bytes']<=1048576):raise RuntimeError('exact fixed bootstrap source reference')
    with path.open('rb') as stream:code=stream.read(item['bytes']+1)
    if len(code)!=item['bytes'] or hashlib.sha256(code).hexdigest()!=item['sha256']:
        raise RuntimeError('metadata bootstrap source changed')
    value=types.ModuleType('ce8_trusted_metadata_bootstrap');value.__file__=str(path)
    exec(compile(code,str(path),'exec'),value.__dict__)
    admission=value.prepare_entry();owner=admission.preparation_owner()
    try:
        owner.register_metadata_window()
        # Exact materializer code is selected, read, hashed and compiled under the
        # same owner. No packet/grant exists or is required before this producer.
        from_source=owner.before('control',operation_attempts=3,metadata_bytes=256)
        materializer=ROOT/'materialize.py'
        size=materializer.stat().st_size
        admission.require(materializer.is_file() and not materializer.is_symlink() and 0<size<=1048576,
          'bounded regular exact metadata producer')
        from_source.finish()
        def read_producer():
            with materializer.open('rb') as stream:return stream.read(size+1)
        blob=owner.run('control',{'operation_attempts':1,'read_bytes':size+1},read_producer)
        admission.require(len(blob)==size,'metadata producer source extent changed')
        item={'path':str(materializer),'bytes':len(blob),'sha256':owner.sha(blob)}
        producer=admission.module(item,'ce8_prepared_metadata_producer',owner=owner)
        return producer.main()
    except BaseException as error:
        if not owner._closed and not owner._root_handoff:owner.abort_owned_entry(error)
        raise

if __name__=='__main__':raise SystemExit(main())
