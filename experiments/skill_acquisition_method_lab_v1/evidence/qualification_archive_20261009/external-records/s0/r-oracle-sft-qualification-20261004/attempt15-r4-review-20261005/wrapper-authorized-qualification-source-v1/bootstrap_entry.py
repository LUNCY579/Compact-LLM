"""Fixed same-process CPU component entry with an explicit trusted caller boundary.

Layer 0 is the existing local Python caller, stdlib, this entry and exact
accounting component. Its startup is unqualified and is not claimed free.
One retained preparation owner exists before admission or later source code
executes. This entry is not a grant and does not qualify the model wrapper.
"""
import hashlib,json,os,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parent
IDENTITIES=ROOT/'ENTRY_IDENTITIES_V2.json'
def require(value,message):
    if value is not True:raise RuntimeError(message)
def exact_ref(item,name):
    require(type(item) is dict and set(item)=={'path','bytes','sha256'}
      and item['path']==str(ROOT/name) and type(item['bytes']) is int and 0<item['bytes']<=1048576
      and type(item['sha256']) is str and len(item['sha256'])==64
      and not set(item['sha256'])-set('0123456789abcdef'),'exact fixed local startup source reference')
def read_exact(item):
    path=Path(item['path'])
    require(path.resolve()==path and all(not part.is_symlink() and not(getattr(part.stat(),'st_file_attributes',0)&1024)
      for part in (path,*path.parents)) and path.stat().st_size==item['bytes'],'regular exact entry source')
    with path.open('rb') as stream:raw=stream.read(item['bytes']+1)
    require(len(raw)==item['bytes'],'exact entry source extent');return raw
def install(name,item,raw):
    require(name not in sys.modules,'entry refuses an existing module alias')
    value=types.ModuleType(name);value.__file__=item['path']
    value.__dict__['__CE8_EXECUTED_SOURCE_SHA256']=item['sha256']
    sys.modules[name]=value
    try:exec(compile(raw,item['path'],'exec'),value.__dict__)
    except BaseException:
        # Preserve the partially executed module; no alias retry/revival.
        value.__dict__['__CE8_ENTRY_LOAD_FAILED']=True
        raise
    return value
def prepare_entry():
    # Fixed metadata entry code is authenticated in the explicitly trusted,
    # unqualified pre-owner layer0. Retain actual code/global identity before
    # any owned PREP dependent registration/read/hash/compile can fail.
    caller=sys._getframe(1);metadata_binding=None
    metadata_path=ROOT/'metadata_entry.py'
    if caller.f_code.co_filename==str(metadata_path):
        require(caller.f_globals.get('__file__')==str(metadata_path)
          and caller.f_code is caller.f_globals['main'].__code__
          and type(sys.modules.get(caller.f_globals.get('__name__'))) is types.ModuleType
          and sys.modules[caller.f_globals['__name__']].__dict__ is caller.f_globals,
          'actual fixed metadata module/code/globals caller')
        with metadata_path.open('rb') as stream:metadata_raw=stream.read(1048577)
        require(len(metadata_raw)<=1048576,'bounded trusted layer0 metadata source')
        compiled=compile(metadata_raw,str(metadata_path),'exec')
        matches=tuple(item for item in compiled.co_consts if type(item) is types.CodeType and item.co_name=='main')
        require(len(matches)==1 and matches[0]==caller.f_code,'fixed actual metadata entry source code')
        metadata_binding=(caller.f_code,caller.f_globals,hashlib.sha256(metadata_raw).hexdigest())
    require(len(sys.argv)==1,'fixed CPU component entry accepts no scope arguments')
    require('accounting' not in sys.modules and 'admission' not in sys.modules,'cold source-owned entry aliases')
    # These exact local inputs are part of the named trusted layer-0 boundary.
    # Their physical startup cost is not a target/component Native result.
    with IDENTITIES.open('rb') as stream:identity_raw=stream.read(16385)
    require(0<len(identity_raw)<=16384,'bounded fixed entry identities')
    def unique(rows):
        result={}
        for key,value in rows:require(key not in result,'duplicate entry identity key');result[key]=value
        return result
    identities=json.loads(identity_raw,object_pairs_hook=unique)
    require(type(identities) is dict and set(identities)=={'schema','accounting_ref','admission_ref','entry_ref'}
      and identities['schema']=='CE8-FIXED-LOCAL-CALLER-ENTRY-IDENTITIES-1','exact entry identity schema')
    for role,name in (('accounting_ref','accounting.py'),('admission_ref','admission.py'),('entry_ref','bootstrap_entry.py')):
        exact_ref(identities[role],name)
    entry_raw=read_exact(identities['entry_ref'])
    require(hashlib.sha256(entry_raw).hexdigest()==identities['entry_ref']['sha256'],'current fixed entry identity')
    accounting_raw=read_exact(identities['accounting_ref'])
    require(hashlib.sha256(accounting_raw).hexdigest()==identities['accounting_ref']['sha256'],'trusted exact accounting component identity')
    accounting=install('accounting',identities['accounting_ref'],accounting_raw)
    owner=accounting.begin_entry_preparation(str(ROOT/'bootstrap_entry.py'),identities['entry_ref']['sha256'])
    if metadata_binding is not None:
        object.__setattr__(owner,'_failure_window_codes',owner._failure_window_codes+((metadata_binding[0],metadata_binding[1]),))
    try:
        # Every subsequent source load is selected before its work, with no new
        # preparation allocation and no change to the later CPU60 origin.
        item=identities['admission_ref']
        raw=owner.run('control',{'operation_attempts':1,'read_bytes':item['bytes']+1},lambda:read_exact(item))
        require(owner.sha(raw)==item['sha256'],'paid exact admission SHA')
        admission=owner.run('control',{'operation_attempts':32,'metadata_bytes':4096},lambda:install('admission',item,raw))
        require(admission.preparation_owner() is owner,'same original entry owner adopted by identity')
        provenance=owner.before('control',operation_attempts=1,metadata_bytes=512)
        admission.__dict__['_ENTRY_IDENTITIES']=identities
        admission.__dict__['_ENTRY_IDENTITIES_REF']={'path':str(IDENTITIES),'bytes':len(identity_raw),'sha256':owner.sha(identity_raw)}
        provenance.finish()
        return admission
    except BaseException as error:
        owner.abort_owned_entry(error)
        raise

def main():
    owner=None
    try:
        admission=prepare_entry();owner=admission.preparation_owner()
        identities=admission._ENTRY_IDENTITIES
        packet,_=admission.fixed('PACKET.json',owner=owner)
        admission.validate_packet(packet)
        require(packet['source_refs']['accounting.py']==identities['accounting_ref']
          and packet['source_refs']['admission.py']==identities['admission_ref']
          and packet['source_refs']['bootstrap_entry.py']==identities['entry_ref']
          and packet['entry_identities_ref']==admission._ENTRY_IDENTITIES_REF,'same exact independently reviewed source entry binding')
        runner=admission.module(packet['source_refs']['run_qualification.py'],'ce8_prepared_fixed_entry_runner',owner=owner)
        return runner.main()
    except BaseException as error:
        if owner is not None and not owner._closed and not owner._root_handoff:owner.abort_owned_entry(error)
        raise

if __name__=='__main__':raise SystemExit(main())
