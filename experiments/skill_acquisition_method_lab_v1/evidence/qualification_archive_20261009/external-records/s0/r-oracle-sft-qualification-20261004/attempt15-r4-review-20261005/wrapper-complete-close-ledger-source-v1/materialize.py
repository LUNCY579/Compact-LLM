"""Create a concrete packet/freeze after root supplies emitted target identities.

Only fixed TARGET_ARTIFACT_INPUT.json is accepted. It carries no review/permit
status. This step is preparation; it never issues a grant or invokes a target.
"""
import hashlib,json,os,sys
import admission as a
def publish(name,value):
    owner=a.preparation_owner()
    raw=a.canonical(value,owner=owner,maximum=131072);a.require(len(raw)<=131072,'bounded create-only packet')
    def write():
        with (a.ROOT/name).open('xb') as stream:
            a.require(stream.write(raw)==len(raw),'complete create-only packet write')
            stream.flush();os.fsync(stream.fileno())
    owner.run('control',{'operation_attempts':3,'write_bytes':131072},write)
    return a.ref(a.ROOT/name,raw,owner=owner)
def source_reference(name,owner):
    def read():
        path=a.ROOT/name
        a.require(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=1048576,
          'bounded regular local execution source')
        with path.open('rb') as stream:raw=stream.read(1048577)
        a.require(0<len(raw)<=1048576,'bounded exact local source')
        return raw
    raw=owner.run('control',{'operation_attempts':3,'read_bytes':1048577,'metadata_bytes':256},read)
    return a.ref(a.ROOT/name,raw,owner=owner)
def finish_metadata(packet_ref,freeze_ref,owner):
    def finish():
        value={'schema':'CE8-PACKET-MATERIALIZATION-LOGICAL-CLOSED-1',
          'packet_ref':packet_ref,'native_source_freeze_ref':freeze_ref,
          'selected':dict(zip(a.FIELDS,owner._selected)),'used':owner.used,
          'origin':owner.origin,'deadline':owner.deadline,'pending':[],'failed':[],'refunds':0,
          'complete_final_envelope_selected_before_snapshot':True,'logical_accounting_closed':True,
          'external_original_process_exit_required':True,'physical_startup_qualified':False,
          'native_qualification':False,'models_loaded':0,'optimizer_updates':0}
        raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
        a.require(len(raw)<=32768,'bounded complete materialization accounting envelope')
        path=a.ROOT/'MATERIALIZATION_ACCOUNTING.json'
        with path.open('xb') as stream:
            a.require(stream.write(raw)==len(raw),'complete metadata accounting write')
            stream.flush();os.fsync(stream.fileno())
        owner.after('control')
        with path.open('rb') as stream:observed=stream.read(32769)
        a.require(observed==raw,'exact durable metadata accounting readback')
        return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(observed).hexdigest()}
    result=owner.run('control',{'operation_attempts':8,'read_bytes':32769,'write_bytes':32768,
      'identity_hash_attempts':1,'identity_hash_bytes':32768,'metadata_bytes':65536},finish)
    object.__setattr__(owner,'_closed',True)
    return result
def main():
    a.require(len(sys.argv)==1,'fixed metadata materializer')
    owner=a.preparation_owner()
    def preflight():
        a.require(all(not os.path.lexists(a.ROOT/name) for name in
          ('PACKET.json','NATIVE_SOURCE_FREEZE.json','MATERIALIZATION_ACCOUNTING.json')),
          'all materializer outputs must be absent before first publication; preserve partial records')
    owner.run('control',{'operation_attempts':3,'metadata_bytes':256},preflight)
    target,tref=a.fixed('TARGET_ARTIFACT_INPUT.json')
    entry,entryref=a.fixed('ENTRY_IDENTITIES_V2.json',maximum=16384)
    a.require(entry['schema']=='CE8-FIXED-LOCAL-CALLER-ENTRY-IDENTITIES-1','fixed local caller entry identities required')
    names=a.SOURCE_NAMES
    sources=owner.run('control',{'operation_attempts':1,'metadata_bytes':4096},dict)
    for name in names:
        sources[name]=source_reference(name,owner)
    construction=owner.before('control',operation_attempts=1,metadata_bytes=8192)
    p={'schema':'CE8-ACTOR-QUALIFICATION-PACKET-1','owner_thread_id':a.OWNER,'source_head':a.HEAD,
      'phase':'CE8_OWNED_NON_ML_OS_FIXTURE','bounds':a.BOUNDS,'final_tail':a.TAIL,'provider_ref':a.PROVIDER,
      'authorization_ref':a.AUTH,'job_census_ref':a.JOB_CENSUS,'attempts':1,'models_allowed':0,'optimizer_updates':0,'gpu_use_allowed':False,
      'frameworks_allowed':[],'scientific_inputs_allowed':[],'launch_ram_bytes':8589934592,'ongoing_ram_bytes':2147483648,
      'max_rss_bytes':536870912,'output_root':str(a.ACTOR_OUTPUT),'target_case':'FIXED_RETAINED_BARE_PYTHON_NORMAL_AND_TIMEOUT',
      'case_names':list(a.CASE_NAMES),'aggregate_partitions':a.partitions(),
      'target_manifest_ref':tref,'target_source_ref':target['actor_source_ref'],'runtime_ref':target['runtime_ref'],
      'source_refs':sources,'entry_identities_ref':entryref,'target_native_qualification_required_before_dispatch':False,
      'qualification_result_scope':'actual target CLR startup, fixed retained child and empty Job/handles/IO; no model root qualification',
      'admission_requirements':'exact independent source/cost review; fresh RAM/boot evidence; fresh create-only source-bound CPU60 grant'}
    construction.finish()
    a.validate_packet(p)
    pref=publish('PACKET.json',p)
    fref=publish('NATIVE_SOURCE_FREEZE.json',{'schema':'CE8-ACTOR-QUALIFICATION-SOURCE-FREEZE-1','packet_ref':pref,
      'source_refs':sources,'target_manifest_ref':tref,'native_execution':False,'models_loaded':0,'optimizer_updates':0})
    finish_metadata(pref,fref,owner)
    return 0
if __name__=='__main__':raise SystemExit(main())
