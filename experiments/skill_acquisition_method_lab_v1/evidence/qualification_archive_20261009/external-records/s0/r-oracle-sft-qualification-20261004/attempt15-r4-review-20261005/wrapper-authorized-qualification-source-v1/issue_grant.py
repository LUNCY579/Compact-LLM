"""Bounded pre-grant preparation and one direct original owner handoff.

No target-native pass is an input. This release proposal requires exact direct
human authorization, independent source review, and all original resource gates.
"""
import os,sys,time
import admission as a
import accounting as cost

def write_fixed(name,value,owner,maximum=131072):
    blob=a.canonical(value,owner=owner,maximum=maximum);path=a.ROOT/name
    def write():
        with path.open('xb') as stream:stream.write(blob);stream.flush();os.fsync(stream.fileno())
    owner.run('control',{'operation_attempts':3,'write_bytes':maximum},write)
    return a.ref(path,blob,owner=owner)

def prepare_and_issue():
    # Fail closed before packet/review/auth reads unless this is the original
    # authenticated entry and live PREP owner. Exact authorization is checked
    # by validate_packet; exact review_binding precedes custody and any grant.
    a.require(a._ENTRY_OWNER_BOUND and cost.entry_preparation_owner() is a.preparation_owner()
      and not a.preparation_owner()._closed and not a.preparation_owner()._root_handoff
      and not a.preparation_owner().fatal,'live exact same-entry original PREP owner required')
    a.require(len(sys.argv)==1,'fixed direct original-owner entry')
    a.require(a._ENTRY_OWNER_BOUND and cost.entry_preparation_owner() is a.preparation_owner()
      and type(getattr(a,'_ENTRY_IDENTITIES',None)) is dict and type(getattr(a,'_ENTRY_IDENTITIES_REF',None)) is dict,'exact same original bootstrap preparation owner required')
    prep=a.preparation_owner()
    p,pref=a.fixed('PACKET.json',owner=prep);target=a.validate_packet(p)
    freeze,fref=a.fixed('NATIVE_SOURCE_FREEZE.json',owner=prep);review,rref=a.fixed('INDEPENDENT_REVIEW_DECISION.json',owner=prep)
    a.require(a.review_binding(p,pref,freeze,fref,review) is True,'exact independent source/all7/PREP review required before any custody or grant')
    a.require('runtime_custody' not in sys.modules and 'ce8_prepared_runtime_custody' not in sys.modules,
      'runtime custody code must use paid exact preparation loader')
    custody_module=a.module(p['source_refs']['runtime_custody.py'],'ce8_prepared_runtime_custody',owner=prep)
    a.require('failure_envelope' not in sys.modules and 'ce8_prepared_failure_envelope' not in sys.modules,'failure exporter must use paid exact PREP source')
    failure_module=a.module(p['source_refs']['failure_envelope.py'],'ce8_prepared_failure_envelope',owner=prep)
    a.require(bool(target['runtime_dependency_refs']) and bool(target['runtime_namespace_roots']),'complete reviewed private runtime inputs required')
    ready,readyref=a.fixed('OWNER_DISPATCH_READINESS.json',owner=prep)
    a.require(ready['owner_thread_id']==a.OWNER and ready['kind']=='READ_ONLY_ORIGINAL_OWNER_RESOURCE_EVIDENCE'
      and 0<=time.time()-ready['observed_unix_seconds']<=120 and ready['available_ram_bytes']>=8589934592,
      'fresh original actual RAM/boot evidence')
    a.require(not os.path.lexists(a.ROOT/'FIXTURE_GRANT.json') and not os.path.lexists(a.OUTPUT),'fresh create-only source grant/output')
    sampler=prep.run('control',{'operation_attempts':64,'metadata_bytes':4096},cost.NativeResourceSampler)
    prep.attach_resources(sampler)
    launch_sample=prep.run('control',{'operation_attempts':3,'metadata_bytes':256},sampler)
    a.require(launch_sample['available_ram_bytes']>=8589934592,'actual source preparation8GiB launch floor')
    # Source modules are raw-paid and loaded during bounded preparation. The
    # same original objects/custody are handed off; no post-origin lazy loader.
    provider=a.module(a.PROVIDER,'ce8_prepared_parent_provider',owner=prep)
    census=a.module(a.JOB_CENSUS,'ce8_prepared_parent_census',owner=prep)
    backend=a.module(p['source_refs']['backend.py'],'ce8_prepared_parent_backend',owner=prep)
    custody=None;owner=None
    try:
        held_refs=prep.run('control',{'operation_attempts':1,'metadata_bytes':512},
          lambda:custody_module.preparation_refs(p,target)+[p['entry_identities_ref']])
        a.require(len(held_refs)<=224,'entry manifest retained inside original224pin subset')
        custody=custody_module.RuntimeCustody.prepare(held_refs,owner=prep,namespaces=target['runtime_namespace_roots'])
        custody.require_live(before_info=lambda:prep.before('control',operation_attempts=1),
          before_namespace=lambda namespace:prep.before('control',operation_attempts=1,metadata_bytes=16384))
        a.require(0<=time.time()-ready['observed_unix_seconds']<=120,'resource evidence fresh after bounded preparation')
        static=a.PreparedStaticViews.create(p,pref,target,custody,(provider,backend,census),prep)
        # Preselect the COMPLETE preliminary publication before construction,
        # accounting capture, JSON, write/fsync and source reference SHA.
        prep_publication=prep.before('control',operation_attempts=8,metadata_bytes=132096,
          write_bytes=131072,identity_hash_attempts=1,identity_hash_bytes=131072)
        prep_record={'schema':'CE8-BOUNDED-ACTOR-PREPARATION-7','owner_thread_id':a.OWNER,'packet_ref':pref,
          'target_manifest_ref':p['target_manifest_ref'],'owner_accounting':prep.receipt(),'read_pin_preparation':custody._preparation,
          'phase':'PRE_GRANT_SEPARATELY_BOUNDED_REVIEWED_PREPARATION','work_claimed_free':False,
          'preparation_closed':False,'publication_selected_before_snapshot':True,
          'pending_budget_constructor_preparation':{'operation_attempts':128,'metadata_bytes':4096},
          'final_preparation_accounting_location':'FRESH_SOURCE_BOUND_FIXTURE_GRANT',
          'metadata_CPU_heap_measurement':None,'models_loaded':0,'optimizer_updates':0}
        a.require(cost.encoded_extent(prep_record)+1<=131072,'selected preliminary PREP record extent')
        prep_raw=a.json.dumps(prep_record,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()+b'\n'
        a.require(len(prep_raw)<=131072,'selected preliminary PREP wire extent');prep.after('control')
        prep_path=a.ROOT/'PRE_GRANT_PREPARATION.json'
        with prep_path.open('xb') as stream:
            a.require(stream.write(prep_raw)==len(prep_raw),'complete preliminary PREP publication')
            stream.flush();os.fsync(stream.fileno())
        prep.after('control')
        prepref={'path':str(prep_path),'bytes':len(prep_raw),'sha256':a.hashlib.sha256(prep_raw).hexdigest()}
        prep_publication.finish()
        # The root owner exists before the FIRST grant dictionary, serializer,
        # SHA, filesystem publication, or immutable permit construction.
        # Retain the partially initialized original owner even if arming the
        # watchdog or its return guard fails. Never lose that root's ledger.
        owner=object.__new__(cost.Budget)
        cost.Budget.__init__(owner,a.partitions(),pin_limit=len(custody._rows),preparation=prep,tail_authority=static.tail_authority)
        a.activate_root(owner)
        ticket=owner.before('control',operation_attempts=1,metadata_bytes=8192)
        # No PREP work occurs after this point. The CPU root's existing grant
        # construction selection owns this direct final counter copy and flag.
        final_preparation={'selected':dict(zip(a.FIELDS,prep._selected)),'used':dict(zip(a.FIELDS,prep._used)),
          'partition_used':{key:dict(zip(a.FIELDS,value)) for key,value in prep._roles.items()},
          'origin':prep.origin,'deadline':prep.deadline,'failed':prep.fatal,'closed':True,
          'all_preparation_work_including_root_constructor_counted':True,
          'trusted_localcaller_startup_qualified':False,'startup_claimed_free':False}
        a.require(not prep.fatal,'failed PREP cannot issue fresh root grant')
        object.__setattr__(prep,'_closed',True)
        g={'schema':'CE8-ACTOR-QUALIFICATION-GRANT-1','owner_thread_id':a.OWNER,'phase':p['phase'],
          'packet_ref':pref,'review_ref':rref,'source_freeze_ref':fref,'owner_readiness_ref':readyref,'authorization_ref':a.AUTH,
          'target_manifest_ref':p['target_manifest_ref'],'target_source_ref':p['target_source_ref'],
          'final_preparation_accounting':final_preparation,'runtime_custody_preparation_ref':prepref,'runtime_custody_owner_pid':os.getpid(),'runtime_custody_pin_count':len(custody._rows),
          'bounds':a.BOUNDS,'final_tail':a.TAIL,'models_allowed':0,'optimizer_updates':0,'gpu_use_allowed':False,
          'output_root':str(a.ACTOR_OUTPUT),'case_names':list(a.CASE_NAMES),
          'case_output_roots':{case:str(a.ACTOR_OUTPUT/case) for case in a.CASE_NAMES},
          'case_selected':{case:a.CASE_SELECTED for case in a.CASE_NAMES},'aggregate_partitions':a.partitions(),
          'physical_origin_monotonic':owner.origin,'physical_deadline_monotonic':owner.deadline,
          'physical_boot_identity':ready['boot_identity'],'target_prior_native_qualification_required':False,'cold_custody_reload_permitted':False}
        a.require(cost.encoded_extent(g)<=8192,'entire grant construction including final PREP counters fits existing8192source envelope')
        ticket.finish();gref=write_fixed('FIXTURE_GRANT.json',g,owner,16384)
        return a.SourceIssuedFixturePermit(a._SEAL,static,g,gref,custody,owner)
    except BaseException as error:
        root_started=owner is not None and getattr(owner,'_root_started',False)
        if root_started:
            owner.mark_failed()
            try:owner.begin_final_tail()
            except BaseException as secondary:error.add_note('tail: '+type(secondary).__name__)
        if custody is not None and not custody._closed:
            try:
                if not root_started:custody.close_all(failure_owner=prep)
                else:a.release_original_pins(custody,owner.pin_close)
            except BaseException as secondary:error.add_note('original pins: '+type(secondary).__name__)
        try:
            if not root_started:prep.failure_receipt(error)
            else:owner.failure_record(error,a.ROOT/'FAILED_ROOT_START.json',envelope=failure_module)
        except BaseException as secondary:error.add_note('failed accounting publication: '+type(secondary).__name__)
        raise

def main():a.require(False,'direct same-owner run_qualification required; metadata grant-and-exit is forbidden')
if __name__=='__main__':raise SystemExit(main())
