"""Write-once terminal evidence copier. No native, PID, GPU, Git or model calls.

Root supplies exact frozen supervisor and separately anchored current death
observations. This program checks recorded source/identity/cleanup/copy evidence;
the independent native auditor remains responsible for successful qualification.
"""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import argparse,hashlib,json,math,os,stat,sys
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1];ROOT=PACKAGE.parents[1]
sys.path.insert(0,str(HERE));import prepare_registration as registration
pure=registration.pure
HEAD='dc4e52ef3a7b6b6cde51f1ab54a68b2d63be5510'
INVENTORY=PACKAGE/'artifacts/short_m0_audit_registrations/SYNTH_SHORT_PHASE3_BATCH04_A1_AUDIT_INVENTORY_001.json'
INVENTORY_SHA='cc28dd3d02a8f550699892428022e0ebf61c2cdcbf2dd9286f948cd1aba4a6ea'
LEAF_NAME='short_m0_atomic_batch04_phase3_attempt1_20261005'
SIGNAL='ROOT_TERMINAL_FROZEN_BYTES_AND_COPY_AUTHORIZED'
TERMINAL={'SUPERVISED_WORKER_EXITED_ZERO','SUPERVISED_WORKER_FAILED','OWNED_WORKER_ABORTED',
    'ADMISSION_DENIED','ADMISSION_PASSED_NO_WORKER','OWNED_WORKER_LAUNCH_FAILED','WHOLE_SCOPE_DEADLINE_EXCEEDED'}
EPOCH_FILETIME=116444736000000000
def hashpin(v):return registration.auditor.hash_pin(v)
def seal(v):return registration.seal(v)
def directory(path,*,parent=None,new=False):
    path=Path(path).absolute()
    if parent is not None:pure.require(path.resolve().parent==Path(parent).resolve(),'exact owned evidence parent')
    if new:
        pure.require(not path.exists() and not path.is_symlink(),'write-once evidence destination must be absent')
        check=path.parent
    else:
        pure.require(path.is_dir(),'existing terminal evidence directory');check=path
    for p in (check,*check.parents):
        meta=p.lstat();pure.require(stat.S_ISDIR(meta.st_mode) and not p.is_symlink()
            and not(getattr(meta,'st_file_attributes',0)&0x400),'regular unaliased evidence directory')
    return path
def exact_json(path,sha,limit=16*1024**2):
    pure.require(hashpin(sha),'independently selected exact file SHA')
    raw=pure.read_bytes(path,limit);pure.require(hashlib.sha256(raw).hexdigest()==sha,'exact externally frozen file bytes '+str(path))
    return pure.loads(raw)
def aware_utc(value):
    pure.require(type(value) is str and len(value)<=32,'bounded UTC identity clock')
    clock=datetime.fromisoformat(value);pure.require(clock.tzinfo is not None and clock.utcoffset().total_seconds()==0,'aware UTC identity clock')
    return clock
def identity(value,*,full):
    pure.require(type(value) is dict and type(value.get('pid')) is int and value['pid']>0
        and type(value.get('create_time')) in (int,float) and math.isfinite(value['create_time']) and value['create_time']>0,'typed exact PID/create-time identity')
    if full:
        pure.require(all(type(value.get(k)) is int and value[k]>0 for k in ('creation_filetime_100ns','process_handle','job_handle'))
            and type(value.get('job_name')) is str and bool(value['job_name'])
            and value.get('assigned_before_resume') is True and value.get('job_kill_on_close') is True
            and value.get('breakaway_allowed') is False,'complete recorded owned Job/FILETIME/handle identity')
    return value
def normalized(value):
    identity(value,full=False)
    pure.require(type(value.get('creation_filetime_100ns')) is int and value['creation_filetime_100ns']>0,'exact raw FILETIME observation identity')
    return {key:value[key] for key in ('pid','creation_filetime_100ns','create_time')}
def identity_roster(leaf,owned):
    result={}
    def add(value):
        item=normalized(value);key=(item['pid'],item['creation_filetime_100ns'])
        pure.require(key not in result or result[key]==item,'no conflicting exact process birth observations');result[key]=item
    add(owned)
    if (leaf/'RESOURCE_OBSERVATIONS.json').exists():
        observations=pure.read_json(leaf/'RESOURCE_OBSERVATIONS.json')
        for row in observations:
            if row.get('kind')=='CPU':
                pure.require(type(row.get('owned_process_identities')) is list,'exact recorded owned process identity roster')
                for value in row['owned_process_identities']:add(value)
    return [result[key] for key in sorted(result)]
def current_death(path,expected_sha,owned,sup,sup_sha,scope,commit,config_sha,leaf):
    pure.require(path is not None and expected_sha is not None,'Root current owned identity death observation required')
    proof=exact_json(path,expected_sha)
    pure.require(proof.get('schema_version')==1 and proof.get('kind')=='ROOT_CURRENT_OWNED_IDENTITY_DEATH_OBSERVATION_V1'
        and proof.get('sha256')==pure.digest({k:v for k,v in proof.items() if k!='sha256'}),'exact self-sealed Root current-death observation schema')
    pure.require(proof.get('identity')==owned and proof.get('owned_identity_sha256')==pure.digest(owned)
        and proof.get('currently_alive') is False and proof.get('observer')=='native_supervisor._pid_alive'
        and proof.get('source_commit')==commit and proof.get('config_file_sha256')==config_sha
        and proof.get('supervisor_receipt_file_sha256')==sup_sha,'externally anchored exact current-death identity/terminal/config/source joins')
    roster=identity_roster(leaf,owned)
    pure.require(proof.get('owned_identities')==roster and proof.get('owned_identities_sha256')==pure.digest(roster)
        and proof.get('identity_checks')==[{'identity':item,'currently_alive':False} for item in roster],
        'all exact root and observed descendant births independently checked currently dead')
    helper=str((PACKAGE/'native_supervisor.py').resolve())
    # Tests supply a fixture-only same-named source path through their scope.
    if helper not in scope['source_pins']:
        matches=[p for p in scope['source_pins'] if Path(p).name=='native_supervisor.py'];pure.require(len(matches)==1,'one pinned observer source');helper=matches[0]
    pure.require(proof.get('observer_source_sha256')==scope['source_pins'][helper]
        and aware_utc(proof['observed_utc'])>=aware_utc(sup['finished_utc']),'current-death observed after exact terminal through pinned observer')
    return {'path':str(Path(path).absolute()),'file_sha256':expected_sha,'body_sha256':proof['sha256'],
        'identity_sha256':pure.digest(owned),'observed_utc':proof['observed_utc'],'all_observed_owned_births':len(roster),'identity_roster_sha256':pure.digest(roster),
        'authority':'Root-supplied externally anchored current observation; preserver performs no process query'}
def terminal_gate(package,leaf,scope,sup,sup_sha,commit,config_sha,death_path,death_sha):
    pure.require(sup.get('status') in TERMINAL and sup.get('partial_outputs_frozen') is True
        and sup.get('sampler_threads_terminal_verified') is True and sup.get('gpu_telemetry_cleanup_verified') is True
        and sup.get('unjoined_sampler_threads')==[],'verified frozen terminal supervisor/sampler/NVML cleanup')
    pure.require(sup.get('output_dir')==str(leaf) and all(type(sup.get(k)) is bool for k in ('worker_attempted','worker_launched','worker_resumed')),
        'exact terminal path/worker lifecycle booleans')
    aware_utc(sup['finished_utc'])
    if 'config_sha256' in sup:pure.require(sup['config_sha256']==config_sha,'recorded exact config identity')
    if 'preflight' in sup:
        pure.require(sup['preflight'].get('source_commit')==commit and sup['preflight'].get('config_sha256')==config_sha,'recorded exact preflight HEAD/config')
    lease=None;lease_info={};owned=None;observation=None
    if sup.get('lease_path') is not None:
        lease=directory(sup['lease_path'],parent=package/'artifacts/_native_gpu_leases')
        pure.require(pure.file_info(lease/'terminal.json')==pure.file_info(leaf/'SUPERVISOR_RECEIPT.json'),'exact frozen lease/supervisor terminal byte pair')
        reservation=pure.read_json(lease/'reservation.json');identity(reservation['supervisor'],full=False)
        pure.require(reservation.get('config_sha256')==config_sha and reservation.get('output_dir')==str(leaf),'exact lease reservation scope/config')
        lease_info={p.name:pure.file_info(p) for p in lease.iterdir() if p.is_file()}
    if sup['worker_launched']:
        pure.require(lease is not None and sup['worker_attempted'] is True and type(sup.get('returncode')) is int
            and sup.get('job_closed_kill_on_close') is True and sup.get('owned_worker_identity_currently_alive') is False,
            'recorded terminal worker return/Job/death evidence')
        owned=identity(pure.read_json(lease/'worker.json'),full=True)
        pure.require(owned==sup.get('worker_identity'),'exact owned worker identity joins lease/supervisor')
        owner=pure.read_json(leaf/'supervisor_ownership.json')
        pure.require(owner.get('worker_pid')==owner.get('root_worker_pid')==owned['pid'] and owner.get('job_name')==owned['job_name']
            and owner.get('supervisor_pid')==reservation['supervisor']['pid']
            and owner.get('supervisor_create_time')==reservation['supervisor']['create_time'] and owner.get('config_sha256')==config_sha,
            'exact recorded raw/lease ownership identities')
        if sup['worker_resumed']:
            resume=pure.read_json(lease/'resume.json')
            pure.require(resume.get('worker_identity')==owned and resume.get('external_monitor_ready_before_resume') is True
                and resume.get('resumed_monotonic')==sup.get('worker_resume_monotonic'),'exact recorded resume/monitor identity')
        else:pure.require(not (lease/'resume.json').exists(),'unresumed terminal must not carry resume contradiction')
        observation=current_death(death_path,death_sha,owned,sup,sup_sha,scope,commit,config_sha,leaf)
    elif sup['worker_attempted']:
        failure=sup.get('launch_failure')
        pure.require(sup['status']=='OWNED_WORKER_LAUNCH_FAILED' and type(failure) is dict and failure.get('cleanup_errors')==[]
            and failure.get('job_closed_kill_on_close') is True and failure.get('resumed_successfully') is False,'exact cleaned failed launch branch')
        if failure.get('worker_pid') is not None:
            pure.require(lease is not None and failure.get('owned_process_handle_terminal_verified') is True
                and type(failure.get('returncode')) is int,'failed created process handle terminal evidence')
            owned=identity(pure.read_json(lease/'worker.json'),full=False)
            pure.require(owned==failure.get('process_identity') and owned['pid']==failure['worker_pid'],'exact failed-created-process identity')
            observation=current_death(death_path,death_sha,owned,sup,sup_sha,scope,commit,config_sha,leaf)
    else:
        pure.require(sup['worker_resumed'] is False and sup.get('worker_identity') is None and sup.get('returncode') is None
            and sup['status'] in ('ADMISSION_DENIED','ADMISSION_PASSED_NO_WORKER'),'exact no-worker terminal branch')
        if lease is not None:pure.require(not (lease/'worker.json').exists() and not (lease/'resume.json').exists(),'no-worker lease identity contradiction')
    return {'lease_path':None if lease is None else str(lease),'lease_files':lease_info,
        'recorded_owned_identity_sha256':None if owned is None else pure.digest(owned),'Root_current_death_observation':observation,
        'no_worker_process_created':owned is None and not sup['worker_launched']}
def raw_inventory(leaf,sup):
    paths=list(leaf.iterdir());pure.require(paths and len(paths)<=2127 and all(p.is_file() for p in paths),'bounded flat raw terminal inventory')
    result={}
    for p in paths:
        pure.require(p.name not in ('preservation_receipt.json','config_snapshot.json') and
            (p.suffix in ('.json','.jsonl') or p.name in ('worker.stdout.log','worker.stderr.log')),'closed raw evidence filenames/types')
        info=pure.file_info(p);limit=64*1024**2 if p.suffix=='.jsonl' else 16*1024**2
        pure.require(info['bytes']<=limit,'bounded raw evidence copy size');result[p.name]=info
    partial=sup['partial_outputs']
    pure.require(type(partial) is list and len({v['path'] for v in partial})==len(partial)
        and {v['path']:{'bytes':v['bytes'],'sha256':v['sha256']} for v in partial}=={n:v for n,v in result.items() if n!='SUPERVISOR_RECEIPT.json'},
        'exact complete frozen supervisor raw output byte map')
    return result
def copy_exact(source,target,expected):
    pure.require(pure.file_info(source)==expected,'stable source before exclusive copy')
    source=Path(source);target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    hasher=hashlib.sha256();total=0
    with source.open('rb') as inp,target.open('xb') as out:
        while True:
            chunk=inp.read(2*1024**2)
            if not chunk:break
            total+=len(chunk);pure.require(total<=expected['bytes'] and out.write(chunk)==len(chunk),'complete bounded exclusive evidence copy');hasher.update(chunk)
        out.flush();os.fsync(out.fileno())
    pure.require({'bytes':total,'sha256':hasher.hexdigest()}==expected==pure.file_info(target)==pure.file_info(source),'stable source/durable copied byte pair')
def preserve(config,leaf,saved,inventory,*,config_sha,inventory_sha,sup_sha,commit,terminal_signal=None,
    current_death_path=None,current_death_sha=None,fixture=False):
    pure.require(terminal_signal==SIGNAL,'Root terminal-frozen copy authorization before reads or writes')
    pure.require(type(commit) is str and len(commit)==40 and all(c in '0123456789abcdef' for c in commit),'exact externally supplied launch HEAD')
    config=Path(config).absolute();inventory=Path(inventory).absolute()
    pure.require(config.resolve().parent==(PACKAGE/'configs').resolve() and inventory.resolve().parent==(PACKAGE/'artifacts/short_m0_audit_registrations').resolve(),'exact owned config/inventory paths')
    leaf=directory(leaf,parent=PACKAGE/'artifacts');saved=directory(saved,parent=PACKAGE/'evidence',new=True)
    scope=exact_json(config,config_sha,8*1024**2);frozen=exact_json(inventory,inventory_sha)
    pure.require(scope['sha256']==pure.digest({k:v for k,v in scope.items() if k!='sha256'})
        and frozen['sha256']==pure.digest({k:v for k,v in frozen.items() if k!='sha256'})
        and frozen.get('source_commit')==commit and frozen.get('config_file_sha256')==config_sha,'exact sealed launch config/inventory anchors')
    pure.require(frozen.get('inventories')=={k:scope[v] for k,v in registration.SECTIONS.items()}
        and frozen.get('counts')==registration.COUNTS,'every frozen source131/runtime65/decoder17/data5/model7 map')
    pure.require(scope['contract']==registration.auditor.VERSION and scope['batch_selector']=={'source_kind':'atomic_RS_DEV','ordinal':4}
        and len(scope['rows'])==64,'reviewed fixed whole64 Phase3 batch04 purpose')
    sup=exact_json(leaf/'SUPERVISOR_RECEIPT.json',sup_sha)
    gate=terminal_gate(PACKAGE,leaf,scope,sup,sup_sha,commit,config_sha,current_death_path,current_death_sha)
    originals=raw_inventory(leaf,sup);config_info=pure.file_info(config);inventory_info=pure.file_info(inventory)
    sources={};mismatches={}
    for path,pin in frozen['inventories']['source'].items():
        p=Path(path);relative=p.relative_to(ROOT).as_posix()
        pure.require(p.suffix in ('.py','.md','.json') and '..' not in Path(relative).parts,'root-relative source-only snapshot path')
        info=pure.file_info(p);sources[relative]=info
        if info['sha256']!=pin:mismatches[relative]={'expected_sha256':pin,'actual_sha256':info['sha256']}
    success=sup['status']=='SUPERVISED_WORKER_EXITED_ZERO'
    if success:
        worker=pure.read_json(leaf/'worker_result.json');phase=pure.read_json(leaf/'PHASE_ACCOUNTING_FINAL.json')
        pure.require(sup['returncode']==0 and sup['worker_resumed'] is True and worker.get('status')=='COMPLETE_SCOPED_SHORT_M0_RAW_COLLECTION_UNSCORED'
            and worker.get('error') is None and worker.get('config_sha256')==config_sha and worker.get('source_commit')==commit
            and phase.get('status')=='COMPLETE_PHASE_OBSERVATION' and not mismatches,'complete successful raw/phase/source terminal before preservation')
        pure.require(type(worker['collection_costs']['attempted_forwards']) is int and len(originals)==worker['collection_costs']['attempted_forwards']+79,
            'exact successful full raw denominator')
    # Only after every admission/cleanup/current-death/inventory check succeeds
    # may the owned evidence leaf be created. A failed copy retains its partial
    # directory without publishing any final receipt or modifying the attempt.
    saved.mkdir()
    for name,info in originals.items():copy_exact(leaf/name,saved/name,info)
    copy_exact(config,saved/'config_snapshot.json',config_info);copy_exact(inventory,saved/'audit_inventory_snapshot.json',inventory_info)
    for relative,info in sources.items():copy_exact(ROOT/relative,saved/'source_snapshots'/relative,info)
    auditor_snapshots={}
    for path,pin in frozen['auditor_dependency_pins'].items():
        p=Path(path);info=pure.file_info(p);pure.require(info['sha256']==pin,'reviewed auditor bytes before snapshot')
        relative=p.relative_to(ROOT).as_posix();copy_exact(p,saved/'auditor_dependency_snapshots'/relative,info);auditor_snapshots[relative]=info
    death_info=None
    if current_death_path is not None:
        death_info=pure.file_info(current_death_path);pure.require(death_info['sha256']==current_death_sha,'Root current-death bytes before snapshot')
        copy_exact(current_death_path,saved/'Root_current_death_observation.json',death_info)
    pure.require(raw_inventory(leaf,sup)==originals and pure.file_info(config)==config_info and pure.file_info(inventory)==inventory_info,'all original raw/config/inventory bytes stable after copy')
    for relative,info in sources.items():pure.require(pure.file_info(ROOT/relative)==info,'all source bytes stable after copy')
    pure.require({name:pure.file_info(saved/name) for name in originals}==originals
        and pure.file_info(saved/'config_snapshot.json')==config_info
        and pure.file_info(saved/'audit_inventory_snapshot.json')==inventory_info
        and {relative:pure.file_info(saved/'source_snapshots'/relative) for relative in sources}==sources,
        'complete final raw/config/inventory/source destination byte map')
    for relative,info in auditor_snapshots.items():
        pure.require(pure.file_info(ROOT/relative)==info==pure.file_info(saved/'auditor_dependency_snapshots'/relative),
            'all original/copied reviewed auditor dependency bytes stable before seal')
    if death_info is not None:pure.require(pure.file_info(current_death_path)==death_info==pure.file_info(saved/'Root_current_death_observation.json'),
        'original/copied externally anchored current-death bytes stable before seal')
    if gate['lease_path'] is not None:
        lease=Path(gate['lease_path']);pure.require({p.name:pure.file_info(p) for p in lease.iterdir() if p.is_file()}==gate['lease_files'],'all original frozen lease bytes stable after copy')
    body={'schema_version':1,'status':'CPU_FIXTURE_TERMINAL_BYTES_PRESERVED_ONLY' if fixture else
        ('SUCCESS_TERMINAL_BYTES_PRESERVED_FULL_AUDIT_PENDING' if success else 'FAILED_OR_NO_WORKER_TERMINAL_BYTES_PRESERVED_NOT_NATIVE_PASS'),
        'recorded_utc':datetime.now(timezone.utc).isoformat(),'source_commit':commit,'raw_directory':str(leaf),'preserved_directory':str(saved),
        'raw_files':originals,'config_snapshot':config_info,'source_snapshots':sources,'audit_inventory_snapshot':inventory_info,
        'auditor_dependency_snapshots':auditor_snapshots,'supervisor_receipt_file_sha256':sup_sha,'terminal_ownership':gate,
        'source_pin_mismatches':mismatches,'source_bytes_match_frozen_launch_pins':not mismatches,'full_native_audit_eligible':success and not fixture,
        'preserver_performed_live_identity_or_OS_probe':False,'current_HEAD_independently_queried_by_preserver':False,
        'native_PASS_obtained':False,'scoring_authorized':False,'training_authorized':False,'study_completed':False}
    body=seal(body);raw=(pure.canonical(body)+'\n').encode('ascii');pure.require(len(raw)<=pure.MAX_JSON,'bounded preservation receipt')
    with (saved/'preservation_receipt.json').open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    pure.require(pure.read_json(saved/'preservation_receipt.json')==body,'exact final preservation receipt reread')
    return body
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root-terminal-frozen',action='store_true')
    parser.add_argument('--supervisor-file-sha256',required=True);parser.add_argument('--current-death-evidence');parser.add_argument('--current-death-file-sha256')
    args=parser.parse_args()
    result=preserve(registration.CONFIG,PACKAGE/'artifacts'/LEAF_NAME,PACKAGE/'evidence'/LEAF_NAME,INVENTORY,
        config_sha=registration.CONFIG_SHA,inventory_sha=INVENTORY_SHA,sup_sha=args.supervisor_file_sha256,commit=HEAD,
        terminal_signal=SIGNAL if args.root_terminal_frozen else None,current_death_path=args.current_death_evidence,current_death_sha=args.current_death_file_sha256)
    print(json.dumps({'status':result['status'],'preservation_body_sha256':result['sha256'],'preserved_directory':result['preserved_directory'],
        'raw_files':len(result['raw_files']),'sources':len(result['source_snapshots']),'full_native_audit_eligible':result['full_native_audit_eligible']}))
if __name__=='__main__':main()
