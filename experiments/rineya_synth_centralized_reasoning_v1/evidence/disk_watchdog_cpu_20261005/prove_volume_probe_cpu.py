"""Bounded read-only CPU sampler comparison with an owned fsync writer fixture.

No model framework, GPU, native qualification, process termination, quota or
host setting change. All writes are exclusive files below this staging leaf.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from volume_disk_probe import VolumeBoundDiskProbe

WORK=Path(__file__).resolve().parent
PACKAGE=WORK.parents[1]
ATTEMPT=PACKAGE/'artifacts/backendq002_recompute_A_attempt1'
sys.path.insert(0,str(PACKAGE))
import native_supervisor as qualified


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def new_json(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
def stats(values):
    ordered=sorted(values)
    return {'count':len(values),'min_seconds':ordered[0],'median_seconds':ordered[len(ordered)//2],
            'p95_seconds':ordered[int((len(ordered)-1)*.95)],'max_seconds':ordered[-1],
            'over_200ms':sum(v>.2 for v in ordered)}


def main():
    evidence_names=('RESOURCE_OBSERVATIONS.json','events.jsonl','SUPERVISOR_RECEIPT.json','supervisor_heartbeat.jsonl')
    source_paths=[WORK/'volume_disk_probe.py',WORK/'test_volume_disk_probe.py',Path(__file__),
                  PACKAGE/'native_supervisor.py',PACKAGE/'backend_qual_worker.py',Path(shutil.__file__)]
    before={str(p):sha(p) for p in source_paths}
    preserved_before={n:sha(ATTEMPT/n) for n in evidence_names}
    output=WORK/'cpu_probe_attempt1_20261005';output.mkdir(exist_ok=False)
    probe=VolumeBoundDiskProbe(output,WORK)
    actual_probe=VolumeBoundDiskProbe(ATTEMPT,PACKAGE/'artifacts')
    require_same=probe.root==actual_probe.root and probe.volume_id==actual_probe.volume_id
    if not require_same:raise ValueError('fixture and failed original output must share exact volume')
    observations=json.loads((ATTEMPT/'RESOURCE_OBSERVATIONS.json').read_bytes())
    cpus=[(i,x) for i,x in enumerate(observations) if x.get('kind')=='CPU']
    ix,worst=max(cpus,key=lambda z:z[1]['sample_completed_monotonic']-z[1]['sample_started_monotonic'])
    previous=cpus[[i for i,(index,x) in enumerate(cpus) if index==ix][0]-1][1]
    heartbeat=[json.loads(line) for line in (ATTEMPT/'supervisor_heartbeat.jsonl').read_text().splitlines()]
    # Heartbeat time is stamped after the completed CPU sample. Cross-clock
    # alignment is approximate and never establishes fsync blocking causality.
    last_hb=heartbeat[-1]['unix_seconds']
    approx_start=last_hb-(worst['sample_completed_monotonic']-worst['sample_started_monotonic'])
    events=[json.loads(line) for line in (ATTEMPT/'events.jsonl').read_text().splitlines()]
    nearby=[e for e in events if approx_start<=datetime.fromisoformat(e['utc']).timestamp()<=last_hb]
    samples=[];writer=[];writer_errors=[]
    def write_fixture():
        try:
            with (output/'owned_CPU_durable_journal.jsonl').open('xb',buffering=0) as stream:
                for ordinal in range(128):
                    record=(json.dumps({'event':'CPU_FIXTURE_attempt_and_return_preserved','ordinal':ordinal,'padding':'x'*256})+'\n').encode()
                    start=time.perf_counter();start_mono=time.monotonic()
                    if stream.write(record)!=len(record):raise OSError('incomplete fixture record')
                    stream.flush();os.fsync(stream.fileno())
                    writer.append({'ordinal':ordinal,'started':start_mono,'completed':time.monotonic(),
                                   'latency_seconds':time.perf_counter()-start,'bytes':len(record)})
                    time.sleep(.025)
        except BaseException as e:writer_errors.append({'type':type(e).__name__,'message':str(e)})
    def phase(name,count):
        due=time.perf_counter()
        for ordinal in range(count):
            started=time.monotonic()
            timing={};results={}
            operations={
                'qualified_output_directory_CPU_snapshot':lambda:qualified._cpu_snapshot(output),
                'qualified_bound_root_CPU_snapshot':lambda:qualified._cpu_snapshot(probe.root),
                'candidate_verified_binding_root_disk_sample':probe.sample}
            names=list(operations);names=names[ordinal%3:]+names[:ordinal%3]
            for op in names:
                t=time.perf_counter();results[op]=operations[op]();timing[op]=time.perf_counter()-t
            # A bracket with unchanged root values allows exact comparisons;
            # concurrent writers can legitimately change free bytes between API
            # calls, so moving brackets are labelled rather than forced equal.
            left=probe.api.space(probe.root)
            leaf=probe.api.space(output)
            right=probe.api.space(probe.root)
            samples.append({'phase':name,'ordinal':ordinal,'started':started,'completed':time.monotonic(),
                            'timing_seconds':timing,'results':results,'read_parity_bracket':{
                                'root_before':left,'output':leaf,'root_after':right,
                                'stable':left==right,'equal_when_stable':leaf==left if left==right else None}})
            due+=.1;time.sleep(max(0,due-time.perf_counter()))
    phase('idle',32)
    thread=threading.Thread(target=write_fixture,name='owned_CPU_fsync_fixture');thread.start()
    try:phase('owned_CPU_fsync_writer',64)
    finally:thread.join(timeout=10)
    if thread.is_alive() or writer_errors or len(writer)!=128:raise ValueError('owned fixture writer incomplete/unjoined')
    probe.verify_binding();actual_probe.verify_binding()
    summary={phase_name:{name:stats([s['timing_seconds'][name] for s in samples if s['phase']==phase_name])
                       for name in samples[0]['timing_seconds']} for phase_name in ('idle','owned_CPU_fsync_writer')}
    stable=[s['read_parity_bracket'] for s in samples if s['read_parity_bracket']['stable']]
    if not stable or not all(s['equal_when_stable'] for s in stable):raise ValueError('stable same-volume API triplet parity failed')
    after={str(p):sha(p) for p in source_paths};preserved_after={n:sha(ATTEMPT/n) for n in evidence_names}
    if before!=after or preserved_before!=preserved_after:raise ValueError('source/failed evidence drift')
    if any(x in sys.modules for x in ('torch','transformers')):raise ValueError('framework import prohibited')
    new_json(output/'probe_observations.json',{'samples':samples,'writer':writer})
    receipt={'schema_version':1,'status':'PASS_BOUNDED_CPU_SAME_VOLUME_PROBE_CONTROLS_NOT_NATIVE_QUALIFICATION',
        'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'source_before':before,'source_after':after,
        'failed_attempt_hashes_before':preserved_before,'failed_attempt_hashes_after':preserved_after,
        'source_primary_references':[
            'https://github.com/python/cpython/blob/v3.11.15/Modules/posixmodule.c#L10790-L10825',
            'https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-getdiskfreespaceexw',
            'https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-getvolumepathnamew'],
        'exact_volume':{'fixture_output':str(output),'failed_output':str(ATTEMPT),'mount_root':probe.root,
                        'volume_GUID':probe.volume_id,'bindings_verified_before_after':True,'no_reparse_ancestry':True},
        'actual_failure':{'receipt_status':json.loads((ATTEMPT/'SUPERVISOR_RECEIPT.json').read_bytes())['status'],
                          'CPU_observations':len(cpus),'worst_observation_index':ix,
                          'worst_window_seconds':worst['sample_completed_monotonic']-worst['sample_started_monotonic'],
                          'worst_phase_latency_seconds':worst['phase_latency_seconds'],
                          'previous_start_gap_seconds':worst['sample_started_monotonic']-previous['sample_started_monotonic'],
                          'approximate_UTC_window':[datetime.fromtimestamp(approx_start,timezone.utc).isoformat(),datetime.fromtimestamp(last_hb,timezone.utc).isoformat()],
                          'nearby_durable_event_count':len(nearby),'nearby_event_types':dict(Counter(e['event'] for e in nearby)),
                          'cause_limit':'323ms measured inside disk-query phase; API wall latency may include scheduling/filter/IO stalls; journal overlap is temporal evidence, not causal proof'},
        'CPU_controls':{'focused_fake_tests':5,'focused_fake_test_seconds':.007,'samples':len(samples),'idle_samples':32,
                        'writer_phase_samples':64,'stable_exact_triplet_parity_samples':len(stable),
                        'moving_space_brackets':len(samples)-len(stable),'writer_returned_fsync_records':len(writer),
                        'writer_joined':True,'writer_total_bytes':sum(x['bytes'] for x in writer),
                        'writer_latency':stats([w['latency_seconds'] for w in writer]),'probe_latencies':summary},
        'resource_policy':{'CPU_freshness_seconds':.2,'CPU_cadence_seconds':.1,'disk_free_GiB':4.,
                           'existing_free_counter':'GetDiskFreeSpaceExW THIRD output total-volume free; shutil.free',
                           'quota_counter':'FIRST output caller availability captured separately; not substituted into existing floor',
                           'numeric_space_cache':False,'all_sample_binding_latency_inside_window':True,
                           'slow_call_still_fails_freshness':True,'original_resource_checks_unchanged':True},
        'proposal_limits':'staged explicit containing-volume-root probe only; measured CPU timings do not guarantee future loaded-host freshness. No fsync removed/batched, no production/native admission or retry.',
        'native_launch_authorized':False,'study_training_qualified':False,'root_source_edits':False,
        'framework_imported':False,'GPU_queried':False,'published_weights_opened':False,
        'raw_observation_file':{'path':str(output/'probe_observations.json'),'sha256':sha(output/'probe_observations.json')}}
    new_json(output/'cpu_probe_receipt.json',receipt)
    print(json.dumps({'receipt':str(output/'cpu_probe_receipt.json'),'receipt_sha256':sha(output/'cpu_probe_receipt.json'),
                      'parity_samples':len(stable),'probe_latencies':summary,'writer_latency':receipt['CPU_controls']['writer_latency']},indent=2))


if __name__=='__main__':main()
