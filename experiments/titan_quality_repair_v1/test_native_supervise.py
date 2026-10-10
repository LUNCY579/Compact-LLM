"""Model-free injected guardian fixtures. No real CLI permit or native payload.
Root runs these only inside its serial hardware-policy Windows Job envelope.
"""
from contextlib import closing
import hashlib
import json
import os
import importlib
import sys
from pathlib import Path
import sqlite3
import stat
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

from . import native_safety as safety,native_supervise as supervisor

ARTIFACT=Path("C:/Users/clclc/BusinessProject/ChatBubble-agent-prompts-20261002/run-artifacts/titan-quality-repair-v1/research-v2")


class FakeLedger:
    def __init__(self): self.charges,self.results=[],[]
    def charge_attempt(self,stage,attempt_id,cost):
        self.charges.append((stage,attempt_id,cost))
        return {"kind":"CHARGED","declared_seconds":cost}
    def record_result(self,attempt_id,status,**kwargs):
        self.results.append((attempt_id,status,kwargs))
        return {"fixture_only":True,"declared_charge_retained":720}


class FakeHardware:
    def __init__(self,path,record): self.handle,self.closed=333,False
    def close(self): self.closed=True


class FakeChild:
    def __init__(self): self.pid,self.done,self.killed=1000,False,False
    def poll(self): return 0 if self.done else None
    def wait(self,timeout=None):
        if not self.done: raise subprocess.TimeoutExpired("fixture",timeout)
        return 0
    def kill(self): self.killed,self.done=True,True


class FakeJob:
    def __init__(self,owner):
        self.owner,self.handle,self.assigned,self.closed,self.kills=owner,444,False,False,0
    def assign(self,child):
        self.assigned=True
        self.owner.active=[child.pid]
    def terminate(self):
        self.kills+=1
        if not self.owner.unresolved:
            self.owner.active=[]
            self.owner.child.done=True
    def close(self): self.closed=True


class FakeTimer:
    def __init__(self,seconds,callback): self.seconds,self.callback,self.started,self.cancelled=seconds,callback,False,False
    def start(self): self.started=True
    def cancel(self): self.cancelled=True


def observation(**updates):
    value={"host_available_bytes":13*safety.GIB,"gpu_available_bytes":7*safety.GIB,
        "disk_available_bytes":30*safety.GIB,"tree_rss_bytes":2**20,"new_output_bytes":0,
        "gpu_compute_pids":[],"foreign_model_processes":[]}
    value.update(updates)
    return value


class GuardianFixtures(unittest.TestCase):
    def setUp(self):
        ARTIFACT.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix="guardian-fixture-",dir=ARTIFACT)
        self.base=Path(self.temp.name).resolve()
        self.assertTrue(self.base.is_relative_to(ARTIFACT.resolve()))
        self.addCleanup(self.temp.cleanup)
        self.runtime=self.base/"runtime"
        self.here=self.base/"source"
        self.here.mkdir()
        names=("native_safety.py","native_supervise.py","native_qualification.py","offline_supervise.py",
               "budget.py","registry.py","schema.py")
        self.manifest={}
        for name in names:
            data=("# private model-free fixture "+name+"\n").encode()
            (self.here/name).write_bytes(data)
            self.manifest[name]=hashlib.sha256(data).hexdigest()
        self.patches=[]
        for name,value in (("RUNTIME_ROOT",self.runtime),("TEMP_LOCK",self.base/"temp-gpu.lock"),
                           ("HARDWARE_LOCK",self.base/"hardware.lock"),("LANE_ROOT",self.base)):
            p=patch.object(safety,name,value)
            p.start()
            self.patches.append(p)
            self.addCleanup(p.stop)
        controller_root=patch.object(supervisor,"RUNTIME_ROOT",self.runtime)
        controller_root.start()
        self.addCleanup(controller_root.stop)
        core=self.base/"core"
        core.mkdir()
        p=patch.object(safety,"CORE",core)
        p.start()
        self.addCleanup(p.stop)
        sidecars={}
        for name in ("CORE.json","config.json","generation_config.json","tokenizer.json","tokenizer_config.json","vocab.json","merges.txt"):
            (core/name).write_text("fixture sidecar "+name,encoding="utf-8")
            sidecars[name]=hashlib.sha256((core/name).read_bytes()).hexdigest()
        self.central=self.base/"central.lock"
        self.crs=self.base/"crs.sqlite3"
        with closing(sqlite3.connect(self.crs)) as db,db:
            db.execute("CREATE TABLE leases(status TEXT)")
            db.execute("CREATE TABLE jobs(key TEXT,state TEXT)")
            db.execute("INSERT INTO leases VALUES('RECONCILED')")
        self.proposal={"experiment":safety.EXPERIMENT,"stage":safety.STAGE,"proposal_version":2,
            "status":"NATIVE_QUALIFICATION_IMPLEMENTED_REVIEW_REQUIRED", "source_checkpoint_sha256":safety.WEIGHT_SHA256,
            "registry_sha256":"a"*64,"source_manifest_sha256":safety.digest(self.manifest),"source_manifest":self.manifest,
            "native_worker_implementation_in_this_milestone":True,"training_sweep_authorized":False,
            "final_evaluation_authorized":False,"promotion_authorized":False,"interpreter":"fixture-python",
            "attempt":{"worker_wall_seconds_max":540,"supervisor_kill_deadline_seconds":600,
                "cleanup_wall_seconds_max":120,"loaded_attempt_cap":3,"qualification_pool_seconds_max":3600,
                "development_path":str(self.base/"DEVELOPMENT.jsonl")},
            "runtime_site_packages":[str(safety.RUNTIME_SITE_PACKAGES)],"source_sidecar_sha256":sidecars,
            "runtime_packages":{n:"fixture-v" for n in ("numpy","tokenizers","psutil","torch","transformers","peft","accelerate","safetensors")}}
        metadata={"declared_weight_sha256":safety.WEIGHT_SHA256,"sidecar_sha256":sidecars}
        metadata_path=self.base/"source-metadata.json"
        metadata_path.write_text(json.dumps(metadata),encoding="utf-8")
        self.proposal["metadata_descriptor"]={"path":str(metadata_path),"sha256":hashlib.sha256(metadata_path.read_bytes()).hexdigest()}
        payload=safety.proposal_payload_digest(self.proposal)
        self.proposal["proposal_payload_sha256"]=payload
        review={"experiment":safety.EXPERIMENT,"scope":"STATIC_CODE_AND_CPU_MECHANICS_ONLY",
            "status":"PASS_FOR_BOUNDED_NATIVE_QUALIFICATION_SOURCE","source_manifest_sha256":safety.digest(self.manifest),
            "proposal_payload_sha256":payload,"native_execution_authorized":False}
        cpu={"experiment":safety.EXPERIMENT,"status":"PASS","source_manifest_sha256":safety.digest(self.manifest),
            "native_model_loads":0,"native_model_forwards":0,"source_unchanged_during_worker":True,
            "job_assigned_before_fixture_imports":True,"fixture_only":True}
        for key,value in (("review_descriptor",review),("cpu_qualification_descriptor",cpu)):
            path=self.base/(key+".json")
            path.write_text(json.dumps(value),encoding="utf-8")
            self.proposal[key]={"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        self.permit={"schema_version":2,"experiment":safety.EXPERIMENT,"stage":safety.STAGE,
            "status":"ISSUED_BY_OPERATOR","fixture_only":False,"attempt_id":"private-fixture-attempt",
            "hardware_slot":"PRIVATE_FIXTURE_NO_HARDWARE","operator_text":"private fixture text never a real permit",
            **{key:self.proposal[key] for key in ("source_checkpoint_sha256","registry_sha256","source_manifest_sha256","proposal_payload_sha256")}}
        self.proposal_path,self.permit_path=self.base/"proposal.json",self.base/"operator.json"
        self.save_bundle()
        self.child,self.active,self.unresolved=FakeChild(),[],False
        self.job=FakeJob(self)
        self.hardware=FakeHardware(None,None)
        self.ledger=FakeLedger()
        self.ticks=10.0
        self.commands=[]
        self.actual_worker=1000

    def save_bundle(self,*,rebind_grant=True):
        self.proposal_path.write_text(json.dumps(self.proposal),encoding="utf-8")
        if rebind_grant:
            self.permit["proposal_file_sha256"]=hashlib.sha256(self.proposal_path.read_bytes()).hexdigest()
        self.permit_path.write_text(json.dumps(self.permit),encoding="utf-8")

    def bundle(self):
        # This private structure is injected into mechanics only. It is not a
        # valid real CLI grant/source manifest and no fake workers are launched.
        return {"proposal":self.proposal,"permit":self.permit,"here":str(self.here),
            "proposal_payload_sha256":self.proposal["proposal_payload_sha256"],
            "proposal_file_sha256":hashlib.sha256(self.proposal_path.read_bytes()).hexdigest(),
            "permit_file_sha256":hashlib.sha256(self.permit_path.read_bytes()).hexdigest(),
            "proposal_path":str(self.proposal_path),"permit_path":str(self.permit_path),
            "review_descriptor":self.proposal["review_descriptor"],
            "cpu_qualification_descriptor":self.proposal["cpu_qualification_descriptor"]}

    def clock(self):
        self.ticks+=.25
        return self.ticks

    def identity(self,pid): return float(pid)+.125

    def run_private(self,**updates):
        attempt=self.runtime/"attempts"/self.permit["attempt_id"]
        def popen(command,**kwargs):
            self.commands.append((command,kwargs))
            return self.child
        def resume(child):
            self.assertTrue(self.job.assigned)
            gate=json.loads((attempt/"BOOTSTRAP_GATE.json").read_bytes())
            worker=self.actual_worker
            self.active=sorted({child.pid,worker})
            safety.write_new(attempt/"WORKER_BOOTSTRAP.json",{
                "experiment":safety.EXPERIMENT,"stage":safety.STAGE,"worker_pid":worker,
                "worker_create_time":self.identity(worker),"ancestors":[{"pid":child.pid,"create_time":self.identity(child.pid)}],
                "actual_parent_job_membership":True,**{k:gate[k] for k in ("attempt_id","lease_token","source_checkpoint_sha256",
                    "source_manifest_sha256","proposal_payload_sha256")}})
        def members(_handle):
            if (attempt/"GO.json").exists() and not self.unresolved:
                self.active=[]
                self.child.done=True
            return list(self.active)
        options=dict(runtime_root=self.runtime,ledger=self.ledger,helpers=None,popen=popen,clock=self.clock,sleep=lambda _:None,
            identity=self.identity,resources=lambda *_:observation(),central=self.central,crs=self.crs,
            hardware_factory=lambda *_:self.hardware,job_factory=lambda:self.job,resume=resume,members=members,timer_factory=FakeTimer)
        options.update(updates)
        return supervisor._supervise_admitted(self.bundle(),safety,**options)

    def test_job_assignment_precedes_payload_and_zero_job_members_release(self):
        result=self.run_private()
        self.assertTrue(result["whole_job_exit_verified"])
        self.assertEqual(result["status"],"NATIVE_WORKER_EXITED")
        self.assertEqual(self.ledger.charges,[("qualification",self.permit["attempt_id"],720)])
        self.assertTrue(self.job.closed and self.hardware.closed)
        self.assertFalse(self.central.exists() or safety.TEMP_LOCK.exists())
        command,kwargs=self.commands[0]
        self.assertTrue(kwargs["creationflags"]&0x4)
        self.assertEqual(kwargs["env"]["CUBLAS_WORKSPACE_CONFIG"],":4096:8")
        self.assertTrue(Path(kwargs["env"]["CUDA_CACHE_PATH"]).is_relative_to(self.runtime))

    def test_hermes_actual_worker_descendant_identity_is_admitted(self):
        self.actual_worker=2000
        result=self.run_private()
        self.assertEqual(result["actual_worker"]["worker_pid"],2000)
        start=json.loads((self.runtime/"attempts"/self.permit["attempt_id"]/safety.START_NAME).read_bytes())
        self.assertEqual((start["launcher_pid"],start["worker_pid"]),(1000,2000))

    def test_missing_or_foreign_launcher_ancestry_refuses(self):
        base={"launcher_pid":1000,"launcher_create_time":1000.125,"experiment":safety.EXPERIMENT,"stage":safety.STAGE,
              "attempt_id":"test","lease_token":"a"*32,"source_checkpoint_sha256":safety.WEIGHT_SHA256,
              "source_manifest_sha256":"b"*64,"proposal_payload_sha256":"c"*64}
        record={**base,"worker_pid":2000,"worker_create_time":2000.125,"actual_parent_job_membership":True,"ancestors":[]}
        with self.assertRaisesRegex(RuntimeError,"LAUNCHER_DESCENDANT"):
            supervisor._validate_bootstrap(record,base,[1000,2000],self.identity)
        record["ancestors"]=[{"pid":1000,"create_time":999.0}]
        with self.assertRaises(RuntimeError): supervisor._validate_bootstrap(record,base,[1000,2000],self.identity)

    def test_job_assignment_error_kills_only_retained_child_handle(self):
        def assignment_fails(_): raise RuntimeError("fixture assignment failure")
        self.job.assign=assignment_fails
        result=self.run_private()
        self.assertTrue(self.child.killed)
        self.assertEqual(result["status"],"NATIVE_SUPERVISOR_STOP")
        self.assertTrue(result["whole_job_exit_verified"])

    def test_unknown_job_exit_retains_shared_token_records(self):
        self.unresolved=True
        def members(_):
            self.ticks+=610
            return list(self.active)
        result=self.run_private(members=members)
        self.assertFalse(result["whole_job_exit_verified"])
        self.assertTrue(result["leases_retained_unresolved_worker"])
        self.assertTrue(self.central.exists() and safety.TEMP_LOCK.exists())
        self.assertTrue(json.loads(safety.TEMP_LOCK.read_bytes())["worker_pids"])
        self.assertEqual(self.ledger.results[0][1],"OUTCOME_UNKNOWN")

    def test_foreign_present_temp_and_central_locks_never_reclaimed(self):
        for path in (self.central,safety.TEMP_LOCK):
            path.write_text("foreign owner",encoding="utf-8")
            result=self.run_private()
            self.assertEqual(result["status"],"BLOCKED_NATIVE_ADMISSION")
            self.assertEqual(path.read_text(encoding="utf-8"),"foreign owner")
            break  # each fixture reserves one immutable attempt; use separate setup for other path

    def test_loaded_marker_cap_includes_killed_incomplete_attempts(self):
        for index in range(3):
            path=self.runtime/"attempts"/f"old-{index}"
            path.mkdir(parents=True)
            (path/safety.LOAD_BEGUN).write_text("{}",encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError,"LOADED_ATTEMPT_CAP"): self.run_private()
        self.assertFalse(self.commands)

    def test_preload_floors_fail_before_child_and_cost_stays_charged(self):
        result=self.run_private(resources=lambda *_:observation(host_available_bytes=11*safety.GIB))
        self.assertEqual(result["status"],"BLOCKED_NATIVE_ADMISSION")
        self.assertFalse(self.commands)
        self.assertEqual(self.ledger.charges[0][2],720)

    def test_source_drift_after_entry_stops_and_retains_end_receipt(self):
        calls=[0]
        def observe(*_):
            calls[0]+=1
            if calls[0]>1: (self.here/"schema.py").write_text("# drift",encoding="utf-8")
            return observation()
        # A non-finished Job forces the watchdog branch after GO.
        result=self.run_private(resources=observe,members=lambda _:list(self.active))
        self.assertEqual(result["status"],"STOP_NATIVE_SOURCE_DRIFT")
        self.assertFalse(result["source_unchanged"])

    def test_real_bundle_refuses_fixture_draft_other_experiment_before_runtime(self):
        for field,value in (("fixture_only",True),("status","DRAFT"),("experiment","ANOTHER_EXPERIMENT")):
            old=self.permit[field]
            self.permit[field]=value
            self.save_bundle()
            with self.assertRaises(ValueError): safety.validate_bundle(self.proposal_path,self.permit_path,here=self.here)
            self.permit[field]=old
        self.assertFalse(self.commands)

    def test_review_cpu_descriptors_bind_source_not_imported_other_lane(self):
        distributions=[types.SimpleNamespace(metadata={"Name":n},version="fixture-v") for n in self.proposal["runtime_packages"]]
        with patch.object(safety.importlib.metadata.Distribution,"discover",return_value=distributions):
            bundle=safety.validate_bundle(self.proposal_path,self.permit_path,here=self.here)
        self.assertEqual(bundle["proposal_payload_sha256"],self.proposal["proposal_payload_sha256"])
        cpu_path=Path(self.proposal["cpu_qualification_descriptor"]["path"])
        value=json.loads(cpu_path.read_bytes())
        value["experiment"]="TRISKILL-GEO-001-PARENTQ"
        cpu_path.write_text(json.dumps(value),encoding="utf-8")
        self.proposal["cpu_qualification_descriptor"]["sha256"]=hashlib.sha256(cpu_path.read_bytes()).hexdigest()
        self.save_bundle()
        with patch.object(safety.importlib.metadata.Distribution,"discover",return_value=distributions):
            with self.assertRaisesRegex(ValueError,"CPU_CONTAINMENT_EVIDENCE_BINDING"):
                safety.validate_bundle(self.proposal_path,self.permit_path,here=self.here)

    def test_proposal_hash_excludes_attestation_cycle_only(self):
        before=safety.proposal_payload_digest(self.proposal)
        changed={**self.proposal,"review_descriptor":{"path":"another","sha256":"0"*64}}
        self.assertEqual(before,safety.proposal_payload_digest(changed))
        changed["attempt"]={**self.proposal["attempt"],"worker_wall_seconds_max":541}
        self.assertNotEqual(before,safety.proposal_payload_digest(changed))

    def test_operator_grant_rejects_proof_descriptor_swap_with_same_payload(self):
        old_payload=self.proposal["proposal_payload_sha256"]
        old_file_sha=self.permit["proposal_file_sha256"]
        cpu_path=Path(self.proposal["cpu_qualification_descriptor"]["path"])
        replacement=self.base/"replacement-cpu-proof.json"
        replacement.write_bytes(cpu_path.read_bytes())
        self.proposal["cpu_qualification_descriptor"]={"path":str(replacement),
            "sha256":hashlib.sha256(replacement.read_bytes()).hexdigest()}
        self.save_bundle(rebind_grant=False)
        self.assertEqual(safety.proposal_payload_digest(self.proposal),old_payload)
        self.assertEqual(self.permit["proposal_file_sha256"],old_file_sha)
        with self.assertRaisesRegex(ValueError,"FULL_PROPOSAL_FILE_BINDING"):
            safety.validate_bundle(self.proposal_path,self.permit_path,here=self.here)

    def test_finite_identity_and_exact_caps_typed(self):
        self.run_private()
        start=json.loads((self.runtime/"attempts"/self.permit["attempt_id"]/safety.START_NAME).read_bytes())
        safety.validate_worker_admission(start)
        for key,bad in (("supervisor_started_monotonic",float("nan")),("worker_create_time",float("inf")),
                        ("worker_pid",True),("cleanup_wall_seconds",True)):
            changed={**start,key:bad}
            with self.assertRaises(ValueError): safety.validate_worker_admission(changed)

    def test_resource_unobservable_foreign_and_posthash_preload_floor(self):
        safety.require_resources(observation(),preload=True)
        with self.assertRaises(RuntimeError): safety.require_resources(observation(gpu_available_bytes=5*safety.GIB),preload=True)
        safety.require_resources(observation(gpu_available_bytes=safety.GIB),preload=False)
        for changed in (observation(host_available_bytes=float("nan")),observation(foreign_model_processes=None),
                        observation(foreign_model_processes=[{"pid":99,"rss_bytes":None}])):
            with self.assertRaises(RuntimeError): safety.require_resources(changed)
        with self.assertRaises(RuntimeError): safety.require_resources(observation(gpu_compute_pids=[200]),own_pids=[100])
        safety.require_resources(observation(gpu_compute_pids=[200],allowed_desktop_gpu_pids=[200]),own_pids=[100])

    def test_atomic_shared_lease_failure_never_publishes_partial_or_replaces_foreign(self):
        attempt=self.runtime/"coordination-attempt"
        attempt.mkdir(parents=True)
        owner={"attempt_dir":str(attempt),"supervisor_pid":10,"supervisor_create_time":10.0,"attempt_id":"write-failure"}
        destination=self.base/"shared-lease"
        real_write=safety.write_new
        def fail_after_create(path,value):
            Path(path).write_bytes(b"partial")
            raise OSError("fixture fsync failure")
        with patch.object(safety,"write_new",side_effect=fail_after_create):
            with self.assertRaises(OSError): safety.acquire_token_lock(destination,"b"*32,owner)
        self.assertFalse(destination.exists())
        destination.write_text("foreign",encoding="utf-8")
        with self.assertRaises(FileExistsError): safety.acquire_token_lock(destination,"c"*32,owner)
        self.assertEqual(destination.read_text(encoding="utf-8"),"foreign")

    def test_byte_finder_executes_manifest_snapshot_not_changed_source_or_cache(self):
        package="_titan_verified_fixture"
        path=self.base/"byte-source"
        path.mkdir()
        (path/"probe.py").write_text("raise RuntimeError('unreviewed source executed')",encoding="utf-8")
        (path/"__pycache__").mkdir()
        (path/"__pycache__"/"probe.cpython-313.pyc").write_bytes(b"unreviewed cache")
        finder=safety.install_verified_packet_sources({"__init__.py":b"", "probe.py":b"VALUE=37\n"},path,package=package)
        try:
            module=importlib.import_module(package+".probe")
            self.assertEqual(module.VALUE,37)
            self.assertEqual(module._native_verified_source_sha256,hashlib.sha256(b"VALUE=37\n").hexdigest())
        finally:
            sys.meta_path.remove(finder)
            for name in list(sys.modules):
                if name==package or name.startswith(package+"."): del sys.modules[name]

    def test_late_source_drift_and_failed_release_are_failed_before_accounting(self):
        real_validate=safety.validate_manifest
        calls=[0]
        def late_drift(manifest,here):
            calls[0]+=1
            if calls[0]>=2: raise ValueError("late fixture source drift")
            return real_validate(manifest,here)
        with patch.object(safety,"validate_manifest",side_effect=late_drift): result=self.run_private()
        self.assertEqual(result["status"],"STOP_NATIVE_SOURCE_DRIFT")
        self.assertEqual(self.ledger.results[0][1],"FAILED")

    def test_isolated_child_command_and_no_site_paths_are_inherited(self):
        self.run_private()
        command,_=self.commands[0]
        self.assertEqual(command[1:4],["-I","-B","-S"])

    def test_nested_reparse_output_is_rejected_before_attempt_or_grant_writes(self):
        real_lstat=Path.lstat
        targets=(self.runtime/"attempts",self.runtime/"used-grants")
        for target in targets:
            def reparse(path,*args,**kwargs):
                if path==target:
                    return types.SimpleNamespace(st_mode=stat.S_IFDIR,st_file_attributes=0x400)
                return real_lstat(path,*args,**kwargs)
            with patch.object(Path,"lstat",reparse):
                with self.assertRaisesRegex(ValueError,"REPARSE_ANCESTOR"): self.run_private()
            self.assertFalse((self.runtime/"attempts"/self.permit["attempt_id"]).exists())
            self.assertFalse((self.runtime/"used-grants").exists())
        for path in (self.runtime,self.base/"budget-ledger"):
            def reparse(path_arg,*args,**kwargs):
                if path_arg==path:
                    return types.SimpleNamespace(st_mode=stat.S_IFDIR,st_file_attributes=0x400)
                return real_lstat(path_arg,*args,**kwargs)
            with patch.object(Path,"lstat",reparse):
                with self.assertRaisesRegex(ValueError,"REPARSE_ANCESTOR"):
                    safety.validate_runtime(path) if path==self.runtime else safety.validate_ledger_output(path)

    def test_crs_expired_live_unknown_and_unreadable_fail_closed(self):
        safety.require_crs_idle(self.crs)
        for status in ("LIVE","RECONCILIATION_REQUIRED","UNKNOWN"):
            with closing(sqlite3.connect(self.crs)) as db,db: db.execute("UPDATE leases SET status=?",(status,))
            with self.assertRaises(RuntimeError): safety.require_crs_idle(self.crs)
        with self.assertRaises(RuntimeError): safety.require_crs_idle(self.base/"missing.sqlite3")

    def test_environment_mutation_refuses_instead_of_changing(self):
        self.run_private()
        start=json.loads((self.runtime/"attempts"/self.permit["attempt_id"]/safety.START_NAME).read_bytes())
        env=supervisor._native_env(Path(start["attempt_dir"]))
        with patch.dict(os.environ,env,clear=True):
            safety.require_worker_environment(start)
            os.environ["CUDA_VISIBLE_DEVICES"]="1"
            with self.assertRaisesRegex(RuntimeError,"ENVIRONMENT_DRIFT"): safety.require_worker_environment(start)

    def test_permit_attempt_replay_and_root_rotation_refuse(self):
        self.run_private()
        with self.assertRaisesRegex(RuntimeError,"ALREADY_USED"): self.run_private()
        with self.assertRaises(ValueError): safety.validate_runtime(self.base/"rotated")

    def test_no_native_bootstrap_method_can_accept_fake_resources_as_grant(self):
        self.permit["fixture_grant"]=True
        self.save_bundle()
        with patch.object(supervisor,"_verified_safety",return_value=safety):
            with self.assertRaises(ValueError): supervisor.supervise(self.proposal_path,self.permit_path,runtime_root=self.runtime)
        self.assertFalse(self.commands)


if __name__=="__main__": unittest.main()
