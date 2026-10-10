"""Future source-bound, bounded native qualification guardian.

The real CLI has no fixture/fake-grant path. Source-bound review and CPU
mechanics evidence plus an issued operator grant are required before locks,
resource admission or payload entry. No native run is authorized by this code.
An explicit human instruction in the owning chat is separately required.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.abc
import importlib.util
import json
import os
import stat
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import uuid

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
RUNTIME_ROOT=Path("C:/Users/clclc/BusinessProject/ChatBubble-agent-prompts-20261002/run-artifacts/titan-quality-repair-v1/native-qualification")
RECEIPT="NATIVE_SUPERVISOR_RECEIPT.json"
RETAINED_HANDLES=[]


def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()


class NoFrameworkImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split(".")[0] in {"torch","transformers","tensorflow","jax","cupy"}:
            raise ImportError("STOP_NATIVE_SUPERVISOR_FRAMEWORK_IMPORT:"+fullname)


def _verified_safety(proposal_path: Path):
    """Bootstrap with stdlib only, then execute the exact manifest-pinned bytes."""
    proposal=json.loads(Path(proposal_path).read_bytes())
    expected=proposal.get("source_manifest",{}).get("native_safety.py")
    path=HERE/"native_safety.py"
    source=path.read_bytes()
    if not isinstance(expected,str) or len(expected)!=64 or hashlib.sha256(source).hexdigest()!=expected:
        raise ValueError("STOP_NATIVE_SAFETY_BOOTSTRAP_PIN")
    name="_titan_native_safety_bootstrap_"+expected[:16]
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    exec(compile(source,str(path),"exec"),module.__dict__)
    return module


def _resume(child):
    """Resume only the owned suspended primary thread after Job assignment."""
    import ctypes
    from ctypes import wintypes as w
    import psutil
    threads=psutil.Process(child.pid).threads()
    if len(threads)!=1: raise RuntimeError("STOP_NATIVE_SUSPENDED_THREAD_IDENTITY")
    k=ctypes.WinDLL("kernel32",use_last_error=True)
    k.OpenThread.argtypes,k.OpenThread.restype=(w.DWORD,w.BOOL,w.DWORD),w.HANDLE
    k.ResumeThread.argtypes,k.ResumeThread.restype=(w.HANDLE,),w.DWORD
    k.CloseHandle.argtypes=(w.HANDLE,)
    handle=k.OpenThread(0x0002,False,threads[0].id)
    if not handle: raise RuntimeError("STOP_NATIVE_OWNED_THREAD_HANDLE")
    try:
        prior=k.ResumeThread(handle)
        if prior!=1: raise RuntimeError("STOP_NATIVE_RESUME_SUSPEND_COUNT")
    finally: k.CloseHandle(handle)


def _bootstrap_output_path(path: Path) -> Path:
    """Stdlib-only error/cache output guard, including Windows junctions."""
    raw=Path(os.path.abspath(os.fspath(path)))
    canonical=Path(os.path.abspath(os.fspath(RUNTIME_ROOT)))
    for item in (raw,*raw.parents,canonical,*canonical.parents):
        try: info=item.lstat()
        except FileNotFoundError: continue
        except OSError as exc: raise ValueError("STOP_NATIVE_OUTPUT_ANCESTOR_UNOBSERVABLE") from exc
        if (stat.S_ISLNK(info.st_mode) or getattr(info,"st_file_attributes",0)&0x400
                or (hasattr(item,"is_junction") and item.is_junction())):
            raise ValueError("STOP_NATIVE_OUTPUT_REPARSE_ANCESTOR")
    physical=raw.resolve()
    if (not raw.is_relative_to(canonical) or not physical.is_relative_to(canonical.resolve())
            or ".codex" in [part.lower() for part in physical.parts]
            or any((parent/".git").exists() for parent in (physical,*physical.parents))):
        raise ValueError("STOP_NATIVE_OWNED_OUTPUT_REDIRECTION")
    return physical


def _native_env(attempt: Path,*,guard=_bootstrap_output_path) -> dict:
    guard(attempt)
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"0","CUBLAS_WORKSPACE_CONFIG":":4096:8",
         "PYTHONDONTWRITEBYTECODE":"1","PYTHONUNBUFFERED":"1","TOKENIZERS_PARALLELISM":"false",
         "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
         "HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","TMP":str(attempt/"tmp"),"TEMP":str(attempt/"tmp"),
         "HF_HOME":str(attempt/"cache/hf"),"HF_HUB_CACHE":str(attempt/"cache/hf/hub"),
         "TORCH_HOME":str(attempt/"cache/torch"),"XDG_CACHE_HOME":str(attempt/"cache"),
         "CUDA_CACHE_PATH":str(attempt/"cache/cuda"),"TORCH_EXTENSIONS_DIR":str(attempt/"cache/torch_extensions"),
         "TORCHINDUCTOR_CACHE_DIR":str(attempt/"cache/torchinductor"),"TRITON_CACHE_DIR":str(attempt/"cache/triton")}
    for path in (attempt/"tmp",attempt/"cache/hf",attempt/"cache/torch",attempt/"cache/cuda",
                 attempt/"cache/torch_extensions",attempt/"cache/torchinductor",attempt/"cache/triton"):
        guard(path)
        path.mkdir(parents=True,exist_ok=True)
    return env


def loaded_attempts(root: Path,safety) -> int:
    return sum(1 for path in (Path(root)/"attempts").glob("*/"+safety.LOAD_BEGUN))


def _validate_bootstrap(record: dict,base: dict,members: list[int],identity) -> dict:
    for key in ("experiment","stage","attempt_id","lease_token","source_checkpoint_sha256",
                "source_manifest_sha256","proposal_payload_sha256"):
        if record.get(key)!=base[key]: raise RuntimeError("STOP_NATIVE_WORKER_BOOTSTRAP_BINDING:"+key)
    pid,created=record.get("worker_pid"),record.get("worker_create_time")
    if type(pid) is not int or type(created) not in (int,float) or pid not in members or identity(pid)!=created:
        raise RuntimeError("STOP_NATIVE_ACTUAL_WORKER_IDENTITY_OR_JOB")
    if record.get("actual_parent_job_membership") is not True:
        raise RuntimeError("STOP_NATIVE_WORKER_BOOTSTRAP_JOB_PROOF")
    launcher=(base["launcher_pid"],base["launcher_create_time"])
    ancestors=record.get("ancestors",[])
    if pid!=launcher[0] and launcher not in {(p.get("pid"),p.get("create_time")) for p in ancestors if isinstance(p,dict)}:
        raise RuntimeError("STOP_NATIVE_WORKER_NOT_OWNED_LAUNCHER_DESCENDANT")
    return {"worker_pid":pid,"worker_create_time":created}


def _supervise_admitted(bundle,safety,*,runtime_root,ledger,helpers,popen=subprocess.Popen,
        clock=time.monotonic,sleep=time.sleep,identity=None,resources=None,central=None,crs=None,
        hardware_factory=None,job_factory=None,resume=_resume,members=None,timer_factory=threading.Timer):
    """Private injected mechanics surface. Real CLI reaches it only after full bundle validation.
    Unit fixtures replace processes/resources/locks; they cannot issue a permit.
    """
    identity=identity or safety.process_created
    observe=resources or safety.resources
    central=Path(central) if central is not None else safety.central_path(REPO)
    crs=Path(crs) if crs is not None else safety.crs_path()
    members=members or safety.job_members
    hardware_factory=hardware_factory or helpers.HardwareHandle
    job_factory=job_factory or helpers.Job
    root=safety.validate_runtime(runtime_root)
    safety.validate_owned_output(root)
    root.mkdir(parents=True,exist_ok=True)
    permit,proposal=bundle["permit"],bundle["proposal"]
    attempt_id=permit["attempt_id"]
    attempt=root/"attempts"/attempt_id
    safety.validate_owned_output(attempt)
    # Reject existing redirected descendants before creating an attempt or
    # marking its grant spent, not only when a later cache is first used.
    for path in (root/"attempts",root/"used-grants",attempt/"tmp",attempt/"cache",attempt/"cache/hf",
                 attempt/"cache/hf/hub",attempt/"cache/torch",attempt/"cache/cuda",
                 attempt/"cache/torch_extensions",attempt/"cache/torchinductor",attempt/"cache/triton"):
        safety.validate_owned_output(path)
    if loaded_attempts(root,safety)>=3: raise RuntimeError("STOP_NATIVE_LOADED_ATTEMPT_CAP")
    used=root/"used-grants"
    safety.validate_owned_output(used)
    used.mkdir(exist_ok=True)
    permit_used=used/(bundle["permit_file_sha256"]+".json")
    safety.validate_owned_output(permit_used)
    if attempt.exists() or permit_used.exists(): raise RuntimeError("STOP_NATIVE_ATTEMPT_OR_PERMIT_ALREADY_USED")
    attempt.mkdir(parents=True,exist_ok=False)
    started=clock()
    token=uuid.uuid4().hex
    base={"experiment":safety.EXPERIMENT,"stage":safety.STAGE,"attempt_id":attempt_id,
          "hardware_slot":permit["hardware_slot"],"source_checkpoint_sha256":safety.WEIGHT_SHA256,
          "registry_sha256":proposal["registry_sha256"],"source_manifest_sha256":proposal["source_manifest_sha256"],
          "proposal_payload_sha256":bundle["proposal_payload_sha256"],"runtime_root":str(root),"attempt_dir":str(attempt),
          "proposal_path":bundle["proposal_path"],"permit_path":bundle["permit_path"],
          "proposal_file_sha256":bundle["proposal_file_sha256"],"permit_file_sha256":bundle["permit_file_sha256"],
          "review_descriptor":bundle["review_descriptor"],"cpu_qualification_descriptor":bundle["cpu_qualification_descriptor"],
          "source_directory":bundle["here"],
          "supervisor_pid":os.getpid(),"supervisor_create_time":identity(os.getpid()),
          "supervisor_started_monotonic":started,**safety.CAPS,"limits":dict(safety.LIMITS),
          "temp_lock":str(safety.TEMP_LOCK),"central_lock":str(central),"hardware_lock":str(safety.HARDWARE_LOCK),
          "crs_path":str(crs),"lease_token":token,"fixture_only":False}
    receipt={**base,"utc_start":utc(),"status":"ADMISSION_PENDING","native_model_calls_claimed_by_supervisor":False}
    hardware=job=child=timer=None
    acquired=[]
    charged=False
    worker=None
    stop_event=threading.Event()
    cleanup_errors=[]
    def timed_kill():
        stop_event.set()
        try:
            if job is not None: job.terminate()
        except Exception as exc: cleanup_errors.append("HARD_WALL_JOB_KILL:"+repr(exc))
    try:
        safety.write_new(permit_used,{"experiment":safety.EXPERIMENT,"attempt_id":attempt_id,
            "permit_sha256":bundle["permit_file_sha256"],"utc":utc(),"cost_declared_seconds":720})
        charge=ledger.charge_attempt("qualification",attempt_id,720)
        receipt["budget_charge"]=charge
        if charge["kind"]!="CHARGED": raise RuntimeError("STOP_NATIVE_QUALIFICATION_BUDGET_EXHAUSTED")
        charged=True
        if base["supervisor_create_time"] is None: raise RuntimeError("STOP_NATIVE_SUPERVISOR_IDENTITY_UNAVAILABLE")
        hardware=hardware_factory(safety.HARDWARE_LOCK,{**base,"utc":utc()})
        safety.require_crs_idle(crs)
        for path in (central,safety.TEMP_LOCK):
            safety.acquire_token_lock(path,token,base)
            acquired.append(path)
        if loaded_attempts(root,safety)>=3: raise RuntimeError("STOP_NATIVE_LOADED_ATTEMPT_CAP_AFTER_OWNERSHIP")
        safety.validate_manifest(proposal["source_manifest"],Path(bundle["here"]))
        observed=observe(attempt,())
        safety.require_resources(observed,preload=True)
        receipt["preload_resources"]=observed
        job=job_factory()
        env=_native_env(attempt,guard=safety.validate_owned_output)
        interpreter=proposal["interpreter"]
        gate_path=attempt/"BOOTSTRAP_GATE.json"
        safety.validate_owned_output(gate_path)
        # Use a stdlib trampoline: it reads a create-only command admission
        # AFTER resumption, then invokes the pinned qualification module.
        trampoline=("import hashlib,json,sys,types; from pathlib import Path; "
            "p=Path(sys.argv[1]); b=p.read_bytes(); g=json.loads(b); q=json.loads(Path(g['proposal_path']).read_bytes()); "
            "h=Path(g['source_directory']); f=h/'native_safety.py'; s=f.read_bytes(); "
            "assert hashlib.sha256(s).hexdigest()==q['source_manifest']['native_safety.py']; "
            "n='experiments.titan_quality_repair_v1.native_safety'; m=types.ModuleType(n); "
            "m.__file__=str(f); m.__package__='experiments.titan_quality_repair_v1'; sys.modules[n]=m; "
            "exec(compile(s,str(f),'exec'),m.__dict__); "
            "raise SystemExit(m.run_verified_worker(p,hashlib.sha256(b).hexdigest()))")
        safety.validate_owned_output(attempt/"worker.log")
        with (attempt/"worker.log").open("xb") as log:
            child=popen([interpreter,"-I","-B","-S","-c",trampoline,str(gate_path)],cwd=str(REPO),env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                creationflags=0x00000004|0x08000000|0x00000200)
            launcher_created=identity(child.pid)
            if launcher_created is None: raise RuntimeError("STOP_NATIVE_LAUNCHER_IDENTITY_UNAVAILABLE")
            job.assign(child)
            for path in acquired: safety.set_worker_pids(path,token,attempt_id,[child.pid])
            base.update({"launcher_pid":child.pid,"launcher_create_time":launcher_created,
                         "job_assigned_before_payload":True,"guardian_admission_passed":True,
                         "hardware_owner_handle":int(hardware.handle),"job_owner_handle":int(job.handle),
                         "job_name":"OWNED_HANDLE_"+token})
            gate_payload=dict(base)
            safety.write_new(gate_path,gate_payload)
            timer=timer_factory(max(0,600-(clock()-started)),timed_kill)
            timer.daemon=True
            timer.start()
            if child.poll() is not None or identity(child.pid)!=launcher_created:
                raise RuntimeError("STOP_NATIVE_LAUNCHER_IDENTITY_BEFORE_RESUME")
            resume(child)
            bootstrap=attempt/"WORKER_BOOTSTRAP.json"
            bootstrap_deadline=min(started+540,clock()+30)
            while not bootstrap.exists():
                if child.poll() is not None or clock()>=bootstrap_deadline or stop_event.is_set():
                    raise RuntimeError("STOP_NATIVE_ACTUAL_WORKER_BOOTSTRAP_MISSING")
                safety.validate_manifest(proposal["source_manifest"],Path(bundle["here"]))
                for path in acquired: safety.check_token_lock(path,token,attempt_id)
                safety.require_crs_idle(crs)
                active=members(int(job.handle))
                safety.require_resources(observe(attempt,active),own_pids=active)
                sleep(.1)
            worker=_validate_bootstrap(json.loads(bootstrap.read_bytes()),base,members(int(job.handle)),identity)
            base.update(worker)
            for path in acquired: safety.set_worker_pids(path,token,attempt_id,[child.pid,worker["worker_pid"]])
            safety.validate_worker_admission(base)
            safety.write_new(attempt/safety.START_NAME,base)
            start_sha=hashlib.sha256((attempt/safety.START_NAME).read_bytes()).hexdigest()
            safety.write_new(attempt/"GO.json",{"start_path":str(attempt/safety.START_NAME),"start_sha256":start_sha,
                "attempt_id":attempt_id,"lease_token":token})
            receipt.update({"actual_worker":worker,"job_assigned_before_native_payload":True,"guardian_admission_passed":True})
            while members(int(job.handle)):
                if stop_event.is_set() or clock()-started>=600:
                    receipt["status"]="STOP_NATIVE_SUPERVISOR_HARD_WALL"
                    timed_kill()
                    break
                if (identity(os.getpid())!=base["supervisor_create_time"]
                        or (identity(worker["worker_pid"]) not in (None,worker["worker_create_time"]))):
                    raise RuntimeError("STOP_NATIVE_PROCESS_IDENTITY_CHANGED")
                safety.validate_manifest(proposal["source_manifest"],Path(bundle["here"]))
                for path in acquired: safety.check_token_lock(path,token,attempt_id)
                safety.require_crs_idle(crs)
                active=members(int(job.handle))
                observed=observe(attempt,active)
                safety.require_resources(observed,own_pids=active)
                receipt["last_resources"]=observed
                sleep(.25)
            if receipt["status"]=="ADMISSION_PENDING": receipt["status"]="NATIVE_WORKER_EXITED"
    except BaseException as exc:
        receipt["status"]="BLOCKED_NATIVE_ADMISSION" if child is None else "NATIVE_SUPERVISOR_STOP"
        receipt["error"]=repr(exc)
        receipt["traceback"]=traceback.format_exc()
        # Assignment can fail with an empty Job and a still-suspended, unassigned
        # child. The retained Popen handle identifies that exact owned process;
        # kill it before treating Job termination as complete cleanup.
        if child is not None and child.poll() is None:
            try: child.kill()  # retained Popen handle, including suspended assignment-failure child
            except Exception as cleanup: cleanup_errors.append("OWNED_LAUNCHER_HANDLE_KILL:"+repr(cleanup))
        if job is not None:
            try: job.terminate()
            except Exception as cleanup: cleanup_errors.append("JOB_TERMINATE:"+repr(cleanup))
    finally:
        if timer is not None: timer.cancel()
        deadline=min(clock()+120,started+720)
        if timer is not None and hasattr(timer,"join"):
            try: timer.join(timeout=max(0,deadline-clock()))
            except Exception as exc: cleanup_errors.append("HARD_WALL_TIMER_JOIN:"+repr(exc))
        active=[]
        while job is not None:
            try: active=members(int(job.handle))
            except Exception as exc:
                active=None
                cleanup_errors.append("JOB_ACTIVE_COUNT_UNKNOWN:"+repr(exc))
                break
            if not active or clock()>=deadline: break
            try: job.terminate()
            except Exception as exc: cleanup_errors.append("CLEANUP_JOB_KILL:"+repr(exc))
            sleep(.1)
        if child is not None:
            try: receipt["launcher_returncode"]=child.wait(timeout=max(0,deadline-clock()))
            except Exception as exc: cleanup_errors.append("LAUNCHER_WAIT:"+repr(exc))
        if receipt["status"]=="NATIVE_WORKER_EXITED" and receipt.get("launcher_returncode")!=0:
            receipt["status"]="NATIVE_WORKER_FAILED"
        dead=(child is None or child.poll() is not None) and active==[]
        receipt["whole_job_exit_verified"]=dead
        receipt["job_active_pids_after_cleanup"]=active
        if dead:
            released={str(path):safety.release_token_lock(path,token,attempt_id) for path in reversed(acquired)}
            receipt["token_leases_released"]=released
            if job is not None:
                try: safety.close_owned_handle(job)
                except Exception as exc: cleanup_errors.append("JOB_CLOSE:"+repr(exc))
            if hardware is not None:
                try: safety.close_owned_handle(hardware)
                except Exception as exc: cleanup_errors.append("HARDWARE_CLOSE:"+repr(exc))
        else:
            receipt["leases_retained_unresolved_worker"]=True
            receipt["owned_handles_retained_until_supervisor_exit"]=True
            RETAINED_HANDLES.extend(value for value in (job,hardware) if value is not None)
        elapsed=clock()-started
        try:
            safety.validate_manifest(proposal["source_manifest"],Path(bundle["here"]))
            receipt["source_unchanged"]=True
        except Exception as exc:
            receipt["source_unchanged"]=False
            receipt["prior_status"]=receipt["status"]
            receipt["status"]="STOP_NATIVE_SOURCE_DRIFT"
            receipt["source_error"]=repr(exc)
        reconciled=(dead and not cleanup_errors and all(receipt.get("token_leases_released",{}).values()))
        receipt["cleanup_reconciled"]=reconciled
        if dead and not reconciled:
            receipt["prior_status_before_cleanup_failure"]=receipt["status"]
            receipt["status"]="STOP_NATIVE_CLEANUP_UNRECONCILED"
        if charged:
            try: receipt["budget_result"]=ledger.record_result(attempt_id,"OUTCOME_UNKNOWN" if not dead else
                ("SUCCEEDED" if receipt["status"]=="NATIVE_WORKER_EXITED" else "FAILED"),observed_seconds=elapsed,
                receipt=str(attempt/RECEIPT))
            except Exception as exc: cleanup_errors.append("BUDGET_SETTLE:"+repr(exc))
        if cleanup_errors and receipt["cleanup_reconciled"]:
            receipt["cleanup_reconciled"]=False
            receipt["prior_status_before_accounting_failure"]=receipt["status"]
            receipt["status"]="STOP_NATIVE_ACCOUNTING_UNRECONCILED"
        receipt.update({"utc_end":utc(),"elapsed_seconds":elapsed,"cleanup_errors":cleanup_errors,
                        "loaded_attempts_ever":loaded_attempts(root,safety),"native_qualification_claimed_by_supervisor":False})
        safety.validate_owned_output(attempt/RECEIPT)
        safety.write_new(attempt/RECEIPT,receipt)
    return receipt


def supervise(proposal_path: Path,permit_path: Path,*,runtime_root=RUNTIME_ROOT):
    if any(name.split(".")[0] in {"torch","transformers","tensorflow","jax","cupy"} for name in sys.modules):
        raise RuntimeError("STOP_NATIVE_SUPERVISOR_ALREADY_IMPORTED_FRAMEWORK")
    blocker=NoFrameworkImports()
    sys.meta_path.insert(0,blocker)
    try:
        safety=_verified_safety(proposal_path)
        bundle=safety.validate_bundle(proposal_path,permit_path)
        if not sys.flags.isolated or not sys.flags.no_site or not sys.flags.dont_write_bytecode:
            raise RuntimeError("STOP_NATIVE_SUPERVISOR_STARTUP_NOT_ISOLATED")
        root=safety.validate_runtime(runtime_root)
        helpers=safety.load_verified_helpers(bundle["sources"])
        # Each dependency was pinned and verified before any execution. Parent
        # imports remain under NoFrameworkImports; checkpoint reads are absent.
        finder=safety.install_verified_packet_sources(bundle["sources"])
        safety.append_reviewed_site_paths(bundle["proposal"])
        import importlib
        budget_module=importlib.import_module("experiments.titan_quality_repair_v1.budget")
        safety.validate_ledger_output(budget_module.CANONICAL_LEDGER_ROOT)
        BudgetLedger=budget_module.BudgetLedger
        try:
            return _supervise_admitted(bundle,safety,runtime_root=root,ledger=BudgetLedger(),helpers=helpers)
        finally:
            if finder in sys.meta_path: sys.meta_path.remove(finder)
    finally:
        sys.meta_path.remove(blocker)


def main(argv=None):
    args=None
    try:
        parser=argparse.ArgumentParser()
        parser.add_argument("--proposal",required=True,type=Path)
        parser.add_argument("--operator-grant",required=True,type=Path)
        parser.add_argument("--runtime-root",type=Path,default=RUNTIME_ROOT)
        args=parser.parse_args(argv)
        result=supervise(args.proposal,args.operator_grant,runtime_root=args.runtime_root)
        print(json.dumps(result,indent=2))
        return 0 if (result["status"]=="NATIVE_WORKER_EXITED" and result["whole_job_exit_verified"]
                     and result["cleanup_reconciled"] and result["source_unchanged"]) else 1
    except BaseException as exc:
        root=RUNTIME_ROOT
        _bootstrap_output_path(root)
        root.mkdir(parents=True,exist_ok=True)
        stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path=root/f"NATIVE_BOOTSTRAP_FAILURE_{stamp}-{os.getpid()}.json"
        _bootstrap_output_path(path)
        with path.open("x",encoding="utf-8") as stream:
            json.dump({"experiment":"TITAN-QUALITY-REPAIR-V1","utc":utc(),"error":repr(exc),
                "traceback":traceback.format_exc(),"native_entry_reached":False},stream,indent=2)
        raise


if __name__=="__main__": sys.exit(main())
