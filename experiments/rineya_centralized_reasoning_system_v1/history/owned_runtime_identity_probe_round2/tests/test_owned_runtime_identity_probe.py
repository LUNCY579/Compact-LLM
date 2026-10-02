"""CPU fixtures only; no actual import experiment, model or canonical host DB."""
import copy
import contextlib
import hashlib
import json
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.rineya_centralized_reasoning_system_v1 import owned_runtime_identity_probe as probe
from experiments.rineya_centralized_reasoning_system_v1.operations import OperationsStore


def completion_fixture(root):
    """Declared fake phase bytes; these never execute framework imports."""
    source = {"commit": "a"*40, "dirty": False, "dirty_status": ""}
    go = {"job_key": probe.KEY, "epoch": "cpu-fixture", "request_hash": "b"*64,
          "binding": {"scope": probe.SCOPE}, "owner": "cpu", "fence": 1,
          "worker_identity": {"pid": 11, "creation_time": "12.0", "command_hash": "c"*64},
          "runtime_identity": {"pid": 12, "creation_time": 13.0}, "source": source,
          "review_refs": [], "launch_intent_ref": {"path": str(root/"LAUNCH_INTENT.json"), "sha256": "d"*64}}
    cfg = {"runtime": {"python": "test-only"}}
    intent = {"binding": go["binding"], "job_key": probe.KEY, "review_refs": go["review_refs"]}
    (root/"LAUNCH_INTENT.json").write_bytes(probe.canonical(intent))
    go["launch_intent_ref"]["sha256"] = probe.native.sha(root/"LAUNCH_INTENT.json")
    context = hashlib.sha256(probe.canonical(probe.authority(go))).hexdigest()
    events, clock = [], 0.0
    def add(phase, **fields):
        nonlocal clock
        events.append({"phase": phase, "authority_sha256": context, "wall_time": clock+1000,
                       "monotonic_time": clock, **fields}); clock += 2.05 if phase in {"idle_begin", "post_import_begin"} else .05
    add("idle_begin", forbidden_imports_present=[]); add("idle_end")
    for i in range(10):
        add("source_begin", index=i)
        add("source_end", index=i, source=source, verify_sources_returned=i in (0,9),
            runtime_metadata=cfg["runtime"] if i in (0,9) else None)
    state = {"cuda_initialized": False, "profile_current_thread": True, "profile_new_python_threads": True,
             "internal_C_side_effect_coverage": False, "generic_C_type_constructor_coverage": False}
    add("torch_import_begin"); add("boundary_armed", profile_current_thread=True, profile_new_python_threads=True)
    add("torch_import_end", boundary=state)
    add("transformers_package_import_begin"); add("transformers_package_import_end", boundary=state)
    add("qwen_symbol_import_begin")
    add("qwen_symbol_import_end", boundary=state,
        config_symbol="transformers.models.qwen3.configuration_qwen3.Qwen3Config",
        model_symbol="transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM",
        transitive_modules=["torch","transformers"])
    add("post_import_begin"); add("post_import_end", boundary=state)
    add("all_registered_phases_returned", model_instances_requested=0, model_binary_reads_requested=0,
        tokenizer_instances_requested=0, tensor_operations_requested=0, model_calls_requested=0,
        native_qualified=False, original_failure_cause_established=False)
    for name in ("GO.json","LAUNCH_ATTEMPT.json"):
        (root/name).write_bytes(probe.canonical(go if name=="GO.json" else {"cpu_fixture":True}))
    containment = {"authority_sha256": context, "go_sha256": probe.native.sha(root/"GO.json"),
        "pid":12, "creation_time":13.0, "query_handle_closed_before_framework":True,
        "query_handle_inheritable":False, "query_handle_access":"JOB_OBJECT_QUERY_ONLY",
        "exact_owned_limit_flags":8192, "in_exact_supervisor_job":True}
    (root/"CHILD_CONTAINMENT.json").write_bytes(probe.canonical(containment))
    (root/"CHILD_EVENTS.jsonl").write_bytes(b"".join(probe.canonical(e)+b"\n" for e in events))
    return go,cfg,events,containment


class IdentityProbeTests(unittest.TestCase):
    def test_unknown_path_wrong_hash_distinct_and_python_serialization_covered(self):
        permitted={"c:/python.exe":"a"*64}
        self.assertFalse(probe.classify_identity("C:/other.exe","a"*64,permitted)["path_registered"])
        self.assertFalse(probe.classify_identity("C:/PYTHON.exe","b"*64,permitted)["hash_matches"])
        self.assertEqual(permitted,{"c:/python.exe":"a"*64})
        for name in ("load","save","_load","_legacy_load","_save","_legacy_save"):
            self.assertTrue(probe.forbidden_call("torch.serialization",name,"call"))

    def test_hash_individual_distinct_bounds_before_reads(self):
        with tempfile.TemporaryDirectory() as td:
            a,b=Path(td)/"a.exe",Path(td)/"b.exe";a.write_bytes(b"abc");b.write_bytes(b"xy")
            small=probe.HashBudget(single=2,distinct=3)
            with self.assertRaises(probe.native.NativeBlocked):small.hash(a)
            self.assertEqual(small.read_bytes,0)
            a.write_bytes(b"ab");self.assertEqual(small.hash(a),hashlib.sha256(b"ab").hexdigest())
            with self.assertRaises(probe.native.NativeBlocked):small.hash(b)
            self.assertEqual((small.distinct_bytes,small.read_bytes),(2,2))

    def identity_fixture(self, fresh=None, failure=None):
        kernel=SimpleNamespace(OpenProcess=lambda *_:1,CloseHandle=lambda *_:True)
        def member(_p,_j,ptr):ptr._obj.value=1;return True
        kernel.IsProcessInJob=member
        before=SimpleNamespace(create_time=lambda:123.0,exe=lambda:"C:/unknown.exe",
                               memory_info=lambda:SimpleNamespace(rss=99))
        budget=probe.HashBudget()
        with patch.object(probe.native,"job_process_ids",return_value=[99]),\
             patch("psutil.Process",side_effect=[before,fresh or before]),\
             patch.object(budget,"hash",side_effect=failure,return_value="a"*64):
            return probe.inventory(SimpleNamespace(kernel=kernel,handle=1),{"runtime_file_sha256":{}},budget)["members"][0]

    def test_hash_failure_retains_resolved_path_time_member_RSS(self):
        row=self.identity_fixture(failure=PermissionError("fixture"))
        self.assertEqual((row["creation_time"],row["executable"],row["rss_bytes"]),(123.0,"C:/unknown.exe",99))
        self.assertTrue(row["exact_owned_job_member"]);self.assertIn("PermissionError",row["query_error"])

    def test_fresh_psutil_object_rejects_identity_drift(self):
        fresh=SimpleNamespace(create_time=lambda:124.0,exe=lambda:"C:/unknown.exe")
        row=self.identity_fixture(fresh=fresh)
        self.assertEqual(row["creation_time"],123.0);self.assertIn("identity drift",row["query_error"])

    def test_complete_phase_bytes_pass_without_hidden_zero_attestation(self):
        with tempfile.TemporaryDirectory() as td:
            go,cfg,_,_=completion_fixture(Path(td));result=probe.validate_child_evidence(td,go,cfg)
            self.assertEqual((result["source_snapshot_calls"],result["verify_sources_calls"],result["runtime_metadata_calls"]),(10,2,2))
            self.assertFalse(result["hidden_internal_operations_attested_zero"])

    def test_missing_duplicate_reordered_wrong_source_counts_boundary_reject(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);go,cfg,events,_=completion_fixture(root)
            changes=[lambda x:x.pop(0),lambda x:x.insert(1,copy.deepcopy(x[0])),lambda x:x.reverse(),
                lambda x:x[3].update(source={"commit":"wrong"}),
                lambda x:x[3].update(verify_sources_returned=False),
                lambda x:x[-2]["boundary"].update(profile_current_thread=False),
                lambda x:x[-1].update(model_calls_requested=True)]
            for change in changes:
                bad=copy.deepcopy(events);change(bad)
                (root/"CHILD_EVENTS.jsonl").write_bytes(b"".join(probe.canonical(e)+b"\n" for e in bad))
                with self.assertRaises(probe.native.NativeBlocked):probe.validate_child_evidence(root,go,cfg)

    def test_old_minimal_markers_missing_containment_wrong_worker_partial_duplicate_JSON_reject(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);go,cfg,_,_=completion_fixture(root)
            minimal=[{"phase":"source_end","index":i} for i in range(10)]+[
                {"phase":"qwen_symbol_import_end","cuda_initialized":False},{"phase":"all_registered_phases_returned"}]
            (root/"CHILD_EVENTS.jsonl").write_bytes(b"".join(probe.canonical(e)+b"\n" for e in minimal))
            with self.assertRaises(probe.native.NativeBlocked):probe.validate_child_evidence(root,go,cfg)
            completion_fixture(root);(root/"CHILD_CONTAINMENT.json").unlink()
            with self.assertRaises(FileNotFoundError):probe.validate_child_evidence(root,go,cfg)
            _,_,_,containment=completion_fixture(root);containment["pid"]=99
            (root/"CHILD_CONTAINMENT.json").write_bytes(probe.canonical(containment))
            with self.assertRaises(probe.native.NativeBlocked):probe.validate_child_evidence(root,go,cfg)
            completion_fixture(root);raw=(root/"CHILD_EVENTS.jsonl").read_bytes()
            for bad in (raw[:-1],raw.replace(b'"phase":"idle_begin"',b'"phase":"idle_begin","phase":"idle_begin"',1)):
                (root/"CHILD_EVENTS.jsonl").write_bytes(bad)
                with self.assertRaises(probe.native.NativeBlocked):probe.validate_child_evidence(root,go,cfg)

    def test_exact_reserved_job_required_other_lease_pending_wrong_epoch_worker_reject(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);store=OperationsStore(root/"state.sqlite3",root/"receipts")
            binding={"scope":probe.SCOPE};source={"commit":"a"*40,"dirty":False}
            lease=store.acquire_lease("other-milestone",120)
            args=SimpleNamespace(epoch="cpu-fixture",contract_sha256="f"*64)
            go={"job_key":probe.KEY,"epoch":args.epoch,"contract_sha256":args.contract_sha256,
                "owner":lease["owner"],"fence":lease["fence"],"binding":binding,"source":source,
                "request_hash":"a"*64,"worker_identity":{"pid":99,"creation_time":"1","command_hash":"b"*64}}
            try:
                with self.assertRaises(Exception):probe.require_probe_job(store,go,args)
                row=store.register_job(probe.KEY,binding,args.epoch);go["request_hash"]=row["request_hash"]
                with self.assertRaises(probe.native.NativeBlocked):probe.require_probe_job(store,go,args)
                store.ready(probe.KEY);store.reserve(probe.KEY,lease["owner"],lease["fence"],60)
                store.mark_running(probe.KEY,lease["owner"],lease["fence"],go["worker_identity"])
                with patch.object(probe,"probe_binding",return_value=binding):
                    probe.require_probe_job(store,go,args)
                    for change in ({"epoch":"wrong"},{"worker_identity":{"pid":100}},{"request_hash":"x"*64}):
                        with self.assertRaises(probe.native.NativeBlocked):probe.require_probe_job(store,{**go,**change},args)
            finally:store.close()

    def test_deadline_independent_of_blocked_sample_and_late_check_rejects(self):
        killed,release=threading.Event(),threading.Event();job=SimpleNamespace(kill=killed.set)
        def slow():
            release.wait(2);return {"rss_bytes":1,"pid_count":1,"host_available_gib":2,"disk_free_gib":5,"log_bytes":0}
        with tempfile.TemporaryDirectory() as td:
            watch=probe.ResourceWatch(job,started=time.monotonic(),output=td,seconds=.1,sample=slow);watch.start()
            try:
                self.assertTrue(killed.wait(1))
                with self.assertRaises(probe.native.NativeBlocked):watch.check()
                self.assertEqual(watch.reason,"PROBE_WALL_LIMIT")
            finally:release.set();watch.close()
        late=probe.ResourceWatch(job,started=time.monotonic()-65,output=".")
        with self.assertRaises(probe.native.NativeBlocked):late.check()

    def test_resource_watch_refuses_without_observer(self):
        killed=threading.Event();job=SimpleNamespace(kill=killed.set)
        bad=lambda:{"rss_bytes":3*probe.native.GIB,"pid_count":1,"host_available_gib":2,"disk_free_gib":5,"log_bytes":0}
        with tempfile.TemporaryDirectory() as td:
            watch=probe.ResourceWatch(job,started=time.monotonic(),output=td,seconds=2,sample=bad);watch.start()
            try:self.assertTrue(killed.wait(1));self.assertIn("RESOURCE_WATCH",watch.reason)
            finally:watch.close()

    def test_cleanup_errors_do_not_skip_other_actions(self):
        actions=[]
        def error(name):
            def fail():actions.append(name);raise OSError(name)
            return fail
        job=SimpleNamespace(close=error("job_close"));notifications=SimpleNamespace(close=error("notification_close"))
        process=SimpleNamespace(poll=lambda:0,wait=lambda **_:0);pipe=SimpleNamespace(close=error("pipe_close"))
        with patch.object(probe.native,"drain_owned_job",return_value={"owned_tree_exited":True}):
            value=probe.cleanup_runtime(job,notifications,process,(pipe,),None)
        self.assertEqual(actions,["job_close","notification_close","pipe_close"]);self.assertEqual(len(value["errors"]),3)

    def test_log_and_contract_drift_fail_before_framework_import(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"events.jsonl";path.write_bytes(b"preserved")
            with patch.object(probe,"MAX_LOG",9):
                with self.assertRaises(probe.native.NativeBlocked):probe.append(path,{"large":"x"*30})
            self.assertEqual(path.read_bytes(),b"preserved")
        with self.assertRaises(probe.native.NativeBlocked):probe.checked_contract("0"*64)
        self.assertNotIn("torch",sys.modules);self.assertNotIn("transformers",sys.modules)

    def test_lost_creation_response_and_restart_retain_full_reservation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);db=root/"state.sqlite3";source={"commit":"a"*40,"dirty":False,"dirty_status":""}
            cfg={"environment":{"selected_identity_variables":[]}}
            native_cfg={"runtime":{"executable":sys.executable},"gpu_process_registry":[]}
            args=SimpleNamespace(contract_sha256="f"*64,review=[],epoch="cpu-fixture",output=root/"results"/"first")
            with patch.object(probe,"PACKAGE",root),patch.object(probe,"host_state_path",return_value=db),\
                 patch.object(probe.native,"committed_snapshot",return_value=source),\
                 patch.object(probe.native,"verify_sources"),patch.object(probe,"checked_reviews"),\
                 patch.object(probe.native,"runtime_metadata",return_value=native_cfg["runtime"]),\
                 patch.object(probe.native,"collect_telemetry",return_value={}),\
                 patch.object(probe.native,"admission_reasons",return_value=[]),patch.object(probe.time,"sleep"),\
                 patch.object(probe.subprocess,"Popen",side_effect=OSError("CPU lost creation response")) as spawn,\
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(probe.supervise(args,cfg,native_cfg),2)
                args.output=root/"results"/"restart"
                self.assertEqual(probe.supervise(args,cfg,native_cfg),2)
                self.assertEqual(spawn.call_count,1)
            store=OperationsStore(db,root/"check_receipts")
            try:
                self.assertEqual(store.get_job(probe.KEY)["state"],"OUTCOME_UNKNOWN")
                self.assertEqual(store.budget_snapshot(args.epoch)["epoch_charged_seconds"],60)
            finally:store.close()

    def test_result_publication_failure_is_durable_unknown_not_unclosed_store(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);db=root/"state.sqlite3";source={"commit":"a"*40,"dirty":False,"dirty_status":""}
            cfg={"environment":{"selected_identity_variables":[]}}
            native_cfg={"runtime":{"executable":sys.executable},"gpu_process_registry":[]}
            args=SimpleNamespace(contract_sha256="f"*64,review=[],epoch="cpu-fixture",output=root/"results"/"publication")
            write=probe.native.write_new
            def fault(path,value):
                if Path(path).name=="PROBE_RESULT.json":raise OSError("CPU disk publication failure")
                return write(path,value)
            with patch.object(probe,"PACKAGE",root),patch.object(probe,"host_state_path",return_value=db),\
                 patch.object(probe.native,"committed_snapshot",return_value=source),\
                 patch.object(probe.native,"verify_sources"),patch.object(probe,"checked_reviews"),\
                 patch.object(probe.native,"runtime_metadata",return_value=native_cfg["runtime"]),\
                 patch.object(probe.native,"collect_telemetry",return_value={}),\
                 patch.object(probe.native,"admission_reasons",return_value=[]),patch.object(probe.time,"sleep"),\
                 patch.object(probe.subprocess,"Popen",side_effect=OSError("CPU lost response")),\
                 patch.object(probe.native,"write_new",side_effect=fault),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(probe.supervise(args,cfg,native_cfg),2)
            error=probe.native.read_json(args.output/"FINALIZATION_ERROR.json")
            self.assertEqual(error["status"],"OUTCOME_UNKNOWN")
            store=OperationsStore(db,root/"check_receipts")
            try:
                self.assertEqual(store.get_job(probe.KEY)["state"],"OUTCOME_UNKNOWN")
                self.assertEqual(store.budget_snapshot(args.epoch)["epoch_charged_seconds"],60)
            finally:store.close()

    def test_admitted_cleanup_failure_persists_unknown_full_cost_without_model(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);db=root/"state.sqlite3";source={"commit":"a"*40,"dirty":False,"dirty_status":""}
            cfg={"environment":{"selected_identity_variables":[]}}
            native_cfg={"runtime":{"executable":sys.executable},"gpu_process_registry":[]}
            args=SimpleNamespace(contract_sha256="f"*64,review=[],epoch="cpu-fixture",output=root/"results"/"cleanup")
            process=SimpleNamespace(pid=99,returncode=0,poll=lambda:0,wait=lambda **_:0,kill=lambda:None)
            def failclose():raise OSError("CPU job close failure")
            job=SimpleNamespace(close=failclose,kill=lambda:None)
            notify=SimpleNamespace(poll=lambda:[],close=lambda:None)
            watch=SimpleNamespace(start=lambda:None,check=lambda:None,close=lambda:None,reason=None,peak=1,records=[])
            member={"pid":100,"creation_time":101.0,"executable":sys._base_executable}
            psprocess=SimpleNamespace(create_time=lambda:100.0,children=lambda **_:[])
            raw={"job_pid_list":[99],"members":[],"observed_rss_bytes":1}
            patches=[patch.object(probe,"PACKAGE",root),patch.object(probe,"host_state_path",return_value=db),
                patch.object(probe.native,"committed_snapshot",return_value=source),
                patch.object(probe.native,"verify_sources"),patch.object(probe,"checked_reviews"),
                patch.object(probe.native,"runtime_metadata",return_value=native_cfg["runtime"]),
                patch.object(probe.native,"collect_telemetry",return_value={}),
                patch.object(probe.native,"admission_reasons",return_value=[]),patch.object(probe.time,"sleep"),
                patch.object(probe.subprocess,"Popen",return_value=process),patch("psutil.Process",return_value=psprocess),
                patch("experiments.opallo_native_collector_v1.supervisor.WindowsJob",return_value=job),
                patch.object(probe,"JobNotifications",return_value=notify),patch.object(probe,"ResourceWatch",return_value=watch),
                patch.object(probe,"inventory",return_value=raw),patch.object(probe.native,"job_process_ids",return_value=[99]),
                patch.object(probe.native,"owned_job_rss_snapshot",return_value={"members":[member]}),
                patch.object(probe.native,"resume_suspended_primary_thread"),
                patch.object(probe.native,"duplicate_owned_job_query_handle",side_effect=RuntimeError("CPU stop before GO")),
                patch.object(probe.native,"drain_owned_job",return_value={"owned_tree_exited":True,"remaining_owned_pids":[]})]
            with contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
                for replacement in patches:stack.enter_context(replacement)
                self.assertEqual(probe.supervise(args,cfg,native_cfg),2)
            result=probe.native.read_json(args.output/"PROBE_RESULT.json")
            self.assertEqual(result["status"],"OUTCOME_UNKNOWN");self.assertTrue(result["cleanup_errors"])
            store=OperationsStore(db,root/"check_receipts")
            try:
                self.assertEqual(store.get_job(probe.KEY)["state"],"OUTCOME_UNKNOWN")
                self.assertEqual(store.budget_snapshot(args.epoch)["epoch_charged_seconds"],60)
            finally:store.close()

    @unittest.skipUnless(os.name=="nt","actual CPU-only exact-Job independent deadline")
    def test_actual_exact_job_deadline_kills_while_observer_sample_blocks(self):
        from experiments.opallo_native_collector_v1.supervisor import WindowsJob
        with tempfile.TemporaryDirectory() as td:
            process=subprocess.Popen([sys.executable,"-B","-c","import time;time.sleep(30)"],
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|4)
            job=None;watch=None;release=threading.Event()
            try:
                job=WindowsJob(process);probe.native.resume_suspended_primary_thread(process)
                until=time.monotonic()+3
                while len(probe.native.job_process_ids(job))<2 and time.monotonic()<until:time.sleep(.02)
                self.assertGreaterEqual(len(probe.native.job_process_ids(job)),2)
                def blocked():
                    release.wait(2);return {"rss_bytes":1,"pid_count":2,"host_available_gib":2,"disk_free_gib":5,"log_bytes":0}
                watch=probe.ResourceWatch(job,started=time.monotonic(),output=td,seconds=.15,sample=blocked);watch.start()
                process.wait(timeout=2);self.assertEqual(watch.reason,"PROBE_WALL_LIMIT")
                self.assertTrue(probe.native.drain_owned_job(job)["owned_tree_exited"])
            finally:
                release.set()
                if watch:watch.close()
                if job:job.close()
                if process.poll() is None:process.kill();process.wait(timeout=5)
                for pipe in (process.stdout,process.stderr):
                    if pipe:pipe.close()

    @unittest.skipUnless(os.name=="nt","actual CPU-only suspended exact-Job fatal-boundary fixture")
    def test_fatal_refusal_cannot_be_swallowed_main_newthread_Ccall_or_failed_publication(self):
        from experiments.opallo_native_collector_v1.supervisor import WindowsJob
        for mode in ("main","thread","publication_failure","c_call","serialization"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as td:
                root=Path(td);marker,refusal=root/"forbidden_effect",root/"refusal.jsonl"
                script=(
                    "import os,sys,threading\nfrom pathlib import Path\n"
                    "from experiments.rineya_centralized_reasoning_system_v1 import owned_runtime_identity_probe as p\n"
                    f"marker=Path({str(marker)!r}); refusal=Path({str(refusal)!r}); mode={mode!r}\n"
                    "assert not any(x in sys.modules for x in ('torch','transformers','tokenizers','safetensors'))\n"
                    "ns={'__name__':'torch.serialization' if mode=='serialization' else 'torch.cuda','marker':marker}\n"
                    "exec('def is_available():\\n marker.write_text(\\\"first\\\")\\ndef init():\\n marker.write_text(\\\"second\\\")\\ndef load():\\n marker.write_text(\\\"load\\\")',ns)\n"
                    "if mode=='publication_failure':\n def fail(*a): raise OSError('CPU publication fixture')\n p.append=fail\n"
                    "if mode=='c_call': p.forbidden_call=lambda module,name,event:event=='c_call' and name=='mkdir'\n"
                    "p.import_boundary(refusal)\n"
                    "def invoke():\n try:\n  if mode=='c_call': os.mkdir(marker)\n  elif mode=='serialization': ns['load']()\n  else: ns['is_available']()\n except BaseException: pass\n ns['init']()\n marker.write_text('continued')\n"
                    "if mode=='thread':\n t=threading.Thread(target=invoke); t.start(); t.join()\nelse: invoke()\n")
                process=subprocess.Popen([sys.executable,"-B","-c",script],cwd=probe.ROOT,
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|4)
                job=None
                try:
                    job=WindowsJob(process);probe.native.resume_suspended_primary_thread(process)
                    stdout,stderr=process.communicate(timeout=8)
                    self.assertEqual(process.returncode,probe.BOUNDARY_EXIT_CODE,stderr.decode(errors="replace"))
                    self.assertFalse(marker.exists())
                    if mode!="publication_failure":self.assertTrue(json.loads(refusal.read_text())["body_execution_blocked"])
                    self.assertTrue(probe.native.drain_owned_job(job)["owned_tree_exited"])
                finally:
                    if job:job.close()
                    if process.poll() is None:process.kill();process.wait(timeout=5)
                    for pipe in (process.stdout,process.stderr):
                        if pipe:pipe.close()


if __name__=="__main__":unittest.main()
