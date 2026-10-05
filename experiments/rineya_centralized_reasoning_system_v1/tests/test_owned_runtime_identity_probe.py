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


class DesktopBoundaryCorrectionTests(unittest.TestCase):
    """Exercise real collector branches through deterministic process adapters."""
    def collect(self, *, role="codex", depth=1, outside="live", orphan=False):
        import psutil
        cfg=probe.native.read_json(probe.CONTRACT)
        profile=probe.checked_desktop_profile(cfg,probe.native.read_json(probe.native.CONTRACT))
        row=next(r for r in profile["updated_rows"] if r["role_id"]==role)
        codex=next(r for r in profile["updated_rows"] if r["role_id"]=="codex")
        principal={"sid_sha256":"a"*64,"session":1}
        records={100+i:{"ctime":1000.0-i,"exe":row["path"] if i==0 else codex["path"],
                        "ppid":100+i+1 if i+1<depth else 9000} for i in range(depth)}
        if orphan: records[100]["ppid"]=8000
        counts={}
        def provider(pid):
            counts[pid]=counts.get(pid,0)+1
            if pid==9000:
                if outside=="denied": raise psutil.AccessDenied(pid)
                if outside=="absent" or (outside=="reuse_absent" and counts[pid]==1): raise psutil.NoSuchProcess(pid)
                return SimpleNamespace(create_time=lambda:1.0 if outside!="reuse_live" or counts[pid]==1 else 2.0,
                                       exe=lambda:r"C:\Windows\explorer.exe",ppid=lambda:1)
            if pid not in records: raise psutil.NoSuchProcess(pid)
            record=records[pid]
            return SimpleNamespace(create_time=lambda:record["ctime"],exe=lambda:record["exe"],ppid=lambda:record["ppid"])
        with patch("psutil.Process",side_effect=provider),\
             patch.object(probe,"stable_desktop_hash",side_effect=lambda _p,d:d),\
             patch.object(probe,"process_principal",return_value=principal):
            witness=probe.collect_desktop_context(profile,row,100,principal)
        return profile,row,principal,witness

    def test_actual_collector_genuine_depth7_and8_boundaries(self):
        for depth in (7,8):
            with self.subTest(depth=depth):
                _,_,_,witness=self.collect(depth=depth)
                self.assertEqual(len(witness["chain"]),depth)
                self.assertTrue(witness["owner_boundary"]["attained"])
                self.assertEqual(witness["owner_boundary"]["reason"],"OBSERVED_NONFAMILY_PARENT")

    def test_actual_collector_depth8_continuation_and_deep_Swift_refuse(self):
        for role in ("codex","swift"):
            with self.subTest(role=role),self.assertRaisesRegex(probe.native.NativeBlocked,"depth exhausted"):
                self.collect(role=role,depth=9)

    def test_missing_false_and_inconsistent_boundary_witness_refuse(self):
        profile,row,principal,witness=self.collect(depth=8)
        cases=[]
        altered=copy.deepcopy(witness);altered.pop("owner_boundary");cases.append(altered)
        for field,value in (("attained",False),("reason","DEPTH_EXHAUSTED"),("owner_pid",999),("owner_ppid",999),
                             ("relation_rechecked",False)):
            altered=copy.deepcopy(witness);altered["owner_boundary"][field]=value;cases.append(altered)
        altered=copy.deepcopy(witness);altered["owner_boundary"]["external_parent"]["executable"]=row["path"];cases.append(altered)
        for altered in cases:
            with self.assertRaises(probe.native.NativeBlocked):
                probe.validate_desktop_context(profile,row,altered,principal,now=altered["completed_at"])

    def test_actual_Codex_and_Swift_absent_external_parent_boundaries(self):
        for role,depth in (("codex",1),("swift",2)):
            with self.subTest(role=role):
                _,_,_,witness=self.collect(role=role,depth=depth,outside="absent")
                boundary=witness["owner_boundary"]
                self.assertEqual(boundary["reason"],"OBSERVED_ABSENT_EXTERNAL_PARENT")
                self.assertTrue(boundary["external_parent"]["absence_rechecked"])
                self.assertTrue(all(member["stable_after_checks"] for member in witness["chain"]))

    def test_actual_AccessDenied_parent_reuse_and_orphanhelper_refuse(self):
        import psutil
        with self.assertRaises(psutil.AccessDenied): self.collect(outside="denied")
        for outside in ("reuse_absent","reuse_live"):
            with self.subTest(outside=outside),self.assertRaises(probe.native.NativeBlocked): self.collect(outside=outside)
        with self.assertRaises(psutil.NoSuchProcess): self.collect(role="swift",orphan=True)
        self.assertNotIn("torch",sys.modules);self.assertNotIn("transformers",sys.modules)


class DesktopProfileTests(unittest.TestCase):
    """Synthetic validator positives; live application intent remains unmeasured."""
    def fixture(self):
        cfg = probe.native.read_json(probe.CONTRACT)
        original = probe.native.read_json(probe.native.CONTRACT)
        profile = probe.checked_desktop_profile(cfg, original)
        row = next(r for r in profile["updated_rows"] if r["role_id"] == "swift")
        owner = {"sid_sha256": "a"*64, "session": 1}
        codex = next(r for r in profile["updated_rows"] if r["role_id"] == "codex")
        chain = [{"pid": 12, "ppid": 11, "creation_time": 12.0, "executable": row["path"], "sha256": row["sha256"],
                  "principal": owner, "stable_after_checks": True},
                 {"pid": 11, "ppid": 1, "creation_time": 11.0, "executable": codex["path"], "sha256": codex["sha256"],
                  "principal": owner, "stable_after_checks": True}]
        witness = {"activity": "PERMITTED_COOPERATIVE_DESKTOP_CONTEXT", "profile_sha256": hashlib.sha256(probe.canonical(profile)).hexdigest(),
                   "authority_sha256": probe.DESKTOP_AUTHORITY_SHA, "began_at": 100.0, "completed_at": 101.0, "chain": chain,
                   "owner_boundary":{"attained":True,"reason":"OBSERVED_ABSENT_EXTERNAL_PARENT","owner_pid":11,
                                     "owner_ppid":1,"relation_rechecked":True,
                                     "external_parent":{"pid":1,"state":"ABSENT","absence_rechecked":True}}}
        return cfg, original, profile, row, owner, witness

    def test_exact_profile_context_and_original_registry_preserved(self):
        cfg, original, profile, row, owner, witness = self.fixture()
        before = copy.deepcopy(original)
        self.assertTrue(probe.validate_desktop_context(profile, row, witness, owner, now=102.0))
        self.assertEqual(original, before)
        self.assertEqual(hashlib.sha256(probe.canonical(probe.registry(original))).hexdigest(), cfg["registry_sha256"])
        self.assertEqual(probe.KEY, "CRS-OWNED-IDENTITY-002")
        self.assertEqual((cfg["limits"]["runs"], cfg["limits"]["wall_seconds"]), (1, 60))
        self.assertFalse(cfg["limits"]["automatic_retry"])
        self.assertFalse(profile["context_policy"]["actual_noncompute_workload_proved"])
        self.assertEqual(profile["context_policy"]["hidden_inprocess_GPU_operations"], "UNMEASURED")
        for source, digest in cfg["source_sha256"].items(): self.assertEqual(probe.native.sha(probe.ROOT/source), digest)
        self.assertEqual(probe.native.sha(probe.native.CONTRACT), cfg["preserved_native"]["native_contract_sha256"])

    def test_missing_unknown_compute_stale_owner_session_PID_parent_and_provenance_refuse(self):
        _, _, profile, row, owner, witness = self.fixture()
        bad = []
        for activity in (None, "UNKNOWN", "GPU_COMPUTE"):
            item = copy.deepcopy(witness); item["activity"] = activity; bad.append(item)
        for field, value in (("began_at", 94.0), ("began_at", 103.0), ("completed_at", 106.0),
                             ("profile_sha256", "0"*64), ("authority_sha256", "0"*64)):
            item = copy.deepcopy(witness); item[field] = value; bad.append(item)
        for index, field, value in ((0,"principal",{"sid_sha256":"b"*64,"session":1}),
                                    (0,"principal",{"sid_sha256":"a"*64,"session":2}),
                                    (0,"stable_after_checks",False), (0,"ppid",999),
                                    (1,"creation_time",13.0), (1,"pid",12),
                                    (1,"sha256","0"*64), (1,"executable",r"C:\unknown.exe")):
            item = copy.deepcopy(witness); item["chain"][index][field] = value; bad.append(item)
        item = copy.deepcopy(witness); item["chain"] = item["chain"][:1]; bad.append(item)
        for item in bad:
            with self.subTest(item=item), self.assertRaises(probe.native.NativeBlocked):
                probe.validate_desktop_context(profile, row, item, owner, now=102.0)
        for principal in ({}, {"sid_sha256":"a"*64,"session":0}):
            with self.assertRaises(probe.native.NativeBlocked): probe.validate_desktop_context(profile,row,witness,principal,now=102)

    def test_profile_duplicates_wildcards_hash_authority_registry_and_scope_drift_refuse(self):
        cfg, original, profile, _, _, _ = self.fixture()
        def load(altered):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                for relative in (cfg["desktop_profile"]["path"], profile["authority"]["path"], profile["context_disposition"]["path"]):
                    target = root/relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes((probe.PACKAGE/relative).read_bytes())
                path = root/cfg["desktop_profile"]["path"]; path.write_bytes(probe.canonical(altered))
                changed = copy.deepcopy(cfg); changed["desktop_profile"]["sha256"] = probe.native.sha(path)
                with patch.object(probe,"PACKAGE",root): probe.checked_desktop_profile(changed, original)
        altered = copy.deepcopy(profile); altered["updated_rows"].append(altered["updated_rows"][0])
        cases = [altered]
        for field, value in (("path",r"C:\*\ChatGPT.exe"),("path",r"C:/wrong/ChatGPT.exe"),("sha256","bad")):
            altered = copy.deepcopy(profile); altered["updated_rows"][0][field] = value; cases.append(altered)
        for field, value in (("original_owned_child_registry_sha256","0"*64),("probe_key","NEW"),
                             ("original_desktop_registry",[])):
            altered=copy.deepcopy(profile); altered[field]=value; cases.append(altered)
        altered=copy.deepcopy(profile); altered["authority"]["sha256"]="0"*64; cases.append(altered)
        for altered in cases:
            with self.assertRaises(probe.native.NativeBlocked): load(altered)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"duplicate.json"; path.write_text('{"a":1,"a":2}')
            with self.assertRaises(probe.native.NativeBlocked): probe.unique_json(path)
            exe=Path(td)/"test.exe"; exe.write_bytes(b"exact")
            self.assertEqual(probe.stable_desktop_hash(exe,hashlib.sha256(b"exact").hexdigest()),hashlib.sha256(b"exact").hexdigest())
            with self.assertRaises(probe.native.NativeBlocked): probe.stable_desktop_hash(exe,"0"*64)
        with self.assertRaises(probe.native.NativeBlocked): probe.checked_contract("0"*64)
        actual_sha=probe.native.sha
        for target in (Path(probe.native.__file__), probe.native.CONTRACT, Path(probe.__file__),
                       probe.PACKAGE/profile["authority"]["path"], probe.PACKAGE/cfg["desktop_profile"]["path"]):
            def drift(path): return "0"*64 if Path(path)==target else actual_sha(path)
            with patch.object(probe.native,"sha",side_effect=drift), self.assertRaises(probe.native.NativeBlocked):
                probe.checked_contract(actual_sha(probe.CONTRACT))
        read=probe.native.read_json
        for field,value in (("runs",2),("wall_seconds",61)):
            altered=copy.deepcopy(cfg);altered["limits"][field]=value
            with patch.object(probe.native,"read_json",side_effect=lambda path:altered if Path(path)==probe.CONTRACT else read(path)),\
                 self.assertRaises(probe.native.NativeBlocked): probe.checked_contract(actual_sha(probe.CONTRACT))

    def test_wrapper_preserves_unknown_compute_and_original_native_configuration(self):
        cfg, original, profile, row, owner, witness = self.fixture()
        now = time.time()
        def telemetry():
            return {"sampled_at":now,"gpu_uuid":original["gpu_uuid"],"host_available_gib":8,"gpu_free_gib":6,
                    "disk_free_gib":10,"process_inventory_available":True,"unresolved_gpu_pids":[],"competing_gpu_pids":[],
                    "unclassified_gpu_pids":[12,99],"processes":[{"pid":12,"name":row["path"]},{"pid":99,"name":r"C:\Chrome.exe"}]}
        before=copy.deepcopy(original)
        with patch.object(probe.native,"collect_telemetry",side_effect=lambda *_a,**_k:telemetry()),\
             patch.object(probe,"process_principal",return_value=owner),\
             patch.object(probe,"collect_desktop_context",return_value=witness):
            actual=probe.collect_probe_telemetry(probe.PACKAGE,cfg,original)
        self.assertEqual(actual["unclassified_gpu_pids"],[99]); self.assertEqual(original,before)
        self.assertTrue(probe.native.admission_reasons(original,actual,current_time=now))
        for field,value in (("sampled_at",now-6),("sampled_at",now+1),("host_available_gib",0),("gpu_free_gib",0),("disk_free_gib",0)):
            altered=telemetry();altered["unclassified_gpu_pids"]=[];altered[field]=value
            self.assertTrue(probe.native.admission_reasons(original,altered,current_time=now))
        with patch.object(probe.native,"collect_telemetry",side_effect=lambda *_a,**_k:telemetry()),\
             patch.object(probe,"process_principal",return_value=owner),\
             patch.object(probe,"collect_desktop_context",side_effect=probe.native.NativeBlocked("unknown context")):
            actual=probe.collect_probe_telemetry(probe.PACKAGE,cfg,original)
        self.assertIn(12,actual["unclassified_gpu_pids"])
        competing=telemetry();competing["competing_gpu_pids"]=[12]
        with patch.object(probe.native,"collect_telemetry",return_value=competing),\
             patch.object(probe,"process_principal",return_value=owner),\
             patch.object(probe,"collect_desktop_context",side_effect=AssertionError("compute must reject before context")):
            actual=probe.collect_probe_telemetry(probe.PACKAGE,cfg,original)
        self.assertIn(12,actual["competing_gpu_pids"]);self.assertIn(12,actual["unclassified_gpu_pids"])

    def test_actual_selected_application_hash_cost_no_GPU_or_framework(self):
        _,_,profile,_,_,_=self.fixture()
        began=time.monotonic(); count=0
        for row in profile["updated_rows"]:
            for path,digest in {row["path"]:row["sha256"],**row["dependency_sha256"]}.items():
                count += Path(path).stat().st_size; probe.stable_desktop_hash(path,digest)
        print(json.dumps({"static_selected_identity_bytes":count,"seconds":time.monotonic()-began,
                          "live_context_or_GPU_feasibility_measured":False}),flush=True)
        self.assertNotIn("torch",sys.modules);self.assertNotIn("transformers",sys.modules)


class IdentityProbeTests(unittest.TestCase):
    def test_unknown_path_wrong_hash_distinct_and_python_serialization_covered(self):
        permitted={"c:/python.exe":"a"*64}
        self.assertFalse(probe.classify_identity("C:/other.exe","a"*64,permitted)["path_registered"])
        self.assertFalse(probe.classify_identity("C:/PYTHON.exe","b"*64,permitted)["hash_matches"])
        self.assertEqual(permitted,{"c:/python.exe":"a"*64})
        for name in ("load","save","_load","_legacy_load","_save","_legacy_save"):
            self.assertTrue(probe.forbidden_call("torch.serialization",name,"call"))

    def test_final_exact_reviewer_roles_reject_builder_aliases_unknown_and_duplicates(self):
        with tempfile.TemporaryDirectory() as td:
            paths=[Path(td)/"science.json",Path(td)/"operations.json"];binding={"scope":probe.SCOPE}
            def reviews(science,operations="/root/exception_ops",science_role="science"):
                for path,role,name in zip(paths,(science_role,"operations"),(science,operations)):
                    path.write_bytes(probe.canonical({"status":"APPROVED_FOR_OWNED_IDENTITY_PROBE",
                        "binding":binding,"role":role,"reviewer":name}))
            roles={"science":"/root/exception_science","operations":"/root/exception_ops"}
            for name in ("scientific_audit","\\root\\scientific_audit","/root/scientific_audit/",
                         " /ROOT/ SCIENTIFIC_AUDIT / ","root","/root/","runtime_audit","data_audit","exception_builder",
                         "/root/exception_builder/","unknown","/other/exception_science","/root/exception_ops"):
                reviews(name)
                with self.assertRaises(probe.native.NativeBlocked):probe.checked_reviews(paths,binding,roles)
            reviews("exception_science"," /ROOT\\EXCEPTION_OPS/ ");probe.checked_reviews(paths,binding,roles)
            reviews("/root/exception_science/","exception_science")
            with self.assertRaises(probe.native.NativeBlocked):probe.checked_reviews(paths,binding,roles)
            reviews("/root/exception_science",science_role="operations")
            with self.assertRaises(probe.native.NativeBlocked):probe.checked_reviews(paths,binding,roles)
            reviews("/root/exception_science")
            with self.assertRaises(probe.native.NativeBlocked):probe.checked_reviews(paths,binding,
                {"science":"/root/exception_ops","operations":"/root/exception_science"})
            with self.assertRaises(probe.native.NativeBlocked):probe.checked_reviews(paths,binding,
                {"science":"/root/runtime_audit","operations":"/root/data_audit"})

    def test_actual_pinned_psutil_path_refusal_and_string_success(self):
        """Actual psutil API only; these disk values are not host admission."""
        import psutil
        self.assertEqual(os.name,'nt');self.assertEqual(psutil.__version__,'7.2.2')
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)
            with self.assertRaises(TypeError):psutil.disk_usage(path)
            actual=psutil.disk_usage(str(path))
            self.assertGreater(actual.total,0);self.assertTrue(0 <= actual.free <= actual.total)
        source=Path(probe.__file__).read_text()
        self.assertIn('psutil.disk_usage(str(self.output))',source)
        self.assertIn('psutil.disk_usage(str(output))',source)
        self.assertNotIn('psutil.disk_usage(self.output)',source)
        self.assertNotIn('psutil.disk_usage(output)',source)
        self.assertEqual(probe.KEY,'CRS-OWNED-IDENTITY-002')

    def test_final_publication_fault_reports_known_state_and_idempotent_no_worker_recovery(self):
        for mode in ("before_release","after_release","lost_write_response"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as td:
                root=Path(td);clock=[10.0];store=OperationsStore(root/"state.sqlite3",root/"receipts",clock=lambda:clock[0])
                binding={"scope":probe.SCOPE};worker={"pid":11,"creation_time":"1","command_hash":"a"*64}
                try:
                    row=store.register_job(probe.KEY,binding,"cpu-fixture");store.ready(probe.KEY)
                    lease=store.acquire_lease("cpu-owner",120);store.reserve(probe.KEY,lease["owner"],lease["fence"],60)
                    store.mark_running(probe.KEY,lease["owner"],lease["fence"],worker);clock[0]=20.0
                    probe.native.write_new(root/"PROBE_RESULT.json",{"status":"SUCCEEDED","binding":binding,
                        "accounting_finalization_in_this_record":False,"native_qualified":False})
                    proof={"observer":"CPU_FINALIZATION_FIXTURE","request_hash":row["request_hash"],
                        "owner":lease["owner"],"fence":lease["fence"],"elapsed_native_seconds":2.6,
                        "worker_identity":worker,"process_exited":True,"exit_observed_at":20.0,
                        "owned_tree_exited":True,"owned_tree_exit":{"owned_tree_exited":True}}
                    ref=store.write_receipt("native_outcome_observation",probe.KEY,proof)
                    store.reconcile(probe.KEY,"SUCCEEDED",2.6,{"native_outcome_proof_ref":ref,
                        "probe_result_sha256":probe.native.sha(root/"PROBE_RESULT.json")},owner=lease["owner"],fence=lease["fence"])
                    if mode!="before_release":store.release_lease(lease["owner"],lease["fence"])
                    original=probe.native.write_new
                    def fault(path,payload):
                        if Path(path).name=="FINALIZATION_RECEIPT.json":
                            if mode=="lost_write_response":original(path,payload)
                            raise OSError("CPU late publication response")
                        return original(path,payload)
                    if mode=="before_release":failure=OSError("CPU release failed before mutation")
                    else:
                        with patch.object(probe.native,"write_new",side_effect=fault):
                            with self.assertRaises(OSError) as caught:probe.recover_finalization(root,store,binding)
                        failure=caught.exception
                    record=probe.record_finalization_failure(root,store,binding,failure,reserved=True)
                    self.assertEqual(record["status"],"PUBLICATION_FAILED_KNOWN_SCHEDULER_OUTCOME")
                    self.assertEqual(record["scheduler_state"],"SUCCEEDED");self.assertEqual(record["actual_charged_seconds"],2.6)
                    self.assertFalse(record["reservation_retained"])
                    self.assertEqual(record["lease_status"],"LIVE" if mode=="before_release" else "RECONCILED")
                    with patch.object(probe.subprocess,"Popen",side_effect=AssertionError("recovery must never dispatch")):
                        recovered=probe.recover_finalization(root,store,binding)
                        again=probe.recover_finalization(root,store,binding)
                    self.assertEqual(recovered,again);self.assertEqual(recovered["recovery_dispatches"],0)
                    self.assertEqual(store.budget_snapshot("cpu-fixture")["epoch_charged_seconds"],2.6)
                    (root/"PROBE_RESULT.json").write_bytes(b"{}")
                    with self.assertRaises(probe.native.NativeBlocked):probe.recover_finalization(root,store,binding)
                finally:store.close()

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
            script="import subprocess,sys,time\nsubprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)'])\ntime.sleep(30)"
            process=subprocess.Popen([sys.executable,"-B","-c",script],
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
