"""No-model fixtures: actual bounded stdlib Job plus synthetic refusal controls."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments.rineya_centralized_reasoning_system_v1 import cpu_native_load as cpu
from experiments.rineya_centralized_reasoning_system_v1.operations import OperationsStore,OperationsError

class CPUSourceRound2Tests(unittest.TestCase):
    def completion(self,root):
        cfg=cpu.native.read_json(cpu.CONTRACT);binding={"scope":cpu.SCOPE};names=cfg["tensor_schedule"]
        result={"binding":binding,"status":"CPU_LOAD_ONLY_COMPLETED","forwards":0,"tokenizers":0,"optimizer_calls":0,
                "serialized_tensors":310,"unique_parameters":596049920,"tensor_sha256":{n:"a"*64 for n in names},
                "device":"cpu","dtype":"bfloat16","CUDA_initialized":False,"asset_sha256":cfg["asset_sha256"],"tied_storage_identity":True,
                "loader":{"cpu_model_allocated":True,"GPU_copy_requested":False,"serialized_tensors":310,
                          "read_buffer_bytes":cpu.CHUNK,"digest_span_bytes_max":cpu.CHUNK},
                "allocation":{"unique_parameters":596049920,"derived_nonpersistent_buffers":[{"name":"model.rotary_emb.inv_freq","device":"cpu","dtype":"float32","bytes":256,"values_initialized_or_qualified":False}]}}
        events=[{"phase":phase,"binding":binding} for phase in ("import_begin","import_end","model_construct_begin","model_construct_end","cpu_allocate_begin","cpu_allocate_end","checkpoint_load_begin")]
        for name in names:events.extend([{"phase":"tensor_copy_begin","binding":binding,"name":name},{"phase":"tensor_copy_end","binding":binding,"name":name,"sha256":"a"*64}])
        events.extend([{"phase":phase,"binding":binding} for phase in ("checkpoint_load_end","loaded_verification_end")])
        return cfg,{"binding":binding},result,events
    def publish(self,root,result,events):
        (root/"CPU_LOAD_RESULT.json").write_bytes(cpu.canonical(result))
        (root/"EVENTS.jsonl").write_bytes(b"".join(cpu.canonical(e)+b"\n" for e in events))
    def test_exact310_schedule_repeat_missing_order_bool_and_device_refuse(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg,go,result,events=self.completion(root);self.publish(root,result,events)
            cpu.validate_completion(root,go,cfg)
            cases=[]
            repeated=copy.deepcopy(events)
            for pair in range(310):repeated[7+2*pair]["name"]=repeated[8+2*pair]["name"]=cfg["tensor_schedule"][0]
            cases.append((result,repeated));cases.append((result,events[:7]+events[9:]))
            shuffled=copy.deepcopy(events);shuffled[7:11]=shuffled[9:11]+shuffled[7:9];cases.append((result,shuffled))
            for field,value in (("forwards",False),("tokenizers",False),("optimizer_calls",False),("CUDA_initialized",True),("device","cuda")):
                altered=copy.deepcopy(result);altered[field]=value;cases.append((altered,events))
            for bad,journal in cases:
                self.publish(root,bad,journal)
                with self.assertRaises(cpu.native.NativeBlocked):cpu.validate_completion(root,go,cfg)
    def test_final_exit_deadline_resource_and_unknown_outward_gate(self):
        arguments={"completion":{},"exit_proof":{"owned_tree_exited":True},"elapsed":119,"expired":False,"resource_reason":None}
        self.assertEqual(cpu.final_success_gate("SUCCEEDED",**arguments),"SUCCEEDED")
        for field,value in (("elapsed",120.1),("expired",True),("resource_reason","RSS"),("completion",None)):
            self.assertEqual(cpu.final_success_gate("SUCCEEDED",**{**arguments,field:value}),"FAILED")
        self.assertEqual(cpu.final_success_gate("SUCCEEDED",**{**arguments,"exit_proof":None}),"OUTCOME_UNKNOWN")
    def test_import_definition_metadata_and_never_device_init_callbacks(self):
        context={"import_phase":True,"definition_source_files":{"pinned"},"cuda_main_source":"pinned"}
        def allowed(name,state=False,flags=3,callback=None,phase=True):
            return cpu.allowed_cuda_import_call("torch.cuda",name,"call",state=state,filename="pinned",code_flags=flags,callback=callback,context={**context,"import_phase":phase})
        self.assertTrue(allowed("<module>",state=None,flags=0));self.assertTrue(allowed("CudaError",flags=0))
        self.assertTrue(allowed("_is_compiled"));self.assertTrue(allowed("is_initialized"))
        self.assertTrue(allowed("_lazy_call",callback="torch.cuda._check_capability"))
        for name in ("is_available","_lazy_init","init","device_count","synchronize","_check_capability"):
            self.assertFalse(allowed(name));self.assertTrue(cpu.forbidden_call("torch.cuda",name,"call"))
        self.assertFalse(allowed("_lazy_call",state=True,callback="torch.cuda._check_capability"))
        self.assertFalse(allowed("_lazy_call",callback="unknown.callback"));self.assertFalse(allowed("_is_compiled",phase=False))
    def test_actual_known_suspended_preJob_failure_handle_cleanup(self):
        process=subprocess.Popen([sys.executable,"-B","-c","import time;time.sleep(30)"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|4)
        try:
            known=cpu.known_launcher(process)
            proof=cpu.cleanup_suspended_launcher(process,known,resumed=False)
            self.assertTrue(proof["owned_tree_exited"]);self.assertIsNotNone(process.poll())
            print(json.dumps({"fixture":"KNOWN_POPEN_SUSPENDED_PREJOB_FAILURE","known":known,"exit_proof":proof,"model_imports":0}),flush=True)
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=3)
    def test_missing_known_handle_preserves_unknown_no_PID_guess(self):
        fake=SimpleNamespace(kill=lambda:self.fail("unknown handle must not be killed"))
        with self.assertRaises(cpu.native.NativeBlocked):cpu.cleanup_suspended_launcher(fake,None,resumed=False)
        with self.assertRaises(cpu.native.NativeBlocked):cpu.cleanup_suspended_launcher(fake,{"pid":999},resumed=True)
        self.assertNotIn("torch",sys.modules);self.assertNotIn("transformers",sys.modules)


class CPULoadTests(unittest.TestCase):
    def test_cpu_scope_stale_RAM_disk_RSS_wrongdevice_and_flags_refuse(self):
        good={"device":"cpu","scope":cpu.SCOPE,"sampled_at":100,"host_available_gib":4,"disk_free_gib":5}
        self.assertEqual(cpu.admission_reasons(good,now=101),[])
        for field,value in (("device","cuda"),("scope","GPU"),("sampled_at",90),("sampled_at",102),
                            ("host_available_gib",3),("disk_free_gib",3),("sampled_at",float("nan"))):
            bad={**good,field:value};self.assertTrue(cpu.admission_reasons(bad,now=101))
        self.assertTrue(cpu.admission_reasons(good,now=101,rss=2.1))
        for flags,limit in ((0x2000,2*cpu.GIB),(0x3200,2*cpu.GIB),(0x2200,cpu.GIB)):
            with self.assertRaises(cpu.native.NativeBlocked):cpu.validate_job_limits({"flags":flags,"job_commit_limit_bytes":limit})
        cpu.validate_job_limits({"flags":0x2200,"job_commit_limit_bytes":2*cpu.GIB})

    def test_CPU_dtype_contiguity_meta_tie_and_fatal_API_decision(self):
        torch=SimpleNamespace(bfloat16="BF16")
        for device,dtype,contiguous in (("cuda","BF16",True),("cpu","FP32",True),("cpu","BF16",False)):
            tensor=SimpleNamespace(device=SimpleNamespace(type=device),dtype=dtype,is_contiguous=lambda:contiguous)
            with self.assertRaises(cpu.native.NativeBlocked):cpu.check_tensor(torch,tensor)
        for module,name,event in (("torch.cuda","is_available","call"),("torch._C","_cuda_init","c_call"),
                                  ("transformers.models.qwen3.modeling_qwen3","forward","call"),
                                  ("transformers.tokenization_utils","__init__","call"),("torch.optim","step","call")):
            self.assertTrue(cpu.forbidden_call(module,name,event))
        self.assertFalse(cpu.forbidden_call("torch","empty","c_call"))
        param=SimpleNamespace(device=SimpleNamespace(type="cpu"),dtype="BF16")
        model=SimpleNamespace(named_parameters=lambda **_: [("weight",param)])
        with self.assertRaises(cpu.native.NativeBlocked):cpu.allocate_cpu_once(torch,model)

    def test_synthetic310_header_exact_spans_wrongdtype_and_duplicate_keys(self):
        tensor=SimpleNamespace(dtype="BF16",shape=[1],is_contiguous=lambda:True)
        states={"t"+str(i):tensor for i in range(310)};states["lm_head.weight"]=tensor
        header={name:{"dtype":"BF16","shape":[1],"data_offsets":[2*i,2*i+2]} for i,name in enumerate(sorted(states.keys()-{"lm_head.weight"}))}
        def stream(value):
            raw=json.dumps(value).encode();payload=struct.pack("<Q",len(raw))+raw+bytes(620)
            return io.BytesIO(payload),len(payload)
        data,size=stream(header);actual,_,payload=cpu.stream_checkpoint._header(data,size,states,SimpleNamespace(bfloat16="BF16"))
        self.assertEqual((len(actual),payload),(310,620))
        for field,value in (("dtype","F32"),("shape",[2]),("data_offsets",[1,3])):
            altered=copy.deepcopy(header);altered["t0"][field]=value;data,size=stream(altered)
            with self.assertRaises(RuntimeError):cpu.stream_checkpoint._header(data,size,states,SimpleNamespace(bfloat16="BF16"))
        with self.assertRaises(RuntimeError):cpu.stream_checkpoint._unique_object([("same",1),("same",2)])

    def test_lease_loss_unknown_full120_and_no_recovery_or_duplicate_dispatch(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);clock=[100.0];store=OperationsStore(root/"temp.sqlite3",root/"receipts",clock=lambda:clock[0])
            binding={"scope":cpu.SCOPE};lease=store.acquire_lease("fixture",180)
            try:
                row=store.register_job(cpu.KEY,binding,"fixture");store.ready(cpu.KEY);store.reserve(cpu.KEY,"fixture",lease["fence"],120)
                store.mark_unknown(cpu.KEY,"synthetic lost creation response")
                self.assertEqual(store.get_job(cpu.KEY)["state"],"OUTCOME_UNKNOWN")
                self.assertEqual(store.budget_snapshot("fixture")["epoch_charged_seconds"],120)
                with self.assertRaises(cpu.native.NativeBlocked):cpu.recover_finalization(root,store,binding)
                with self.assertRaises(OperationsError):store.reserve(cpu.KEY,"another",lease["fence"],120)
                with self.assertRaises(OperationsError):store._require_lease("wrong",lease["fence"])
                go={"owner":"fixture","fence":lease["fence"]}
                with self.assertRaises(cpu.native.NativeBlocked):cpu.require_job(store,go,SimpleNamespace(epoch="fixture"))
            finally:store.close()

    @unittest.skipUnless(os.name=="nt","actual Windows Job required")
    def test_actual_suspended_CPU_Job_commit_RSS_deadline_empty_exit(self):
        cfg=cpu.native.read_json(cpu.CONTRACT)
        process=job=watch=None
        with tempfile.TemporaryDirectory() as td:
            output=Path(td);began=time.monotonic()
            try:
                process=subprocess.Popen([sys.executable,"-B","-c","import time;time.sleep(30)"],
                        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|4)
                job=cpu.CPUJob(process)
                self.assertEqual(job.limit_witness["flags"],0x2200)
                self.assertEqual(job.limit_witness["job_commit_limit_bytes"],2*cpu.GIB)
                self.assertEqual(cpu.native.job_process_ids(job),[process.pid])
                raw=cpu.observe_guard(job,cfg,output,"fixture_before_resume",cpu.identity.HashBudget())
                self.assertEqual(raw["job_pid_list"],[process.pid])
                cpu.native.resume_suspended_primary_thread(process)
                watch=cpu.identity.ResourceWatch(job,started=time.monotonic(),output=output,seconds=.15,
                        sample=lambda:time.sleep(2));watch.start()
                process.wait(timeout=3)
                self.assertIsNotNone(watch.reason)
                proof=cpu.native.drain_owned_job(job)
                self.assertTrue(proof["owned_tree_exited"]);self.assertEqual(cpu.native.job_process_ids(job),[])
                print(json.dumps({"fixture":"ACTUAL_CPU_JOB_NO_MODEL","flags":job.limit_witness["flags"],
                      "Job_commit_limit_bytes":2*cpu.GIB,"initial_RSS_bytes":raw["observed_rss_bytes"],
                      "independent_deadline_reason":watch.reason,"owned_tree_exit":proof,"seconds":time.monotonic()-began}),flush=True)
            finally:
                if watch:watch.close()
                if job:job.kill();job.close()
                if process is not None and process.poll() is None:process.kill();process.wait(timeout=3)
        self.assertNotIn("torch",sys.modules);self.assertNotIn("transformers",sys.modules)
