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
