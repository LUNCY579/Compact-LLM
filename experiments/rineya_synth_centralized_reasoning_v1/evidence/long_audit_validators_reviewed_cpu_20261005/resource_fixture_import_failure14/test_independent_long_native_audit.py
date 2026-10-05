"""Pure generated JSON/token fixtures only; cannot emit native audit authority."""
from copy import deepcopy
from pathlib import Path
import sys,unittest
HERE=Path(__file__).resolve().parent
STAGED=HERE.name=="long_native_audit_work" and HERE.parent.name=="artifacts"
PACKAGE=HERE.parents[1] if STAGED else HERE.parent
CODE=HERE if STAGED else PACKAGE
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(CODE))
import auditor as a
class Controls(unittest.TestCase):
    def fixture(self,p=992,role="trace1",length=32):
        row={"prompt_tokens":p,"prompt_ids":[1]*p};scope={"role":role,"row":row,"short_parity_row":{"prompt_ids":[1]*419},
            "recipe":{"prefix_stress_ids":[1]*31 if role=="prefix_stress" else None}}
        scope["dose"]=a.contract.work(row,role)
        trace=[{"token":2,"logprob":-.25,"prefix_length":p+i} for i in range(length)]
        if length<32:trace[-1]["token"]=151645
        capsule={"trace":trace,"greedy_EOS_respected":role!="prefix_stress","full32_greedy_coverage":role!="prefix_stress" and length==32,"stress_is_greedy":False}
        roster=[("long_native_dense_parity_forward",[1]*419),("long_native_zeroB_parity_forward",[1]*419)]
        if role=="bulk":roster.append(("long_native_bulk_forward",row["prompt_ids"]+[t["token"] for t in trace[:-1]]))
        elif role=="prefix_stress":roster.extend(("long_native_declared_prefix_stress_forward",row["prompt_ids"]+[1]*i) for i in range(32))
        else:roster.extend(("long_native_greedy_forward",row["prompt_ids"]+[t["token"] for t in trace[:i]]) for i in range(length))
        capsule["physical_dose"]={"attempted_calls":len(roster),"input_positions":sum(len(ids) for _,ids in roster),
            "records":[{"operation":name,"input_ids_sha256":a.digest(ids),"input_positions":len(ids)} for name,ids in roster]}
        return scope,capsule
    def test_all_registered_windows_complete_exact_physical_prefix_caps_fixture_only(self):
        for p in (992,2016,4064):
            for role in a.contract.ROLES:
                scope,cap=self.fixture(p,role)
                a.validate_trace(scope,cap);value=a.validate_physical(scope,cap)
                self.assertEqual(3 if role=="bulk" else 34,value["attempted_calls"]);self.assertLessEqual(value["input_positions"],134144)
                self.assertNotIn("status",value)
    def test_earlyEOS_denominator_truthful_and_hidden_earlyEOS_or_truncation_rejected(self):
        scope,cap=self.fixture(length=3);info=a.validate_trace(scope,cap);self.assertEqual("eos",info["finish"]);self.assertFalse(info["full32_greedy_coverage"])
        self.assertEqual(5,a.validate_physical(scope,cap)["attempted_calls"])
        cap["trace"][0]["token"]=151645
        with self.assertRaises(ValueError):a.validate_trace(scope,cap)
        scope,cap=self.fixture(length=3);cap["trace"][-1]["token"]=2
        with self.assertRaises(ValueError):a.validate_trace(scope,cap)
    def test_stress_not_laundered_as_greedy_or_earlyEOS_coverage(self):
        scope,cap=self.fixture(role="prefix_stress");cap["trace"][0]["token"]=151645
        self.assertEqual("declared_prefix_stress",a.validate_trace(scope,cap)["finish"]);a.validate_physical(scope,cap)
        cap["full32_greedy_coverage"]=True
        with self.assertRaises(ValueError):a.validate_trace(scope,cap)
        scope,cap=self.fixture(role="prefix_stress");scope["recipe"]["prefix_stress_ids"][0]=151645
        with self.assertRaises(ValueError):a.validate_trace(scope,cap)
    def test_any_extra_model_call_wrong_prefix_operation_or_positions_rejected(self):
        for mutate in (lambda c:c["physical_dose"]["records"].append(deepcopy(c["physical_dose"]["records"][0])),
            lambda c:c["physical_dose"]["records"][2].update(input_ids_sha256="a"*64),
            lambda c:c["physical_dose"]["records"][2].update(operation="unregistered_forward"),
            lambda c:c["physical_dose"].update(input_positions=1)):
            scope,cap=self.fixture();mutate(cap)
            with self.assertRaises(ValueError):a.validate_physical(scope,cap)
    def test_typed_token_LP_prefix_and_actual32_denominator(self):
        for key,value in (("token",True),("token",151936),("logprob",float("nan")),("logprob",.1),("prefix_length",True)):
            scope,cap=self.fixture();cap["trace"][0][key]=value
            with self.assertRaises(ValueError):a.validate_trace(scope,cap)
    def test_cumulative_operations_no_erased_prior_or_rollback_or_partial(self):
        prior={"load":{"attempted":1,"returned":1,"failed":0,"pending":0}}
        now={"load":{"attempted":2,"returned":2,"failed":0,"pending":0},"forward":{"attempted":34,"returned":34,"failed":0,"pending":0}}
        self.assertEqual(prior["load"],a.delta(now,prior)["load"])
        with self.assertRaises(ValueError):a.delta({"forward":now["forward"]},prior)
        with self.assertRaises(ValueError):a.counts({"bad":{"attempted":1,"returned":0,"failed":0,"pending":1}})
    def bulk(self):
        scope,cap=self.fixture(role="bulk",length=3)
        previous=[{"capsule_chain":{"capsule":{"trace":deepcopy(cap["trace"])}}} for _ in range(2)]
        records=[{"step":i,"position":scope["row"]["prompt_tokens"]-1+i,"selected_token":step["token"],"trace_logprob":step["logprob"],
            "argmax_token":step["token"],"argmax_logprob":step["logprob"],"selected_logprob":step["logprob"],"absolute_error":0.} for i,step in enumerate(cap["trace"])]
        cap["comparison"]={"bulk_steps":records,"actual_compared_steps":3,"argmax_differences":[],"maximum_error":0.,"mean_error":0.}
        return scope,cap,previous
    def test_complete_actual_bulk_perstep_maxmean_and_argmax_fixtures_only(self):
        self.assertTrue(a.validate_bulk_records(*self.bulk()))
    def test_summary_only_current_plumbing_cannot_issue_complete_bulk_audit(self):
        scope,cap,previous=self.bulk();cap["comparison"].pop("bulk_steps")
        with self.assertRaisesRegex(ValueError,"every-step"):a.validate_bulk_records(scope,cap,previous)
    def test_bulk_missing_step_wrongLP_wrongerror_or_repeatedtrace_mismatch_rejected(self):
        for kind in ("step","LP","error","repeat","token"):
            scope,cap,previous=self.bulk()
            if kind=="step":cap["comparison"]["bulk_steps"].pop()
            elif kind=="LP":cap["comparison"]["bulk_steps"][0]["selected_logprob"]=-1.
            elif kind=="error":cap["comparison"]["bulk_steps"][0]["absolute_error"]=1.
            elif kind=="token":cap["comparison"]["bulk_steps"][0]["argmax_token"]=3
            else:previous[1]["capsule_chain"]["capsule"]["trace"][0]["token"]=3
            with self.assertRaises(ValueError):a.validate_bulk_records(scope,cap,previous)
    def test_signal_and_complete_externalanchors_required_before_IO(self):
        with self.assertRaises(ValueError):a.validate_anchors(None,{})
        with self.assertRaises(ValueError):a.validate_anchors(a.SIGNAL,{})
    def test_no_fixture_nativePASS_or_framework_import(self):
        self.assertFalse(any(n in sys.modules for n in ("torch","transformers","tokenizers","jinja2")))
        scope,cap=self.fixture();self.assertNotIn("status",cap);self.assertNotIn("execution_scope",cap)
class ResourceFixtureControls(unittest.TestCase):
    """Synthetic Windows-shaped observations. No Windows or GPU API is called."""
    @staticmethod
    def write(path,value):path.write_text(canonical(value)+"\n",encoding="utf-8",newline="\n")
    def fixture(self,root):
        package=Path(root);leaf=package/"artifacts/fixture";leaf.mkdir(parents=True)
        lease=package/"artifacts/_native_gpu_leases/fixture";lease.mkdir(parents=True)
        identity={"pid":100,"create_time":1.,"creation_filetime_100ns":10,"assigned_before_resume":True,
            "job_kill_on_close":True,"breakaway_allowed":False,"job_name":"Local\\SYNTH_NQ_JOB_"+32*"a"}
        owner={"worker_pid":100,"root_worker_pid":100,"supervisor_pid":300,"supervisor_create_time":3.,
            "job_name":identity["job_name"],"config_sha256":"a"*64}
        ownership={"actual_worker_pid":200,"actual_parent_pid":100,"launcher_root_pid":100,"supervisor_pid":300,
            "job_name":identity["job_name"],"owned_job_membership_verified":True,"query_job_handle_closed_before_framework_import":True}
        self.write(leaf/"supervisor_ownership.json",owner);self.write(leaf/"stage_worker_ownership_verified.json",ownership)
        def cpu(now,owned=True):
            items=[{"pid":100,"create_time":1.,"creation_filetime_100ns":10,"rss_bytes":100000000,
                "sample_started_monotonic":now,"sample_completed_monotonic":now}]
            if owned:items.append({"pid":200,"create_time":2.,"creation_filetime_100ns":20,"rss_bytes":100000000,
                "sample_started_monotonic":now,"sample_completed_monotonic":now})
            return {"sample_started_monotonic":now,"sample_completed_monotonic":now,"host_available_bytes":5*1024**3,
                "disk_free_bytes":10*1024**3,"worker_tree_rss_bytes":sum(x["rss_bytes"] for x in items) if owned else 0,
                "owned_process_samples":items if owned else[],
                "owned_process_identities":[{k:x[k] for k in ("pid","create_time","creation_filetime_100ns")} for x in items] if owned else[]}
        def gpu(now):return {"GPU_index":0,"gpu_uuid":"GPU_FIXTURE_NO_NATIVE_RESULT","gpu_observation_id":"fixture_"+str(now),
            "gpu_sample_started_monotonic":now,"gpu_sample_completed_monotonic":now,"gpu_total_bytes":8*1024**3,"gpu_free_bytes":5*1024**3}
        admissions=[{**cpu(t,False),**gpu(t)} for t in (7.,9.)]
        sup={"bounds":{"worker_process_tree_RSS_GiB":2.0,"system_available_reserve_GiB":1.25,"preworker_available_GiB":3.3125,
            "GPU_free_at_admission_GiB":4.,"GPU_free_during_work_GiB":.5,"disk_free_GiB":4.,"whole_scope_seconds":600,"external_watchdog_milliseconds":100},
            "cadence":{"CPU_watchdog_seconds":.1,"CPU_freshness_seconds":.2,"GPU_query_interval_seconds":.5,"GPU_query_timeout_seconds":2.,
                "GPU_freshness_seconds":2.,"admission_interval_seconds":2.,"admission_deadline_seconds":30.,"admission_samples":2},
            "admission_only":False,"worker_attempted":True,"worker_launched":True,"worker_resumed":True,
            "owned_handle_alive_verified_after_launch":True,"preflight_revalidated_before_current_observations":True,
            "job_closed_kill_on_close":True,"sampler_threads_terminal_verified":True,"gpu_telemetry_cleanup_verified":True,
            "owned_worker_identity_currently_alive":False,"unjoined_sampler_threads":[],"admission_observations":admissions,
            "dispatch_observation":admissions[-1],"worker_identity":identity,"worker_resume_monotonic":10.05,"deadline_monotonic":607.,
            "config_sha256":"a"*64,"lease_path":str(lease),"elapsed_seconds":4.}
        resources=[]
        for i,t in enumerate((10.,10.1,10.2)):
            row={"kind":"CPU",**cpu(t)}
            if i==0:row["phase"]="worker_suspended_startup"
            resources.append(row)
        resources.extend({"kind":"GPU",**gpu(t)} for t in (10.,10.2))
        self.write(leaf/"RESOURCE_OBSERVATIONS.json",resources)
        (leaf/"supervisor_heartbeat.jsonl").write_text("".join(canonical({"supervisor_pid":300,"unix_seconds":20.+i*.1})+"\n" for i in range(3)),encoding="utf-8",newline="\n")
        self.write(leaf/"worker_GPU_final.json",{"CUDA_initialized":True,"peak_allocated_bytes":1024**3,"peak_reserved_bytes":2*1024**3,"free_bytes_final":2*1024**3})
        self.write(lease/"worker.json",identity)
        self.write(lease/"resume.json",{"worker_identity":identity,"external_monitor_ready_before_resume":True,
            "resumed_monotonic":10.05,"deadline_monotonic":607.})
        self.write(lease/"reservation.json",{"config_sha256":"a"*64,"output_dir":str(leaf),"lease_id":"fixture","GPU_index":0,
            "supervisor":{"pid":300,"create_time":3.}})
        self.write(leaf/"SUPERVISOR_RECEIPT.json",sup);(lease/"terminal.json").write_bytes((leaf/"SUPERVISOR_RECEIPT.json").read_bytes())
        return package,leaf,sup,resources
    def test_complete_fixture_resources_and_owned_lease_schema_only(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            package,leaf,sup,_=self.fixture(root)
            with patch.object(a,"PACKAGE",package):value=a.verify_resources({},sup,leaf)
            self.assertEqual(3,value["CPU_samples"]);self.assertEqual(2,value["GPU_samples"])
            self.assertNotIn("status",value);self.assertEqual({100:1.,200:2.,300:3.},value["identities"])
    def test_returned_GPU_amount_CPU_chronology_identity_RSS_and_gap_negatives(self):
        from unittest.mock import patch
        mutations=(lambda rows:rows[-1].update(gpu_free_bytes=1),
            lambda rows:rows[1].update(sample_completed_monotonic=9.),
            lambda rows:rows[1]["owned_process_identities"][0].update(create_time=99.),
            lambda rows:rows[1].update(worker_tree_rss_bytes=999),
            lambda rows:rows[-1].update(gpu_uuid="foreign_GPU"))
        for mutate in mutations:
            with tempfile.TemporaryDirectory(dir=HERE) as root:
                package,leaf,sup,rows=self.fixture(root);mutate(rows);self.write(leaf/"RESOURCE_OBSERVATIONS.json",rows)
                with patch.object(a,"PACKAGE",package):
                    with self.assertRaises(ValueError):a.verify_resources({},sup,leaf)
    def test_admission_cleanup_prelaunch_or_lease_rebind_cannot_pass(self):
        from unittest.mock import patch
        for key,value in (("job_closed_kill_on_close",False),("unjoined_sampler_threads",["pending"]),
            ("worker_resumed",False),("failure","stale_external_CPU_watchdog")):
            with tempfile.TemporaryDirectory(dir=HERE) as root:
                package,leaf,sup,_=self.fixture(root);sup[key]=value
                with patch.object(a,"PACKAGE",package):
                    with self.assertRaises(ValueError):a.verify_resources({},sup,leaf)
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            package,leaf,sup,_=self.fixture(root)
            self.write(Path(sup["lease_path"])/"terminal.json",{"foreign":True})
            with patch.object(a,"PACKAGE",package):
                with self.assertRaisesRegex(ValueError,"terminal lease"):a.verify_resources({},sup,leaf)



if __name__=="__main__":unittest.main()
