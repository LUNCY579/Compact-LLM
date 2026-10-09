"""Pure generated JSON/token fixtures only; cannot emit native audit authority."""
from copy import deepcopy
from pathlib import Path
import sys,unittest,tempfile,json,importlib.util,subprocess
HERE=Path(__file__).resolve().parent
STAGED=HERE.name=="long_native_audit_work" and HERE.parent.name=="artifacts"
PACKAGE=HERE.parents[1] if STAGED else HERE.parent
CODE=HERE if STAGED else PACKAGE
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(CODE))
AUDITOR=CODE/("auditor.py" if STAGED else "long_native_auditor.py")
spec=importlib.util.spec_from_file_location("explicit_independent_long_native_auditor",AUDITOR)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
assert Path(a.__file__).resolve()==AUDITOR.resolve()
import long_native_core as fixture_core
from rs_contract import canonical
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
        script="""import sys,importlib.util
from pathlib import Path
sys.path.insert(0,sys.argv[2])
spec=importlib.util.spec_from_file_location('isolated_long_native_auditor',Path(sys.argv[1]))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert Path(module.__file__).resolve()==Path(sys.argv[1]).resolve()
assert not any(n in sys.modules for n in ('torch','transformers','tokenizers','jinja2'))
assert module.NATIVE_DRIVER_READY is False
"""
        dependencies=Path(a.contract.__file__).resolve().parent
        child=subprocess.run([sys.executable,"-I","-B","-c",script,str(AUDITOR),str(dependencies)],capture_output=True,text=True,timeout=30)
        self.assertEqual(0,child.returncode,child.stderr)
        scope,cap=self.fixture();self.assertNotIn("status",cap);self.assertNotIn("execution_scope",cap)

    def test_exact_durable_native_admission_CPU_schema_only(self):
        scope={"recipe":{"sha256":"a"*64},"role":"trace1"}
        result={"native_admission":{"binding":{"purpose":a.contract.PURPOSE,"recipe_sha256":"a"*64,"role":"trace1","seed":101},
            "owned_supervision_verified":True,"model_load_and_constructor_in_scope":True,"observed_monotonic":10.,"expires_monotonic":10.2}}
        sup={"worker_resume_monotonic":9.,"deadline_monotonic":600.};self.assertTrue(a.validate_native_admission(scope,result,sup))
        for key,value in (("observed_monotonic",True),("expires_monotonic",float("nan")),("expires_monotonic",10.3),("owned_supervision_verified",False)):
            current=deepcopy(result);current["native_admission"][key]=value
            with self.assertRaises(ValueError):a.validate_native_admission(scope,current,sup)
        current=deepcopy(result);current["native_admission"]["binding"]["role"]="bulk"
        with self.assertRaises(ValueError):a.validate_native_admission(scope,current,sup)
    def test_native_driver_pending_even_with_wellformed_externalfixture_anchors(self):
        anchors={key:"a"*64 for key in ("inventory_file_sha256","config_file_sha256","result_file_sha256","supervisor_file_sha256",
            "preservation_body_sha256","encoder_registration_file_sha256","recipe_sha256")}
        anchors.update(source_commit="b"*40,context_cap=4096,role="trace1")
        with self.assertRaisesRegex(ValueError,"driver remains pending"):a.audit("UNUSED","UNUSED","UNUSED","UNUSED",terminal_signal=a.SIGNAL,expected=anchors)
        self.assertFalse(a.NATIVE_DRIVER_READY)

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




class StateAndChainFixtureControls(unittest.TestCase):
    """Native-shaped metadata fixtures; no tensor/RNG/encoder/model is invoked."""
    def state(self):
        from native_loader import _source_shapes,_adapter_shapes
        shapes=_adapter_shapes();model={name:"a"*64 for name in _source_shapes()}
        scope={"source_pins":{"CPU_SOURCE":"a"*64},"runtime_pins":{"CPU_RUNTIME":"b"*64},"data_pins":{"CPU_DATA":"c"*64},
            "model_pins":{"model.safetensors":"d"*64},"tokenizer_binding":{"model_dir":"CPU_NONEXISTENT","assets":{"tokenizer.json":{"sha256":"e"*64}},
                "binding_sources":{"CPU_BINDING":"f"*64}},"recipe":{"sha256":"1"*64},"row":{"prompt_tokens":992}}
        maps={"source":scope["source_pins"],"runtime":scope["runtime_pins"],"data":scope["data_pins"],"model":scope["model_pins"],
            "encoder_assets":{str(Path("CPU_NONEXISTENT")/"tokenizer.json"):"e"*64},"encoder_bindings":scope["tokenizer_binding"]["binding_sources"]}
        encoder={"short_hash_kind":"SHA256_RENDERED_UTF8_CHAT_TEMPLATE","long_hash_kind":"SHA256_CANONICAL_MESSAGES_JSON",
            "short_tokens":419,"long_tokens":992,"full_retokenization_matches":True,"tokenizer_binding_sha256":a.digest(scope["tokenizer_binding"]),"native_long_context_qualified":False}
        hashes={name:"2"*64 for name in shapes};rng={"python_sha256":"3"*64,"cpu":"4"*64,"cuda0":"5"*64};owner={"job_name":"CPU_FIXTURE_NO_REAL_JOB"}
        common={"recipe_sha256":"1"*64,"source_byte_maps_sha256":a.digest(maps),"model_source_map_sha256":a.digest(model),
            "encoder_proof_sha256":a.digest(encoder),"dense_start_rng":rng,"inference_start_rng":rng,"initial_adapter_sha256":a.digest(hashes),"seed":101}
        inventory=[{"name":name,"shape":shape,"dtype":"float32","device":"cuda:0","requires_grad":True,"parameters":a.math.prod(shape)} for name,shape in shapes.items()]
        base={"verified_source_tensors":310,"frozen_base_unchanged":True,"unique_base_parameters":596049920,"base_dtype":"bfloat16","embedding_head_tie":True}
        active={"model.norm":34,"model.rotary_emb":34}
        for i in range(28):active.update({"model.layers.%d%s"%(i,suffix):34 for suffix in ("",".input_layernorm",".post_attention_layernorm",".self_attn.q_norm",".self_attn.k_norm",".self_attn",".mlp")})
        start={"kind":"NATIVE_OBSERVED_LONG_START","identity_common":common,"owned_scope_identity":owner,"actual_pin_maps":maps,"encoder":encoder,
            "loader":{"source_tensor_sha256":model,"unique_base_tensors":310,"serialized_tensors":310,"unique_base_parameters":596049920,
                "base_dtype":"bfloat16","base_frozen":True,"embedding_head_tie":True,"memory_mapped":False,"cpu_model_allocated":False,
                "FP32_base_copy_allocated":False,"blocking_chunk_copies":True,"asset_hashes":{"model.safetensors":{"sha256":"d"*64}}},
            "adapters":{"inventory":inventory,"rank":8,"alpha":16,"dropout":0.,"qv_site_count":56,"layer_count":28,"trainable_tensors":112,"trainable_parameters":1146880},
            "adapter_byte_hashes_before":hashes,"actual_dense_zeroB_parity":{"equal":True,"maximum_error":0.},
            "zeroB":{"all_B_zero":True,"qv_site_count":56,"trainable_tensors":112},
            "compute":{"base_parameter_objects_and_pointers_preserved":True,"unique_base_tensors":310,"unique_base_parameters":596049920,
                "base_storage_dtype":"bfloat16","activation_dtype":"float32","linear_site_count":197,"embedding_site_count":1}}
        end={"kind":"NATIVE_OBSERVED_LONG_END","identity_common":common,"actual_pin_maps_after":maps,"RNG_unchanged":True,"RNG_after":rng,
            "adapter_byte_hashes_after":hashes,"zeroB_after":{"all_B_zero":True,"qv_site_count":56,"trainable_tensors":112},"zeroB_unchanged":True,
            "base310_unchanged":True,"base_postcheck":base,"activation_observations":active}
        capsule={"start_proof":start,"end_proof":end,"physical_dose":{"attempted_calls":34}}
        # Canonical JSON reorders mapping keys; ordered inventory stays separate.
        return scope,json.loads(canonical(capsule)),owner
    def test_complete_metadata112_310_RNG_198_and_canonical_map_order_fixture_only(self):
        scope,capsule,owner=self.state();result=a.validate_observed_state(scope,capsule,owner)
        self.assertEqual(112,result["adapter_hashes"]);self.assertEqual(310,result["base_hashes"])
        self.assertFalse(result["raw_RNG_encoding_or_restore_verified"]);self.assertNotIn("status",result)
    def test_hash_roster_dtype_rng_parity_activation_and_source_mutations_rejected(self):
        for kind in ("adapter","order","rng","parity","active","source","base"):
            scope,capsule,owner=self.state()
            if kind=="adapter":capsule["end_proof"]["adapter_byte_hashes_after"].pop(next(iter(capsule["end_proof"]["adapter_byte_hashes_after"])))
            elif kind=="order":capsule["start_proof"]["adapters"]["inventory"].reverse()
            elif kind=="rng":capsule["end_proof"]["RNG_after"]["cpu"]="a"*64
            elif kind=="parity":capsule["start_proof"]["actual_dense_zeroB_parity"]["maximum_error"]=.001
            elif kind=="active":capsule["end_proof"]["activation_observations"]["model.norm"]=33
            elif kind=="base":capsule["end_proof"]["base_postcheck"]["verified_source_tensors"]=309
            else:capsule["end_proof"]["actual_pin_maps_after"]["source"]["CPU_SOURCE"]="b"*64
            with self.assertRaises(ValueError):a.validate_observed_state(scope,capsule,owner)
    def node(self):
        scope={"recipe":{"sha256":"a"*64}}
        payload={"purpose":a.contract.PURPOSE,"recipe_sha256":"a"*64,
            "posted_scope_costs":{"operations":{"load":{"attempted":1,"returned":1,"failed":0,"pending":0}},"physical_calls":34,"input_positions":130534}}
        payload["sha256"]=a.digest(payload)
        # Root production arithmetic builds fixture; auditor recomputes locally.
        terminal={"capsule_sha256":payload["sha256"],"costs":fixture_core.project(fixture_core.project(payload["posted_scope_costs"],"long_capsule_publish"),"long_terminal_publish")}
        promotion={"capsule_sha256":payload["sha256"],"terminal_sha256":a.digest(terminal),"costs":fixture_core.project(terminal["costs"],"long_capsule_promote")}
        node={"kind":a.contract.KIND,"recipe_sha256":"a"*64,"capsule":payload,"capsule_publication":{},"terminal":terminal,
            "terminal_publication":{},"promotion":promotion,"promotion_publication":{},"cumulative_costs":promotion["costs"]}
        node["sha256"]=a.digest(node);return scope,node
    def test_independent_projection_matches_fixture_and_resigned_erasure_rejected(self):
        scope,node=self.node();self.assertEqual(node["capsule"],a.chain_payload(node,scope["recipe"]))
        node["capsule"]["posted_scope_costs"]["operations"].clear();node["capsule"]["sha256"]=a.digest({k:v for k,v in node["capsule"].items() if k!="sha256"})
        node["terminal"]["capsule_sha256"]=node["capsule"]["sha256"];node["promotion"]["capsule_sha256"]=node["capsule"]["sha256"]
        node["promotion"]["terminal_sha256"]=a.digest(node["terminal"]);node["sha256"]=a.digest({k:v for k,v in node.items() if k!="sha256"})
        with self.assertRaisesRegex(ValueError,"independent posted"):a.chain_payload(node,scope["recipe"])
    def test_promotion_rollback_pending_and_wrong_recipe_denied(self):
        scope,node=self.node();node["promotion"]["costs"]["physical_calls"]-=1;node["sha256"]=a.digest({k:v for k,v in node.items() if k!="sha256"})
        with self.assertRaises(ValueError):a.chain_payload(node,scope["recipe"])
        scope,node=self.node();scope["recipe"]["sha256"]="b"*64
        with self.assertRaises(ValueError):a.chain_payload(node,scope["recipe"])
        with self.assertRaises(ValueError):a.project_costs({"operations":{"pending":{"attempted":1,"returned":0,"failed":0,"pending":1}},"physical_calls":0,"input_positions":0},"long_capsule_publish")

if __name__=="__main__":unittest.main()
