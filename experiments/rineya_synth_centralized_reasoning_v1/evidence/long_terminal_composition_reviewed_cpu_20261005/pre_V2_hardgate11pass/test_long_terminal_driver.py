"""Whole-driver generated JSON fixtures only; no native PASS or model execution."""
from copy import deepcopy
from pathlib import Path
import hashlib,importlib.util,json,sys,unittest
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1] if HERE.name=="long_native_terminal_driver_work" and HERE.parent.name=="artifacts" else HERE.parent
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(HERE))
import driver as d
spec=importlib.util.spec_from_file_location("fixture_state_builders",PACKAGE/"tests/test_independent_long_native_audit.py")
fixture_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture_module)
class Controls(unittest.TestCase):
    def frame(self):
        state_builder=fixture_module.StateAndChainFixtureControls()
        scope,capsule,owner=state_builder.state()
        scope.update({"schema_version":1,"purpose":d.v.contract.PURPOSE,"authorization":deepcopy(d.v.contract.AUTHORIZATION),"role":"trace1",
            "owned_qualification_authorized":True,"staged_native_dispatch_forbidden":False,
            "native_launch_authorized":False,"native_long_context_qualified":False,"study_training_authorized":False,"study_warmstart_allowed":False,
            "guard_enrollment_authorized":False,"scoring_authorized":False,"resources":deepcopy(d.v.contract.native.RESOURCE_BOUNDS),
            "tolerances":deepcopy(d.v.contract.native.TOLERANCES),"references":None,"encoder_registration":{"path":"CPU_ENCODER_NOT_CREATED","sha256":"c"*64}})
        physical_scope,trace_cap=fixture_module.Controls().fixture()
        scope.update({key:physical_scope[key] for key in ("row","short_parity_row","dose")})
        scope["recipe"]={"sha256":"1"*64,"context_cap":1024,"prefix_stress_ids":None}
        scope=d.seal(scope)
        capsule.update(trace_cap);capsule.update({"kind":d.v.contract.KIND,"purpose":d.v.contract.PURPOSE,"recipe_sha256":"1"*64,"role":"trace1",
            "previous_capsule_sha256":None,"comparison":None,"CPU_fixture":False,
            "native_launch_authorized":False,"native_long_context_qualified":False,"study_training_authorized":False,"scoring_authorized":False})
        posted={"operations":{"model_construct":{"attempted":1,"returned":1,"failed":0,"pending":0}},"physical_calls":34,
            "input_positions":trace_cap["physical_dose"]["input_positions"]}
        capsule["posted_scope_costs"]=posted;capsule=d.seal(capsule)
        terminal={"capsule_sha256":capsule["sha256"],"costs":d.v.project_costs(d.v.project_costs(posted,"long_capsule_publish"),"long_terminal_publish")}
        promotion={"capsule_sha256":capsule["sha256"],"terminal_sha256":d.digest(terminal),"costs":d.v.project_costs(terminal["costs"],"long_capsule_promote")}
        chain=d.seal({"kind":d.v.contract.KIND,"recipe_sha256":"1"*64,"capsule":capsule,"capsule_publication":{},
            "terminal":terminal,"terminal_publication":{},"promotion":promotion,"promotion_publication":{},"cumulative_costs":promotion["costs"]})
        proof={"binding":{"purpose":d.v.contract.PURPOSE,"recipe_sha256":"1"*64,"role":"trace1","seed":101},
            "owned_supervision_verified":True,"model_load_and_constructor_in_scope":True,"observed_monotonic":10.,"expires_monotonic":10.2}
        result=d.seal({"status":"COMPLETE_NATIVE_LONG_SCOPE_PENDING_INDEPENDENT_AUDIT","execution_scope":d.v.contract.NATIVE,"role":"trace1",
            "recipe_sha256":"1"*64,"config_sha256":"a"*64,"source_commit":"b"*40,"wall_seconds":1.,"capsule_chain":chain,
            "actual_costs":promotion["costs"],"physical_dose":trace_cap["physical_dose"],"partial_trace":trace_cap["trace"],"native_admission":proof,
            "native_launch_authorized":False,"native_long_context_qualified":False,"study_training_authorized":False,"guard_enrollment_authorized":False,"scoring_authorized":False})
        sup={"status":"SUPERVISED_WORKER_EXITED_ZERO","returncode":0,"partial_outputs_frozen":True,"job_closed_kill_on_close":True,
            "sampler_threads_terminal_verified":True,"gpu_telemetry_cleanup_verified":True,"owned_worker_identity_currently_alive":False,
            "unjoined_sampler_threads":[],"config_sha256":"a"*64,"preflight":{"source_commit":"b"*40},"elapsed_seconds":2.,
            "worker_resume_monotonic":9.,"deadline_monotonic":600.}
        anchors={key:"a"*64 for key in ("inventory_file_sha256","config_file_sha256","result_file_sha256","supervisor_file_sha256",
            "preservation_body_sha256","encoder_registration_file_sha256","recipe_sha256")}
        anchors.update(encoder_registration_file_sha256="c"*64,recipe_sha256="1"*64,source_commit="b"*40,context_cap=1024,role="trace1")
        maps={"source":scope["source_pins"],"runtime":scope["runtime_pins"],"model":scope["model_pins"],"data":scope["data_pins"],
            "encoder_assets":{str(Path(scope["tokenizer_binding"]["model_dir"])/name):item["sha256"] for name,item in scope["tokenizer_binding"]["assets"].items()},
            "encoder_bindings":scope["tokenizer_binding"]["binding_sources"]}
        inventory=d.seal({"kind":d.v.INVENTORY_KIND,"source_commit":"b"*40,"recipe_sha256":"1"*64,"role":"trace1","context_cap":1024,
            "maps":maps,"counts":{name:len(value) for name,value in maps.items()},"previous_references":None,
            "encoder_registration":scope["encoder_registration"],"validator_file_sha256":d.VALIDATOR_SHA256})
        return {"scope":scope,"result":result,"supervisor":sup,"inventory":inventory,"owner":owner,"previous":[]},anchors
    def sign_result(self,frame):frame["result"]=d.seal({k:v for k,v in frame["result"].items() if k!="sha256"})
    def test_whole_generated_frame_passes_only_fixture_status(self):
        frame,anchors=self.frame();result=d.validate_frame(frame,anchors,CPU_fixture_mode=True)
        self.assertEqual(d.CPU_KIND,result["kind"]);self.assertEqual("CPU_FIXTURE_CHECKS_ONLY",result["status"])
        self.assertFalse(result["native_AUDIT_KIND_PASS_issued"]);self.assertFalse(result["resume_authority"]);self.assertFalse(result["native_ready"])
        self.assertNotEqual(d.v.contract.AUDIT_KIND,result["kind"])
    def test_source_model_encoder_map_omission_hash_substitution_and_validator_pin(self):
        for kind in ("source","model","encoder_assets","validator"):
            frame,anchors=self.frame();inventory=frame["inventory"]
            if kind=="validator":inventory["validator_file_sha256"]="e"*64
            else:inventory["maps"][kind]=deepcopy(inventory["maps"][kind]);inventory["maps"][kind].pop(next(iter(inventory["maps"][kind])))
            frame["inventory"]=d.seal({k:v for k,v in inventory.items() if k!="sha256"})
            with self.assertRaises(ValueError):d.validate_frame(frame,anchors,CPU_fixture_mode=True)
    def test_failed_terminal_cleanup_expired_admission_or_wrong_HEAD_rejects_whole_frame(self):
        for mutate in (lambda f:f["supervisor"].update(status="OWNED_WORKER_ABORTED"),
            lambda f:f["supervisor"].update(job_closed_kill_on_close=False),
            lambda f:f["result"]["native_admission"].update(expires_monotonic=10.3),
            lambda f:f["supervisor"]["preflight"].update(source_commit="c"*40)):
            frame,anchors=self.frame();mutate(frame);self.sign_result(frame)
            with self.assertRaises(ValueError):d.validate_frame(frame,anchors,CPU_fixture_mode=True)
    def test_state_RNG_112_roster_parity_activation_and_physical_wrong_prefix(self):
        for kind in ("rng","adapter","parity","activation","physical"):
            frame,anchors=self.frame();caps=frame["result"]["capsule_chain"]["capsule"]
            if kind=="rng":caps["end_proof"]["RNG_after"]["cpu"]="a"*64
            elif kind=="adapter":caps["end_proof"]["adapter_byte_hashes_after"].pop(next(iter(caps["end_proof"]["adapter_byte_hashes_after"])))
            elif kind=="parity":caps["start_proof"]["actual_dense_zeroB_parity"]["equal"]=False
            elif kind=="activation":caps["end_proof"]["activation_observations"]["model.norm"]=0
            else:caps["physical_dose"]["records"][2]["input_ids_sha256"]="b"*64
            # Immediate mutation is caught at whole capsule publication/body boundary.
            self.sign_result(frame)
            with self.assertRaises(ValueError):d.validate_frame(frame,anchors,CPU_fixture_mode=True)
    def test_laundered_fixture_capsule_and_partial_costs_denied(self):
        for kind in ("fixture","pending","rollback"):
            frame,anchors=self.frame();result=frame["result"]
            if kind=="fixture":result["capsule_chain"]["capsule"]["CPU_fixture"]=True
            elif kind=="pending":result["actual_costs"]["operations"]["model_construct"].update(returned=0,pending=1)
            else:result["actual_costs"]["physical_calls"]-=1
            self.sign_result(frame)
            with self.assertRaises(ValueError):d.validate_frame(frame,anchors,CPU_fixture_mode=True)
    def test_native_entry_hard_rejects_before_IO_or_output(self):
        frame,anchors=self.frame()
        with self.assertRaisesRegex(ValueError,"PASS emission disabled"):
            d.audit_terminal("UNUSED","UNUSED","UNUSED","UNUSED",terminal_signal=d.v.SIGNAL,anchors=anchors,output_path="UNUSED")
        self.assertFalse(d.NATIVE_PASS_ENABLED);self.assertFalse(d.ENCODER_V2_CONSUMPTION_REVIEWED)
    def test_package_only_or_resealed_CPU_encoder_cannot_be_final(self):
        frame,anchors=self.frame();encoder=d.seal({"kind":"SYNTH_LONG_ENCODER_RUNTIME_REGISTRATION_V1","runtime_pins":{},
            "complete_runtime_execution_closure_established":False});stdlib=d.seal({"kind":"STAGED_CPU_STDLIB_RUNTIME_SUPERSET_NOT_NATIVE_REGISTRATION_V1"})
        with self.assertRaisesRegex(ValueError,"complete encoder"):d.validate_registration_pending(frame["scope"],encoder,stdlib,anchors)
    def test_current_context_or_role_or_missing_previous_is_rejected(self):
        for kind in ("context","role","prior"):
            frame,anchors=self.frame()
            if kind=="context":anchors["context_cap"]=4096
            elif kind=="role":anchors["role"]="bulk"
            else:frame["scope"]["references"]=[{"fake_previous":True}];frame["scope"]=d.seal({k:v for k,v in frame["scope"].items() if k!="sha256"})
            with self.assertRaises(ValueError):d.validate_frame(frame,anchors,CPU_fixture_mode=True)

    def whole(self):
        frame,anchors=self.frame();sup=frame["supervisor"];owner=frame["owner"]
        owner.update(worker_pid=100,root_worker_pid=100,supervisor_pid=300,supervisor_create_time=3.,config_sha256=anchors["config_file_sha256"])
        identity={"pid":100,"create_time":1.,"creation_filetime_100ns":10,"job_name":owner["job_name"],"assigned_before_resume":True,
            "job_kill_on_close":True,"breakaway_allowed":False}
        sup.update(worker_identity=identity,bounds={"worker_process_tree_RSS_GiB":2.0,"system_available_reserve_GiB":1.25,
            "preworker_available_GiB":3.3125,"GPU_free_at_admission_GiB":4.,"GPU_free_during_work_GiB":.5,"disk_free_GiB":4.,
            "whole_scope_seconds":600,"external_watchdog_milliseconds":100},
            cadence={"CPU_watchdog_seconds":.1,"CPU_freshness_seconds":.2,"GPU_query_interval_seconds":.5,"GPU_query_timeout_seconds":2.,
                "GPU_freshness_seconds":2.,"admission_interval_seconds":2.,"admission_deadline_seconds":30.,"admission_samples":2})
        def cpu(t):
            items=[{"pid":100,"create_time":1.,"creation_filetime_100ns":10,"rss_bytes":100,
                "sample_started_monotonic":t,"sample_completed_monotonic":t}]
            return {"kind":"CPU","sample_started_monotonic":t,"sample_completed_monotonic":t,"host_available_bytes":5*1024**3,
                "disk_free_bytes":10*1024**3,"worker_tree_rss_bytes":100,"owned_process_samples":items,
                "owned_process_identities":[{k:item[k] for k in ("pid","create_time","creation_filetime_100ns")} for item in items]}
        def gpu(t):return {"kind":"GPU","GPU_index":0,"gpu_uuid":"CPU_FIXTURE_UUID","gpu_observation_id":"fixture_"+str(t),
            "gpu_sample_started_monotonic":t,"gpu_sample_completed_monotonic":t,"gpu_total_bytes":8*1024**3,"gpu_free_bytes":5*1024**3}
        frame.update(resources=[cpu(10.),gpu(10.),cpu(10.1),gpu(10.2)],
            admissions=[{**cpu(t),**gpu(t)} for t in (7.,9.)],heartbeat=[{"supervisor_pid":300,"unix_seconds":20.+i*.1} for i in range(2)],
            ownership={"job_name":owner["job_name"],"launcher_root_pid":100,"actual_worker_pid":100,"owned_job_membership_verified":True,
                "query_job_handle_closed_before_framework_import":True},direct_GPU={"CUDA_initialized":True,"peak_allocated_bytes":0,
                "peak_reserved_bytes":0,"free_bytes_final":5*1024**3})
        frame["lease"]={"worker":identity,"resume":{"worker_identity":identity,"external_monitor_ready_before_resume":True,
            "resumed_monotonic":sup["worker_resume_monotonic"],"deadline_monotonic":sup["deadline_monotonic"]},
            "reservation":{"config_sha256":anchors["config_file_sha256"],"supervisor":{"pid":300,"create_time":3.}},"terminal":deepcopy(sup)}
        raw=lambda value:(d.canonical(value)+"\n").encode()
        source=b"CPU_FIXTURE_SOURCE_NOT_EXECUTED";source_sha=hashlib.sha256(source).hexdigest()
        frame["scope"]["source_pins"]={"CPU_SOURCE":source_sha};maps=frame["inventory"]["maps"];maps["source"]=frame["scope"]["source_pins"]
        state=frame["result"]["capsule_chain"]["capsule"]
        state["start_proof"]["owned_scope_identity"]=deepcopy(owner)
        state["start_proof"]["actual_pin_maps"]["source"]=frame["scope"]["source_pins"]
        state["end_proof"]["actual_pin_maps_after"]["source"]=frame["scope"]["source_pins"]
        state["start_proof"]["identity_common"]["source_byte_maps_sha256"]=d.digest(state["start_proof"]["actual_pin_maps"])
        state["end_proof"]["identity_common"]=deepcopy(state["start_proof"]["identity_common"])
        frame["scope"]=d.seal({k:v for k,v in frame["scope"].items() if k!="sha256"})
        node=frame["result"]["capsule_chain"];node["capsule"]=d.seal({k:v for k,v in state.items() if k!="sha256"})
        node["terminal"]["capsule_sha256"]=node["capsule"]["sha256"];node["promotion"]["capsule_sha256"]=node["capsule"]["sha256"]
        node["promotion"]["terminal_sha256"]=d.digest(node["terminal"]);frame["result"]["capsule_chain"]=d.seal({k:v for k,v in node.items() if k!="sha256"})
        self.sign_result(frame);frame["inventory"]=d.seal({k:v for k,v in frame["inventory"].items() if k!="sha256"})
        encoder=d.seal({"kind":"CPU_CANDIDATE_NO_NATIVE_REGISTRATION","runtime_pins":{},"complete_runtime_execution_closure_established":False})
        frame["bytes"]={name:raw(value) for name,value in (("config",frame["scope"]),("result",frame["result"]),
            ("supervisor",frame["supervisor"]),("inventory",frame["inventory"]),("encoder",encoder))}
        frame["selected_paths"]={name:name for name in frame["bytes"]}
        for role,key in (("config","config_file_sha256"),("result","result_file_sha256"),("supervisor","supervisor_file_sha256"),
            ("inventory","inventory_file_sha256"),("encoder","encoder_registration_file_sha256")):anchors[key]=hashlib.sha256(frame["bytes"][role]).hexdigest()
        # Bound hashes are updated in all selected objects before final bytes.
        config_hash=anchors["config_file_sha256"]
        frame["result"]["config_sha256"]=frame["supervisor"]["config_sha256"]=frame["lease"]["reservation"]["config_sha256"]=config_hash
        frame["lease"]["terminal"]=deepcopy(frame["supervisor"]);self.sign_result(frame)
        frame["bytes"]["result"]=raw(frame["result"]);frame["bytes"]["supervisor"]=raw(frame["supervisor"])
        anchors["result_file_sha256"]=hashlib.sha256(frame["bytes"]["result"]).hexdigest();anchors["supervisor_file_sha256"]=hashlib.sha256(frame["bytes"]["supervisor"]).hexdigest()
        frame["original_raw_bytes"]={"worker_result.json":frame["bytes"]["result"],"SUPERVISOR_RECEIPT.json":frame["bytes"]["supervisor"]}
        frame["preserved_raw_bytes"]=deepcopy(frame["original_raw_bytes"]);frame["source_bytes"]={"CPU_SOURCE":source};frame["source_snapshot_bytes"]=deepcopy(frame["source_bytes"])
        frame["preservation"]=d.seal({"raw_files":{name:{"bytes":len(value),"sha256":hashlib.sha256(value).hexdigest()} for name,value in frame["original_raw_bytes"].items()}})
        anchors["preservation_body_sha256"]=frame["preservation"]["sha256"];frame["CPU_fixture_driver"]=True
        return frame,anchors
    def test_whole_fixture_bytes_resource_state_cost_pipeline_stays_CPU_KIND(self):
        frame,anchors=self.whole();report=d.check_fixture_terminal(frame,anchors)
        self.assertEqual(d.CPU_KIND,report["kind"]);self.assertFalse(report["native_AUDIT_KIND_PASS_issued"])
        self.assertTrue(report["all_supplied_byte_anchors_checked"]);self.assertFalse(report["actual_native_file_IO_identity_or_encoder_registration_validated"])
    def test_whole_driver_byte_or_source_snapshot_substitution_denied(self):
        for section,key in (("bytes","result"),("preserved_raw_bytes","worker_result.json"),("source_snapshot_bytes","CPU_SOURCE")):
            frame,anchors=self.whole();frame[section][key]+=b"changed"
            with self.assertRaises(ValueError):d.check_fixture_terminal(frame,anchors)
    def test_whole_driver_resource_GPUfloor_RSS_identity_or_lease_failure(self):
        for kind in ("GPU","RSS","identity","lease"):
            frame,anchors=self.whole()
            if kind=="GPU":frame["resources"][1]["gpu_free_bytes"]=1
            elif kind=="RSS":frame["resources"][0]["worker_tree_rss_bytes"]=999
            elif kind=="identity":frame["resources"][0]["owned_process_identities"][0]["create_time"]=9.
            else:frame["lease"]["terminal"]={"forged":True}
            with self.assertRaises(ValueError):d.check_fixture_terminal(frame,anchors)

if __name__=="__main__":unittest.main()
