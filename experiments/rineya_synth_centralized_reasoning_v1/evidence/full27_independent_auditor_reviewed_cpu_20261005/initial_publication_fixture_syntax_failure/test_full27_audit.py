"""CPU fixtures for pure auditor validators; never emit native audit/resume authority."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(PACKAGE))
import auditor as a
from rs_contract import canonical,digest
class Controls(unittest.TestCase):
    def setUp(self):
        self.scope={"sha256":"","contract":a.contract.VERSION,"purpose":a.contract.PURPOSE,"authorization":deepcopy(a.contract.AUTHORIZATION),
            "owned_qualification_authorized":True,"profile":a.contract.PROFILE,"staged_native_dispatch_forbidden":False,
            "native_launch_authorized":False,"study_training_authorized":False,"study_warmstart_allowed":False,
            "continuous27_bitwise_parity_established":False,"qualification_checkpoints_never_final_study":True,
            "resources":deepcopy(a.contract.native.RESOURCE_BOUNDS),"tolerances":deepcopy(a.contract.native.TOLERANCES),
            "base_compute_profile":"CPU_FIXTURE_NOT_NATIVE","checkpoint_amendment":{"CPU_fixture":True},"worker_ram_amendment":{"CPU_fixture":True}}
        maps={kind:{"fixture_%s_%d"%(kind,i):"a"*64 for i in range(count)} for kind,count in (("source",31),("runtime",83),("data",8),("model",7))}
        for section,key in a.SECTIONS.items():self.scope[key]=maps[section]
        self.scope["recipe"]={"qualification_source_pins":maps["source"],"purpose":a.contract.PURPOSE,"study_warmstart_allowed":False}
        self.scope["recipe"]["sha256"]=digest(self.scope["recipe"]);self.sign(self.scope)
        self.config=PACKAGE/"configs/FIXTURE_NEVER_CREATED.json"
        self.inventory={"kind":a.INVENTORY_KIND,"schema_version":1,"source_commit":"b"*40,
            "root_path":str(a.ROOT.resolve()),"package_path":str(PACKAGE.resolve()),"config_path":str(self.config.resolve()),
            "config_file_sha256":"c"*64,"inventories":maps,"counts":{"source":31,"runtime":83,"data":8,"model":7},
            "profile_body_sha256":digest({k:self.scope[k] for k in ("profile","base_compute_profile","checkpoint_amendment","worker_ram_amendment")})}
        self.sign(self.inventory);self.sup={"preflight":{"source_commit":"b"*40}}
    @staticmethod
    def sign(value):value["sha256"]=digest({k:v for k,v in value.items() if k!="sha256"})
    def check_admission(self):return a.admit_inventory(self.scope,self.sup,self.inventory,self.config,"c"*64)
    def test_exact_maps_CPU_fixture_only_positive(self):
        self.assertEqual({"source":31,"runtime":83,"data":8,"model":7},self.check_admission())
        self.assertFalse(self.scope["study_warmstart_allowed"])
    def test_same_lengths_wrong_source_hash_is_denied(self):
        self.scope["source_pins"]=deepcopy(self.scope["source_pins"]);self.scope["source_pins"]["fixture_source_0"]="d"*64;self.sign(self.scope)
        with self.assertRaisesRegex(ValueError,"frozen path/hash"):self.check_admission()
    def test_source_commit_config_profile_wrong_authority_denied(self):
        original=deepcopy(self.inventory)
        for key,value in (("source_commit","e"*40),("config_file_sha256","d"*64),("profile_body_sha256","d"*64)):
            self.inventory=deepcopy(original);self.inventory[key]=value;self.sign(self.inventory)
            with self.assertRaises(ValueError):self.check_admission()
        self.inventory=original;self.scope["study_warmstart_allowed"]=True;self.sign(self.scope)
        with self.assertRaisesRegex(ValueError,"warmstart"):self.check_admission()
    def test_old_inventory_or_resigned_resource_lowering_denied(self):
        self.inventory["counts"]["runtime"]=82;self.sign(self.inventory)
        with self.assertRaises(ValueError):self.check_admission()
        self.setUp();self.scope["resources"]["GPU_free_during_work_GiB"]=.1;self.sign(self.scope)
        with self.assertRaisesRegex(ValueError,"resource/numerical"):self.check_admission()
    def test_terminal_signal_and_all_independent_anchors_required_before_IO(self):
        with self.assertRaisesRegex(ValueError,"preservation signal"):a.audit("missing","missing","missing","missing",terminal_signal=None,expected={})
        with self.assertRaisesRegex(ValueError,"trust anchors"):a.audit("missing","missing","missing","missing",terminal_signal=a.SIGNAL,expected={})
    def test_delta_does_not_erase_constructor_or_doublecount_previous(self):
        prior={"load":dict(attempted=1,returned=1,failed=0,pending=0)}
        current={"load":dict(attempted=2,returned=2,failed=0,pending=0),"step":dict(attempted=9,returned=9,failed=0,pending=0)}
        self.assertEqual({"load":prior["load"],"step":current["step"]},a.counts_delta(current,prior))
        with self.assertRaisesRegex(ValueError,"disappear"):a.counts_delta({"step":current["step"]},prior)
        current["load"]["returned"]=0
        with self.assertRaises(ValueError):a.counts_delta(current,prior)
    def events(self,path,items):path.write_text("".join(canonical(item)+"\n" for item in items),encoding="utf-8",newline="\n")
    def test_journal_attempt_return_publication_exact_bytes_CPU_only(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            leaf=Path(temporary);payload={"CPU_fixture_only":True,"native_PASS":False};raw=(canonical(payload)+"\n").encode()
            pub=leaf/"fixture.json";pub.write_bytes(raw)
            descriptor={"path":str(pub),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"payload_sha256":digest(payload),"durable":True}
            events=leaf/"events.jsonl";self.events(events,[{"event":"attempt","id":1,"operation":"fixture","utc":"2026-10-05T00:00:00+00:00"},
                {"event":"qualification_measurement_preserved","name":"fixture","publication":descriptor,"utc":"2026-10-05T00:00:00+00:00"},
                {"event":"returned","id":1,"operation":"fixture","utc":"2026-10-05T00:00:00+00:00"}])
            value=a.validate_events(events,leaf);self.assertEqual(1,value["API_calls"]);self.assertEqual(payload,value["publications"][0]["payload"])
            pub.write_bytes(b"forged")
            with self.assertRaises(ValueError):a.validate_events(events,leaf)
    def test_pending_failed_duplicate_or_noncontiguous_events_denied(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            leaf=Path(temporary);events=leaf/"events.jsonl";attempt={"event":"attempt","id":1,"operation":"fixture"}
            variants=[[attempt],[attempt,{"event":"failed","id":1,"operation":"fixture"}],
                [attempt,{"event":"returned","id":1,"operation":"fixture"},{"event":"returned","id":1,"operation":"fixture"}],
                [{"event":"attempt","id":2,"operation":"fixture"}]]
            for items in variants:
                self.events(events,items)
                with self.assertRaises(ValueError):a.validate_events(events,leaf)
    def test_publication_escape_symlink_duplicate_JSON_and_partial_line_denied(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            leaf=Path(temporary);outside=HERE/"not_real_owned_file";events=leaf/"events.jsonl"
            self.events(events,[{"event":"qualification_measurement_preserved","name":"fixture","utc":"2026-10-05T00:00:00+00:00",
                "publication":{"path":str(outside),"bytes":1,"sha256":"a"*64,"payload_sha256":"b"*64,"durable":True}}])
            with self.assertRaises(ValueError):a.validate_events(events,leaf)
            events.write_bytes(b'{"event":"attempt","event":"returned"}\n')
            with self.assertRaises(ValueError):a.validate_events(events,leaf)
            events.write_bytes(b'{"event":"attempt","id":1,"operation":"fixture"}')
            with self.assertRaises(ValueError):a.validate_events(events,leaf)
    def reference(self):
        names=["fixture_%d"%i for i in range(112)];witness={"parameter_order":names,"adapters":{name:{"shape":[1]} for name in names}}
        previous={"sha256":"a"*64};rows=[{"total_tokens":3} for _ in range(20)]
        report={"passed":True,"count":112,"atol":1e-6,"rtol":1e-5,"reference_state_preserved_sha256":"a"*64,
            "scope":"UNQUALIFIED_NATIVE_BACKEND_QUALIFICATION_ADAPTER","native_launch_authorized":False,"study_warmstart_authorized":False,
            "reference_rows":20,"reference_input_positions":60,
            "errors":[{"name":name,"shape":[1],"finite":True,"passed":True,"maximum_absolute_error":0.} for name in names]}
        return report,previous,rows,witness
    def test_reference112_exact_preserved_cursor_shapes_rows_tolerances_CPU_schema_only(self):
        values=self.reference();self.assertTrue(a.verify_reference(*values))
        for key,value in (("count",111),("atol",.1),("reference_input_positions",59),("reference_state_preserved_sha256","b"*64)):
            current=deepcopy(values);current[0][key]=value
            with self.assertRaises(ValueError):a.verify_reference(*current)
        current=deepcopy(values);current[0]["errors"][1]["name"]=current[0]["errors"][0]["name"]
        with self.assertRaises(ValueError):a.verify_reference(*current)
        current=deepcopy(values);current[0]["errors"][0]["maximum_absolute_error"]=float("nan")
        with self.assertRaises(ValueError):a.verify_reference(*current)
    def test_failure_terminal_cannot_be_relabelled_native_success_or_resume(self):
        worker={"status":"FAILED_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED","sha256":""};self.sign(worker)
        sup={"status":"OWNED_WORKER_ABORTED","returncode":137,"partial_outputs_frozen":True}
        with self.assertRaisesRegex(ValueError,"successful frozen"):a.verify_terminal({},worker,sup,HERE)
    def test_inert_fixture_tests_cannot_create_native_PASS_or_import_framework(self):
        self.assertFalse(any(n.split(".",1)[0] in a.chunks.FRAMEWORKS for n in sys.modules))
        self.assertNotIn("status",self.scope)
        with self.assertRaises(ValueError):a.validate_anchors(a.SIGNAL,{"CPU_fixture":True})

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


class PublicationFixtureControls(unittest.TestCase):
    """Generated JSON/region/schema fixtures, deliberately not native archives."""
    def fixture(self,leaf,chunk=1,previous=None):
        leaf=Path(leaf);leaf.mkdir(parents=True,exist_ok=True);arm="M1"
        updates=[{"update_ordinal":i,"learning_rate":a.contract.schedule.learning_rates()[i-1],
            "input_tokens":60,"supervised_tokens":20,"occurrence_row_ids":["CPU_FIXTURE_%03d"%j for j in range((i-1)*20,i*20)]} for i in range(1,28)]
        registration={"arms":{arm:{"updates":updates}},"sha256":"CPU_SCHEMA_FIXTURE_NEVER_NATIVE"}
        recipe={"purpose":a.contract.PURPOSE,"arm":arm,"execution_scope":a.chunks.FAKE_SCOPE,"cohort_sha256":"a"*64,"backend_source_sha256":"b"*64}
        recipe["sha256"]=digest(recipe);binding=a.chunks.binding(registration,arm,chunk*9);groups=180 if chunk==1 else 200
        amendment={"counts":{"no_grad":56,"forward":groups*28,"recompute":groups*28,"per_layer_no_grad":2,
            "per_layer_forward":groups,"per_layer_recompute":groups}}
        dose={"maximum_forward_calls":groups+2,"expected_forward_input_positions":(groups+2)*3,
            "train_forwards":180,"reference_forwards":0 if chunk==1 else20,"zero_parity_forwards":2,
            "reference_autograd_calls":0 if chunk==1 else20,"native_base_content_checks":15 if chunk==1 else19}
        scope={"chunk":chunk,"arm":arm,"registration":registration,"recipe":recipe,"checkpoint_amendment":amendment,
            "dose":dose,"rows":[{"total_tokens":3,"supervised_tokens":1} for _ in range(540)]}
        region_rows=[];identity=0;layers={}
        for layer in range(28):
            layers[str(layer)]={}
            for phase,count in (("no_grad",2),("forward",groups)):
                for _ in range(count):
                    identity+=1;region_rows.append({"region_id":identity,"layer":layer,"phase":phase,"status":"returned","elapsed_seconds":0.})
                    if phase=="forward":region_rows.append({"region_id":identity,"layer":layer,"phase":"recompute","status":"returned","elapsed_seconds":0.})
            for phase,count in (("no_grad",2),("forward",groups),("recompute",groups)):
                layers[str(layer)][phase]={"attempted":count,"returned":count,"failed":0,"pending":0,"elapsed_seconds":0.}
        region=leaf/"decoder_regions.jsonl";region.write_text("".join(canonical(row)+"\n" for row in region_rows),encoding="utf-8",newline="\n")
        decoder={"use_reentrant":False,"checkpoint_early_stop":False,"preserve_rng_state":True,"failed_or_pending":False,
            "removed":True,"counters":{"forward_regions":groups*28,"recompute_regions":groups*28,"no_grad_regions":56},
            "layers":layers,"region_file":{"path":str(region),"rows":len(region_rows),**a.file_info(region)}}
        witness={"sha256":"c"*64,"parameter_order":["CPU_FIXTURE_GRAD_%d"%i for i in range(112)],
            "adapters":{"CPU_FIXTURE_GRAD_%d"%i:{"shape":[1]} for i in range(112)},"CPU_fixture_not_native_tensors":True}
        reference=None
        if chunk>1:
            reference={"scope":"UNQUALIFIED_NATIVE_BACKEND_QUALIFICATION_ADAPTER","native_launch_authorized":False,
                "study_warmstart_authorized":False,"passed":True,"count":112,"atol":1e-6,"rtol":1e-5,
                "reference_state_preserved_sha256":previous["qualification_witness"]["sha256"],
                "reference_rows":20,"reference_input_positions":60,
                "errors":[{"name":name,"shape":[1],"finite":True,"passed":True,"maximum_absolute_error":0.} for name in witness["parameter_order"]]}
        physical={"attempted_calls":groups+2,"input_positions":(groups+2)*3,
            "operations":{"train_forwards":180,"reference_forwards":0 if chunk==1 else20,"zero_parity_forwards":2}}
        evidence={"purpose":a.contract.PURPOSE,"binding":binding,"witness":witness,"reference_gradient_comparison":reference,
            "physical_dose":physical,"decoder_work":decoder,"cleanup_completed":True,"study_warmstart_allowed":False,
            "continuous27_bitwise_parity_established":False,"CPU_fake_control":False,"CPU_schema_fixture_never_native_audit":True}
        evidence["sha256"]=digest(evidence)
        archive=leaf/("qualification_C%d.pt"%(chunk*9));archive.write_bytes(b"CPU_SCHEMA_ONLY_NOT_A_ZIP_OR_NATIVE_STATE")
        metadata={"binding":binding,"arm":arm,"seed":101,"scope":a.chunks.FAKE_SCOPE,"recipe_sha256":recipe["sha256"],
            "cohort_sha256":recipe["cohort_sha256"],"backend_source_sha256":recipe["backend_source_sha256"],
            "native_launch_authorized":False,"training_authorized":False,"native_resume_qualified":False,"qualification_STATE1_used":False,
            "fresh_start_attestation_sha256":"d"*64}
        descriptor={"schema_version":1,"kind":a.chunks.BACKEND_KIND,"path":str(archive),**a.file_info(archive),
            "metadata":metadata,"metadata_sha256":digest(metadata)}
        protocol={name:dict(attempted=0,returned=0,failed=0,pending=0) for name in a.chunks.OPERATIONS}
        protocol_counts={"gradient_reset":chunk*9,"forward":chunk*180,"completion_loss":chunk*180,"backward":chunk*180,
            "finite_gradient_clip":chunk*9,"optimizer_step":chunk*9,"checkpoint_save":chunk,"checkpoint_reload":chunk-1,
            "checkpoint_publish":chunk-1,"terminal_costs_publish":chunk-1,"checkpoint_promote":chunk-1}
        for name,n in protocol_counts.items():protocol[name].update(attempted=n,returned=n)
        names={"native_base_content_verify":15 if chunk==1 else19,"training_forward":180,"completion_CE_selected_NLL_check":180,
            "finite_gradient_clip":9,"training_backward":180,"gradient_reset":9,"optimizer_step":9,"consumed_gradient_clear":9,
            "zero_parity_base_forward":1,"zero_parity_adapter_forward":1,"reference_forward":0 if chunk==1 else20,
            "reference_autograd_grad":0 if chunk==1 else20,"qualification_gradient_compare":int(chunk>1),
            "model_construct":1,"checkpoint_load":1,"tensor_copy":310,"checkpoint_write":1,"checkpoint_serialize":1,
            "qualification_owned_hooks_remove":1,"qualification_decoder_regions_write":1,"qualification_decoder_regions_verify":1}
        for layer in range(28):
            for phase,count in (("no_grad",2),("forward",groups),("recompute",groups)):names["checkpoint_layer%02d_%s"%(layer,phase)]=count
        delta={name:dict(attempted=n,returned=n,failed=0,pending=0) for name,n in names.items() if n}
        before={} if previous is None else previous["costs"]["primitive_operations"]
        posted={"schema_version":1,"scope":a.chunks.KIND,"protocol":protocol,
            "positions":{"input_attempted":binding["input_tokens"],"input_returned":binding["input_tokens"],
                "supervised_attempted":binding["supervised_tokens"],"supervised_returned":binding["supervised_tokens"]},
            "primitive_operations":a.chunks._merge(before,delta)}
        envelope={"schema_version":1,"kind":a.chunks.KIND,"arm":arm,"seed":101,"chunk_ordinal":chunk,
            "recipe":recipe,"binding":binding,"backend_descriptor":descriptor,"qualification_evidence":evidence,
            "posted_save_costs":posted,"native_resume_qualified":False,"training_authorized":False}
        envelope["sha256"]=digest(envelope)
        terminal_costs=a.chunks._project(a.chunks._project(posted,"checkpoint_publish"),"terminal_costs_publish")
        terminal={"envelope_sha256":envelope["sha256"],"costs":terminal_costs,"costs_sha256":digest(terminal_costs)}
        promoted=a.chunks._project(terminal_costs,"checkpoint_promote")
        promotion={"envelope_sha256":envelope["sha256"],"durable_terminal_sha256":digest(terminal),"terminal_costs":promoted}
        publications=[]
        def publish(name,payload):
            path=leaf/(name+".json");raw=(canonical(payload)+"\n").encode();path.write_bytes(raw)
            item={"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"payload_sha256":digest(payload),"durable":True}
            publications.append({"name":name,"publication":item,"payload":deepcopy(payload),"line":100000+len(publications)})
            return item
        publish("zero_parity",{"equal":True,"max_error":0.,"all_B_zero":True,"CPU_fixture_only":True})
        if chunk>1:
            publish("restored_boundary_witness",previous["qualification_witness"]);publish("reference_gradient_comparison",reference)
        publish("detached_boundary_witness",witness)
        publication=publish("checkpoint_publish",envelope);terminal_publication=publish("terminal_costs_publish",terminal)
        promotion_publication=publish("checkpoint_promote",promotion)
        resume={"schema_version":1,"kind":a.chunks.KIND,"status":"SUCCESSFUL_FULL27_QUALIFICATION_CHUNK_PROTOCOL_ONLY",
            "envelope":envelope,"publication":publication,"durable_terminal":terminal,"terminal_publication":terminal_publication,
            "promotion":promotion,"promotion_publication":promotion_publication,"terminal_costs":promoted}
        resume["sha256"]=digest(resume)
        worker={"resume_descriptor":resume,"costs":promoted,"qualification_witness":witness,"physical_dose":physical}
        current=a.counts_delta(promoted["primitive_operations"],before);ordered=[];ordinal=0;line=0
        for name,item in current.items():
            for _ in range(item["returned"]):
                ordinal+=1;line+=1;ordered.append((ordinal,name,"attempt",line));line+=1;ordered.append((ordinal,name,"returned",line))
        worker["operation_accounting"]={"scope":"registered_API_boundaries;internal_ATen_operations_not_individually_metered",
            "operations":current,"last_call":ordinal}
        events={"publications":publications,"counts":current,"API_calls":ordinal,"ordered":ordered}
        return scope,worker,events
    def test_complete_posted_terminal_promotion_chain_fixture_and_second_samearm_scope(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            first=Path(root)/"1";scope,worker,events=self.fixture(first)
            proof=a.verify_publication_chain(scope,worker,events,first,None)
            self.assertEqual(182,proof["physical"]["attempted_calls"]);self.assertNotIn("status",proof)
            second=Path(root)/"2";scope2,worker2,events2=self.fixture(second,2,worker)
            proof2=a.verify_publication_chain(scope2,worker2,events2,second,worker)
            self.assertEqual(202,proof2["physical"]["attempted_calls"])
            self.assertEqual(18,proof2["envelope"]["binding"]["completed_updates"])
    def test_changed_publication_or_latest_cost_rollback_denied(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            scope,worker,events=self.fixture(root)
            worker["costs"]["positions"]["input_returned"]-=1
            with self.assertRaises(ValueError):a.verify_publication_chain(scope,worker,events,root,None)
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            scope,worker,events=self.fixture(root)
            Path(worker["resume_descriptor"]["publication"]["path"]).write_bytes(b"wrong")
            with self.assertRaises(ValueError):a.verify_publication_chain(scope,worker,events,root,None)
    def test_missing_pair_failed_region_or_native_cost_omission_denied(self):
        for variant in ("pair","pending","cost"):
            with tempfile.TemporaryDirectory(dir=HERE) as root:
                scope,worker,events=self.fixture(root)
                if variant=="cost":events["counts"].pop("model_construct")
                else:
                    path=Path(root)/"decoder_regions.jsonl";lines=path.read_text().splitlines();rows=[json.loads(line) for line in lines]
                    if variant=="pair":rows[2]["region_id"]=999999
                    else:rows[2]["status"]="pending"
                    path.write_text("".join(canonical(row)+"\n" for row in rows),encoding="utf-8",newline="\n")
                with self.assertRaises(ValueError):a.verify_publication_chain(scope,worker,events,root,None)
    def test_cleanup_after_publication_or_physical_wrong_denominator_denied(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            scope,worker,events=self.fixture(root)
            events["ordered"]=[(i,n,status,200000 if n=="qualification_owned_hooks_remove" else line) for i,n,status,line in events["ordered"]]
            with self.assertRaisesRegex(ValueError,"before publication"):a.verify_publication_chain(scope,worker,events,root,None)
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            scope,worker,events=self.fixture(root);worker["physical_dose"]["attempted_calls"]=181
            with self.assertRaises(ValueError):a.verify_publication_chain(scope,worker,events,root,None)

if __name__=="__main__":unittest.main()
