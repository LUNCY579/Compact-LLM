"""Staged terminal-driver composition; CPU fixtures never native audit authority."""
from copy import deepcopy
from pathlib import Path
import hashlib,importlib.util,json,math,sys
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1] if HERE.name=="long_native_terminal_driver_work" and HERE.parent.name=="artifacts" else HERE
VALIDATOR=PACKAGE/"long_native_auditor.py"
VALIDATOR_SHA256="d04b7bea64272017277213107b2c619f3b9e5c47b810042446bcd1a925bec56a"
require=lambda value,message: None if value else (_ for _ in ()).throw(ValueError(message))
require(VALIDATOR.is_file() and not VALIDATOR.is_symlink()
    and hashlib.sha256(VALIDATOR.read_bytes()).hexdigest()==VALIDATOR_SHA256,"exact independently reviewed integrated validator bytes")
sys.path.insert(0,str(PACKAGE))
spec=importlib.util.spec_from_file_location("pinned_independent_long_native_validator",VALIDATOR)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
canonical,digest=v.canonical,v.digest
NATIVE_PASS_ENABLED=False
ENCODER_V2_CONSUMPTION_REVIEWED=False
CPU_KIND="CPU_LONG_TERMINAL_DRIVER_FIXTURE_CHECK_NOT_NATIVE_AUDIT"
def seal(value):return {**value,"sha256":digest(value)}
def closed_success(scope,result,sup,anchors):
    v.validate_anchors(v.SIGNAL,anchors);v.checksum(scope);v.checksum(result)
    require(scope["recipe"]["sha256"]==anchors["recipe_sha256"] and scope["recipe"]["context_cap"]==anchors["context_cap"]
        and scope["role"]==anchors["role"] and scope["purpose"]==v.contract.PURPOSE
        and scope["authorization"]==v.contract.AUTHORIZATION,"externally selected exact recipe/context/role/purpose")
    require(scope["owned_qualification_authorized"] is True and scope["staged_native_dispatch_forbidden"] is False
        and all(scope[k] is False for k in ("native_launch_authorized","native_long_context_qualified","study_training_authorized",
            "study_warmstart_allowed","guard_enrollment_authorized","scoring_authorized")),"explicit qualification-only scope")
    require(scope["resources"]==v.contract.native.RESOURCE_BOUNDS and scope["tolerances"]==v.contract.native.TOLERANCES
        and scope["dose"]==v.contract.work(scope["row"],scope["role"]),"unchanged dose/numerical/resource contract")
    require(result["status"]=="COMPLETE_NATIVE_LONG_SCOPE_PENDING_INDEPENDENT_AUDIT" and result["execution_scope"]==v.contract.NATIVE
        and result["role"]==scope["role"] and result["recipe_sha256"]==scope["recipe"]["sha256"]
        and result["source_commit"]==sup["preflight"]["source_commit"]==anchors["source_commit"]
        and result["config_sha256"]==sup["config_sha256"]==anchors["config_file_sha256"],"complete same-purpose native result/config/source")
    require(sup["status"]=="SUPERVISED_WORKER_EXITED_ZERO" and sup["returncode"]==0 and sup["partial_outputs_frozen"] is True
        and all(sup[k] is True for k in ("job_closed_kill_on_close","sampler_threads_terminal_verified","gpu_telemetry_cleanup_verified"))
        and sup["owned_worker_identity_currently_alive"] is False and sup["unjoined_sampler_threads"]==[],"completed owned terminal only")
    require(0<result["wall_seconds"]<=sup["elapsed_seconds"]<600
        and not any(result.get(k) for k in ("error","cleanup_error")) and not any(sup.get(k) for k in ("failure","containment_failure","sampler_shutdown_error")),
        "no failed/partial/deadline/cleanup contradiction")
    return True
def validate_inventory(scope,inventory,anchors):
    v.checksum(inventory)
    require(inventory["kind"]==v.INVENTORY_KIND and inventory["source_commit"]==anchors["source_commit"]
        and inventory["recipe_sha256"]==anchors["recipe_sha256"] and inventory["role"]==anchors["role"]
        and inventory["context_cap"]==anchors["context_cap"],"externally bound exact complete inventory")
    expected={"source":scope["source_pins"],"runtime":scope["runtime_pins"],"model":scope["model_pins"],"data":scope["data_pins"],
        "encoder_assets":{str(Path(scope["tokenizer_binding"]["model_dir"])/name):item["sha256"] for name,item in scope["tokenizer_binding"]["assets"].items()},
        "encoder_bindings":scope["tokenizer_binding"]["binding_sources"]}
    require(inventory["maps"]==expected and inventory["counts"]=={name:len(value) for name,value in expected.items()},
        "all exact externally selected path/hash maps, never count-only")
    require(inventory["previous_references"]==scope["references"]
        and inventory["encoder_registration"]==scope["encoder_registration"],"explicit previous-role/encoder external anchors")
    require(inventory["validator_file_sha256"]==VALIDATOR_SHA256,"pinned independently reviewed validator")
    return True
def validate_registration_pending(scope,encoder,stdlib,anchors):
    """Recognize evidence, but never treat CPU candidates as final V2 authority."""
    v.checksum(encoder);v.checksum(stdlib)
    require(scope["encoder_registration"]["sha256"]==anchors["encoder_registration_file_sha256"],"exact independently selected encoder registration file")
    require(encoder["kind"]=="SYNTH_LONG_ENCODER_RUNTIME_REGISTRATION_V1"
        and encoder.get("complete_runtime_execution_closure_established") is True,"complete encoder execution closure required")
    require(type(encoder.get("runtime_pins")) is dict and all(v.hash64(pin) for pin in encoder["runtime_pins"].values()),"typed encoder package byte roster")
    require(stdlib["kind"]=="STAGED_CPU_STDLIB_RUNTIME_SUPERSET_NOT_NATIVE_REGISTRATION_V1","known staged stdlib candidate, not native authorization")
    require(not ENCODER_V2_CONSUMPTION_REVIEWED,"V2 consumption remains separately reviewed")
    return {"status":"PENDING_REVIEWED_ENCODER_STDLIB_V2_CONSUMPTION","native_authority":False}
def validate_previous(scope,previous,anchors):
    wanted=list(v.contract.ROLES[:v.contract.ROLES.index(scope["role"])])
    require(len(previous)==len(wanted) and len(scope["references"] or[])==len(wanted),"every independently audited prior role")
    old_costs={"operations":{},"physical_calls":0,"input_positions":0};old_chain=None;jobs=set();items=[]
    for expected_role,frame,reference in zip(wanted,previous,scope["references"] or[]):
        require(set(reference)=={"result_path","result_sha256","supervisor_path","supervisor_sha256","audit_path","audit_sha256"},"closed independent prior file anchors")
        result,sup,audit=frame["result"],frame["supervisor"],frame["audit"]
        require(frame["result_file_sha256"]==reference["result_sha256"] and frame["supervisor_file_sha256"]==reference["supervisor_sha256"]
            and frame["audit_file_sha256"]==reference["audit_sha256"],"exact trusted predecessor result/SUP/audit files")
        v.checksum(result);v.checksum(audit)
        require(result["role"]==audit["role"]==sup["preflight"]["role"]==expected_role
            and result["execution_scope"]==v.contract.NATIVE and result["status"]=="COMPLETE_NATIVE_LONG_SCOPE_PENDING_INDEPENDENT_AUDIT"
            and result["recipe_sha256"]==audit["recipe_sha256"]==sup["preflight"]["recipe"]["sha256"]==scope["recipe"]["sha256"],
            "same source/context/recipe actual predecessor role")
        require(audit["kind"]==v.contract.AUDIT_KIND and audit["status"]=="PASS_NATIVE_LONG_SCOPE_NUMERICAL_MECHANICS_ONLY"
            and audit["purpose"]==v.contract.PURPOSE and audit["result_sha256"]==reference["result_sha256"]
            and audit["supervisor_sha256"]==reference["supervisor_sha256"] and audit["config_sha256"]==result["config_sha256"]==sup["config_sha256"]
            and audit["source_commit"]==result["source_commit"]==sup["preflight"]["source_commit"]
            and audit["study_training_authorized"] is False and audit["scoring_authorized"] is False,"actual externally selected native predecessor audit, no authority transfer")
        require(sup["status"]=="SUPERVISED_WORKER_EXITED_ZERO" and sup["returncode"]==0
            and all(sup[k] is True for k in ("job_closed_kill_on_close","sampler_threads_terminal_verified","gpu_telemetry_cleanup_verified"))
            and sup["owned_worker_identity_currently_alive"] is False,"prior Job/resources/worker terminal")
        capsule=v.chain_payload(result["capsule_chain"],scope["recipe"])
        require(capsule["CPU_fixture"] is False and capsule["role"]==expected_role and capsule["previous_capsule_sha256"]==old_chain,
            "native-only prior capsule and exact latest whole lineage")
        new=result["capsule_chain"]["cumulative_costs"]
        v.delta(new["operations"],old_costs["operations"])
        require(new["physical_calls"]>=old_costs["physical_calls"] and new["input_positions"]>=old_costs["input_positions"],"whole cumulative physical dose no rollback")
        job=capsule["start_proof"]["owned_scope_identity"]["job_name"]
        require(job not in jobs,"each independently fresh owned predecessor Job");jobs.add(job)
        old_costs=new;old_chain=result["capsule_chain"]["sha256"];items.append(result)
    return items,jobs
def validate_frame(frame,anchors,*,CPU_fixture_mode=False):
    """Composition over selected immutable JSON; does not authenticate file IO."""
    require(type(CPU_fixture_mode) is bool,"explicit fixture provenance")
    scope,result,sup=frame["scope"],frame["result"],frame["supervisor"]
    closed_success(scope,result,sup,anchors);validate_inventory(scope,frame["inventory"],anchors)
    v.validate_native_admission(scope,result,sup)
    previous,jobs=validate_previous(scope,frame["previous"],anchors)
    proof=v.validate_chain_memory(scope,result,previous);capsule=proof["capsule"]
    require(capsule["start_proof"]["owned_scope_identity"]["job_name"] not in jobs,"current scope must have a new owned Job")
    trace=v.validate_trace(scope,capsule);physical=v.validate_physical(scope,capsule)
    observed=v.validate_observed_state(scope,capsule,frame["owner"])
    if previous:
        require(capsule["start_proof"]["identity_common"]==previous[0]["capsule_chain"]["capsule"]["start_proof"]["identity_common"],
            "actual same independently observed source/model/encoder/dense-start/inference-start RNG and initial112 bytes")
    if scope["role"]=="bulk":
        v.validate_bulk_records(scope,capsule,previous)
        require(result["bulk_comparison"]==capsule["comparison"],"durable actual bulk every-step comparison result")
    else:require(capsule["comparison"] is None,"no bulk numerical summary on trace/stress roles")
    report={"kind":CPU_KIND if CPU_fixture_mode else "LONG_TERMINAL_CHECK_PREVIEW_NO_NATIVE_AUDIT",
        "status":"CPU_FIXTURE_CHECKS_ONLY" if CPU_fixture_mode else "CHECKED_JSON_PROVENANCE_PENDING_NATIVE_DRIVER",
        "role":scope["role"],"context_cap":anchors["context_cap"],"trace":trace,"physical":physical,"state_hash_observations":observed,
        "current_operations":proof["current_operations"],"native_AUDIT_KIND_PASS_issued":False,
        "resume_authority":False,"scoring_authorized":False,"study_training_authorized":False,
        "native_ready":False,"raw_RNG_restore_or_tensor_reread_verified":False}
    return seal(report)
def check_preserved_bytes(frame,original,preserved,config_info,inventory_info):
    """Pure exact byte descriptors; caller's native IO/authentication remains gated."""
    pres=frame["preservation"];v.checksum(pres)
    require(original==pres["raw_files"]==preserved,"every original/preserved raw file byte descriptor")
    require(config_info==pres["config_snapshot"] and inventory_info==pres["audit_inventory_snapshot"],"immutable selected config/inventory byte snapshots")
    require(set(original)=={item["path"] for item in frame["supervisor"]["partial_outputs"]}|{"SUPERVISOR_RECEIPT.json"},"complete frozen native raw roster")
    return True
def audit_terminal(attempt,preserved,config,inventory,*,terminal_signal,anchors,output_path=None):
    v.validate_anchors(terminal_signal,anchors)
    require(NATIVE_PASS_ENABLED and ENCODER_V2_CONSUMPTION_REVIEWED,
        "native terminal PASS emission disabled until reviewed complete encoder+stdlib V2 registration and actual terminal IO/identity driver")
    raise ValueError("actual native terminal IO driver is not enabled; no NativeAUDIT_KIND/PASS or resume authority")

def validate_resource_frame(frame):
    """Pure returned-record validation; never queries OS, devices or processes."""
    sup,owner=frame["supervisor"],frame["owner"];resources=frame["resources"]
    expected_bounds={"worker_process_tree_RSS_GiB":2.0,"system_available_reserve_GiB":1.25,"preworker_available_GiB":3.3125,
        "GPU_free_at_admission_GiB":4.0,"GPU_free_during_work_GiB":.5,"disk_free_GiB":4.0,"whole_scope_seconds":600,"external_watchdog_milliseconds":100}
    expected_cadence={"CPU_watchdog_seconds":.1,"CPU_freshness_seconds":.2,"GPU_query_interval_seconds":.5,
        "GPU_query_timeout_seconds":2.0,"GPU_freshness_seconds":2.0,"admission_interval_seconds":2.,"admission_deadline_seconds":30.,"admission_samples":2}
    require(sup["bounds"]==expected_bounds and sup["cadence"]==expected_cadence,"all fixed resource/freshness/cadence gates")
    admissions=frame["admissions"];require(len(admissions)==2,"two current admission observations")
    require(all(row["gpu_free_bytes"]>=4*1024**3 and row["host_available_bytes"]>=3.3125*1024**3 and row["disk_free_bytes"]>=4*1024**3
        for row in admissions) and admissions[1]["sample_started_monotonic"]-admissions[0]["sample_started_monotonic"]>=2,"unchanged fresh admission amounts/interval")
    cpu=[r for r in resources if r["kind"]=="CPU"];gpu=[r for r in resources if r["kind"]=="GPU"]
    require(cpu and gpu and len(cpu)+len(gpu)==len(resources),"complete returned CPU/GPU record inventory")
    identities={}
    for row in cpu:
        require(all(type(row[k]) is int and row[k]>=0 for k in ("host_available_bytes","worker_tree_rss_bytes","disk_free_bytes"))
            and row["host_available_bytes"]>=1.25*1024**3 and row["worker_tree_rss_bytes"]<=2*1024**3 and row["disk_free_bytes"]>=4*1024**3,
            "typed returned host/RSS/disk floors")
        require(row["sample_started_monotonic"]<=row["sample_completed_monotonic"] and row["sample_completed_monotonic"]-row["sample_started_monotonic"]<=.2,"CPU sample service/freshness")
        require(sum(item["rss_bytes"] for item in row["owned_process_samples"])==row["worker_tree_rss_bytes"]
            and row["owned_process_identities"]==[{k:item[k] for k in ("pid","create_time","creation_filetime_100ns")} for item in row["owned_process_samples"]],"owned PID/createTime/FILETIME/RSS closure")
        for item in row["owned_process_samples"]:
            require(type(item["pid"]) is int and item["pid"]>0 and type(item["creation_filetime_100ns"]) is int and item["creation_filetime_100ns"]>0
                and row["sample_started_monotonic"]<=item["sample_started_monotonic"]<=item["sample_completed_monotonic"]<=row["sample_completed_monotonic"],"owned process typed identity/timing")
            require(identities.setdefault(item["pid"],item["create_time"])==item["create_time"],"no reused sampled identity")
    uuid=admissions[0]["gpu_uuid"]
    for row in gpu:
        require(type(row["GPU_index"]) is int and row["GPU_index"]==0 and row["gpu_uuid"]==uuid and bool(row["gpu_observation_id"])
            and type(row["gpu_free_bytes"]) is int and type(row["gpu_total_bytes"]) is int and .5*1024**3<=row["gpu_free_bytes"]<=row["gpu_total_bytes"],
            "typed current same-device GPU floor")
        require(row["gpu_sample_started_monotonic"]<=row["gpu_sample_completed_monotonic"] and row["gpu_sample_completed_monotonic"]-row["gpu_sample_started_monotonic"]<=2,"GPU service/freshness")
    for rows,start,end,maxage in ((cpu,"sample_started_monotonic","sample_completed_monotonic",.2),(gpu,"gpu_sample_started_monotonic","gpu_sample_completed_monotonic",2.)):
        require(all(b[start]>=a[start] and b[end]-a[start]<=maxage for a,b in zip(rows,rows[1:])),"returned sample cadence/publication freshness")
    hb=frame["heartbeat"]
    require(hb and all(item["supervisor_pid"]==owner["supervisor_pid"] for item in hb)
        and all(0<=b["unix_seconds"]-a["unix_seconds"]<=.2 for a,b in zip(hb,hb[1:])),"same supervisor heartbeat chronology/freshness")
    identity=sup["worker_identity"];ownership=frame["ownership"]
    require(identity["assigned_before_resume"] is True and identity["job_kill_on_close"] is True and identity["breakaway_allowed"] is False
        and identity["pid"]==owner["worker_pid"] and identity["job_name"]==owner["job_name"]==ownership["job_name"]
        and ownership["owned_job_membership_verified"] is True and ownership["query_job_handle_closed_before_framework_import"] is True
        and ownership["launcher_root_pid"]==identity["pid"] and ownership["actual_worker_pid"] in identities,"actual fresh owned interpreter/launcher Job")
    resume=frame["lease"]["resume"];reservation=frame["lease"]["reservation"]
    require(frame["lease"]["worker"]==identity==resume["worker_identity"] and resume["external_monitor_ready_before_resume"] is True
        and resume["resumed_monotonic"]==sup["worker_resume_monotonic"] and resume["deadline_monotonic"]==sup["deadline_monotonic"]
        and reservation["config_sha256"]==sup["config_sha256"] and reservation["supervisor"]=={"pid":owner["supervisor_pid"],"create_time":owner["supervisor_create_time"]}
        and frame["lease"]["terminal"]==sup,"durable owned reservation/worker/resume/terminal chain")
    direct=frame["direct_GPU"]
    require(direct["CUDA_initialized"] is True and all(type(direct[k]) is int and direct[k]>=0 for k in ("peak_allocated_bytes","peak_reserved_bytes","free_bytes_final"))
        and direct["peak_reserved_bytes"]>=direct["peak_allocated_bytes"] and direct["free_bytes_final"]>=.5*1024**3,"direct terminal allocator/current free observations")
    return {"CPU_samples":len(cpu),"GPU_samples":len(gpu),"observed_owned_identities":identities,"OS_query_performed":False}
def check_frame_bytes(frame,anchors):
    """All supplied bytes authenticate to independently selected external anchors."""
    blobs=frame["bytes"];selected=frame["selected_paths"]
    for role,key in (("config","config_file_sha256"),("result","result_file_sha256"),("supervisor","supervisor_file_sha256"),
        ("inventory","inventory_file_sha256"),("encoder","encoder_registration_file_sha256")):
        raw=blobs[selected[role]];require(type(raw) is bytes and hashlib.sha256(raw).hexdigest()==anchors[key],"external exact "+role+" bytes")
    payloads={"config":frame["scope"],"result":frame["result"],"supervisor":frame["supervisor"],"inventory":frame["inventory"]}
    for role,payload in payloads.items():
        decoded=json.loads(blobs[selected[role]],object_pairs_hook=v.contract.native._pairs,parse_constant=v.contract.native._bad_constant)
        require(canonical(decoded)==canonical(payload),"byte-selected "+role+" object cannot be substituted")
    pres=frame["preservation"];v.checksum(pres);require(pres["sha256"]==anchors["preservation_body_sha256"],"externally selected preservation body")
    original={name:{"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()} for name,raw in frame["original_raw_bytes"].items()}
    preserved={name:{"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()} for name,raw in frame["preserved_raw_bytes"].items()}
    require(original==preserved==pres["raw_files"],"all immutable original/preserved raw bytes, no missing/extra outputs")
    source={name:{"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()} for name,raw in frame["source_bytes"].items()}
    snapshots={name:{"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()} for name,raw in frame["source_snapshot_bytes"].items()}
    require(source==snapshots and {name:item["sha256"] for name,item in source.items()}==frame["scope"]["source_pins"],"all source bytes/current preserved snapshots exact")
    return True
def check_fixture_terminal(frame,anchors):
    """Whole fixture path is hard-labelled and cannot issue declared native KIND."""
    require(frame.get("CPU_fixture_driver") is True,"fixture driver label required")
    check_frame_bytes(frame,anchors)
    report=validate_frame(frame,anchors,CPU_fixture_mode=True)
    resource=validate_resource_frame(frame)
    return seal({**{k:value for k,value in report.items() if k!="sha256"},"resource_fixture":resource,
        "all_supplied_byte_anchors_checked":True,"actual_native_file_IO_identity_or_encoder_registration_validated":False})

