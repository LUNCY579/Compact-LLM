"""Staged independent full27 terminal audit. No native execution or study authority."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,math,sys
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1] if HERE.name=="full27_audit_work" and HERE.parent.name=="artifacts" else HERE
ROOT=PACKAGE.parents[1]
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(HERE))
import full27_contract as contract
import full27_chunk as chunks
from rs_contract import canonical,digest,require
SIGNAL="ROOT_TERMINAL_AND_PRESERVED"
INVENTORY_KIND="SYNTH_FULL27_INDEPENDENT_AUDIT_INVENTORY_V1"
SECTIONS={"source":"source_pins","runtime":"runtime_pins","data":"data_pins","model":"model_pins"}
MAX_JSON=16*1024**2
MAX_EVENTS_BYTES=32*1024**2
CLOSED=("production_training_authorized","study_training_authorized","study_warmstart_allowed",
        "DEV_collection_authorized","scoring_authorized","protected_bank_access","quality_gain_established",
        "full_resumed9_qualified","full27_qualified","continuous27_bitwise_parity_established","automatic_retry_authorized")
def file_info(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),"regular immutable file required")
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for raw in iter(lambda:stream.read(1024**2),b""):value.update(raw)
    return {"bytes":path.stat().st_size,"sha256":value.hexdigest()}
def read_json(path,limit=MAX_JSON):
    path=Path(path);require(path.is_file() and not path.is_symlink(),"regular JSON file")
    with path.open("rb") as stream:raw=stream.read(limit+1)
    require(len(raw)<=limit,"bounded audit JSON")
    return json.loads(raw.decode("utf-8"),object_pairs_hook=contract.native._pairs,parse_constant=contract.native._bad_constant)
def checksum(value):
    require(type(value) is dict and value.get("sha256")==digest({k:v for k,v in value.items() if k!="sha256"}),"canonical body checksum")
def safe_child(path,leaf):
    path=Path(path);require(path.is_absolute() and not path.is_symlink() and path.is_file()
        and path.resolve().parent==Path(leaf).resolve(),"exact owned leaf file")
    return path
def inventory_path(path):
    path=Path(path).absolute();parent=PACKAGE/"artifacts/full27_audit_registrations"
    require(parent.is_dir() and not parent.is_symlink() and path.is_file() and not path.is_symlink()
            and path.resolve().parent==parent.resolve(),"fixed owned ignored audit inventory")
    return path
def admit_inventory(scope,sup,inventory,config_path,config_sha256):
    checksum(scope);checksum(inventory)
    require(inventory.get("kind")==INVENTORY_KIND and inventory.get("schema_version")==1,"known independent inventory")
    require(inventory.get("root_path")==str(ROOT.resolve()) and inventory.get("package_path")==str(PACKAGE.resolve())
        and inventory.get("config_path")==str(Path(config_path).resolve()) and inventory.get("config_file_sha256")==config_sha256,
        "independently selected root/package/config byte anchor")
    require(inventory.get("source_commit")==sup.get("preflight",{}).get("source_commit"),"trusted exact launch commit")
    require(scope.get("contract")==contract.VERSION and scope.get("purpose")==contract.PURPOSE
        and scope.get("authorization")==contract.AUTHORIZATION and scope.get("owned_qualification_authorized") is True
        and scope.get("profile")==contract.PROFILE and scope.get("staged_native_dispatch_forbidden") is False,
        "only integrated explicitly owned full27 qualification")
    require(all(scope.get(k) is False for k in ("native_launch_authorized","study_training_authorized","study_warmstart_allowed","continuous27_bitwise_parity_established"))
        and scope.get("qualification_checkpoints_never_final_study") is True,"qualification never study warmstart")
    require(scope.get("resources")==contract.native.RESOURCE_BOUNDS and scope.get("tolerances")==contract.native.TOLERANCES,
        "unchanged exact resource/numerical gates")
    inventories=inventory.get("inventories");counts=inventory.get("counts")
    require(type(inventories) is dict and set(inventories)==set(SECTIONS) and counts=={"source":31,"runtime":83,"data":8,"model":7},
        "independently frozen complete source31/runtime83/data8/model7 maps")
    for section,key in SECTIONS.items():
        require(type(inventories[section]) is dict and inventories[section]==scope.get(key)
            and len(inventories[section])==counts[section],"every frozen path/hash "+section)
    require(scope["recipe"]["qualification_source_pins"]==inventories["source"]
        and scope["recipe"]["sha256"]==digest({k:v for k,v in scope["recipe"].items() if k!="sha256"})
        and scope["recipe"]["purpose"]==contract.PURPOSE and scope["recipe"]["study_warmstart_allowed"] is False,
        "distinct qualification recipe exact executing source closure")
    require(inventory.get("profile_body_sha256")==digest({k:scope[k] for k in ("profile","base_compute_profile","checkpoint_amendment","worker_ram_amendment")}),
        "trusted complete compute/checkpoint/RAM profile")
    return deepcopy(counts)
def validate_events(path,leaf):
    require(file_info(path)["bytes"]<=MAX_EVENTS_BYTES,"bounded event stream")
    counts={};attempts={};returned={};publications=[];measurements=[];last_id=0;ordered=[];active=[];files=set()
    with Path(path).open("rb") as stream:
        for index,raw in enumerate(stream,1):
            require(len(raw)<=65536 and raw.endswith(b"\n"),"complete bounded journal record")
            event=json.loads(raw.decode(),object_pairs_hook=contract.native._pairs,parse_constant=contract.native._bad_constant)
            kind=event.get("event")
            if kind=="attempt":
                identity=event.get("id");name=event.get("operation")
                require(type(identity) is int and identity==last_id+1 and type(name) is str and name,"contiguous named API attempt")
                last_id=identity;attempts[identity]=name
                active.append(identity)
                item=counts.setdefault(name,dict(attempted=0,returned=0,failed=0,pending=0));item["attempted"]+=1;item["pending"]+=1
                ordered.append((identity,name,"attempt",index))
            elif kind in ("returned","failed"):
                identity=event.get("id");name=event.get("operation")
                require(identity in attempts and identity not in returned and attempts[identity]==name,"exact once attempted API outcome")
                require(active and active.pop()==identity,"synchronous nested API return order")
                returned[identity]=kind;item=counts[name];item["pending"]-=1;item[kind]+=1
                ordered.append((identity,name,kind,index))
            elif kind=="qualification_measurement_preserved":
                require(set(event)=={"event","name","publication","utc"} and type(event["name"]) is str,"closed qualification publication event")
                publication=event["publication"];safe_child(publication["path"],leaf)
                require(publication["path"] not in files,"no duplicate advertised evidence file");files.add(publication["path"])
                payload=read_json(publication["path"]);chunks._verify_publication(publication,payload)
                publications.append({"name":event["name"],"publication":deepcopy(publication),"payload":payload,"line":index})
            elif kind=="measurement_preserved":
                require(type(event.get("path")) is str and "/" not in event["path"] and "\\" not in event["path"],"bounded stage evidence filename")
                path=safe_child(str(Path(leaf)/event["path"]),leaf)
                require(str(path) not in files,"no duplicate advertised evidence file");files.add(str(path))
                require(file_info(path)["sha256"]==event["sha256"],"stage evidence event bytes")
                measurements.append({"path":path.name,"sha256":event["sha256"],"line":index})
            else:raise ValueError("unknown/failure journal event forbidden on successful audit")
    require(not active and all(item["failed"]==item["pending"]==0 for item in counts.values()) and len(attempts)==len(returned),"all recorded APIs returned without partial costs")
    return {"counts":counts,"API_calls":last_id,"publications":publications,"measurements":measurements,"ordered":ordered}
def verify_terminal(scope,worker,sup,leaf):
    checksum(worker)
    require(sup.get("status")=="SUPERVISED_WORKER_EXITED_ZERO" and sup.get("returncode")==0
        and sup.get("partial_outputs_frozen") is True,"successful frozen supervisor only; failed/partial cannot resume")
    require(worker.get("status")=="COMPLETE_QUALIFICATION_CHUNK_STUDY_UNAUTHORIZED"
        and worker.get("scope")==chunks.NATIVE_SCOPE and worker.get("qualification_purpose")==contract.PURPOSE
        and worker.get("arm")==scope["arm"] and worker.get("chunk_ordinal")==scope["chunk"],"complete same-arm native qualification worker")
    require(all(worker.get(k) is False for k in CLOSED),"closed worker study/scoring/quality authority")
    require(worker.get("recipe")==scope["recipe"] and worker.get("config_sha256")==sup.get("config_sha256")
        and worker.get("source_commit")==sup.get("preflight",{}).get("source_commit"),"worker/source/config joins")
    require(worker.get("prevalidated_whole_occurrences")==180 and worker.get("completed_updates")==scope["chunk"]*9,"nine full completed updates")
    expected=scope["registration"]["arms"][scope["arm"]]["updates"][(scope["chunk"]-1)*9:scope["chunk"]*9]
    summary=worker.get("update_summaries")
    require(type(summary) is list and len(summary)==9,"nine update summaries")
    for actual,want in zip(summary,expected):
        require(actual["update_ordinal"]==want["update_ordinal"] and actual["learning_rate"]==want["learning_rate"]
            and all(type(actual[k]) in (int,float) and math.isfinite(actual[k]) and actual[k]>=0 for k in ("mean_per_sequence_loss","gradient_norm")),
            "exact global LR/update with finite observed loss/preclip norm")
    require(type(worker.get("wall_seconds")) in (int,float) and math.isfinite(worker["wall_seconds"])
        and type(sup.get("elapsed_seconds")) in (int,float) and math.isfinite(sup["elapsed_seconds"])
        and 0<worker["wall_seconds"]<=sup["elapsed_seconds"]<600,"worker and whole scope within unchanged600s")
    return chunks.binding(scope["registration"],scope["arm"],scope["chunk"]*9)

def verify_scope(scope):
    require(scope["arm"] in ("M1","M2") and type(scope["chunk"]) is int and scope["chunk"] in (1,2,3),"fixed arm/chunk")
    rows=contract.packet(scope["registration"],scope["arm"])
    require(scope["rows"]==rows and scope["source_packet_sha256"]==digest(rows),"all540 exact frozen rows/order before outputs")
    require(scope["recipe"]==chunks.recipe(scope["registration"],scope["arm"],chunks.NATIVE_SCOPE)
        and scope["dose"]==contract.expected_work(scope["registration"],scope["arm"],scope["chunk"],rows),"current exact source/recipe/physical work")
    require(scope["checkpoint_amendment"]==contract.checkpoint_binding(scope["chunk"])
        and scope["worker_ram_amendment"]==contract.prerequisite.worker_ram_binding(),"unchanged checkpoint/RAM profile")
    require(scope["global_learning_rates"]==contract.schedule.learning_rates(),"global27 learning rates")
    require(scope["source_pins"]==contract.source_pins(),"complete current root qualification source31 bytes")
    require(scope["inference_qualification"]==contract.prerequisite.inference_binding()
        and scope["completed_B_prerequisite"]==contract.verify_completed_B(),"byte-bound audited prerequisite mechanics only")
    return rows
def counts_delta(current,previous):
    chunks._counts(current);chunks._counts(previous);delta={}
    require(all(name in current for name in previous),"prior primitive costs cannot disappear")
    for name,item in current.items():
        old=previous.get(name,dict(attempted=0,returned=0,failed=0,pending=0))
        require(all(item[k]>=old[k] for k in item),"cumulative primitive rollback")
        diff={k:item[k]-old[k] for k in item}
        require(diff["attempted"]==diff["returned"]+diff["failed"]+diff["pending"],"conserved current primitive delta")
        if any(diff.values()):delta[name]=diff
    return delta
def verify_previous(scope,inventory):
    previous=scope["previous"]
    if scope["chunk"]==1:
        require(previous is None and scope["previous_authority"] is None and scope["latest_costs"] is None
            and inventory.get("previous_authority") is None,"first chunk fresh/no warmstart")
        return None
    authority=scope["previous_authority"]
    require(authority==inventory.get("previous_authority"),"independently selected exact previous result/supervisor/audit pins")
    require(contract.verify_previous(authority,scope["registration"],scope["arm"],scope["chunk"],scope["recipe"])==previous,
        "same-arm previous successful native chunk audited and source-bound")
    require(scope["latest_costs"]==previous["costs"],"authoritative latest cumulative costs")
    return previous
def verify_publication_chain(scope,worker,events,leaf,previous):
    chunk=scope["chunk"];binding=chunks.binding(scope["registration"],scope["arm"],chunk*9)
    resume=worker["resume_descriptor"]
    # Validate this completed boundary as the previous boundary of the next
    # ordinal. _validate_resume has no dispatch path and permits completed27.
    envelope=chunks._validate_resume(resume,scope["registration"],scope["arm"],chunk+1,scope["recipe"],worker["costs"])
    evidence=envelope["qualification_evidence"]
    require(evidence.get("CPU_fake_control") is False and evidence.get("cleanup_completed") is True
        and evidence.get("continuous27_bitwise_parity_established") is False,"native evidence complete but study authority closed")
    require(evidence["binding"]==binding and worker["qualification_witness"]==evidence["witness"]
        and worker["physical_dose"]==evidence["physical_dose"],"detached exact saved boundary and physical dose")
    descriptor=envelope["backend_descriptor"];safe_child(descriptor["path"],leaf)
    require(Path(descriptor["path"]).name=="qualification_C%d.pt"%(chunk*9),"private qualification checkpoint only")
    publication_names=["zero_parity"]+(["restored_boundary_witness","reference_gradient_comparison"] if chunk>1 else[])+[
        "detached_boundary_witness","checkpoint_publish","terminal_costs_publish","checkpoint_promote"]
    pubs=events["publications"]
    require([item["name"] for item in pubs]==publication_names,"exact closed qualification publication sequence")
    by_name={item["name"]:item for item in pubs}
    for name,key in (("checkpoint_publish","publication"),("terminal_costs_publish","terminal_publication"),("checkpoint_promote","promotion_publication")):
        require(by_name[name]["publication"]==resume[key],"event to envelope/terminal/promotion exact file")
    require(by_name["detached_boundary_witness"]["payload"]==evidence["witness"],"published saved witness exact")
    parity=by_name["zero_parity"]["payload"]
    require(parity.get("equal") is True and parity.get("max_error")==0. and parity.get("all_B_zero") is True,"physical fresh zeroB parity exact")
    if chunk>1:
        expected=previous["resume_descriptor"]["envelope"]["qualification_evidence"]["witness"]
        require(by_name["restored_boundary_witness"]["payload"]==expected,"fresh actual restored boundary exactly prior detached state")
        reference=by_name["reference_gradient_comparison"]["payload"]
        require(reference==evidence["reference_gradient_comparison"],"reference report in durable envelope and own file")
        verify_reference(reference,expected,scope["rows"][(chunk-1)*180:(chunk-1)*180+20],evidence["witness"])
    else:require(evidence.get("reference_gradient_comparison") is None,"first chunk no reference dose")
    expected_physical={"attempted_calls":scope["dose"]["maximum_forward_calls"],
        "input_positions":scope["dose"]["expected_forward_input_positions"],
        "operations":{k:scope["dose"][k] for k in ("train_forwards","reference_forwards","zero_parity_forwards")}}
    require(evidence["physical_dose"]==expected_physical,"all training/reference/parity physical calls and positions")
    decoder=deepcopy(evidence["decoder_work"]);region=decoder.pop("region_file")
    contract.verify_decoder_file(decoder,region,scope["checkpoint_amendment"],leaf)
    old={} if previous is None else previous["costs"]["primitive_operations"]
    delta=counts_delta(worker["costs"]["primitive_operations"],old)
    require(delta==events["counts"],"every journal API joins current primitive delta; no erased loads/double count")
    accounting=worker["operation_accounting"]
    require(accounting=={"scope":"registered_API_boundaries;internal_ATen_operations_not_individually_metered",
        "operations":events["counts"],"last_call":events["API_calls"]},"worker current API accounting exact")
    require(all(v["failed"]==v["pending"]==0 for v in delta.values()),"no failed/pending primitive costs")
    expected_calls={"native_base_content_verify":scope["dose"]["native_base_content_checks"],
        "training_forward":180,"completion_CE_selected_NLL_check":180,"finite_gradient_clip":9,"training_backward":180,"gradient_reset":9,"optimizer_step":9,"consumed_gradient_clear":9,
        "zero_parity_base_forward":1,"zero_parity_adapter_forward":1,"reference_forward":scope["dose"]["reference_forwards"],
        "reference_autograd_grad":scope["dose"]["reference_autograd_calls"],
        "qualification_gradient_compare":int(chunk>1),"model_construct":1,"checkpoint_load":1,"tensor_copy":310,
        "checkpoint_write":1,"checkpoint_serialize":1,"qualification_owned_hooks_remove":1,
        "qualification_decoder_regions_write":1,"qualification_decoder_regions_verify":1}
    # Loss/clip protocol wrappers and actual native primitives stay distinct.
    for name,count in expected_calls.items():
        require(delta.get(name,dict(attempted=0,returned=0,failed=0,pending=0))==dict(attempted=count,returned=count,failed=0,pending=0),
            "exact current native work "+name)
    for index in range(28):
        for phase in ("no_grad","forward","recompute"):
            count=scope["checkpoint_amendment"]["counts"]["per_layer_"+phase]
            require(delta.get("checkpoint_layer%02d_%s"%(index,phase))==dict(attempted=count,returned=count,failed=0,pending=0),
                "all28 actual region primitive calls")
    returned_lines={name:max(line for _,n,status,line in events["ordered"] if n==name and status=="returned")
        for name in ("qualification_owned_hooks_remove","qualification_decoder_regions_verify")}
    require(max(returned_lines.values())<by_name["checkpoint_publish"]["line"],"verified hook cleanup/region bytes before publication")
    return {"envelope":envelope,"evidence":evidence,"descriptor":descriptor,"delta":delta,"physical":expected_physical}
def verify_reference(report,restored,rows,saved_witness):
    require(report.get("scope")=="UNQUALIFIED_NATIVE_BACKEND_QUALIFICATION_ADAPTER"
        and report.get("native_launch_authorized") is False and report.get("study_warmstart_authorized") is False,"reference control authority remains closed")
    require(report.get("passed") is True and report.get("count")==112 and report.get("atol")==1e-6 and report.get("rtol")==1e-5,
        "all112 exact reference tolerances")
    require(report.get("reference_state_preserved_sha256")==restored["sha256"]
        and report.get("reference_rows")==20 and report.get("reference_input_positions")==sum(r["total_tokens"] for r in rows),
        "reference20 complete rows/positions and exact preserved prior state")
    errors=report.get("errors");order=saved_witness["parameter_order"]
    require(type(errors) is list and len(errors)==112 and [item.get("name") for item in errors]==order,
        "all112 unique ordered reference gradient names")
    for item,name in zip(errors,order):
        require(item.get("shape")==saved_witness["adapters"][name]["shape"] and item.get("finite") is True and item.get("passed") is True
            and type(item.get("maximum_absolute_error")) in (int,float) and math.isfinite(item["maximum_absolute_error"]) and item["maximum_absolute_error"]>=0,
            "finite passed correct-shape per-gradient control")
    return True

def verify_resources(scope,sup,leaf):
    bounds={"worker_process_tree_RSS_GiB":2.0,"system_available_reserve_GiB":1.25,"preworker_available_GiB":3.3125,
        "GPU_free_at_admission_GiB":4.0,"GPU_free_during_work_GiB":.5,"disk_free_GiB":4.0,"whole_scope_seconds":600,
        "external_watchdog_milliseconds":100}
    cadence={"CPU_watchdog_seconds":.1,"CPU_freshness_seconds":.2,"GPU_query_interval_seconds":.5,
        "GPU_query_timeout_seconds":2.0,"GPU_freshness_seconds":2.0,"admission_interval_seconds":2.,
        "admission_deadline_seconds":30.,"admission_samples":2}
    require(sup.get("bounds")==bounds and sup.get("cadence")==cadence,"unchanged exact supervised amounts/cadence/freshness")
    require(sup.get("admission_only") is False and all(sup.get(k) is True for k in (
        "worker_attempted","worker_launched","worker_resumed","owned_handle_alive_verified_after_launch",
        "preflight_revalidated_before_current_observations","job_closed_kill_on_close",
        "sampler_threads_terminal_verified","gpu_telemetry_cleanup_verified")) and
        sup.get("owned_worker_identity_currently_alive") is False and sup.get("unjoined_sampler_threads")==[],
        "one completed owned launch/Job/sampler/NVML terminal closure")
    require(not any(sup.get(k) for k in ("failure","termination_error","containment_failure","sampler_shutdown_error","gpu_telemetry_cleanup_error")),
        "no contradictory terminal cleanup failure")
    admissions=sup.get("admission_observations")
    require(type(admissions) is list and len(admissions)==2,"two fresh native admissions")
    uuid=admissions[0]["gpu_uuid"]
    def cpu_fresh(row,now):
        vals=[row.get(k) for k in ("sample_started_monotonic","sample_completed_monotonic")]
        require(all(type(v) in (int,float) and math.isfinite(v) for v in vals)
            and vals[0]<=vals[1]<=now and 0<=now-vals[0]<=.2
            and all(type(row.get(k)) is int and row[k]>=0 for k in ("host_available_bytes","worker_tree_rss_bytes","disk_free_bytes")),"fresh typed chronological CPU envelope")
    def gpu_fresh(row,now):
        vals=[row.get(k) for k in ("gpu_sample_started_monotonic","gpu_sample_completed_monotonic")]
        require(all(type(v) in (int,float) and math.isfinite(v) for v in vals) and vals[0]<=vals[1]<=now and 0<=now-vals[0]<=2
            and row.get("GPU_index")==0 and type(row.get("GPU_index")) is int
            and type(row.get("gpu_uuid")) is str and bool(row["gpu_uuid"]) and row["gpu_uuid"]==uuid
            and type(row.get("gpu_observation_id")) is str and bool(row["gpu_observation_id"])
            and type(row.get("gpu_free_bytes")) is int
            and type(row.get("gpu_total_bytes")) is int and 0<=row["gpu_free_bytes"]<=row["gpu_total_bytes"],"fresh typed pinned GPU envelope")
    for row in admissions+[sup["dispatch_observation"]]:
        now=max(row["sample_completed_monotonic"],row["gpu_sample_completed_monotonic"])
        cpu_fresh(row,now);gpu_fresh(row,now)
        require(row["gpu_free_bytes"]>=4*1024**3 and row["host_available_bytes"]>=3.3125*1024**3
            and row["disk_free_bytes"]>=4*1024**3 and row["worker_tree_rss_bytes"]==0,"preworker amount floors")
    require(admissions[1]["sample_started_monotonic"]-admissions[0]["sample_started_monotonic"]>=2.
        and sup["dispatch_observation"]["gpu_observation_id"]==admissions[-1]["gpu_observation_id"],"fixed two-observation admission")
    owner=read_json(Path(leaf)/"supervisor_ownership.json");identity=sup["worker_identity"]
    ownership=read_json(Path(leaf)/"stage_worker_ownership_verified.json")
    require(identity["assigned_before_resume"] is True and identity["job_kill_on_close"] is True
        and identity["breakaway_allowed"] is False and owner["worker_pid"]==identity["pid"]
        and owner["job_name"]==identity["job_name"]==ownership["job_name"]
        and owner["config_sha256"]==sup["config_sha256"] and ownership["supervisor_pid"]==owner["supervisor_pid"]
        and ownership["launcher_root_pid"]==identity["pid"] and ownership["owned_job_membership_verified"] is True
        and ownership["query_job_handle_closed_before_framework_import"] is True,"owned launcher/actual child/Job before framework")
    resources=read_json(Path(leaf)/"RESOURCE_OBSERVATIONS.json")
    require(type(resources) is list and resources and all(row.get("kind") in ("CPU","GPU") for row in resources),"complete bounded CPU/GPU resource record inventory")
    cpu=[r for r in resources if r["kind"]=="CPU"];gpu=[r for r in resources if r["kind"]=="GPU"]
    require(len(cpu)>=2 and len(gpu)>=2 and cpu[0].get("phase")=="worker_suspended_startup"
        and cpu[0]["sample_completed_monotonic"]<=sup["worker_resume_monotonic"],"external ready monitor while owned worker suspended")
    observed={}
    for row in cpu:
        cpu_fresh(row,row["sample_completed_monotonic"])
        require(row["host_available_bytes"]>=1.25*1024**3 and row["worker_tree_rss_bytes"]<=2*1024**3
            and row["disk_free_bytes"]>=4*1024**3,"all returned CPU/RSS/disk amount gates")
        keys=("pid","create_time","creation_filetime_100ns")
        require([{k:item[k] for k in keys} for item in row["owned_process_samples"]]==row["owned_process_identities"]
            and sum(item["rss_bytes"] for item in row["owned_process_samples"])==row["worker_tree_rss_bytes"],"owned identity/RSS sum exact")
        for item in row["owned_process_samples"]:
            require(all(type(item[k]) is int and item[k]>0 for k in ("pid","creation_filetime_100ns"))
                and type(item["create_time"]) in (int,float) and math.isfinite(item["create_time"])
                and type(item["rss_bytes"]) is int and item["rss_bytes"]>=0
                and row["sample_started_monotonic"]<=item["sample_started_monotonic"]<=item["sample_completed_monotonic"]<=row["sample_completed_monotonic"],
                "all per-process identities and timing typed")
            previous=observed.setdefault(item["pid"],item["create_time"]);require(previous==item["create_time"],"no reused identity silently merged")
    for row in gpu:
        gpu_fresh(row,row["gpu_sample_completed_monotonic"]);require(row["gpu_free_bytes"]>=.5*1024**3,"all runtime GPU amount gates")
    for rows,start,end,age in ((cpu,"sample_started_monotonic","sample_completed_monotonic",.2),(gpu,"gpu_sample_started_monotonic","gpu_sample_completed_monotonic",2.)):
        require(all(b[start]>=a[start] and b[end]-a[start]<=age for a,b in zip(rows,rows[1:])),
            "returned sample cadence/publication gaps remain within registered freshness")
    hb=[];heartbeat=Path(leaf)/"supervisor_heartbeat.jsonl"
    require(file_info(heartbeat)["bytes"]<=1024**2,"bounded heartbeat")
    with heartbeat.open("rb") as stream:
        for raw in stream:
            require(len(raw)<=1024 and raw.endswith(b"\n"),"complete heartbeat line")
            value=json.loads(raw,object_pairs_hook=contract.native._pairs);require(value["supervisor_pid"]==owner["supervisor_pid"],"heartbeat supervisor identity");hb.append(value)
    require(len(hb)>=2 and all(0<=b["unix_seconds"]-a["unix_seconds"]<=.2 for a,b in zip(hb,hb[1:])),"returned heartbeat cadence within CPU freshness")
    direct=read_json(Path(leaf)/"worker_GPU_final.json")
    require(direct.get("CUDA_initialized") is True and not direct.get("diagnostic_error")
        and all(type(direct.get(k)) is int and direct[k]>=0 for k in ("peak_allocated_bytes","peak_reserved_bytes","free_bytes_final"))
        and direct["peak_reserved_bytes"]>=direct["peak_allocated_bytes"] and direct["free_bytes_final"]>=.5*1024**3,
        "direct successful terminal GPU diagnostics and same work floor")
    lease=Path(sup["lease_path"]);require(lease.is_dir() and not lease.is_symlink() and lease.resolve().parent==(PACKAGE/"artifacts/_native_gpu_leases").resolve(),"owned durable lease directory")
    require(read_json(lease/"worker.json")==identity,"lease worker identity")
    resumed=read_json(lease/"resume.json")
    require(resumed["worker_identity"]==identity and resumed["external_monitor_ready_before_resume"] is True
        and resumed["resumed_monotonic"]==sup["worker_resume_monotonic"] and resumed["deadline_monotonic"]==sup["deadline_monotonic"],"assigned-before-resume/ready-before-execute lease")
    require(file_info(lease/"terminal.json")==file_info(Path(leaf)/"SUPERVISOR_RECEIPT.json"),"durable terminal lease same exact supervisor bytes")
    reservation=read_json(lease/"reservation.json")
    require(reservation["config_sha256"]==sup["config_sha256"] and reservation["output_dir"]==str(Path(leaf))
        and reservation["lease_id"]==lease.name and reservation["GPU_index"]==0
        and reservation["supervisor"]=={"pid":owner["supervisor_pid"],"create_time":owner["supervisor_create_time"]},"same durable scope reservation")
    observed[owner["supervisor_pid"]]=owner["supervisor_create_time"]
    require(ownership["actual_worker_pid"] in observed,"actual child independently observed in owned Job samples")
    return {"CPU_samples":len(cpu),"GPU_samples":len(gpu),"minimum_sampled_GPU_free_bytes":min(r["gpu_free_bytes"] for r in gpu),
        "minimum_host_available_bytes":min(r["host_available_bytes"] for r in cpu),"maximum_owned_tree_RSS_bytes":max(r["worker_tree_rss_bytes"] for r in cpu),
        "minimum_disk_free_bytes":min(r["disk_free_bytes"] for r in cpu),"direct_terminal_GPU":direct,"identities":observed,
        "CPU_max_sample_seconds":max(r["sample_completed_monotonic"]-r["sample_started_monotonic"] for r in cpu),
        "GPU_max_sample_seconds":max(r["gpu_sample_completed_monotonic"]-r["gpu_sample_started_monotonic"] for r in gpu),
        "CPU_max_gap_seconds":max(b["sample_started_monotonic"]-a["sample_started_monotonic"] for a,b in zip(cpu,cpu[1:])),
        "GPU_max_gap_seconds":max(b["gpu_sample_started_monotonic"]-a["gpu_sample_started_monotonic"] for a,b in zip(gpu,gpu[1:])),
        "heartbeat_rows":len(hb),"whole_scope_seconds":sup["elapsed_seconds"]}
def verify_loss_stage(measurements,events,leaf,scope):
    checks=[]
    for item in measurements:
        if item["path"].startswith("stage_loss_"):
            loss=read_json(Path(leaf)/item["path"])
            require(type(loss.get("ce")) in (int,float) and math.isfinite(loss["ce"]) and loss["ce"]>=0
                and type(loss.get("selected_logsoftmax")) in (int,float) and math.isfinite(loss["selected_logsoftmax"])
                and type(loss.get("absolute_error")) in (int,float) and math.isfinite(loss["absolute_error"])
                and 0<=loss["absolute_error"]<=2e-6 and set(loss)=={"ce","selected_logsoftmax","absolute_error","supervised_positions"}
                and math.isclose(abs(loss["ce"]-loss["selected_logsoftmax"]),loss["absolute_error"],rel_tol=1e-12,abs_tol=1e-12)
                and type(loss["supervised_positions"]) is int and loss["supervised_positions"]>0,
                "stored reference CE/selected NLL numerical check")
            checks.append(loss)
    require(len(checks)==scope["dose"]["reference_forwards"],"only independent reference row numerical stages are durably recorded")
    rows=scope["rows"][(scope["chunk"]-1)*180:(scope["chunk"]-1)*180+len(checks)]
    require([loss["supervised_positions"] for loss in checks]==[row["supervised_tokens"] for row in rows],"reference EOS supervision denominator for each ordered row")
    return {"stored_reference_checks":len(checks),"training_CE_checks_returned":180,
        "limits":"Training row CE values are not separately persisted by this backend; the frozen checked-loss implementation and180 returned CE-check calls enforce the existing2e-6 gate. No model math is rerun by this audit."}
def validate_anchors(signal,expected):
    require(signal==SIGNAL,"root terminal and preservation signal required before any heavy read")
    fields={"inventory_file_sha256","profile_body_sha256","config_file_sha256","result_file_sha256",
        "supervisor_file_sha256","preservation_body_sha256","source_commit","arm","chunk"}
    require(type(expected) is dict and set(expected)==fields,"all independently selected trust anchors required")
    require(all(type(expected[k]) is str and len(expected[k])==64 and all(c in "0123456789abcdef" for c in expected[k])
        for k in fields-{"source_commit","arm","chunk"}) and type(expected["source_commit"]) is str
        and len(expected["source_commit"])==40 and all(c in "0123456789abcdef" for c in expected["source_commit"])
        and expected["arm"] in ("M1","M2") and type(expected["chunk"]) is int and expected["chunk"] in (1,2,3),"exact external byte/HEAD/arm/chunk anchors")
    return True
def audit(attempt_path,preserved_path,config_path,inventory_registration_path,*,terminal_signal,expected,output_path=None):
    """Only a complete real native terminal may emit the declared audit KIND.

    Expected hashes and HEAD are selected by the root independently of these
    files. A checksum/hash alone never authenticates self-signed authority.
    CPU fixtures call pure validators; they cannot invoke a native PASS shortcut.
    """
    validate_anchors(terminal_signal,expected)
    require(not any(n.split(".",1)[0] in chunks.FRAMEWORKS for n in sys.modules),"no framework/model/replay in independent audit")
    leaf=Path(attempt_path).absolute();saved=Path(preserved_path).absolute();config_path=Path(config_path).absolute()
    require(leaf.resolve().parent==(PACKAGE/"artifacts").resolve() and leaf.is_dir() and not leaf.is_symlink(),"owned actual attempt leaf")
    require(saved.resolve().parent==(PACKAGE/"evidence").resolve() and saved.is_dir() and not saved.is_symlink(),"owned preserved terminal evidence leaf")
    require(config_path.resolve().parent==(PACKAGE/"configs").resolve() and not config_path.is_symlink(),"owned frozen actual config")
    inventory_registration_path=inventory_path(inventory_registration_path)
    for path,key in ((inventory_registration_path,"inventory_file_sha256"),(config_path,"config_file_sha256"),
        (leaf/"worker_result.json","result_file_sha256"),(leaf/"SUPERVISOR_RECEIPT.json","supervisor_file_sha256")):
        require(file_info(path)["sha256"]==expected[key],"independently supplied exact "+key)
    scope=read_json(config_path);sup=read_json(leaf/"SUPERVISOR_RECEIPT.json");worker=read_json(leaf/"worker_result.json")
    inventory=read_json(inventory_registration_path);pres=read_json(saved/"preservation_receipt.json");checksum(pres)
    require(pres["sha256"]==expected["preservation_body_sha256"] and pres["raw_directory"]==str(leaf)
        and pres["preserved_directory"]==str(saved),"exact independently anchored preserved original leaf")
    require(inventory["source_commit"]==pres["source_commit"]==sup["preflight"]["source_commit"]==expected["source_commit"]
        and scope["arm"]==expected["arm"] and scope["chunk"]==expected["chunk"],"trusted exact launchHEAD/arm/chunk")
    require(inventory["profile_body_sha256"]==expected["profile_body_sha256"],"independently supplied exact profile anchor")
    counts=admit_inventory(scope,sup,inventory,config_path,expected["config_file_sha256"])
    verify_scope(scope)
    require(sup["preflight"]=={**scope,"config_path":str(config_path.resolve()),"config_sha256":expected["config_file_sha256"],
        "source_commit":expected["source_commit"],"worker_path":str(PACKAGE/"full27_worker.py")},"complete frozen actual preflight")
    binding=verify_terminal(scope,worker,sup,leaf);previous=verify_previous(scope,inventory)
    native_names={p.name for p in leaf.iterdir() if p.is_file()}
    require(all(p.is_file() and not p.is_symlink() for p in leaf.iterdir()) and native_names==set(pres["raw_files"]),"exact complete raw inventory/no directories/links")
    def closure():
        return {"raw":{name:file_info(leaf/name) for name in pres["raw_files"]},
            "preserved_raw":{name:file_info(saved/"raw"/name) for name in pres["raw_files"]},
            **{section:{path:file_info(Path(scope["model_dir"])/path if section=="model" else Path(path))
                for path in scope[key]} for section,key in SECTIONS.items()},
            "source_snapshots":{name:file_info(saved/"source_snapshots"/name) for name in pres["source_snapshots"]},
            "config":file_info(config_path),"preserved_config":file_info(saved/"config_snapshot.json"),
            "inventory":file_info(inventory_registration_path),"preserved_inventory":file_info(saved/"audit_inventory_snapshot.json"),
            "preservation_receipt":file_info(saved/"preservation_receipt.json")}
    before=closure()
    require(before["raw"]==before["preserved_raw"]==pres["raw_files"]
        and before["config"]==before["preserved_config"]==pres["config_snapshot"]
        and before["inventory"]==before["preserved_inventory"]==pres["audit_inventory_snapshot"],"all original/preserved raw/config/inventory bytes")
    require(before["source_snapshots"]==pres["source_snapshots"] and len(before["source_snapshots"])==31,"complete31 preserved source bytes")
    for section,key in SECTIONS.items():
        for path,item in before[section].items():require(item["sha256"]==scope[key][path],"all independently frozen current "+section+" bytes")
    for path,item in before["source"].items():
        require(before["source_snapshots"][Path(path).relative_to(ROOT).as_posix()]==item,"source snapshot exact current source")
    require({entry["path"]:{k:entry[k] for k in ("bytes","sha256")} for entry in sup["partial_outputs"]}==
        {name:item for name,item in before["raw"].items() if name!="SUPERVISOR_RECEIPT.json"},"exact terminal frozen raw inventory")
    event_data=validate_events(leaf/"events.jsonl",leaf)
    publication=verify_publication_chain(scope,worker,event_data,leaf,previous)
    loader_proof=verify_loader_and_ordering(scope,publication,event_data,leaf)
    numerical=verify_loss_stage(event_data["measurements"],event_data,leaf,scope)
    resource=verify_resources(scope,sup,leaf)
    import psutil
    live={}
    for pid,created in resource.pop("identities").items():
        try:live[str(pid)]=abs(psutil.Process(pid).create_time()-created)<.001
        except psutil.NoSuchProcess:live[str(pid)]=False
    require(not any(live.values()),"independent exact current owned/supervisor identities dead")
    import symbolic_checkpoint
    state=symbolic_checkpoint.verify_checkpoint(publication["descriptor"]["path"],publication["descriptor"],
        publication["evidence"]["witness"],scope["registration"],scope["arm"],scope["chunk"]*9)
    after=closure();require(before==after,"all byte closures stable before-after")
    def compact(value):return {name:{"entries":len(items),"manifest_sha256":digest(items)} if name in SECTIONS or name in (
        "raw","preserved_raw","source_snapshots") else items for name,items in value.items()}
    result={"schema_version":1,"kind":contract.AUDIT_KIND,"status":"PASS_FULL27_QUALIFICATION_CHUNK_MECHANICS_ONLY",
        "purpose":contract.PURPOSE,"recorded_at_utc":datetime.now(timezone.utc).isoformat(),"arm":scope["arm"],"chunk_ordinal":scope["chunk"],
        "source_commit":expected["source_commit"],"config_path":str(config_path),"config_file_sha256":expected["config_file_sha256"],
        "result_sha256":expected["result_file_sha256"],"supervisor_sha256":expected["supervisor_file_sha256"],
        "trusted_inventory_file_sha256":expected["inventory_file_sha256"],"trusted_profile_body_sha256":expected["profile_body_sha256"],
        "trusted_preservation_body_sha256":expected["preservation_body_sha256"],"exact_source_inventory_counts":counts,
        "binding":binding,"physical_dose":publication["physical"],"returned_current_API_calls":event_data["API_calls"],
        "cumulative_costs_sha256":digest(worker["costs"]),"qualification_witness_sha256":publication["evidence"]["witness"]["sha256"],
        "checkpoint_descriptor":publication["descriptor"],"symbolic_archive_verification":state,"numerical_scope":numerical,"loader_and_execution_order":loader_proof,"resources":resource,
        "cleanup":{"owned_process_identities_currently_alive":live,"all_owned_job_sampler_NVML_lease_and_byte_links_verified":True},
        "hash_closure":{"before":compact(before),"after":compact(after),"all_equal":True},
        "qualification_chunk_mechanics_verified":True,"study_warmstart_allowed":False,"study_training_authorized":False,
        "DEV_collection_authorized":False,"scoring_authorized":False,"full_resumed9_qualified":False,"full27_qualified":False,
        "continuous27_bitwise_parity_established":False,"CPU_fixture_used":False,"audit_framework_model_GPU_execution":False,
        "model_math_rerun":False,"protected_bank_access":False,"quality_gain_established":False,"M0_M1_M2_comparison_completed":False,
        "conclusion":"Exactly one source-bound nine-update qualification chunk completed. Checkpoint bytes and current/previous costs are closed for this same-arm qualification lineage. No study initialization, capability, transfer, retention, continuous27 parity or general native-fit authority is granted."}
    result["sha256"]=digest(result)
    if output_path is not None:
        output_path=Path(output_path).absolute()
        require(output_path.resolve().parent==(PACKAGE/"evidence").resolve() and output_path.suffix==".json"
            and not output_path.exists() and not output_path.is_symlink(),"new write-once owned independent audit receipt")
        import os
        with output_path.open("xb") as stream:
            raw=(canonical(result)+"\n").encode();require(stream.write(raw)==len(raw),"complete independent audit write");stream.flush();os.fsync(stream.fileno())
        require(read_json(output_path)==result,"reread emitted independent audit checksum")
    return result

def verify_loader_and_ordering(scope,publication,events,leaf):
    measured={item["path"]:item for item in events["measurements"]}
    require("stage_loader.json" in measured and "stage_worker_ownership_verified.json" in measured,"durable exact loader and owned interpreter evidence")
    loader=read_json(Path(leaf)/"stage_loader.json");witness=publication["evidence"]["witness"]
    require(loader.get("unique_base_tensors")==310 and loader.get("unique_base_parameters")==596049920
        and loader.get("serialized_tensors")==310 and loader.get("base_dtype")=="bfloat16"
        and loader.get("base_frozen") is True and loader.get("embedding_head_tie") is True
        and loader.get("memory_mapped") is False and loader.get("cpu_model_allocated") is False
        and loader.get("FP32_base_copy_allocated") is False and loader.get("blocking_chunk_copies") is True,
        "exact qualified denseBF16 loader/base/tie/storage profile")
    source=loader.get("source_tensor_sha256")
    require(type(source) is dict and len(source)==310 and all(type(value) is str and len(value)==64
        and all(ch in "0123456789abcdef" for ch in value) for value in source.values())
        and witness["base_source_tensor_hash_map_sha256"]==digest(source)
        and witness["base_frozen_source_tensors"]==310 and witness["base_parameters"]==596049920,
        "detached state witness exact frozen native310 base source map")
    require(set(loader["asset_hashes"])==set(scope["model_pins"]) and
        all(loader["asset_hashes"][name]["sha256"]==pin for name,pin in scope["model_pins"].items()),"all source model/tokenizer asset bytes loader→config")
    line=lambda name,status:[index for _,operation,result,index in events["ordered"] if operation==name and result==status]
    require(len(line("completion_CE_selected_NLL_check","returned"))==180 and len(line("finite_gradient_clip","returned"))==9,
        "exact returned checked-loss and norm operations")
    if scope["chunk"]>1:
        require(len(line("checkpoint_bitwise_restore_verify","returned"))==1 and
            max(line("checkpoint_bitwise_restore_verify","returned"))<min(line("reference_forward","attempt")),
            "actual bitwise tensor/RNG restore before independent reference")
        require(max(line("reference_autograd_grad","returned"))<min(line("training_forward","attempt")),"independent reference20 before native update")
        compares=line("qualification_gradient_compare","returned");backwards=line("training_backward","returned")
        require(len(compares)==1 and backwards[19]<compares[0]<line("finite_gradient_clip","attempt")[0],
            "all112 reference gradients compared after20backwards and beforeclip10/19")
        require(publication["delta"].get("checkpoint_adapter_restore")==dict(attempted=1,returned=1,failed=0,pending=0)
            and publication["delta"].get("checkpoint_optimizer_restore")==dict(attempted=1,returned=1,failed=0,pending=0)
            and all(publication["delta"].get(name)==dict(attempted=1,returned=1,failed=0,pending=0) for name in
                ("checkpoint_python_RNG_restore","checkpoint_CPU_RNG_restore","checkpoint_CUDA0_RNG_restore")),
            "all112 factors/fullAdam/PythonCPUCUDA0 actual restore calls")
    for operation,count in (("checkpoint_read_verified_bytes",1+int(scope["chunk"]>1)),
        ("checkpoint_weights_only_load",1+int(scope["chunk"]>1)),("qualification_saved_archive_state_hash_witness",1),
        ("native_base_content_verify",scope["dose"]["native_base_content_checks"])):
        require(publication["delta"].get(operation)==dict(attempted=count,returned=count,failed=0,pending=0),"exact serialized/base verification primitive "+operation)
    require(line("checkpoint_write","returned")[0]<line("qualification_saved_archive_state_hash_witness","returned")[0]
        <next(item["line"] for item in events["publications"] if item["name"]=="checkpoint_publish"),
        "serialized bytes compared to detached witness before durable promotion lineage")
    return {"loaded_source_tensors":310,"base_parameters":596049920,"source_tensor_hash_map_sha256":digest(source),
        "every_returned_training_CE_call_enforced_pinned_gate":True,"reference_before_training_and_preclip_verified":scope["chunk"]>1,
        "restore_proof_scope":"serialized/native bitwise state and detached witness; no model replay by auditor"}
