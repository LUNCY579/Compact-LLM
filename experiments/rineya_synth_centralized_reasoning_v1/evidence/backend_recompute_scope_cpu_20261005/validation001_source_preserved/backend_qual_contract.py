"""Bound A9+continuous10/BfreshC9+10 qualification, never study authority.

Staged modules cannot dispatch native work. Reviewed integration and a new clean,
committed write-once config with explicit owned-qualification authorization are
required. Selection reads all200 complete frozen M1 TRAIN occurrences before
outputs. Native7 resource/numerical gates remain exact.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from importlib.metadata import distribution

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1] if HERE.name in ("backend_qual_work","backend_recompute_scope_work") and HERE.parent.name=="artifacts" else HERE
ROOT=PACKAGE.parents[1]
for location in (PACKAGE,HERE):
    if str(location) in sys.path: sys.path.remove(str(location))
    sys.path.insert(0,str(location))
import native_contract as native
import study_schedule as schedule
from native_loader import ASSET_PINS,MODEL_DIR
from rs_contract import canonical,digest,require

ID="SYNTH-BACKENDQ-002-FULLRECOMPUTE"
VERSION="synth-backend-crossprocess-full-recompute-qualification-2"
SCOPE="backend_crossprocess_native_qualification_only"
WITNESS_KIND="SYNTH_BACKEND_QUAL_WITNESS_NOT_STUDY_CHECKPOINT"
AUTHORIZATION={"scope":SCOPE,"owner_authority":"user_goal_and_delegated_decisions_2026-10-04",
               "production_training":False,"protected_bank_access":False,"paid_services":False,
               "DEV_collection":False,"scoring":False,"study_training":False,"study_warmstart":False}
SOURCE_NAMES=("backend_qual_contract.py","backend_qual_core.py","backend_qual_native.py","backend_qual_worker.py","backend_qual_supervisor.py")
AUDIT=PACKAGE/"evidence/infq001_attempt1_independent_audit_20261004.json"
AUDIT_SHA="b655ec5d883d398714564a0d72e2f987ec061489eaec803e49bb234513f7965e"
AUDIT_BODY_SHA="cbc8c16b91e7a227e338df7ddedc030c5831afd556a06a7d3f6a68557e5f2a31"
INFERENCE_CONFIG=PACKAGE/"configs/SYNTH_INFQ001_TRAIN4.json"
INFERENCE_CONFIG_SHA="a26897bfbe36ce284221c6f4182c487d0d76287e708fe84de75f4e57fc2a9a79"
INF_EVIDENCE=PACKAGE/"evidence/infq001_attempt1_20261004"
NATIVE7_EVIDENCE=PACKAGE/"evidence/nq001_attempt7_20261003"
PROFILE="BF16_RESIDENT_FP32_ACTIVATIONS_QV8_V1"
CHECKPOINT_PROFILE="BF16_RESIDENT_FP32_QV8_EVAL28_FULL_NONREENTRANT_RECOMPUTE_V2"
INSTALLER_SHA="1cf3cad34dfb05a73c6930fc29ac72538b3c38a15ad0772a8b5378eef55aa04f"
CPU_PROOF=PACKAGE/"artifacts/checkpoint_successor_work/cpu_full_recompute_identity_successor_20261005.json"
CPU_PROOF_SHA="1e238147e3cb168f853a44e03c1c23f4668c8e25fbfa4dd5ebabc06b2f918f7f"
CHECKPOINT_PRIMARY_FILES={"torch":("torch/utils/checkpoint.py","torch/utils/_pytree.py","torch/utils/_python_dispatch.py",
    "torch/fx/traceback.py","torch/_functorch/_aot_autograd/functional_utils.py","torch/testing/_internal/logging_tensor.py",
    "torch/_dynamo/__init__.py","torch/_dynamo/eval_frame.py","torch/_dynamo/decorators.py"),
    "transformers":("transformers/modeling_layers.py","transformers/modeling_utils.py","transformers/models/qwen3/modeling_qwen3.py")}


def checkpoint_runtime_files():
    paths=[Path(distribution(package).locate_file(name)).resolve() for package,names in CHECKPOINT_PRIMARY_FILES.items() for name in names]
    require(all(path.is_file() and not path.is_symlink() for path in paths),"current local checkpoint/decoder primary dependency closure")
    return {str(path):native.filehash(path) for path in paths}


def checkpoint_binding(phase):
    require(native.filehash(PACKAGE/"checkpoint_successor_profile.py")==INSTALLER_SHA and native.filehash(CPU_PROOF)==CPU_PROOF_SHA,"reviewed installer/CPU proof bytes changed")
    proof=native.read_json(CPU_PROOF)
    require(proof.get("child_returncode")==0 and proof.get("CUDA_initialized") is False and proof.get("published_weights_opened") is False
            and proof.get("all112_grad_Adam_moments_RNG_byte_equal") is True and proof.get("continuous_vs_restored_update2_bitwise_equal") is True
            and proof["aggregate_full_regions"]=={"forward_regions":1680,"recompute_regions":1680,"all_recomputations_normal_exit":True,"failed":0,"pending":0},"bounded actual tinyCPU full-recompute proof required")
    groups=200 if phase=="A" else 40
    return {"profile":CHECKPOINT_PROFILE,"base_compute_profile":PROFILE,"authority_transfer":"Native7/INFQ base-compute helpers only; checkpoint profile still requires new owned A/B qualification",
            "installer_path":str(PACKAGE/"checkpoint_successor_profile.py"),"installer_sha256":INSTALLER_SHA,"CPU_proof_path":str(CPU_PROOF),"CPU_proof_file_sha256":CPU_PROOF_SHA,
            "decoder_layers":28,"evaluation_mode":True,"dropout":0.0,"actual_use_cache":False,"use_reentrant":False,"checkpoint_early_stop":False,"preserve_rng_state":True,
            "no_grad_bypass_preserved":True,"counts":{"no_grad":56,"forward":groups*28,"recompute":groups*28,"per_layer_no_grad":2,"per_layer_forward":groups,"per_layer_recompute":groups},
            "work_is_additional_not_training_dose":"Recomputed decoder calls are separate physical work; no extra rows, labels, backward/optimizer updates or token-budget permission",
            "duration_scope":"actual original decoder-function calls; controller/model wall remains separate", "native_profile_fit_qualified":False,"study_training_authorized":False}


def selected_packet():
    registration=schedule.read_json(PACKAGE/"STUDY_DEV_REGISTRATION_DRAFT.json")
    schedule.validate_registration(registration)
    _,_,pools,records=schedule._inputs()
    ids=[identity for update in registration["arms"]["M1"]["updates"][:10] for identity in update["occurrence_row_ids"]]
    require(len(ids)==200,"whole frozen first10 updates/200 occurrences")
    rows=[]
    for identity in ids:
        row=deepcopy(pools["M1"][identity]);schedule.validate_token_row(row)
        require(records[identity]["arm"]=="M1" and records[identity]["split"]=="train" and records[identity]["skill"] in ("R","S"),"atomic M1 TRAIN only")
        require(all(0<=token<151936 for token in row["input_ids"]),"native vocabulary bounds")
        rows.append(row)
    return registration,rows


def cursor(registration,completed):
    require(type(completed) is int and completed in (9,10),"qualification cursor only9/10")
    updates=registration["arms"]["M1"]["updates"][:completed]
    ids=[identity for update in updates for identity in update["occurrence_row_ids"]]
    return {"completed_updates":completed,"consumed_occurrences":completed*20,"input_tokens":sum(u["input_tokens"] for u in updates),
            "supervised_tokens":sum(u["supervised_tokens"] for u in updates),"consumed_order_sha256":digest(ids),
            "next_update_ordinal":completed+1,"learning_rate":schedule.learning_rates()[completed-1]}


def source_pins():
    paths=[PACKAGE/name for name in ("native_contract.py","native_loader.py","native_mechanics.py","native_precision.py",
           "native_process.py","native_nvml.py","native_supervisor.py","native_worker.py","inference_qual_worker.py","inference_contract.py","inference_component.py",
           "study_torch_backend.py","study_native_chunk.py","study_schedule.py","rs_contract.py")]
    paths += [HERE/name if name in ("backend_qual_contract.py","backend_qual_worker.py") else PACKAGE/name for name in SOURCE_NAMES]
    paths += [PACKAGE/"checkpoint_successor_profile.py"]
    paths += sorted((HERE/"tests").glob("test_backend_qual*.py")) if HERE!=PACKAGE else sorted((PACKAGE/"tests").glob("test_backend_qual*.py"))
    paths += [HERE/"test_backend_recompute_scope.py"] if HERE!=PACKAGE else sorted((PACKAGE/"tests").glob("test_backend_recompute_scope*.py"))
    require(all(path.is_file() and not path.is_symlink() for path in paths),"complete qualification source inventory")
    return {str(path.resolve()):native.filehash(path) for path in paths}


def inference_binding():
    require(native.filehash(AUDIT)==AUDIT_SHA and native.filehash(INFERENCE_CONFIG)==INFERENCE_CONFIG_SHA,"qualified inference audit/config bytes changed")
    audit=native.read_json(AUDIT)
    require(audit.get("sha256")==AUDIT_BODY_SHA and audit["sha256"]==digest({k:v for k,v in audit.items() if k!="sha256"})
            and audit.get("status")=="PASS_SHORT_SCOPED_NATIVE_INFERENCE_MECHANICS","independent inference qualification required")
    raw=audit["hash_closure"]["raw_preserved_hashes"]
    for name,item in raw.items():
        path=INF_EVIDENCE/name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size==item["bytes"] and native.filehash(path)==item["sha256"],"preserved inference evidence changed")
    native7=audit["hash_closure"]["Native7_binding"]
    require(native7["status"]=="PASS_SHORT_SCOPED_NATIVE_MECHANICS" and native7["profile"]==PROFILE,"Native7 profile qualification")
    for path,pin in native7["qualified_helper_pins"].items(): require(native.filehash(Path(path))==pin,"qualified Native7 helper drift")
    return {"audit_path":str(AUDIT),"audit_file_sha256":AUDIT_SHA,"audit_body_sha256":AUDIT_BODY_SHA,"status":audit["status"],
            "config_path":str(INFERENCE_CONFIG),"config_file_sha256":INFERENCE_CONFIG_SHA,"source_commit":"5c85071bdf6b5cd4001fd7f75c2330fc514536ca",
            "preserved_raw_sha256":digest(raw),"qualified_helper_pins":native7["qualified_helper_pins"],
            "historical_snapshot_is_not_current_source_qualification":True,"authority_transfer":"compute_profile_only_no_study_or_DEV_permission"}


def _observed_duration(path,operation):
    starts={};durations=[]
    with Path(path).open() as stream:
        for line in stream:
            row=json.loads(line)
            if row.get("operation")!=operation: continue
            if row.get("event")=="attempt": starts[row["id"]]=row["utc"]
            elif row.get("event")=="returned" and row["id"] in starts:
                durations.append((datetime.fromisoformat(row["utc"])-datetime.fromisoformat(starts[row["id"]])).total_seconds())
    require(len(durations)==1,"one authoritative full-base timing observation")
    return durations[0]


def _qualification_recipe(registration,rows,sources):
    value={"kind":"BACKEND_QUALIFICATION_ONLY_NEVER_STUDY_WARMSTART","id":ID,"arm":"M1","seed":101,
           "backend_source_sha256":native.filehash(PACKAGE/"study_torch_backend.py"),
           "registration_sha256":registration["sha256"],"cohort_sha256":digest([row["row_id"] for row in rows]),
           "packet_sha256":digest(rows),"source_pins":sources,"source_decision_pins":registration["source_and_decision_pins"],
           "checkpoint_amendment":{"profile":CHECKPOINT_PROFILE,"installer_sha256":INSTALLER_SHA,"CPU_proof_file_sha256":CPU_PROOF_SHA,
                "use_reentrant":False,"checkpoint_early_stop":False,"preserve_rng_state":True,"evaluation_mode":True,
                "phase_decoder_counts":{"A":{"no_grad":56,"forward":5600,"recompute":5600},"B":{"no_grad":56,"forward":1120,"recompute":1120}}},
           "global_LR":schedule.learning_rates()[:10],"phase_design":"A9_plus_continuous10_BfreshC9_plus10_reference20",
           "study_warmstart_allowed":False}
    value["sha256"]=digest(value)
    return value


def validate_packet(scope):
    require(scope.get("id")==ID and scope.get("phase") in ("A","B") and scope.get("authorization")==AUTHORIZATION,"exact backend qualification authority/phase")
    require(scope.get("resources")==native.RESOURCE_BOUNDS and scope.get("tolerances")==native.TOLERANCES,"unchanged resource/numerical gates")
    require(scope.get("profile")==CHECKPOINT_PROFILE and scope.get("base_compute_profile")==PROFILE and scope.get("checkpoint_amendment")==checkpoint_binding(scope["phase"]),"exact distinct full-recompute profile/proof/work binding")
    require(scope.get("production_training_authorized") is False and scope.get("study_warmstart_allowed") is False
            and scope.get("DEV_collection_authorized") is False and scope.get("quality_gain_established") is False
            and scope.get("protected_bank_access") is False,"qualification cannot grant study/scoring/banks authority")
    registration,rows=selected_packet()
    require(scope.get("rows")==rows and scope.get("source_packet_sha256")==digest(rows),"all200 whole frozen rows before outputs")
    require(scope["c9_binding"]==cursor(registration,9) and scope["c10_binding"]==cursor(registration,10),"exact C9/C10 cursor/input/supervised/globalLR")
    recipe=scope["qualification_recipe"]
    require(recipe["sha256"]==digest({k:v for k,v in recipe.items() if k!="sha256"}) and recipe["kind"]=="BACKEND_QUALIFICATION_ONLY_NEVER_STUDY_WARMSTART"
            and recipe["cohort_sha256"]==digest([r["row_id"] for r in rows]),"distinct qualification recipe/cohort")
    expected={"A":{"train_forwards":200,"reference_forwards":0,"zero_parity_forwards":2,"maximum_forward_calls":202,"native_optimizer_steps":10,"native_backwards":200,"reference_autograd_calls":0},
              "B":{"train_forwards":20,"reference_forwards":20,"zero_parity_forwards":2,"maximum_forward_calls":42,"native_optimizer_steps":1,"native_backwards":20,"reference_autograd_calls":20}}[scope["phase"]]
    require(all(scope["dose"].get(key)==value for key,value in expected.items()) and scope["dose"]["maximum_forward_input_positions"]==102400
            and scope["dose"]["context_cap"]==512,"explicit qualification dose caps")
    return registration,deepcopy(rows)


def build_scope(phase,*,predecessor=None,authorize_owned_qualification=False):
    require(phase in ("A","B") and type(authorize_owned_qualification) is bool,"phase A/B explicit owned-qualification flag")
    registration,rows=selected_packet();sources=source_pins();qualified=inference_binding()
    require((phase=="A" and predecessor is None) or (phase=="B" and type(predecessor) is dict),"B requires prior successful A; no fresh B/default checkpoint")
    first_length=rows[0]["total_tokens"]
    training=rows if phase=="A" else rows[180:200]
    reference=[] if phase=="A" else rows[180:200]
    dose={"train_forwards":len(training),"reference_forwards":len(reference),"zero_parity_forwards":2,
          "maximum_forward_calls":202 if phase=="A" else 42,"maximum_forward_input_positions":102400,"context_cap":512,
          "expected_forward_input_positions":sum(r["total_tokens"] for r in training+reference)+2*first_length,
          "native_optimizer_steps":10 if phase=="A" else 1,"native_backwards":len(training),"reference_autograd_calls":len(reference),
          "training_input_positions":sum(r["total_tokens"] for r in training),"training_supervised_positions":sum(r["supervised_tokens"] for r in training),
          "maximum_row_tokens":max(r["total_tokens"] for r in rows),"whole_source_occurrences_prevalidated":200,
          "witness10_is_not_a_checkpoint":True,"C9_checkpoints_saved":1 if phase=="A" else 0}
    observed=[_observed_duration(INF_EVIDENCE/"events.jsonl","verify_frozen_base"),_observed_duration(NATIVE7_EVIDENCE/"events.jsonl","frozen_base_postcheck")]
    require(round(observed[0],3)==6.874 and round(observed[1],3)==45.900,"preregistered raw-event timing observations changed")
    prior=native.read_json(PACKAGE/"configs/SYNTH_NQ001_PROC7.json")
    body={"schema_version":1,"id":ID,"contract":VERSION,"phase":phase,"profile":CHECKPOINT_PROFILE,"base_compute_profile":PROFILE,
          "checkpoint_amendment":checkpoint_binding(phase),"authorization":deepcopy(AUTHORIZATION),
          "owned_qualification_authorized":authorize_owned_qualification,"staged_native_dispatch_forbidden":HERE!=PACKAGE,
          "resources":deepcopy(native.RESOURCE_BOUNDS),"tolerances":deepcopy(native.TOLERANCES),"runtime_versions":deepcopy(native.RUNTIME),
          "offline_environment":deepcopy(native.OFFLINE_ENV),"precision":deepcopy(prior["precision"]),"source_pins":sources,
          "runtime_pins":{**deepcopy(prior["runtime_pins"]),**checkpoint_runtime_files()},"checkpoint_runtime_pins":checkpoint_runtime_files(),
          "data_pins":deepcopy(prior["data_pins"]),"model_dir":str(MODEL_DIR),"model_pins":dict(ASSET_PINS),
          "inference_qualification":qualified,"qualification_recipe":_qualification_recipe(registration,rows,sources),"seed":101,"arm":"M1",
          "rows":rows,"source_packet_sha256":digest(rows),"global_learning_rates":schedule.learning_rates()[:10],"dose":dose,
          "c9_binding":cursor(registration,9),"c10_binding":cursor(registration,10),"predecessor":deepcopy(predecessor),
          "resource_feasibility":{"basic_backend_first9_plus_C9_full_base_checks":12,"first9_plus_C9_including_parity_and_witness_checks":14,
               "A_total_full_base_checks":16,"B_total_full_base_checks":9,
               "check_count_derivation":{"A":"2constructor+10steps+1saveC9+1zeroParity+2witnesses","B":"2constructor+2restore+1step10+1zeroParity+2witnesses+1referenceFinal"},
               "observed_check_seconds":observed,"first9_check_only_seconds_range":[14*min(observed),14*max(observed)],
               "A_all_check_only_seconds_range":[16*min(observed),16*max(observed)],"B_all_check_only_seconds_range":[9*min(observed),9*max(observed)],
               "timing_source_pins":{str(path):native.filehash(path) for path in (INF_EVIDENCE/"events.jsonl",NATIVE7_EVIDENCE/"events.jsonl")},
               "scope_fit_qualified":False,"failure":"STOP_PRESERVE_NO_RETRY_NO_DROPPING_OR_LIMIT_RELAXATION"},
          "qualification_scope":"sustained9_plus_one_fresh_process_continuation_only_not_full_resumed9_or27",
          "production_training_authorized":False,"study_warmstart_allowed":False,"DEV_collection_authorized":False,
          "protected_bank_access":False,"quality_gain_established":False,"full_resumed9_qualified":False,"full27_qualified":False}
    validate_packet(body)
    return body


def verify_descriptor_bytes(descriptor):
    require(type(descriptor) is dict and set(descriptor)=={"schema_version","kind","path","bytes","sha256","metadata","metadata_sha256"},"C9 exact backend descriptor")
    path=Path(descriptor["path"])
    require(path.is_absolute() and not path.is_symlink() and "state1" not in str(path).lower() and type(descriptor["bytes"]) is int and 0<descriptor["bytes"]<=32*1024**2,"private bounded C9; STATE1 forbidden")
    require(path.stat().st_size==descriptor["bytes"] and native.filehash(path)==descriptor["sha256"] and descriptor["metadata_sha256"]==digest(descriptor["metadata"]),"C9 bytes/metadata closure")
    return deepcopy(descriptor)


def verify_predecessor(predecessor,scope,*,native_supervised=True):
    require(type(predecessor) is dict and set(predecessor)=={"result_path","result_sha256","supervisor_path","supervisor_sha256"},"successful A result and terminal supervisor required")
    path=Path(predecessor["result_path"])
    require(path.is_absolute() and path.is_file() and not path.is_symlink() and native.filehash(path)==predecessor["result_sha256"],"A result bytes changed")
    result=native.read_json(path)
    expected_status="PASS_SCOPED_BACKEND_A9_CONT10_MECHANICS" if native_supervised else "PASS_CPU_FAKE_BACKEND_A9_CONT10_CONTROL"
    require(result.get("status")==expected_status and result.get("phase")=="A"
            and result.get("qualification_recipe_sha256")==scope["qualification_recipe"]["sha256"]
            and result.get("source_packet_sha256")==scope["source_packet_sha256"] and result.get("study_warmstart_allowed") is False,
            "only successful same recipe/source A qualification may seed B")
    require(result.get("native_execution_performed") is native_supervised and result.get("CPU_fake_control") is (not native_supervised),"CPU fake result cannot become native predecessor authority")
    require(result["dose"]["attempted_calls"]==202 and result["dose"]["input_positions"]==scope["c10_binding"]["input_tokens"]+2*scope["rows"][0]["total_tokens"]
            and result["dose"]["operations"]=={"train":{"attempted_calls":200,"input_positions":scope["c10_binding"]["input_tokens"]},
                "reference":{"attempted_calls":0,"input_positions":0},"zero_parity":{"attempted_calls":2,"input_positions":2*scope["rows"][0]["total_tokens"]}},"A exact202 physical forwards/whole positions")
    require(set(result["protocol_operations"])=={"gradient_reset","forward","loss","backward","clip","step","checkpoint_save","checkpoint_restore","reference_control"},"A protocol cost closure")
    for name,count in (("gradient_reset",10),("forward",200),("loss",200),("backward",200),("clip",10),("step",10),("checkpoint_save",1),("checkpoint_restore",0),("reference_control",0)):
        require(result["protocol_operations"][name]=={"attempted":count,"returned":count,"failed":0,"pending":0},"A complete protocol "+name)
    require(result["sha256"]==digest({k:v for k,v in result.items() if k!="sha256"}),"A result checksum")
    counts=result["primitive_operations"]
    require(type(counts) is dict and counts and all(set(item)=={"attempted","returned","failed","pending"}
            and all(type(n) is int and n>=0 for n in item.values()) and item["attempted"]==item["returned"] and not item["failed"] and not item["pending"] for item in counts.values()),"failed/pending A cannot seed B")
    descriptor=verify_descriptor_bytes(result["checkpoint_c9"])
    require(descriptor["kind"]=="SYNTH_STUDY_TORCH_BOUNDARY_V1" and descriptor["metadata"].get("recipe_sha256")==scope["qualification_recipe"]["sha256"]
            and descriptor["metadata"].get("cohort_sha256")==scope["qualification_recipe"]["cohort_sha256"] and descriptor["metadata"].get("binding")==scope["c9_binding"],"C9 qualification-only recipe/cohort/cursor")
    require(result["witness10"].get("kind")==WITNESS_KIND and result["witness10"].get("binding")==scope["c10_binding"]
            and result["witness9"].get("kind")==WITNESS_KIND and result["witness9"].get("binding")==scope["c9_binding"],"complete qualification-only W9/W10")
    if native_supervised:
        sup_path=Path(predecessor["supervisor_path"])
        require(sup_path.is_absolute() and sup_path.is_file() and not sup_path.is_symlink() and native.filehash(sup_path)==predecessor["supervisor_sha256"],"A supervisor bytes changed")
        supervisor=native.read_json(sup_path)
        require(supervisor.get("status")=="SUPERVISED_WORKER_EXITED_ZERO" and supervisor.get("returncode")==0
                and supervisor.get("job_closed_kill_on_close") is True and supervisor.get("owned_worker_identity_currently_alive") is False
                and supervisor.get("sampler_threads_terminal_verified") is True and supervisor.get("gpu_telemetry_cleanup_verified") is True,"A supervised terminal containment required")
        require(path.parent==sup_path.parent and path.resolve().parent.parent==(PACKAGE/"artifacts").resolve(),"A evidence must be one owned output leaf")
        require(Path(descriptor["path"]).resolve().parent==path.resolve().parent,"C9 must belong to A owned leaf")
        require(result["config_sha256"]==supervisor.get("config_sha256")==supervisor.get("preflight",{}).get("config_sha256")
                and supervisor["preflight"].get("phase")=="A" and supervisor["preflight"].get("qualification_recipe",{}).get("sha256")==scope["qualification_recipe"]["sha256"],"A result/supervisor frozen config/source binding")
    return result


def _head(require_clean=True):
    value=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
    if require_clean:
        require(not subprocess.run(["git","status","--porcelain"],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip(),"qualification requires clean committed root sources/config")
    return value


def write_scope(path,phase,*,predecessor=None,authorize_owned_qualification=False):
    require(HERE==PACKAGE,"staging cannot create native dispatch configs")
    require(authorize_owned_qualification is True,"explicit owned backend qualification authorization required")
    target=Path(path).absolute()
    require(target.parent.resolve()==(PACKAGE/"configs").resolve() and not target.exists() and not target.is_symlink(),"new root write-once config")
    _head(True);body=build_scope(phase,predecessor=predecessor,authorize_owned_qualification=True)
    if phase=="B": verify_predecessor(predecessor,body)
    body["sha256"]=digest(body)
    with target.open("xb",buffering=0) as stream:
        raw=(canonical(body)+"\n").encode();require(stream.write(raw)==len(raw),"short scope write");os.fsync(stream.fileno())
    return body


def preflight(path,*,require_clean=True):
    require(HERE==PACKAGE,"staged native dispatch forbidden before independent review/root integration")
    filename=Path(path).absolute()
    require(filename.parent.resolve()==(PACKAGE/"configs").resolve() and filename.is_file() and not filename.is_symlink(),"owned backend qualification config")
    body=native.read_json(filename)
    require(body.get("sha256")==digest({k:v for k,v in body.items() if k!="sha256"}) and body.get("owned_qualification_authorized") is True,"owned qualification checksum/explicit authorization")
    expected=build_scope(body["phase"],predecessor=body["predecessor"],authorize_owned_qualification=True)
    require(canonical({k:v for k,v in body.items() if k!="sha256"})==canonical(expected),"frozen qualification source/row/authority/dose contract changed")
    require({**native.runtime_files(),**checkpoint_runtime_files()}==body["runtime_pins"] and checkpoint_runtime_files()==body["checkpoint_runtime_pins"],"executing interpreter/current checkpoint runtime closure changed")
    for path,pin in body["runtime_pins"].items(): require(native.filehash(Path(path))==pin,"runtime byte drift")
    for name,pin in body["model_pins"].items(): require(native.filehash(MODEL_DIR/name)==pin,"native model/tokenizer bytes changed")
    if body["phase"]=="B": verify_predecessor(body["predecessor"],body)
    return {**body,"config_path":str(filename.resolve()),"config_sha256":native.filehash(filename),"source_commit":_head(require_clean),
            "worker_path":str(PACKAGE/"backend_qual_worker.py")}
