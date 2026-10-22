"""Static registered bindings only; no checkpoint byte reads or native API."""
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3];PKG=ROOT/"experiments/rineya_centralized_reasoning_system_v1"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
registration=json.loads((PKG/"authorization/DELEGATED_CPU_NATIVE_LOAD_FEASIBILITY_001.json").read_text())
assert sha(PKG/"authorization/DELEGATED_CPU_NATIVE_LOAD_FEASIBILITY_001.json")=="e66d181f01a2346617b5433623859e71b518eefe45c896e117d097c2ddeeeb86"
paths=["experiments/rineya_centralized_reasoning_system_v1/cpu_native_load.py",
       "experiments/rineya_centralized_reasoning_system_v1/tests/test_cpu_native_load.py",
       "experiments/rineya_centralized_reasoning_system_v1/native.py",
       "experiments/rineya_centralized_reasoning_system_v1/operations.py",
       "experiments/rineya_centralized_reasoning_system_v1/owned_runtime_identity_probe.py",
       "experiments/opallo_native_collector_v1/supervisor.py",
       "experiments/post_training_mimo_centralized_v1/scripts/stream_checkpoint.py"]
source={relative:sha(ROOT/relative) for relative in paths}
cfg={"schema_version":1,"contract_id":"CRS-CPU-NATIVE-LOAD-FEASIBILITY-V1","scope":"EXACT_BF16_CPU_LOAD_ONLY_FIT",
     "job_key":"CRS-CPU-LOAD-FIT-001","status":"SOURCE_CANDIDATE_NOT_REVIEWED_NOT_LAUNCHED","device":"cpu","dtype":"bfloat16",
     "authority_path":"authorization/DELEGATED_CPU_NATIVE_LOAD_FEASIBILITY_001.json",
     "authority_sha256":"e66d181f01a2346617b5433623859e71b518eefe45c896e117d097c2ddeeeb86",
     "source_sha256":source,"model_dir":registration["model_dir_read_only"],"asset_sha256":registration["asset_sha256"],
     "runtime":registration["runtime"],"runtime_file_sha256":registration["runtime_file_sha256"],
     "registry_sha256":"15158c7e894334c1434be492d8a9dc3e22dd620b112b078516b6adc00b190969",
     "limits":{"wall_seconds":120,"runs":1,"automatic_retry":False,"tree_rss_gib":2.0,"job_commit_gib":2.0,
               "host_reserve_gib":1.25,"preworker_host_gib":3.3125,"disk_free_gib":4.0,"freshness_seconds":5,"threads":2},
     "dose":{"meta_constructions":1,"CPU_allocations":1,"checkpoint_loads":1,"serialized_tensors":310,
             "unique_parameters":596049920,"forwards":0,"tokenizers":0,"optimizer_calls":0},
     "CPU_buffer_span_max_bytes":2097152,"CPU_nonpersistent_rotary_buffer_bytes":256,
     "CPU_nonpersistent_buffer_values_initialized_or_qualified":False,
     "new_CPU_source_round":1,"new_CPU_source_round_limit":3,"old_GPU_round3_quarantine_preserved":True,
     "source_reviewers":{"science":"/root/exception_science","operations":"/root/exception_ops"},
     "owned_CPU_CUDA_API_boundary":"Pre-import fatal current/newPython-thread profiler; no CUDA calls or backend mutations",
     "owned_GPU_participation_refused":True,"hidden_internal_C_activity":"UNMEASURED",
     "unrelated_GPU_capacity_outside_reviewed_CPU_resource_scope":True,"old_GPU_free_floor_4GiB_unchanged":True,
     "borrowed_helpers":{"stream_checkpoint":"_header/_read_exact_into only; no GPU-oriented load_streamed receipt",
        "identity_probe":"raw inventory, hash budget, append, read_jsonl, reviewer namespace, independent RSS/deadline watch only",
        "native":"read/hash/source/assets/runtime, explicit Job query/duplicate, PID list, resume and owned drain only",
        "WindowsJob":"subclass adds exact0x2200 Job commit ceiling before resume"},
     "qualification":{"CPU_load_fit":"UNMEASURED","GPU_native":False,"reader":False,"training":False,"efficacy":False},
     "budget":{"epoch_seconds":7200,"rolling24h_seconds":14400,"historical_seconds":47.42100000000573,
               "same_canonical_host_state":True,"unknown120s_full_hold":True},"scope_review_refs":registration["review_refs"]}
with (PKG/"contracts/CPU_NATIVE_LOAD_FEASIBILITY_V1.json").open("x",encoding="utf-8",newline="\n") as handle:handle.write(json.dumps(cfg,indent=2)+"\n")
print(json.dumps({"contract_sha256":sha(PKG/"contracts/CPU_NATIVE_LOAD_FEASIBILITY_V1.json"),"source_sha256":source},indent=2))
