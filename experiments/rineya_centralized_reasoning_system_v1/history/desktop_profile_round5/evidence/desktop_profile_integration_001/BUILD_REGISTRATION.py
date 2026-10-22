"""One static registration build. No application/GPU/framework/DB access."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PKG = ROOT / "experiments/rineya_centralized_reasoning_system_v1"
PRIMARY = Path("C:/Users/clclc/BusinessProject/ChatBubble-rineya-centralized-reasoning-system-v1") / "experiments/rineya_centralized_reasoning_system_v1"
OUT = Path(__file__).resolve().parent
assert not (OUT / "FREEZE.json").exists(), "Frozen registration must not be rebuilt"
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes((json.dumps(value, indent=2, ensure_ascii=False)+"\n").encode("utf-8"))
for relative, expected in {
    "owned_runtime_identity_probe.py":"52e15881991f617fcb41f89cead7c379cc279a9a4da9c790b25501f03cae0e35",
    "tests/test_owned_runtime_identity_probe.py":"86d2a08743acff8916d908f758ef404ec4e487d8cb8c9c703995d413d8f50793",
    "contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_002_V2.json":"d7b2e5bc7673580c59a07ed5de27e8548bf13c5829fedb6e907dc9c12345a409",
}.items():
    assert digest(PRIMARY/relative) == expected
    target=OUT/"source_before"/relative; target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes((PRIMARY/relative).read_bytes())
for name in ("COOPERATIVE_CONTEXT_DISPOSITION.json", "STATIC_ANCESTRY_TO_OWNER_002.json", "RECONCILIATION.json"):
    target=OUT/"source"/name;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes((PRIMARY/"evidence/desktop_profile_preintegration_002"/name).read_bytes())
for name in ("AUTONOMOUS_CONTINUATION_SCI_ADVISORY_001.md", "AUTONOMOUS_CONTINUATION_OPS_ADVISORY_001.md"):
    (OUT/"source"/name).write_bytes((PRIMARY/"audits"/name).read_bytes())
draft=read(PRIMARY/"resolution_proposals/author_exception_002_desktop_profile/DESKTOP_PROFILE_DRAFT.json")
native=read(PKG/"contracts/NATIVE_DIAGNOSTIC_V1.json")
rows=draft["proposed_updated_rows"]
codex=rows[1]["path"]
anchors=[r for r in native["gpu_process_registry"] if r["path"].endswith(("SearchHost.exe","WidgetBoard.exe"))]
assert len(anchors)==2
for role,row in zip(("webview","codex","swift"),rows):
    row["role_id"]=role
    row["owner_roots"]=[a["path"] for a in anchors] if role=="webview" else [codex]
    row["allowed_ancestry_paths"]=list(dict.fromkeys([row["path"],*row["owner_roots"]]))
profile={"schema_version":1,"probe_key":"CRS-OWNED-IDENTITY-002","status":"FROZEN_CANDIDATE_PENDING_INDEPENDENT_SOURCE_REVIEW",
         "authority":{"path":"authorization/OWNER_EXCEPTION_DESKTOP_PROFILE_GRANTED_2026-10-04.json","sha256":"479b6c5284da978c491a6477a9064f21b585fa29e9991bf84d75960c380d18f0"},
         "context_disposition":{"path":"evidence/desktop_profile_integration_001/source/COOPERATIVE_CONTEXT_DISPOSITION.json","sha256":"b2a64e3bd3471d42db805d06cba44c1c4ffed6af79ff19dc9c2f4bb94954f49e"},
         "original_native_contract_sha256":draft["original_native_contract_sha256"],
         "original_owned_child_registry_sha256":draft["original_owned_child_registry_sha256"],
         "original_desktop_registry":native["gpu_process_registry"],"updated_rows":rows,
         "inspection_only_original_anchors":anchors,
         "inspection_only_ancestor_sha256":{r["path"]:r["sha256"] for r in rows+anchors},
         "context_policy":{"max_depth":8,"freshness_seconds":5,"actual_noncompute_workload_proved":False,
                           "hidden_inprocess_GPU_operations":"UNMEASURED","inspection_pins_are_permissions":False,
                           "kind":"MEASURED_OWNERSHIP_AND_COOPERATIVE_DESKTOP_CONTEXT_ONLY",
                           "known_compute_or_unknown_observed_PID_context_refused":True,
                           "native_or_owned_child_registry_inheritance":False}}
profile_path=PKG/"contracts/DESKTOP_GPU_IDENTITIES_PROBE002_V1.json"
write(profile_path,profile)
cfg=read(PKG/"contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_002_V2.json")
cfg["status"]="OWNER_APPROVED_FIFTH_ROUND_CANDIDATE_PENDING_SINGLE_TARGETED_CPU_TEST_AND_FRESH_REVIEWS"
cfg["implementation_sha256"]=digest(PKG/"owned_runtime_identity_probe.py")
cfg["source_sha256"]["experiments/rineya_centralized_reasoning_system_v1/owned_runtime_identity_probe.py"]=digest(PKG/"owned_runtime_identity_probe.py")
cfg["source_sha256"]["experiments/rineya_centralized_reasoning_system_v1/tests/test_owned_runtime_identity_probe.py"]=digest(PKG/"tests/test_owned_runtime_identity_probe.py")
cfg["desktop_profile"]={"path":"contracts/DESKTOP_GPU_IDENTITIES_PROBE002_V1.json","sha256":digest(profile_path)}
cfg["builder"]="/root/desktop_profile_builder"
cfg["owner_exception_desktop_profile"]={"authority":profile["authority"],"same_milestone_fifth_round":True,
                                         "additional_source_profile_test_review_rounds":1,"additional_worker_attempts":0,
                                         "one_remaining_fixed_KEY002_run":True,"no_material_further_repair_in_this_round":True,
                                         "parent_V2_sha256":digest(PKG/"contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_002_V2.json")}
cfg["cpu_validation"]={"status":"PENDING_SINGLE_TARGETED_CPU_INVOCATION","wall_seconds_max":60,
                       "result_path":"evidence/desktop_profile_integration_001/TEST_RESULT.json","native_qualification":False,
                       "canonical_host_database_access":False,"live_GPU_context":False}
write(PKG/"contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_002_V3.json",cfg)
print(json.dumps({"profile_sha256":digest(profile_path),"V3_sha256":digest(PKG/"contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_002_V3.json"),
                  "probe_sha256":digest(PKG/"owned_runtime_identity_probe.py"),"tests_sha256":digest(PKG/"tests/test_owned_runtime_identity_probe.py")},indent=2))
