"""Create-only sixth-round source registration. No framework/GPU/DB actions."""
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
PKG=ROOT/"experiments/rineya_centralized_reasoning_system_v1"
PRIMARY=Path("C:/Users/clclc/BusinessProject/ChatBubble-rineya-centralized-reasoning-system-v1")/"experiments/rineya_centralized_reasoning_system_v1"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert not (OUT/"FREEZE.json").exists()
assert sha(PKG/"authorization/DELEGATED_DESKTOP_BOUNDARY_CORRECTION_006.json")=="84f56a6f566f988ce402e9342676f7db6988607194e3b12a37a4d22ff5a5f939"
assert sha(PKG/"contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_002_V3.json")=="3747f47c5788791336eb4679d6c2b4f16d7221faaac22e97d92cf8d2367ac86f"
assert sha(PKG/"contracts/DESKTOP_GPU_IDENTITIES_PROBE002_V1.json")=="b433c5d1108678638a91304e71fe893fd4ae031c47ca3e325c7547463755a03d"
source=OUT/"source";source.mkdir(exist_ok=True)
for relative in ("authorization/DELEGATED_DESKTOP_BOUNDARY_CORRECTION_006.json",
                 "audits/DESKTOP_PROFILE_INTEGRATION_SCI_REVIEW_001.md","audits/DESKTOP_PROFILE_INTEGRATION_OPS_REVIEW_001.md"):
    with (source/Path(relative).name).open("xb") as handle:handle.write((PRIMARY/relative).read_bytes())
cfg=json.loads((PKG/"contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_002_V3.json").read_text())
cfg["status"]="CUMULATIVE_ROUND6_BOUNDARY_CORRECTION_CANDIDATE_PENDING_SINGLE_TARGETED_CPU_AND_FRESH_REVIEWS"
cfg["implementation_sha256"]=sha(PKG/"owned_runtime_identity_probe.py")
for relative in ("owned_runtime_identity_probe.py","tests/test_owned_runtime_identity_probe.py"):
    cfg["source_sha256"]["experiments/rineya_centralized_reasoning_system_v1/"+relative]=sha(PKG/relative)
cfg["delegated_boundary_correction_006"]={"authority_path":"authorization/DELEGATED_DESKTOP_BOUNDARY_CORRECTION_006.json",
    "authority_sha256":"84f56a6f566f988ce402e9342676f7db6988607194e3b12a37a4d22ff5a5f939",
    "cumulative_source_review_round":6,"same_milestone":True,"rejected_round5_preserved":True,
    "parent_V3_sha256":"3747f47c5788791336eb4679d6c2b4f16d7221faaac22e97d92cf8d2367ac86f",
    "findings":["DP5-SCI-001","DP5-OPS-001"],"extra_worker_attempts":0,"depth_limit_unchanged":8,
    "CPU_invocations_max":1,"CPU_wall_seconds_max":60,"further_material_repair_in_round":False}
cfg["cpu_validation"]={"status":"PENDING_SINGLE_SIXTH_ROUND_TARGETED_CPU_INVOCATION",
    "wall_seconds_max":60,"result_path":"evidence/desktop_boundary_correction_006/TEST_RESULT.json",
    "native_qualification":False,"canonical_host_database_access":False,"live_GPU_context":False}
path=PKG/"contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_002_V4.json"
with path.open("x",encoding="utf-8",newline="\n") as handle:handle.write(json.dumps(cfg,indent=2)+"\n")
print(json.dumps({"V4_sha256":sha(path),"probe_sha256":cfg["implementation_sha256"],
                  "tests_sha256":sha(PKG/"tests/test_owned_runtime_identity_probe.py")},indent=2))
