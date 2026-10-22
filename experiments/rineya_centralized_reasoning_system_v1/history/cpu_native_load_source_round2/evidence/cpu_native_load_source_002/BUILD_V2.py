import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3];PKG=ROOT/"experiments/rineya_centralized_reasoning_system_v1"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
cfg=json.loads((PKG/"contracts/CPU_NATIVE_LOAD_FEASIBILITY_V1.json").read_text())
cfg["contract_id"]="CRS-CPU-NATIVE-LOAD-FEASIBILITY-V2";cfg["new_CPU_source_round"]=2
cfg["status"]="ROUND2_SOURCE_CANDIDATE_PENDING_ONE_TARGETED_FIXTURE_AND_FRESH_REVIEWS"
cfg["source_round2_authority_sha256"]="af6f1d6506899a328845709fac2e9ee42f484effe00cf0c17f9e1d6ec1806229"
names=["model.embed_tokens.weight","model.norm.weight"]
parts=["input_layernorm.weight","post_attention_layernorm.weight","self_attn.q_norm.weight","self_attn.k_norm.weight",
       "self_attn.q_proj.weight","self_attn.k_proj.weight","self_attn.v_proj.weight","self_attn.o_proj.weight",
       "mlp.gate_proj.weight","mlp.up_proj.weight","mlp.down_proj.weight"]
names.extend("model.layers."+str(layer)+"."+part for layer in range(28) for part in parts)
cfg["tensor_schedule"]=sorted(names);assert len(set(names))==310
cuda=Path("C:/Users/clclc/BusinessProject/.venv-mq001/Lib/site-packages/torch/cuda")
cfg["CUDA_definition_source_sha256"]={str(path):sha(path) for path in sorted(cuda.rglob("*.py"))}
cfg["CUDA_main_source"]=str(cuda/"__init__.py")
cfg["runtime_file_sha256"].update(cfg["CUDA_definition_source_sha256"])
cfg["runtime_file_sha256"][str(cuda.parent/"__init__.py")]=sha(cuda.parent/"__init__.py")
cfg["CUDA_import_metadata_policy"]={"phase":"IMPORT_ONLY","required_state":False,"_is_compiled":"pinned hasattr compiled symbol only",
    "is_initialized":"False short circuit, no bad-fork callback","_lazy_call":"known3 callbacks queue-only, stateFalse entry/return, callback bodies remain fatal",
    "definition_events":"exact pinned module/class code only, nested device calls remain fatal","hidden_C_side_effects":"UNMEASURED"}
for relative in cfg["source_sha256"]:cfg["source_sha256"][relative]=sha(ROOT/relative)
with (PKG/"contracts/CPU_NATIVE_LOAD_FEASIBILITY_V2.json").open("x",encoding="utf-8",newline="\n") as handle:handle.write(json.dumps(cfg,indent=2)+"\n")
print(json.dumps({"contract_sha256":sha(PKG/"contracts/CPU_NATIVE_LOAD_FEASIBILITY_V2.json"),"source_sha256":cfg["source_sha256"]},indent=2))
