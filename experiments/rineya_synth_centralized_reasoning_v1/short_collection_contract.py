"""Staged one-batch M0 raw collection contract; no framework or scoring import.

write_scope/preflight deliberately reject staging. Native dispatch requires
separate reviewed integration, clean committed sources/config and admission.
The fixed CPU plan supplies public prompt arrays only; targets are never read.
"""
from __future__ import annotations
from copy import deepcopy
import importlib.metadata
from pathlib import Path
import os
import subprocess
import sys
import ast
import sysconfig

HERE = Path(__file__).resolve().parent
STAGED = HERE.name == "short_ram_successor_work" and HERE.parent.name == "artifacts"
PACKAGE = HERE.parents[1] if STAGED else HERE
ROOT = PACKAGE.parents[1]
sys.path.insert(0, str(PACKAGE))
import inference_contract as qualified
import native_contract as native
from inference_component import ROW_FIELDS, validate_row
from native_loader import ASSET_PINS, MODEL_DIR
from rs_contract import canonical, digest, require

VERSION = "synth-short-M0-physical-RAM-coherent-monitor-staged-2"
PURPOSE = "short_M0_raw_collection_one_fixed_batch_only"
PLAN_DIR = PACKAGE / "evidence/short_collection_plan_cpu_20261004"
PLAN_PATH = PLAN_DIR / "short_M0_collection_plan_cpu.json"
PLAN_BODY_SHA256 = "d0a9e71668badb7d3a9d49fe6b8d64bb156cefc41e0178132a4497974d3426db"
PLAN_FILE_SHA256 = "fafd166bf9f980bb13dbb4f3bc8bb81fa85abb077fba548795d000471d02d0e4"
PLAN_CODE_SHA256 = "9ea1302aa13875ab08a9cde2bdcdfe771a7d84bd16095e08166cddcbfd4573e4"
INFERENCE_AUDIT_SHA256 = "b655ec5d883d398714564a0d72e2f987ec061489eaec803e49bb234513f7965e"
INFERENCE_AUDIT = PACKAGE / "evidence/infq001_attempt1_independent_audit_20261004.json"
AUTHORIZATION = {"scope": PURPOSE, "raw_DEV_collection": True, "owner_authority": "explicit_owner_goal_and_frozen_run_manifest",
                 "production_training": False, "scoring": False, "prior_correct_enrollment": False,
                 "language": False, "long_context": False, "protected_bank_access": False, "paid_services": False}
SOURCE_NAMES = ("short_collection_contract.py", "short_collection_worker.py", "short_collection_supervisor.py", "short_ram_monitor.py")
ROWS_PER_BATCH, CONTEXT_CAP, GENERATION_CAP, FORWARD_CAP = 64, 512, 32, 2050
HELPER_PINS = {
    "native_supervisor.py": "edf2de5b0917150c2c79e6a555dc0399995d12859028e62a7f7ece86b767e691",
    "inference_qual_worker.py": "1172838c31c1b9440337707826573012e06f9a6a26b606a62a43976d19e99f1e",
    "native_process.py": "3d36770a36be5ef9db2c9ab159b069e3dde6c6e61210c58513c5afa7c4e70540",
    "worker_ram_probe.py": "e4ec5e19241b877b99cc1ed933514c6fb09721658d91666df900a9a1e00f75a4",
}
RAM_PROOF = PACKAGE / "evidence/worker_ram_probe_cpu_20261005/CPU_ram_callback_comparison_20261005.json"
RAM_PROOF_SHA = "4e1b9784258d4a80c4246d88c69a0de65426f50b64a203bcd9d56f717a3ed0a2"
RAM_PRESERVATION = RAM_PROOF.with_name("preservation_receipt.json")
RAM_PRESERVATION_SHA = "a265e279b51504ea2237034d450fb07faa4608d26d75eea83d0c95a298b71127"
RAM_PRIMARY = PACKAGE / "evidence/available_ram_primary_source_review_20261005.json"
RAM_PRIMARY_SHA = "156106209e38413373bfba45c3469eb67641fcc0ba6bb6ed5575bbcd8fbd7255"


def successor_profile():
    """Small CPU source/receipt binding only; no model/runtime-wide byte IO."""
    for name, pin in HELPER_PINS.items():
        require(native.filehash(PACKAGE/name) == pin, "unchanged helper/proven callback byte pin: " + name)
    receipts = {}
    for path, pin in ((RAM_PROOF, RAM_PROOF_SHA), (RAM_PRESERVATION, RAM_PRESERVATION_SHA), (RAM_PRIMARY, RAM_PRIMARY_SHA)):
        require(native.filehash(path) == pin, "RAM CPU/primary evidence pin")
        receipts[str(path.resolve())] = pin
    proof = native.read_json(RAM_PROOF)
    require(proof["status"] == "PASS_MATCHED512_DEFAULT_VS_DIRECT_RAM_CALLBACK_CPU_COMPONENT_ONLY"
            and [f["factor"] for f in proof["factors"]] == ["default_psutil", "candidate_direct"], "proven matched CPU factors")
    expected = {"attempted": 512, "returned": 512, "failed": 0, "pending": 0}
    require(all(f["guard"]["counts"] == expected and all(c["counts"] == expected for c in f["callbacks"].values())
                for f in proof["factors"]), "complete guard/callback CPU proof")
    def body(path):
        nodes = [n for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef) and n.name == "available_bytes"]
        require(len(nodes) == 1, "one RAM callback body")
        return ast.dump(nodes[0], include_attributes=False)
    require(body(PACKAGE/"worker_ram_probe.py") == body(RAM_PROOF.parent/"worker_ram_probe.py"), "same proven fresh physical-RAM callback body")
    ast_path = (Path(sysconfig.get_path("stdlib"))/"ast.py").resolve()
    return {"id": "SHORT_M0_PHYSICAL_RAM_COHERENT_EXTERNAL_CHECK_V2", "helper_pins": deepcopy(HELPER_PINS),
        "CPU_evidence_pins": receipts, "runtime_extension_pins": {str(ast_path): native.filehash(ast_path)},
        "available_callback": "worker_ram_probe.available_bytes; fresh physical available every unchanged worker guard call",
        "monitor": "collector-specific ShortRamMonitor; copy locked snapshots then capture check clock",
        "CPU_freshness_seconds": .2, "CPU_cadence_seconds": .1, "all_floors_and_numerics_unchanged": True,
        "abort_diagnostics": "bounded selected clock/sample/branch/thread/CPU-phase record written after containment; diagnostic IO cannot replace abort reason",
        "batch04_causality_established": False, "native_collection_qualified": False,
        "source_review_required_before_integration": True, "native_launch_authorized": False}


def merge_runtime_inventory(core, profile):
    """Retain the qualified64 byte pins and add only the executed AST source."""
    require(type(core) is dict and len(core) == 64, "exact unchanged Native7 runtime64 required before additive extension")
    require(profile.get("id") == "SHORT_M0_PHYSICAL_RAM_COHERENT_EXTERNAL_CHECK_V2", "known successor profile")
    extension = profile.get("runtime_extension_pins")
    require(type(extension) is dict and len(extension) == 1 and all(Path(path).name == "ast.py" for path in extension),
            "exact singleton AST runtime extension")
    require(not set(core).intersection(extension), "AST extension cannot replace qualified runtime pins")
    require(all(type(pin) is str and len(pin) == 64 and all(c in "0123456789abcdef" for c in pin)
                for pin in [*core.values(), *extension.values()]), "typed SHA256 runtime inventory")
    return {**core, **extension}


def integrated():
    require(not STAGED and HERE == PACKAGE and (PACKAGE / "native_contract.py").is_file(),
            "staged preparation cannot authorize dispatch; reviewed package integration required")


def _closed(value):
    require(all(value.get(k) is False for k in ("native_launch_authorized", "scoring_authorized",
        "production_training_authorized", "language_qualification_established", "long_context_qualification_established",
        "prior_correct_enrollment_performed", "protected_bank_access", "confirmation_eligible")), "CPU plan permissions closed")


def plan():
    require(native.filehash(PLAN_PATH) == PLAN_FILE_SHA256 and native.filehash(PLAN_DIR/"collection_plan.py") == PLAN_CODE_SHA256,
            "exact frozen plan/generator source byte pins")
    body = native.read_json(PLAN_PATH)
    require(body.get("sha256") == digest({k:v for k,v in body.items() if k != "sha256"}) == PLAN_BODY_SHA256
            and body.get("status") == "CPU_COLLECTION_PLAN_NATIVE_UNAUTHORIZED", "exact frozen CPU plan checksum")
    _closed(body)
    require(body["rows"] == 1152 and body["rows_per_batch"] == 64 and body["context_cap"] == 512
            and body["maximum_generated_tokens"] == 32 and set(body["batches"]) == {"atomic_RS_DEV", "task_guard_DEV"},
            "fixed complete short population; no language/long-context substitution")
    ordered = []
    for kind, expected in (("atomic_RS_DEV", 384), ("task_guard_DEV", 768)):
        source_relative, source_sha, count = body["source_pins"][kind]
        require(count == expected and native.filehash(PACKAGE / source_relative) == source_sha, "public prompt source byte pin")
        source = native.read_json(PACKAGE / source_relative)
        require(source.get("native_launch_authorized") is False and len(source["rows"]) == expected, "complete public source inventory")
        source_rows = []
        for item in source["rows"]:
            if kind == "atomic_RS_DEV":
                row = deepcopy(item)
            else:
                require(set(item) == {"row_id", "messages", "input_ids", "attention_mask", "prompt_sha256",
                                     "prompt_token_sha256", "prompt_tokens"}
                        and item["prompt_sha256"] == digest(item["messages"])
                        and item["prompt_token_sha256"] == digest(item["input_ids"]), "public task prompt/message/token identity")
                row = {"row_id": item["row_id"], "prompt_sha256": item["prompt_sha256"],
                       "prompt_ids": deepcopy(item["input_ids"]), "prompt_tokens": item["prompt_tokens"],
                       "attention_mask": deepcopy(item["attention_mask"])}
            source_rows.append(validate_row(row))
        batches = body["batches"][kind]
        require(len(batches) == expected // 64, "all fixed64 batches retained")
        for ordinal, batch in enumerate(batches, 1):
            rows = batch["rows"]
            require(batch["ordinal"] == ordinal and batch["source_kind"] == kind
                    and batch["start"] == (ordinal-1)*64 and batch["stop"] == ordinal*64
                    and rows == source_rows[(ordinal-1)*64:ordinal*64] and len(rows) == 64
                    and batch["source_packet_sha256"] == digest(rows), "exact whole-row batch/order/source binding")
            for row in rows: validate_row(row)
            positions = sum(32*r["prompt_tokens"] + 496 for r in rows)
            require(batch["maximum_capture_forwards"] == 2048 and batch["maximum_capture_input_positions"] == positions
                    and batch["maximum_zero_parity_forwards"] == 2 and batch["maximum_zero_parity_input_positions"] == 2*rows[0]["prompt_tokens"]
                    and batch["maximum_total_forwards"] == 2050 and batch["maximum_total_input_positions"] == positions+2*rows[0]["prompt_tokens"]
                    and batch["native_launch_authorized"] is False and batch["scoring_authorized"] is False, "registered larger per-batch dose only")
            ordered.extend(r["row_id"] for r in rows)
    require(len(ordered) == len(set(ordered)) == 1152 and digest(ordered) == body["row_order_sha256"], "complete population denominator and order")
    return body


def select_batch(body, kind, ordinal):
    require(kind in ("atomic_RS_DEV", "task_guard_DEV") and type(ordinal) is int
            and 1 <= ordinal <= (6 if kind == "atomic_RS_DEV" else 12), "one existing fixed batch selector")
    selected = deepcopy(body["batches"][kind][ordinal-1])
    require(len(selected["rows"]) == 64 and all(set(r) == ROW_FIELDS for r in selected["rows"]), "64 prompt-only five-field rows")
    return selected


def _tokenizer_runtime():
    distribution = importlib.metadata.distribution("tokenizers")
    require(distribution.version == "0.22.2", "pinned local generated-only decoder runtime")
    package = Path(distribution.locate_file("tokenizers"))
    # tokenizers.__init__ imports implementations, whose imports execute the
    # Python decoder/processor wrappers. Bind the complete installed package
    # source inventory rather than only the entrypoint and Rust extension.
    paths = sorted(set(package.rglob("*.py")) | set(package.rglob("*.pyd")))
    require(package/"__init__.py" in paths and package/"implementations/__init__.py" in paths
            and any(path.suffix == ".pyd" for path in paths) and all(path.is_file() for path in paths),
            "complete decoder Python source/binary closure")
    return {str(path.resolve()): native.filehash(path) for path in paths}


def build_scope(kind, ordinal):
    cpu_plan = plan(); batch = select_batch(cpu_plan, kind, ordinal)
    prior, native7 = qualified._native7()
    require(native.filehash(INFERENCE_AUDIT) == INFERENCE_AUDIT_SHA256, "exact independent inference audit byte pin")
    audit = native.read_json(INFERENCE_AUDIT)
    require(audit.get("status") == "PASS_SHORT_SCOPED_NATIVE_INFERENCE_MECHANICS"
            and audit.get("short_scoped_native_inference_mechanics_verified") is True
            and audit.get("source_commit") and audit.get("scoring_authorized") is False
            and audit.get("full_DEV_collection_qualified") is False, "independently passed short inference mechanics only")
    raw_evidence = PACKAGE / "evidence/infq001_attempt1_20261004"
    for name, entry in audit["hash_closure"]["raw_preserved_hashes"].items():
        require(native.filehash(raw_evidence / name) == entry["sha256"] and (raw_evidence / name).stat().st_size == entry["bytes"], "inference1 immutable evidence closure")
    passed = native.read_json(raw_evidence/"SUPERVISOR_RECEIPT.json")["preflight"]
    for name in ("inference_component.py", "inference_contract.py", "inference_qual_worker.py", "qualification_supervisor.py"):
        path = (PACKAGE/name).resolve()
        require(native.filehash(path) == passed["source_pins"][str(path)], "passed inference1 implementation drift: " + name)
    require(dict(ASSET_PINS) == passed["model_pins"] and qualified.PROFILE == passed["profile"],
            "passed inference1 model/tokenizer/compute profile binding")
    runtime = native.runtime_files()
    require(runtime == prior["runtime_pins"], "qualified Native7 runtime drift")
    sources = native.package_sources()
    for path in PACKAGE.joinpath("tests").glob("*.py"): sources[str(path.resolve())] = native.filehash(path)
    for name in SOURCE_NAMES:
        path = HERE / name; require(path.is_file(), "complete collector implementation required")
        sources[str(path.resolve())] = native.filehash(path)
    for path in HERE.glob("test_*.py"): sources[str(path.resolve())] = native.filehash(path)
    source_paths = [PLAN_PATH, PLAN_DIR/"collection_plan.py", INFERENCE_AUDIT]
    source_paths += [PACKAGE/parts[0] for parts in cpu_plan["source_pins"].values()]
    data = {str(path.resolve()):native.filehash(path) for path in source_paths}
    for name, pin in ASSET_PINS.items(): require(native.filehash(MODEL_DIR/name) == pin, "model/tokenizer asset drift")
    RAM_profile = successor_profile()
    runtime = merge_runtime_inventory(runtime, RAM_profile)
    scope = {"schema_version":1,"id":"SYNTH-SHORT-M0-RAM2-%s-%02d"%(kind,ordinal),"contract":VERSION,
        "authorization":deepcopy(AUTHORIZATION),"profile":qualified.PROFILE,"resources":deepcopy(native.RESOURCE_BOUNDS),
        "tolerances":deepcopy(native.TOLERANCES),"runtime_versions":{**native.RUNTIME,"tokenizers":"0.22.2"},
        "runtime_pins":runtime,"decoder_runtime_pins":_tokenizer_runtime(),"offline_environment":deepcopy(native.OFFLINE_ENV),
        "precision":deepcopy(prior["precision"]),"source_pins":sources,"data_pins":data,"native7":native7,
        "inference1_audit_sha256":native.filehash(INFERENCE_AUDIT),"inference1_scope_transfer":"compute/numerics_only_new64row_scope_requires_supervised_completion",
        "model_dir":str(MODEL_DIR.resolve()),"model_pins":dict(ASSET_PINS),"seed":101,"model_role":"fresh_dense_M0_zero_B_only_no_STATE1_or_study_checkpoint",
        "cpu_plan_sha256":cpu_plan["sha256"],"batch_selector":{"source_kind":kind,"ordinal":ordinal},"batch":batch,
        "short_RAM_monitor_profile": RAM_profile,
        "rows":batch["rows"],"source_packet_sha256":batch["source_packet_sha256"],
        "dose":{"context_cap":512,"maximum_generated_tokens":32,"rows":64,"maximum_capture_forwards":2048,
                "maximum_capture_input_positions":batch["maximum_capture_input_positions"],"maximum_parity_forwards":2,
                "maximum_parity_input_positions":batch["maximum_zero_parity_input_positions"],"maximum_forward_calls":2050,
                "maximum_forward_input_positions":batch["maximum_total_input_positions"],"optimizer_steps":0,"backwards":0},
        "collection_is_not_grader_authority":True,"correctness_certified":False,"retention_certified":False,
        "training_authorized":False,"scoring_authorized":False,"language_qualification_established":False,
        "long_context_qualification_established":False,"prior_correct_enrollment_performed":False,
        "protected_bank_access":False,"paid_services":False,"confirmation_eligible":False,"quality_gain_established":False}
    return scope


def _committed():
    status = subprocess.run(["git","status","--porcelain"],cwd=ROOT,capture_output=True,text=True,check=True).stdout
    require(not status.strip(), "clean committed collector/config required")
    return subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()


def write_scope(path, kind, ordinal):
    integrated();_committed()
    path=Path(path).absolute()
    require(path.parent.resolve()==(PACKAGE/"configs").resolve() and path.suffix==".json"
            and not path.exists() and not path.is_symlink(),"new write-once owned collection config")
    body=build_scope(kind,ordinal);body["sha256"]=digest(body)
    raw=(canonical(body)+"\n").encode();require(len(raw)<=native.MAX_JSON,"bounded collection config")
    with path.open("xb",buffering=0) as stream:require(stream.write(raw)==len(raw),"complete config write");os.fsync(stream.fileno())
    return body


def preflight(path):
    integrated()
    path=Path(path).absolute()
    require(path.parent.resolve()==(PACKAGE/"configs").resolve() and path.is_file() and not path.is_symlink(),"owned collector config")
    body=native.read_json(path)
    require(body.get("sha256")==digest({k:v for k,v in body.items() if k!="sha256"}),"collection config checksum")
    selection=body["batch_selector"]
    require(canonical({k:v for k,v in body.items() if k!="sha256"})==canonical(build_scope(selection["source_kind"],selection["ordinal"])),"current fixed batch/source/runtime/profile contract drift")
    commit=_committed()
    return {**body,"worker_path":str(PACKAGE/"short_collection_worker.py"),"config_path":str(path.resolve()),
            "config_sha256":native.filehash(path),"source_commit":commit,
            "bindings":{"source":body["source_pins"],"runtime":body["runtime_pins"],"decoder":body["decoder_runtime_pins"],"data":body["data_pins"],"model":body["model_pins"]}}
