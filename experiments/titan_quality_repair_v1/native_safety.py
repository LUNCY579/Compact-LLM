"""Model-free, fail-closed native qualification admission and live ownership.

Issued JSON cannot authenticate a human by itself: the operator must separately
authorize this exact bounded run in the owning chat. No permit is issued here.
No Torch/model import, checkpoint read, lock reclaim or worker launch occurs at
module import. Fixtures may exercise private/injected mechanics, never real CLI
admission with a draft, fake or fixture grant.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
import ctypes
import datetime as dt
import hashlib
import importlib.util
import importlib.metadata
import importlib.abc
import importlib.machinery
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
import types

EXPERIMENT = "TITAN-QUALITY-REPAIR-V1"
STAGE = "native_qualification"
WEIGHT_SHA256 = "a4d73353db9a91cbba7b4cd7c8fe4fe6822670d10fe523b26a6d651bb9509567"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUSINESS = Path("C:/Users/clclc/BusinessProject")
LANE_ROOT = BUSINESS / "ChatBubble-agent-prompts-20261002/run-artifacts/titan-quality-repair-v1"
RUNTIME_ROOT = LANE_ROOT / "native-qualification"
HARDWARE_LOCK = BUSINESS / "ChatBubble-agent-prompts-20261002/hardware-worker.lock"
TEMP_LOCK = Path("C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock")
LEASE_LABEL = "titan-quality-repair-v1:native-qualification"
RUNTIME_SITE_PACKAGES=Path("C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Lib/site-packages")
CORE=BUSINESS/"Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Transition-1.0.0-devfreeze"
LOAD_BEGUN = "LOAD_BEGUN.json"
START_NAME = "NATIVE_SUPERVISOR_START.json"
LIMITS = {"host_admission_gib": 12, "gpu_admission_gib": 6, "disk_admission_gib": 20,
          "host_reserve_gib": 1.5, "gpu_reserve_gib": 0.5, "tree_rss_max_gib": 10,
          "output_bytes_max": 6 * 2**30}
CAPS = {"supervisor_hard_wall_seconds": 600, "worker_wall_seconds": 540,
        "cleanup_wall_seconds": 120, "declared_attempt_cost_seconds": 720}
GIB = 2**30
HEX = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def read_pinned(path: Path, expected: str) -> bytes:
    source = Path(path).read_bytes()
    if not isinstance(expected, str) or HEX.fullmatch(expected) is None or hashlib.sha256(source).hexdigest() != expected:
        raise ValueError("STOP_NATIVE_FILE_BINDING:" + str(path))
    return source


def write_new(path: Path, value: dict):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def proposal_payload_digest(proposal: dict) -> str:
    # Attestations bind this payload. Excluding their descriptors avoids a
    # circular proposal->review digest->proposal digest dependency.
    excluded = {"proposal_payload_sha256", "proposal_sha256", "review_descriptor",
                "cpu_qualification_descriptor", "operator_grant_descriptor"}
    return digest({k: v for k, v in proposal.items() if k not in excluded})


def validate_runtime(path: Path) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raise ValueError("STOP_NATIVE_RUNTIME_PATH")
    reject_reparse_ancestors(raw)
    reject_reparse_ancestors(RUNTIME_ROOT)
    resolved = raw.resolve()
    if (resolved != RUNTIME_ROOT.resolve() or ".codex" in [p.lower() for p in resolved.parts]
            or any((p / ".git").exists() for p in (resolved, *resolved.parents))):
        raise ValueError("STOP_NATIVE_RUNTIME_ROOT_ROTATION_OR_SOURCE_OUTPUT")
    return resolved


def reject_reparse_ancestors(path: Path):
    """Inspect only this named path's ancestors; never traverse linked trees."""
    for item in (Path(path),*Path(path).parents):
        try: info=item.lstat()
        except FileNotFoundError: continue
        except OSError as exc: raise ValueError("STOP_NATIVE_OUTPUT_ANCESTOR_UNOBSERVABLE") from exc
        if (stat.S_ISLNK(info.st_mode) or getattr(info,"st_file_attributes",0)&getattr(stat,"FILE_ATTRIBUTE_REPARSE_POINT",0x400)
                or (hasattr(item,"is_junction") and item.is_junction())):
            raise ValueError("STOP_NATIVE_OUTPUT_REPARSE_ANCESTOR")


def validate_owned_output(path: Path) -> Path:
    raw=Path(os.path.abspath(os.fspath(path)))
    canonical=Path(os.path.abspath(os.fspath(RUNTIME_ROOT)))
    reject_reparse_ancestors(raw)
    root=validate_runtime(canonical)
    resolved=raw.resolve()
    if (not raw.is_relative_to(canonical) or not resolved.is_relative_to(root)
            or ".codex" in [p.lower() for p in resolved.parts]
            or any((p/".git").exists() for p in (resolved,*resolved.parents))):
        raise ValueError("STOP_NATIVE_OWNED_OUTPUT_REDIRECTION")
    return resolved


def validate_ledger_output(path: Path) -> Path:
    raw=Path(os.path.abspath(os.fspath(path)))
    reject_reparse_ancestors(raw)
    reject_reparse_ancestors(LANE_ROOT)
    resolved=raw.resolve()
    if (not raw.is_relative_to(Path(os.path.abspath(os.fspath(LANE_ROOT))))
            or not resolved.is_relative_to(LANE_ROOT.resolve()) or ".codex" in [p.lower() for p in resolved.parts]
            or any((p/".git").exists() for p in (resolved,*resolved.parents))):
        raise ValueError("STOP_NATIVE_LEDGER_OUTPUT_REDIRECTION")
    return resolved


def validate_manifest(manifest: dict, here: Path = HERE) -> dict[str, bytes]:
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("STOP_EMPTY_NATIVE_SOURCE_MANIFEST")
    current = {p.name for p in Path(here).iterdir() if p.is_file()}
    if set(manifest) != current:
        raise ValueError("STOP_NATIVE_MANIFEST_INCOMPLETE_OR_NEW_SOURCE")
    sources = {}
    for name, expected in manifest.items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("STOP_NATIVE_SOURCE_MANIFEST_PATH")
        path=Path(here)/name
        if path.suffix.lower() not in {".py",".md",".json",".txt"} or path.stat().st_size>4*2**20:
            raise ValueError("STOP_NATIVE_MANIFEST_PAYLOAD_OR_LARGE_FILE")
        sources[name] = read_pinned(path, expected)
    for name in ("native_safety.py", "native_supervise.py", "native_qualification.py", "offline_supervise.py",
                 "budget.py", "registry.py", "schema.py"):
        if name not in sources:
            raise ValueError("STOP_NATIVE_MANIFEST_MISSING_IMPLEMENTATION:" + name)
    return sources


def reject_fixture_or_draft(value: dict):
    if not isinstance(value, dict):
        raise ValueError("STOP_NATIVE_DOCUMENT_NOT_OBJECT")
    if any(value.get(key) is not False for key in ("fixture_only",) if key in value):
        raise ValueError("STOP_FIXTURE_NATIVE_DOCUMENT")
    if any(value.get(key) for key in ("fixture_grant", "fake", "draft", "synthetic_permit")):
        raise ValueError("STOP_DRAFT_FAKE_OR_FIXTURE_PERMIT")
    status = str(value.get("status", "")).upper()
    if any(word in status for word in ("DRAFT", "FAKE", "FIXTURE", "PENDING", "BLOCKED")):
        raise ValueError("STOP_DRAFT_FAKE_OR_FIXTURE_PERMIT")


def _descriptor(proposal: dict, key: str) -> tuple[dict, dict]:
    descriptor = proposal.get(key)
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise ValueError("STOP_NATIVE_MISSING_EVIDENCE_DESCRIPTOR:" + key)
    path = Path(descriptor["path"])
    if not path.is_absolute() or not path.resolve().is_relative_to(LANE_ROOT.resolve()):
        raise ValueError("STOP_NATIVE_EVIDENCE_OUTSIDE_OWNED_LANE")
    value = json.loads(read_pinned(path, descriptor["sha256"]))
    # A real measured CPU mechanics fixture is appropriate containment evidence,
    # not a native/model permit. Its binding and zero-model fields are checked by
    # the caller. Fixture *grants* remain unconditionally prohibited.
    if key != "cpu_qualification_descriptor": reject_fixture_or_draft(value)
    return descriptor, value


def validate_bundle(proposal_path: Path, permit_path: Path, *, here: Path = HERE) -> dict:
    """Only a schema-2 issued operator grant plus matching real evidence opens admission.
    This performs small code/metadata reads only; never reads checkpoint tensors.
    """
    proposal_source, permit_source = Path(proposal_path).read_bytes(), Path(permit_path).read_bytes()
    proposal, permit = json.loads(proposal_source), json.loads(permit_source)
    reject_fixture_or_draft(proposal)
    reject_fixture_or_draft(permit)
    if (proposal.get("experiment") != EXPERIMENT or proposal.get("stage") != STAGE
            or type(proposal.get("proposal_version")) is not int or proposal["proposal_version"] != 2
            or proposal.get("source_checkpoint_sha256") != WEIGHT_SHA256
            or proposal.get("native_worker_implementation_in_this_milestone") is not True
            or proposal.get("training_sweep_authorized") is not False
            or proposal.get("final_evaluation_authorized") is not False or proposal.get("promotion_authorized") is not False):
        raise ValueError("STOP_NATIVE_PROPOSAL_SCOPE_OR_IMPLEMENTATION")
    payload_sha = proposal_payload_digest(proposal)
    if proposal.get("proposal_payload_sha256") != payload_sha:
        raise ValueError("STOP_NATIVE_PROPOSAL_PAYLOAD_DIGEST")
    manifest = proposal.get("source_manifest")
    if proposal.get("source_manifest_sha256") != digest(manifest):
        raise ValueError("STOP_NATIVE_SOURCE_MANIFEST_DIGEST")
    sources = validate_manifest(manifest, here)
    attempt=proposal.get("attempt",{})
    if (attempt.get("worker_wall_seconds_max")!=540 or attempt.get("supervisor_kill_deadline_seconds")!=600
            or attempt.get("cleanup_wall_seconds_max")!=120 or attempt.get("loaded_attempt_cap")!=3
            or attempt.get("qualification_pool_seconds_max")!=3600):
        raise ValueError("STOP_NATIVE_PROPOSAL_ENVELOPE")
    data_path=Path(attempt.get("development_path",""))
    if not data_path.is_absolute() or not data_path.resolve().is_relative_to(LANE_ROOT.resolve()):
        raise ValueError("STOP_NATIVE_DEVELOPMENT_OUTSIDE_OWNED_LANE")
    if (type(permit.get("schema_version")) is not int or permit["schema_version"] != 2
            or permit.get("experiment") != EXPERIMENT or permit.get("stage") != STAGE
            or permit.get("status") != "ISSUED_BY_OPERATOR" or permit.get("fixture_only") is not False):
        raise ValueError("STOP_NATIVE_REQUIRES_ISSUED_OPERATOR_GRANT")
    for key in ("source_checkpoint_sha256", "registry_sha256", "source_manifest_sha256", "proposal_payload_sha256"):
        if permit.get(key) != proposal.get(key) or not isinstance(permit.get(key), str) or HEX.fullmatch(permit[key]) is None:
            raise ValueError("STOP_NATIVE_OPERATOR_BINDING:" + key)
    # The payload excludes review/CPU descriptors to avoid attestation cycles.
    # The final operator grant binds the complete reviewed file separately, so
    # replacing either proof after issuance requires a new human-bound grant.
    actual_proposal_file_sha256=hashlib.sha256(proposal_source).hexdigest()
    if (not isinstance(permit.get("proposal_file_sha256"),str)
            or HEX.fullmatch(permit["proposal_file_sha256"]) is None
            or permit["proposal_file_sha256"]!=actual_proposal_file_sha256):
        raise ValueError("STOP_NATIVE_OPERATOR_FULL_PROPOSAL_FILE_BINDING")
    if (not isinstance(permit.get("attempt_id"), str) or IDENTIFIER.fullmatch(permit["attempt_id"]) is None
            or not isinstance(permit.get("hardware_slot"), str) or not permit["hardware_slot"].strip()
            or not isinstance(permit.get("operator_text"), str) or len(permit["operator_text"].strip()) < 12):
        raise ValueError("STOP_NATIVE_OPERATOR_ATTEMPT_SLOT_OR_TEXT")
    versions=proposal.get("runtime_packages")
    required={"numpy","tokenizers","psutil","torch","transformers","peft","accelerate","safetensors"}
    if not isinstance(versions,dict) or set(versions)!=required or any(not isinstance(v,str) for v in versions.values()):
        raise ValueError("STOP_NATIVE_RUNTIME_METADATA_UNQUALIFIED")
    sites=proposal.get("runtime_site_packages")
    if (not isinstance(sites,list) or len(sites)!=1 or not isinstance(sites[0],str)
            or not Path(sites[0]).is_absolute() or Path(sites[0]).resolve()!=RUNTIME_SITE_PACKAGES.resolve()):
        raise ValueError("STOP_NATIVE_RUNTIME_SITE_PATH")
    distributions={d.metadata["Name"].lower().replace("-","_"):d.version
                   for d in importlib.metadata.Distribution.discover(path=sites)}
    for package,expected in versions.items():
        actual=distributions.get(package)
        if actual is None: raise ValueError("STOP_NATIVE_RUNTIME_PACKAGE_MISSING:"+package)
        if actual!=expected: raise ValueError("STOP_NATIVE_RUNTIME_PACKAGE_DRIFT:"+package)
    review_descriptor, review = _descriptor(proposal, "review_descriptor")
    if (review.get("experiment") != EXPERIMENT or review.get("scope") != "STATIC_CODE_AND_CPU_MECHANICS_ONLY"
            or review.get("status") != "PASS_FOR_BOUNDED_NATIVE_QUALIFICATION_SOURCE"
            or review.get("native_execution_authorized") is not False
            or review.get("source_manifest_sha256") != proposal["source_manifest_sha256"]
            or review.get("proposal_payload_sha256") != payload_sha):
        raise ValueError("STOP_NATIVE_REVIEW_SCOPE_OR_BINDING")
    metadata_descriptor,metadata=_descriptor(proposal,"metadata_descriptor")
    sidecars=proposal.get("source_sidecar_sha256")
    sidecar_names={"CORE.json","config.json","generation_config.json","tokenizer.json","tokenizer_config.json","vocab.json","merges.txt"}
    if (not isinstance(sidecars,dict) or set(sidecars)!=sidecar_names
            or metadata.get("declared_weight_sha256")!=WEIGHT_SHA256
            or metadata.get("sidecar_sha256")!=sidecars):
        raise ValueError("STOP_NATIVE_ORIGINAL_SOURCE_METADATA_BINDING")
    for name,expected in sidecars.items(): read_pinned(CORE/name,expected)
    cpu_descriptor, cpu = _descriptor(proposal, "cpu_qualification_descriptor")
    if (cpu.get("experiment") != EXPERIMENT or cpu.get("status") != "PASS"
            or cpu.get("source_manifest_sha256") != proposal["source_manifest_sha256"]
            or type(cpu.get("native_model_loads")) is not int or cpu["native_model_loads"] != 0
            or type(cpu.get("native_model_forwards")) is not int or cpu["native_model_forwards"] != 0
            or cpu.get("source_unchanged_during_worker") is not True
            or cpu.get("job_assigned_before_fixture_imports") is not True):
        raise ValueError("STOP_NATIVE_CPU_CONTAINMENT_EVIDENCE_BINDING")
    return {"proposal": proposal, "permit": permit, "sources": sources,
            "proposal_path": str(Path(proposal_path).resolve()), "permit_path": str(Path(permit_path).resolve()),
            "proposal_file_sha256": hashlib.sha256(proposal_source).hexdigest(),
            "permit_file_sha256": hashlib.sha256(permit_source).hexdigest(),
            "review_descriptor": review_descriptor, "cpu_qualification_descriptor": cpu_descriptor,
            "metadata_descriptor":metadata_descriptor,
            "proposal_payload_sha256": payload_sha, "here": str(Path(here).resolve())}


class VerifiedPacketFinder(importlib.abc.MetaPathFinder,importlib.abc.Loader):
    """Compile only the admitted bytes; never consult packet .pyc or PYTHONPATH."""
    def __init__(self,sources,here,package="experiments.titan_quality_repair_v1",*,frameworks_allowed=False):
        self.sources,self.here,self.package,self.frameworks_allowed=sources,Path(here),package,frameworks_allowed
    def find_spec(self,fullname,path=None,target=None):
        if not self.frameworks_allowed and fullname.split(".")[0] in {"torch","transformers","tensorflow","jax","cupy"}:
            raise ImportError("STOP_NATIVE_FRAMEWORK_BEFORE_ACTUAL_GUARDIAN")
        if fullname==self.package:
            return importlib.util.spec_from_loader(fullname,self,is_package=True)
        if fullname.startswith(self.package+"."):
            leaf=fullname[len(self.package)+1:]
            if "." in leaf or leaf+".py" not in self.sources: raise ImportError("STOP_NATIVE_UNMANIFESTED_PACKET_MODULE")
            return importlib.util.spec_from_loader(fullname,self)
        return None
    def create_module(self,spec): return None
    def exec_module(self,module):
        name="__init__.py" if module.__name__==self.package else module.__name__.rsplit(".",1)[1]+".py"
        source=self.sources[name]
        module.__file__=str(self.here/name)
        module.__package__=self.package
        if name=="__init__.py": module.__path__=[str(self.here)]
        module._native_verified_source_sha256=hashlib.sha256(source).hexdigest()
        exec(compile(source,module.__file__,"exec"),module.__dict__)


def install_verified_packet_sources(sources,here=HERE,*,frameworks_allowed=False,preserve_safety=False,
                                    package="experiments.titan_quality_repair_v1"):
    finder=VerifiedPacketFinder(sources,here,package,frameworks_allowed=frameworks_allowed)
    for name in list(sys.modules):
        if name==package or name.startswith(package+"."):
            if preserve_safety and name==package+".native_safety": continue
            del sys.modules[name]
    parent=package.rsplit(".",1)[0] if "." in package else None
    if parent and parent not in sys.modules:
        namespace=types.ModuleType(parent)
        namespace.__path__=[str(Path(here).parent)]
        sys.modules[parent]=namespace
    sys.meta_path.insert(0,finder)
    return finder


def append_reviewed_site_paths(proposal):
    sites=proposal.get("runtime_site_packages")
    if not isinstance(sites,list) or len(sites)!=1 or Path(sites[0]).resolve()!=RUNTIME_SITE_PACKAGES.resolve():
        raise ValueError("STOP_NATIVE_RUNTIME_SITE_PATH")
    for path in proposal["runtime_site_packages"]:
        if path not in sys.path: sys.path.append(path)  # no site.py, user-site or .pth execution


def run_verified_worker(gate_path: Path,gate_sha: str):
    if not sys.flags.isolated or not sys.flags.no_site or not sys.flags.dont_write_bytecode:
        raise RuntimeError("STOP_NATIVE_WORKER_STARTUP_NOT_ISOLATED")
    gate=json.loads(read_pinned(gate_path,gate_sha))
    bundle=validate_bundle(Path(gate["proposal_path"]),Path(gate["permit_path"]))
    finder=install_verified_packet_sources(bundle["sources"],preserve_safety=True)
    append_reviewed_site_paths(bundle["proposal"])
    import importlib
    importlib.import_module("experiments.titan_quality_repair_v1")
    worker=importlib.import_module("experiments.titan_quality_repair_v1.native_qualification")
    sys.argv=["native_qualification","--bootstrap-gate",str(gate_path),"--bootstrap-gate-sha256",gate_sha]
    return worker.main()


def load_verified_helpers(sources: dict[str, bytes], here: Path = HERE):
    name = "_titan_native_verified_offline_helpers_" + hashlib.sha256(sources["offline_supervise.py"]).hexdigest()[:16]
    path = Path(here) / "offline_supervise.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(sources["offline_supervise.py"], str(path), "exec"), module.__dict__)
    module.MEMORY_CAP = int(LIMITS["tree_rss_max_gib"] * GIB)
    return module


def process_created(pid: int) -> float | None:
    import psutil
    try:
        return psutil.Process(pid).create_time()
    except psutil.NoSuchProcess:
        return None
    except psutil.AccessDenied as exc:
        raise RuntimeError("STOP_NATIVE_PROCESS_IDENTITY_UNAVAILABLE") from exc


def central_path(repo: Path = REPO) -> Path:
    value = subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=repo, text=True, timeout=10).strip()
    return Path(value) / "rineya-central-gpu0.lock"


def crs_path() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        raise RuntimeError("STOP_NATIVE_CRS_PATH_UNAVAILABLE")
    return Path(base) / "Rineya/CRS-001/host_state.sqlite3"


def require_crs_idle(path: Path):
    if not Path(path).is_file():
        raise RuntimeError("STOP_NATIVE_CRS_UNAVAILABLE")
    try:
        with closing(sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True, timeout=1)) as db:
            db.execute("PRAGMA query_only=ON")
            if db.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise RuntimeError("STOP_NATIVE_CRS_CORRUPT")
            leases = db.execute("SELECT status FROM leases").fetchall()
            active = db.execute("SELECT key FROM jobs WHERE state IN ('RESERVED','RUNNING','OUTCOME_UNKNOWN')").fetchall()
            if any(status != "RECONCILED" for status, in leases) or active:
                raise RuntimeError("STOP_NATIVE_CRS_ACTIVE_OR_UNRECONCILED")
    except (sqlite3.Error, OSError) as exc:
        raise RuntimeError("STOP_NATIVE_CRS_INCOMPATIBLE_OR_UNAVAILABLE") from exc


def acquire_token_lock(path: Path, token: str, owner: dict):
    if any(p.is_symlink() for p in (Path(path), *Path(path).parents)):
        raise RuntimeError("STOP_NATIVE_LOCK_SYMLINK")
    # Fully written/fsynced owned staging precedes atomic exclusive publication.
    # A write failure can leave only an owned attempt artifact, never a partial
    # shared lease. Hard-link failure refuses; no rename/replace fallback exists.
    stage=Path(owner["attempt_dir"])/("LEASE_STAGING_"+Path(path).name+"_"+token+".json")
    write_new(stage, {"label": LEASE_LABEL, "token": token, "created": time.time(),
                     "pid": owner["supervisor_pid"], "create_time": owner["supervisor_create_time"],
                     "worker_pids": [], "attempt_id": owner["attempt_id"]})
    os.link(stage,path)


def check_token_lock(path: Path, token: str, attempt: str):
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise RuntimeError("STOP_NATIVE_LEASE_UNREADABLE") from exc
    if (not isinstance(value, dict) or value.get("token") != token or value.get("label") != LEASE_LABEL
            or value.get("attempt_id") != attempt):
        raise RuntimeError("STOP_NATIVE_LEASE_OWNERSHIP_LOST")
    return value


def release_token_lock(path: Path, token: str, attempt: str) -> bool:
    try:
        check_token_lock(path, token, attempt)
        Path(path).unlink()
        return True
    except FileNotFoundError:
        return True
    except (OSError, RuntimeError):
        return not Path(path).exists()


def set_worker_pids(path: Path,token: str,attempt: str,pids):
    value=check_token_lock(path,token,attempt)
    value["worker_pids"]=list(pids)
    temporary=Path(str(path)+"."+token+".tmp")
    write_new(temporary,value)
    try: os.replace(temporary,path)
    finally:
        if temporary.exists(): temporary.unlink()


def output_bytes(root: Path) -> int:
    total = 0
    for path in Path(root).rglob("*"):
        try:
            if path.is_file(): total += path.stat().st_size
        except FileNotFoundError:
            pass
    return total


def resources(root: Path, members=()) -> dict:
    import psutil
    rss = 0
    for pid in members:
        try: rss += psutil.Process(pid).memory_info().rss
        except psutil.NoSuchProcess: pass
        except psutil.AccessDenied as exc: raise RuntimeError("STOP_NATIVE_TREE_RSS_UNAVAILABLE") from exc
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                      text=True, timeout=5).strip().splitlines()
        if len(out) != 1: raise ValueError("card count")
        gpu_free = int(out[0].strip()) * 2**20
        compute = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                          text=True, timeout=5).splitlines()
        gpu_pids = [int(value.strip()) for value in compute if value.strip().isdigit()]
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError("STOP_NATIVE_GPU_INVENTORY_UNAVAILABLE") from exc
    foreign = []
    desktops=[]
    allowed_executables={Path("C:/Windows/System32/dwm.exe").resolve(),Path("C:/Windows/explorer.exe").resolve()}
    for pid in set(gpu_pids)-set(members)-{os.getpid()}:
        try:
            process=psutil.Process(pid)
            created=process.create_time()
            executable=Path(process.exe()).resolve()
            if executable in allowed_executables and process.create_time()==created:
                desktops.append(pid)
            else:
                foreign.append({"pid":pid,"executable":str(executable),"observed_foreign_gpu_compute":True})
        except (psutil.NoSuchProcess,psutil.AccessDenied,OSError) as exc:
            foreign.append({"pid":pid,"observed_gpu_identity_unavailable":True,"error":repr(exc)})
    model_names = ("python", "torch", "llama", "ollama", "vllm", "comfy", "rineya", "train")
    for process in psutil.process_iter(["pid", "name", "memory_info"]):
        info = process.info
        if info["pid"] in set(members) | {os.getpid()}: continue
        name = (info.get("name") or "").lower()
        memory = info.get("memory_info")
        if not name and (memory is None or memory.rss>=512*2**20):
            foreign.append({"pid":info["pid"],"name":None,"rss_bytes":None if memory is None else memory.rss,
                            "unobservable_process_identity":True})
            continue
        if not any(part in name for part in model_names): continue
        if memory is None or memory.rss >= 512 * 2**20:
            foreign.append({"pid":info["pid"],"name":info.get("name"),"rss_bytes":None if memory is None else memory.rss})
    return {"host_available_bytes": psutil.virtual_memory().available, "gpu_available_bytes": gpu_free,
            "disk_available_bytes": shutil.disk_usage(root).free, "tree_rss_bytes": rss,
            "new_output_bytes": output_bytes(root), "gpu_compute_pids": gpu_pids,
            "allowed_desktop_gpu_pids":desktops,
            "foreign_model_processes":foreign,
            "opaque_wddm_compute_pids_not_used_as_default_ownership_evidence":True}


def require_resources(value: dict, *, preload=False, own_pids=()):
    for key in ("host_available_bytes","gpu_available_bytes","disk_available_bytes","tree_rss_bytes","new_output_bytes"):
        if type(value.get(key)) not in (int,float) or not math.isfinite(value[key]) or value[key]<0:
            raise RuntimeError("STOP_NATIVE_RESOURCE_OBSERVATION_UNAVAILABLE")
    if not isinstance(value.get("foreign_model_processes"),list):
        raise RuntimeError("STOP_NATIVE_FOREIGN_MODEL_INVENTORY_UNAVAILABLE")
    for key in ("gpu_compute_pids","allowed_desktop_gpu_pids"):
        if not isinstance(value.get(key,[]),list) or any(type(pid) is not int or pid<=0 for pid in value.get(key,[])):
            raise RuntimeError("STOP_NATIVE_GPU_PID_INVENTORY_UNAVAILABLE")
    host = LIMITS["host_admission_gib" if preload else "host_reserve_gib"] * GIB
    gpu = LIMITS["gpu_admission_gib" if preload else "gpu_reserve_gib"] * GIB
    if (value["host_available_bytes"] < host or value["gpu_available_bytes"] < gpu
            or value["disk_available_bytes"] < LIMITS["disk_admission_gib"] * GIB
            or value["tree_rss_bytes"] > LIMITS["tree_rss_max_gib"] * GIB
            or value["new_output_bytes"] > LIMITS["output_bytes_max"]
            or value.get("foreign_model_processes")
            or set(value.get("gpu_compute_pids",[]))-set(own_pids)-set(value.get("allowed_desktop_gpu_pids",[]))):
        raise RuntimeError("STOP_NATIVE_RESOURCES_OR_FOREIGN_GPU_OWNER")


def _kernel():
    if os.name != "nt": raise RuntimeError("STOP_NATIVE_REQUIRES_WINDOWS")
    from ctypes import wintypes as w
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.argtypes, k.OpenProcess.restype = (w.DWORD,w.BOOL,w.DWORD), w.HANDLE
    k.GetCurrentProcess.restype = w.HANDLE
    k.CloseHandle.argtypes = (w.HANDLE,)
    k.DuplicateHandle.argtypes = (w.HANDLE,w.HANDLE,w.HANDLE,ctypes.POINTER(w.HANDLE),w.DWORD,w.BOOL,w.DWORD)
    k.IsProcessInJob.argtypes = (w.HANDLE,w.HANDLE,ctypes.POINTER(w.BOOL))
    k.QueryInformationJobObject.argtypes = (w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD,ctypes.POINTER(w.DWORD))
    k.GetFinalPathNameByHandleW.argtypes = (w.HANDLE,w.LPWSTR,w.DWORD,w.DWORD)
    return k


@contextmanager
def duplicate_parent_handle(parent_pid: int, handle: int):
    from ctypes import wintypes as w
    k, duplicate = _kernel(), w.HANDLE()
    parent = k.OpenProcess(0x0040, False, parent_pid)
    if not parent: raise RuntimeError("STOP_NATIVE_OWNER_HANDLE_UNAVAILABLE")
    try:
        if not k.DuplicateHandle(parent, w.HANDLE(handle), k.GetCurrentProcess(), ctypes.byref(duplicate), 0, False, 2):
            raise RuntimeError("STOP_NATIVE_OWNER_HANDLE_DUPLICATION_FAILED")
        yield k, duplicate
    finally:
        if duplicate.value: k.CloseHandle(duplicate)
        k.CloseHandle(parent)


def job_members(handle: int) -> list[int]:
    from ctypes import wintypes as w
    k = _kernel()
    size = 8 + 4096 * ctypes.sizeof(ctypes.c_size_t)
    buffer, returned = ctypes.create_string_buffer(size), w.DWORD()
    if not k.QueryInformationJobObject(w.HANDLE(handle), 3, buffer, size, ctypes.byref(returned)):
        raise RuntimeError("STOP_NATIVE_JOB_MEMBERSHIP_UNAVAILABLE")
    assigned, count = ctypes.c_uint32.from_buffer(buffer,0).value, ctypes.c_uint32.from_buffer(buffer,4).value
    if assigned > 4096 or count != assigned:
        raise RuntimeError("STOP_NATIVE_JOB_MEMBERSHIP_INCOMPLETE")
    return [ctypes.c_size_t.from_buffer(buffer,8+i*ctypes.sizeof(ctypes.c_size_t)).value for i in range(count)]


def close_owned_handle(owner):
    # Reviewed helper close() historically ignored the Win32 BOOL. Preserve a
    # failed close as an unresolved ownership error rather than claiming release.
    if hasattr(owner,"kernel"):
        if owner.handle is not None:
            if not owner.kernel.CloseHandle(owner.handle): raise RuntimeError("STOP_NATIVE_OWNED_HANDLE_CLOSE_FAILED")
            owner.handle=None
    else:
        owner.close()  # private injected fixture only; real helper always has kernel


def verify_owner_handles(admission: dict, worker_mode: bool) -> list[int]:
    from ctypes import wintypes as w
    with duplicate_parent_handle(admission["supervisor_pid"], admission["hardware_owner_handle"]) as (k, hardware):
        path = ctypes.create_unicode_buffer(32768)
        if not k.GetFinalPathNameByHandleW(hardware, path, len(path), 0):
            raise RuntimeError("STOP_NATIVE_HARDWARE_HANDLE_UNAVAILABLE")
        observed = path.value.removeprefix("\\\\?\\")
        if Path(observed).resolve() != HARDWARE_LOCK.resolve():
            raise RuntimeError("STOP_NATIVE_HARDWARE_HANDLE_WRONG_OBJECT")
    with duplicate_parent_handle(admission["supervisor_pid"], admission["job_owner_handle"]) as (k, job):
        process = k.GetCurrentProcess() if worker_mode else k.OpenProcess(0x0100, False, admission["worker_pid"])
        if not process: raise RuntimeError("STOP_NATIVE_WORKER_PROCESS_UNAVAILABLE")
        flag = w.BOOL()
        try:
            if not k.IsProcessInJob(process, job, ctypes.byref(flag)) or not flag.value:
                raise RuntimeError("STOP_NATIVE_WORKER_OUTSIDE_OWN_JOB")
            return job_members(int(job.value))
        finally:
            if not worker_mode: k.CloseHandle(process)


def validate_worker_admission(value: dict) -> dict:
    if not isinstance(value, dict) or value.get("experiment") != EXPERIMENT or value.get("stage") != STAGE:
        raise ValueError("STOP_NATIVE_WORKER_ADMISSION_SCOPE")
    for key in ("source_checkpoint_sha256", "registry_sha256", "source_manifest_sha256", "proposal_payload_sha256",
                "proposal_file_sha256", "permit_file_sha256"):
        if not isinstance(value.get(key), str) or HEX.fullmatch(value[key]) is None:
            raise ValueError("STOP_NATIVE_WORKER_ADMISSION_DIGEST:"+key)
    if value["source_checkpoint_sha256"] != WEIGHT_SHA256:
        raise ValueError("STOP_NATIVE_WORKER_SOURCE")
    for key, expected in CAPS.items():
        if type(value.get(key)) is not int or value[key] != expected:
            raise ValueError("STOP_NATIVE_WORKER_CAP:"+key)
    if value.get("limits") != LIMITS or value.get("fixture_only") is not False:
        raise ValueError("STOP_NATIVE_WORKER_LIMITS_OR_FIXTURE")
    if value.get("job_assigned_before_payload") is not True or value.get("guardian_admission_passed") is not True:
        raise ValueError("STOP_NATIVE_WORKER_NO_ASSIGNED_JOB_ADMISSION")
    for key in ("supervisor_pid", "worker_pid", "hardware_owner_handle", "job_owner_handle"):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise ValueError("STOP_NATIVE_WORKER_IDENTITY_OR_HANDLE")
    for key in ("supervisor_create_time", "worker_create_time", "supervisor_started_monotonic"):
        if type(value.get(key)) not in (int, float) or not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError("STOP_NATIVE_WORKER_CREATE_TIME")
    if (not isinstance(value.get("attempt_id"),str) or IDENTIFIER.fullmatch(value["attempt_id"]) is None
            or not isinstance(value.get("hardware_slot"),str) or not value["hardware_slot"].strip()):
        raise ValueError("STOP_NATIVE_WORKER_ATTEMPT_OR_SLOT")
    root = validate_runtime(value["runtime_root"])
    if Path(value["attempt_dir"]).resolve() != root / "attempts" / value["attempt_id"]:
        raise ValueError("STOP_NATIVE_WORKER_ATTEMPT_PATH")
    if Path(value["temp_lock"]) != TEMP_LOCK or Path(value["hardware_lock"]) != HARDWARE_LOCK:
        raise ValueError("STOP_NATIVE_WORKER_SHARED_LOCK_PATH")
    if not isinstance(value.get("lease_token"),str) or len(value["lease_token"]) != 32:
        raise ValueError("STOP_NATIVE_WORKER_TOKEN")
    return value


def require_worker_environment(value: dict):
    attempt=Path(value["attempt_dir"])
    expected={"CUDA_VISIBLE_DEVICES":"0","CUBLAS_WORKSPACE_CONFIG":":4096:8","PYTHONDONTWRITEBYTECODE":"1",
        "TOKENIZERS_PARALLELISM":"false","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
        "HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","TMP":str(attempt/"tmp"),"TEMP":str(attempt/"tmp"),
        "HF_HOME":str(attempt/"cache/hf"),"HF_HUB_CACHE":str(attempt/"cache/hf/hub"),
        "TORCH_HOME":str(attempt/"cache/torch"),"XDG_CACHE_HOME":str(attempt/"cache"),
        "CUDA_CACHE_PATH":str(attempt/"cache/cuda"),"TORCH_EXTENSIONS_DIR":str(attempt/"cache/torch_extensions"),
        "TORCHINDUCTOR_CACHE_DIR":str(attempt/"cache/torchinductor"),"TRITON_CACHE_DIR":str(attempt/"cache/triton")}
    if any(os.environ.get(key)!=expected_value for key,expected_value in expected.items()):
        raise RuntimeError("STOP_NATIVE_WORKER_ENVIRONMENT_DRIFT")


def bootstrap_worker(gate_path: Path, gate_sha256: str, *, timeout=30) -> tuple[dict, "Guardian"]:
    """Bind a real CPython descendant before releasing any native payload.

    Hermes' launcher may be a different PID. The supervisor admits the actual
    worker from its create-only bootstrap receipt and exact Job membership.
    """
    gate=json.loads(read_pinned(gate_path,gate_sha256))
    bundle=validate_bundle(Path(gate["proposal_path"]),Path(gate["permit_path"]))
    for key in ("proposal_payload_sha256","proposal_file_sha256","permit_file_sha256"):
        if gate.get(key)!=bundle[key]: raise ValueError("STOP_NATIVE_BOOTSTRAP_BINDING")
    if gate.get("job_assigned_before_payload") is not True:
        raise RuntimeError("STOP_NATIVE_BOOTSTRAP_JOB_NOT_ASSIGNED")
    if process_created(gate["supervisor_pid"]) != gate["supervisor_create_time"]:
        raise RuntimeError("STOP_NATIVE_BOOTSTRAP_SUPERVISOR_IDENTITY")
    import psutil
    current=psutil.Process(os.getpid())
    actual={**gate,"worker_pid":os.getpid(),"worker_create_time":current.create_time()}
    validate_worker_admission(actual)
    members=verify_owner_handles(actual,True)
    if os.getpid() not in members: raise RuntimeError("STOP_NATIVE_BOOTSTRAP_OUTSIDE_JOB")
    for path in (gate["temp_lock"],gate["central_lock"]):
        check_token_lock(Path(path),gate["lease_token"],gate["attempt_id"])
    ancestors=[{"pid":p.pid,"create_time":p.create_time()} for p in current.parents()]
    write_new(Path(gate["attempt_dir"])/"WORKER_BOOTSTRAP.json",{
        "experiment":EXPERIMENT,"stage":STAGE,"worker_pid":os.getpid(),"worker_create_time":current.create_time(),
        "ancestors":ancestors,**{k:gate[k] for k in ("attempt_id","lease_token","source_checkpoint_sha256",
            "source_manifest_sha256","proposal_payload_sha256")},"actual_parent_job_membership":True})
    go_path=Path(gate["attempt_dir"])/"GO.json"
    deadline=time.monotonic()+min(timeout,30)
    while not go_path.exists():
        if time.monotonic()>=deadline: raise RuntimeError("STOP_NATIVE_BOOTSTRAP_GO_TIMEOUT")
        if process_created(gate["supervisor_pid"])!=gate["supervisor_create_time"]:
            raise RuntimeError("STOP_NATIVE_BOOTSTRAP_SUPERVISOR_LOST")
        time.sleep(.05)
    go=json.loads(go_path.read_bytes())
    if go.get("lease_token")!=gate["lease_token"] or go.get("attempt_id")!=gate["attempt_id"]:
        raise RuntimeError("STOP_NATIVE_BOOTSTRAP_GO_TOKEN")
    if Path(go["start_path"]).resolve()!=Path(gate["attempt_dir"])/START_NAME:
        raise RuntimeError("STOP_NATIVE_BOOTSTRAP_GO_PATH")
    guardian=Guardian(Path(go["start_path"]),go["start_sha256"],worker_mode=True)
    if guardian.admission["worker_pid"]!=os.getpid(): raise RuntimeError("STOP_NATIVE_BOOTSTRAP_WORKER_CHANGED")
    guardian.check(stage="before_native_payload")
    return guardian.admission,guardian


class Guardian:
    """Actual live verifier; production workers require this exact bound check method."""
    def __init__(self, start_path: Path, start_sha256: str, *, worker_mode=True):
        self.start_path, self.start_sha256, self.worker_mode = Path(start_path), start_sha256, worker_mode
        self.admission = validate_worker_admission(json.loads(read_pinned(self.start_path,start_sha256)))
        self.bundle = validate_bundle(Path(self.admission["proposal_path"]),Path(self.admission["permit_path"]))
        for key in ("proposal_payload_sha256","proposal_file_sha256","permit_file_sha256"):
            if self.admission[key] != self.bundle[key]: raise ValueError("STOP_NATIVE_ADMISSION_BUNDLE_BINDING")
        for key in ("attempt_id","hardware_slot"):
            if self.admission[key] != self.bundle["permit"][key]: raise ValueError("STOP_NATIVE_ADMISSION_OPERATOR_BINDING")
        self._local_resource_check=None
        self.cached_resource_ownership=None

    def bind_local_resource_check(self, check):
        """Only the real admitted framework may bind its torch/psutil reserve check.
        Ownership/source/permit checks remain in this Guardian. The check is not
        an admission substitute and cannot change registered resource thresholds.
        """
        owner=getattr(check,"__self__",None)
        import importlib
        framework=importlib.import_module("experiments.titan_quality_repair_v1.native_qualification").TorchFramework
        if (not callable(check) or type(owner) is not framework
                or getattr(check,"__func__",None) is not framework.resource_snapshot):
            raise ValueError("STOP_NATIVE_LOCAL_RESOURCE_CHECK_NOT_ACTUAL_FRAMEWORK")
        self._local_resource_check=check

    def check(self, *, stage: str) -> dict:
        value = self.admission
        if self.worker_mode: require_worker_environment(value)
        cheap=stage in {"each_forward","checkpoint_recompute_layer"}
        preload=stage in {"before_load_begun","load_source","before_dense_reload_begun","reload_dense"}
        read_pinned(self.start_path,self.start_sha256)
        if not cheap:
            validate_manifest(self.bundle["proposal"]["source_manifest"])
            read_pinned(value["proposal_path"],value["proposal_file_sha256"])
            read_pinned(value["permit_path"],value["permit_file_sha256"])
            for descriptor in (self.bundle["review_descriptor"],self.bundle["cpu_qualification_descriptor"],self.bundle["metadata_descriptor"]):
                read_pinned(descriptor["path"],descriptor["sha256"])
            for name,expected in self.bundle["proposal"]["source_sidecar_sha256"].items():
                read_pinned(CORE/name,expected)
        if (process_created(value["supervisor_pid"]) != value["supervisor_create_time"]
                or process_created(value["worker_pid"]) != value["worker_create_time"]
                or (self.worker_mode and os.getpid() != value["worker_pid"])):
            raise RuntimeError("STOP_NATIVE_PROCESS_IDENTITY_LOST")
        wall=CAPS["worker_wall_seconds"] if self.worker_mode else CAPS["supervisor_hard_wall_seconds"]
        if time.monotonic()-value["supervisor_started_monotonic"] >= wall:
            raise RuntimeError("STOP_NATIVE_WORKER_OR_SUPERVISOR_WALL")
        for path in (value["temp_lock"],value["central_lock"]):
            check_token_lock(Path(path),value["lease_token"],value["attempt_id"])
        if not cheap:
            if Path(value["central_lock"]) != central_path(): raise RuntimeError("STOP_NATIVE_CENTRAL_PATH_CHANGED")
            require_crs_idle(Path(value["crs_path"]))
        members = verify_owner_handles(value,self.worker_mode)
        if not cheap:
            actual=resources(Path(value["attempt_dir"]),members)
            require_resources(actual,own_pids=members,preload=preload)
            self.cached_resource_ownership={"foreign_model_processes":actual["foreign_model_processes"],
                                           "gpu_compute_pids":actual["gpu_compute_pids"],
                                           "allowed_desktop_gpu_pids":actual["allowed_desktop_gpu_pids"]}
        if stage=="before_framework_import":
            for finder in sys.meta_path:
                if type(finder) is VerifiedPacketFinder:
                    finder.frameworks_allowed=True
        if self.worker_mode and self._local_resource_check is not None:
            if self.cached_resource_ownership is None:
                raise RuntimeError("STOP_NATIVE_NO_PRIOR_FULL_RESOURCE_PROOF")
            observed=self._local_resource_check()
            require_resources(observed,own_pids=members,preload=preload)
        elif cheap:
            raise RuntimeError("STOP_NATIVE_CHEAP_CHECK_WITHOUT_LOCAL_RESOURCE_PROOF")
        else:
            observed = actual
        return {"passed": True, "phase": stage, "experiment": EXPERIMENT, "stage": STAGE,
                "source_weight_sha256":value["source_checkpoint_sha256"],
                **{k:value[k] for k in ("source_checkpoint_sha256","registry_sha256","source_manifest_sha256",
                    "proposal_payload_sha256","attempt_id","hardware_slot","worker_pid","worker_create_time","supervisor_pid")},
                "job_bound": True,"leases_owned": True,"permit_verified": True,"fixture_only": False,
                "resources": observed,"job_member_pids":members}
