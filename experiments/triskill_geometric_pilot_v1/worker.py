"""Bounded screen worker; run only inside the separately qualified supervisor.

This file implements no process launcher or Windows containment repairs. A
create-only handoff receipt from that lane plus actual Job membership is required.
The default command is harmless preflight, with no torch import or model load.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import datetime as dt
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _reject_bootstrap_path_substitution(path: Path):
    # Check before resolving paths so a link/junction cannot hide its target.
    path = Path(os.path.abspath(path))
    for component in reversed((path, *path.parents)):
        try:
            info = component.lstat()
        except OSError as exc:
            raise ValueError(f"prospective source missing/inaccessible: {path}") from exc
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"prospective source path substitution: {component}")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"prospective source is not a regular file: {path}")


def _load_binding(freeze):
    """Bootstrap only the precisely named, supervisor-pinned binding source."""
    entry = freeze.get("dependency_binding", {}).get("roles", {}).get("binding", {})
    path = HERE / "dependency_binding.py"
    relative = "experiments/triskill_geometric_pilot_v1/dependency_binding.py"
    if entry.get("kind") != "code" or entry.get("path") != relative:
        raise ValueError("incomplete prospective dependency binding")
    _reject_bootstrap_path_substitution(path)
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if entry.get("sha256") != digest:
        raise ValueError("prospective binding source digest mismatch")
    existing = sys.modules.get("dependency_binding")
    if existing is not None:
        origin = getattr(existing, "__file__", None)
        spec_origin = getattr(getattr(existing, "__spec__", None), "origin", None)
        if not origin or os.path.normcase(os.path.abspath(origin)) != os.path.normcase(str(path)) or (spec_origin and os.path.normcase(os.path.abspath(spec_origin)) != os.path.normcase(str(path))):
            raise ValueError("cached binding module path mismatch")
    name = f"_triskill_pilot_binding_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ValueError("binding module spec unavailable")
    module = importlib.util.module_from_spec(spec)
    module._triskill_source_binding = (str(path), digest)
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    _reject_bootstrap_path_substitution(path)
    if sha256(path) != digest:
        raise ValueError("binding source changed during validation")
    return module

LOCK = Path("C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock")
ARMS = ("BASE", "T", "H", "TH", "IDENTITY", "ROTATION", "RANDOM", "PROMPT", "REVERSE")
MAX_GENERATIONS = 64
MAX_FORWARDS = 1600
MAX_TOKEN_POSITIONS = 20000
WALL_SECONDS = 600
INPUT_SLOTS = 256
OUTPUT_SLOTS = 24
PROMPT_COMPOSITION = (
    "Track updates and whether uncertainty survives. Answer only if determined. "
)


class Budget:
    def __init__(self, now=time.monotonic):
        self.now = now
        self.started = now()
        self.forwards = self.completed_forwards = self.token_positions = self.generations = 0
        self.completed_generations = 0

    def check(self):
        if self.now() - self.started >= WALL_SECONDS:
            raise RuntimeError("STOP_WALL_BUDGET")

    def forward(self, positions: int):
        self.check()
        if self.forwards + 1 > MAX_FORWARDS or self.token_positions + positions > MAX_TOKEN_POSITIONS:
            raise RuntimeError("STOP_FORWARD_OR_TOKEN_BUDGET")
        self.forwards += 1
        self.token_positions += positions

    def generation(self):
        self.check()
        if self.generations >= MAX_GENERATIONS:
            raise RuntimeError("STOP_GENERATION_BUDGET")
        self.generations += 1

    def receipt(self):
        return {"model_forwards": self.forwards, "token_positions_including_padding": self.token_positions,
                "completed_model_forwards": self.completed_forwards,
                "screen_generations_requested": self.generations,
                "completed_screen_generations": self.completed_generations,
                "elapsed_seconds": self.now() - self.started}


def preflight(owner_token=None):
    import psutil
    rec = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "host_ram_free_gib": psutil.virtual_memory().available / 2**30,
           "disk_free_gib": shutil.disk_usage(HERE).free / 2**30,
           "lease": None, "python_jobs": [], "model_forwards": 0, "model_loads": 0}
    if LOCK.exists():
        try:
            rec["lease"] = json.loads(LOCK.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rec["lease"] = "unreadable; fail closed"
    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            name = (proc.info["name"] or "").lower()
            if any(word in name for word in ("python", "ollama", "llama", "vmmem")):
                rec["python_jobs"].append({"pid": proc.pid, "name": name,
                                           "rss_bytes": proc.info["memory_info"].rss})
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            rec.setdefault("process_visibility_incomplete", True)
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu",
                              "--format=csv,noheader,nounits"], check=True,
                             capture_output=True, text=True, timeout=10).stdout.strip()
        rec["gpu_raw"] = gpu
        fields = gpu.split(",")
        if len(fields) != 4:
            raise ValueError("expected exactly one GPU")
        rec["gpu_free_gib"] = float(fields[2]) / 1024
        holders = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                  "--format=csv,noheader"], check=True,
                                 capture_output=True, text=True, timeout=10).stdout.strip()
        rec["gpu_holders_raw"] = holders
        rec["gpu_attribution_incomplete"] = "Insufficient Permissions" in holders
        # A compute-holder record does not establish whether it is a model.
        # Fail closed rather than infer ownership from host RSS or free VRAM.
        rec["foreign_gpu_compute_pids"] = [int(line.split(",",1)[0]) for line in holders.splitlines()
            if line.split(",",1)[0].strip().isdigit() and int(line.split(",",1)[0]) != os.getpid()]
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        rec["gpu_error"] = str(exc)
    reasons = []
    if rec["host_ram_free_gib"] < 5: reasons.append("RAM below original 5GiB reserve")
    if rec["disk_free_gib"] < 6: reasons.append("disk below original 6GiB reserve")
    if rec.get("gpu_free_gib", 0) < 6: reasons.append("GPU below conservative 6GiB preload threshold")
    owned = isinstance(rec["lease"],dict) and rec["lease"].get("pid") == os.getpid() and rec["lease"].get("token") == owner_token
    if rec["lease"] is not None and not owned: reasons.append("shared GPU lease held/unreadable")
    if rec.get("gpu_attribution_incomplete") or rec.get("gpu_error"):
        reasons.append("GPU ownership attribution incomplete")
    if rec.get("foreign_gpu_compute_pids"):
        reasons.append("foreign GPU compute PIDs; exclusive model ownership not established")
    if rec.get("process_visibility_incomplete"):
        reasons.append("research process visibility incomplete")
    mine = {os.getpid(), os.getppid()}
    if any(p["pid"] not in mine and p["rss_bytes"] >= 512*1024**2 for p in rec["python_jobs"]):
        reasons.append("foreign research/model process has >=512MiB RSS")
    rec["blockers"] = reasons
    rec["status"] = "BLOCKED" if reasons else "RESOURCES_CLEAR_CONTAINMENT_STILL_REQUIRED"
    return rec


def validate_containment(path: Path, expected_sha: str, owner_verifier=None):
    """Delegate actual Job/supervisor proof to the independently qualified owner.

    Receipt claims plus membership in an arbitrary Job are insufficient. No
    verifier exists in this research lane, and the CLI cannot invent one.
    """
    if owner_verifier is None:
        raise RuntimeError("STOP_CONTAINMENT_INTEGRATION_UNQUALIFIED")
    if len(expected_sha) != 64 or sha256(path) != expected_sha:
        raise ValueError("containment receipt digest mismatch")
    proof=owner_verifier(path,expected_sha,os.getpid(),WALL_SECONDS)
    if not isinstance(proof,dict) or proof.get("qualified") is not True or proof.get("worker_pid") != os.getpid() or not 0 < proof.get("hard_wall_seconds",601) <= WALL_SECONDS:
        raise ValueError("owner verifier did not establish live-worker containment")
    return proof


def prompt_system(row, arm):
    return PROMPT_COMPOSITION+row.system if arm == "PROMPT" and "rules" in row.metadata else row.system


def directional_rotation(rotation, arm):
    """Reversing parent order inverts the operator, not the behavioral answer."""
    if arm == "REVERSE":
        return rotation.T
    if arm in ("ROTATION", "RANDOM"):
        return rotation
    return None


def screen_schedule(rows):
    """Nine controls within the original 64-generation envelope.

    Identity and reverse are measured on four chain rows only; their parent
    retention remains unmeasured. All comparisons preserve explicit coverage.
    """
    parent_rows=tuple(row for row in rows if row.task_kind in ("transition_parent","answerability_parent"))
    chain_rows=tuple(row for row in rows if "rules" in row.metadata)
    if len(parent_rows) != 4 or len(chain_rows) != 4 or len(rows) != 8:
        raise ValueError("frozen screening allocation changed")
    schedule=[(arm,parent_rows) for arm in ("BASE","T","H")]
    schedule += [(arm,chain_rows if arm in ("BASE","T","H","IDENTITY","REVERSE") else rows) for arm in ARMS]
    if sum(len(unit_rows) for _,unit_rows in schedule) != MAX_GENERATIONS:
        raise ValueError("control allocation exceeds the original generation envelope")
    return tuple(schedule)


def _validate_freeze(path: Path, expected_sha: str):
    _reject_bootstrap_path_substitution(path)
    freeze_source=path.read_bytes()
    if not expected_sha or hashlib.sha256(freeze_source).hexdigest() != expected_sha:
        raise ValueError("supervisor-pinned prospective freeze digest required")
    freeze=json.loads(freeze_source)
    required={"worker.py","geometry.py","benchmark.py","dependency_binding.py","geometry.npz","geometry.json",
              "screen.json","development.json","PREREG.md","BENCHMARK_MATRIX.md",
              "FOLLOWUP_PREREG.md","FUTURE_PROTOCOL.md","SOURCES_VERIFIED.md"}
    if not required <= set(freeze.get("files",{})):
        raise ValueError("incomplete prospective freeze membership")
    for relative,expected in freeze["files"].items():
        # Local artifacts remain local: the dependency manifest alone admits
        # the two precisely identified renderer/DEV-fixture paths.
        if not isinstance(relative,str) or Path(relative).name != relative:
            raise ValueError(f"prospective freeze path mismatch: {relative}")
        target=HERE/relative
        _reject_bootstrap_path_substitution(target)
        if sha256(target) != expected:
            raise ValueError(f"prospective freeze mismatch: {relative}")
    binding=_load_binding(freeze)
    bound=binding.validate_dependencies(freeze.get("dependency_binding"))
    geometry=bound.load_local("geometry")
    required_assets={"config.json","generation_config.json","tokenizer.json","tokenizer_config.json","vocab.json","merges.txt"}
    if not required_assets <= set(freeze.get("source_sidecars",{})):
        raise ValueError("source tokenizer/config freeze missing")
    for name,expected in freeze["source_sidecars"].items():
        if name not in required_assets:
            raise ValueError(f"source sidecar changed: {name}")
        sidecar=geometry.SOURCES["BASE"][0].parent/name
        _reject_bootstrap_path_substitution(sidecar)
        if sha256(sidecar) != expected:
            raise ValueError(f"source sidecar changed: {name}")
    for package,expected in freeze.get("runtime_packages",{}).items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"runtime package changed: {package}")
    # Reject mutation during the checks, before any renderer fixture self-test.
    bound.verify_current()
    if sha256(path) != expected_sha:
        raise ValueError("prospective freeze changed during validation")
    return freeze,bound,geometry


def validate_freeze(path: Path, expected_sha: str):
    return _validate_freeze(path,expected_sha)[0]


@contextmanager
def exclusive_lease():
    # No waiting/reclaiming/terminating a foreign owner. Existing convention only.
    token = uuid.uuid4().hex
    now = time.time()
    rec = {"created": now, "pid": os.getpid(), "token": token, "label": "triskill-geo001",
           "job": "nine-arm 64-generation bounded zero-training screen", "worker_pids": [os.getpid()],
           "orphan_after_unix": now + 900, "cleanup_complete": False}
    with LOCK.open("x", encoding="utf-8") as stream:
        json.dump(rec, stream)
    try:
        yield rec
    finally:
        try:
            current = json.loads(LOCK.read_text(encoding="utf-8"))
            if current.get("token") == token and rec["cleanup_complete"]:
                LOCK.unlink()
        except (OSError, ValueError):
            pass


def load_arm_weights(model, arm, torch, geometry):
    """Replace only in-memory parameters; H uses the realized compressed update."""
    target = "TH" if arm in ("TH", "IDENTITY", "ROTATION", "REVERSE", "RANDOM", "PROMPT") else arm
    files = {name: geometry.TensorFile(path) for name, (path, _) in geometry.SOURCES.items()}
    try:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if target == "H":
                    value = files["BASE"].tensor(name) + files["TH"].tensor(name) - files["T"].tensor(name)
                else:
                    value = files[target].tensor(name)
                if tuple(value.shape) != tuple(param.shape):
                    raise ValueError(f"parameter shape mismatch {name}")
                param.copy_(torch.from_numpy(value.astype("float32")).to(param.device))
    finally:
        for stream in files.values(): stream.close()


def run_screen(out: Path, containment_path: Path, containment_sha: str, freeze_path: Path,
               freeze_sha: str, owner_verifier=None):
    """Narrow bootstrap: native functions come only from pinned source bytes.

    The qualified supervisor must pin this bootstrap source before executing
    it. Self-validation does not authenticate an arbitrary malicious bootstrap.
    Same-path imported worker objects are never used as native implementations.
    """
    started=time.monotonic()
    _reject_bootstrap_path_substitution(freeze_path)
    source=freeze_path.read_bytes()
    if not freeze_sha or hashlib.sha256(source).hexdigest() != freeze_sha:
        raise ValueError("supervisor-pinned prospective freeze digest required")
    freeze=json.loads(source)
    binding=_load_binding(freeze)
    bound=binding.validate_dependencies(freeze.get("dependency_binding"))
    worker=bound.load_worker()
    bound.verify_worker_execution(worker)
    return worker._run_bound_screen(out,containment_path,containment_sha,freeze_path,freeze_sha,
                                    owner_verifier=owner_verifier,_started=started)


def _run_bound_screen(out: Path, containment_path: Path, containment_sha: str, freeze_path: Path,
                      freeze_sha: str, owner_verifier=None, _started=None):
    """Single model load; all arms share padded greedy fixed-slot inference."""
    module=sys.modules.get(__name__)
    if module is None or getattr(module,"_triskill_source_binding",None) != (str(HERE/"worker.py"),sha256(HERE/"worker.py")):
        raise RuntimeError("STOP_UNBOUND_NATIVE_WORKER")
    budget = Budget()  # External supervisor ceiling also includes startup and load.
    if _started is not None: budget.started=_started
    budget.check()
    containment = validate_containment(containment_path, containment_sha, owner_verifier)
    freeze,bound,geometry = _validate_freeze(freeze_path,freeze_sha)
    bound.verify_worker_execution(module)
    benchmark=bound.load_local("benchmark")
    build_development,build_screen=benchmark.build_development,benchmark.build_screen
    score_response,compare_outputs=benchmark.score_response,benchmark.compare_outputs
    import numpy as np
    snapshot = preflight()
    if snapshot["blockers"]: raise RuntimeError(json.dumps(snapshot))
    if out.exists(): raise FileExistsError("create-only screen output exists")
    geo_receipt = json.loads((HERE / "geometry.json").read_text(encoding="utf-8"))
    if geo_receipt["status"] != "PREPARED" or sha256(HERE / "geometry.npz") != geo_receipt["geometry_sha256"]:
        raise ValueError("geometry is absent/commuting/unverified")
    for path, expected in geometry.SOURCES.values():
        if sha256(path) != expected: raise ValueError("source weight digest changed")
    data = np.load(HERE / "geometry.npz", allow_pickle=False)
    renderer=bound.load_renderer()
    renderer.template_selftest()  # Pinned original DEV bytes; no protected bank.
    render=renderer.render
    rows = build_screen()
    if len(rows) != 8: raise ValueError("screen membership changed")
    if [row.to_dict() for row in rows] != json.loads((HERE/"screen.json").read_text(encoding="utf-8")):
        raise ValueError("screen generator differs from frozen membership")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    result = {"experiment": "TRISKILL-GEO-001-GAPFIX", "status": "STARTED", "confirmation": False,
              "study_mode": "mechanical_exploratory", "natural_causal_mechanism_claim": False,
              "construction_uses_C_labels": False, "direction_selection_adapted_from_C": False,
              "quality_claim": False, "training_examples": 0, "model_loads": 0,
              "resources_before": snapshot, "containment": containment, "arms": {},
              "freeze_sha256": sha256(freeze_path), "geometry_sha256": geo_receipt["geometry_sha256"],
              "dependency_binding": bound.receipt(),
              "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                          "dtype": "float32", "attention": "eager", "fixed_input_slots": INPUT_SLOTS,
                          "fixed_output_slots": OUTPUT_SLOTS, "greedy": True, "tf32": False}}
    def execute(lease_rec):
        result["lease"] = lease_rec
        # Check again after exclusive acquisition and immediately before load.
        budget.check()
        resources=preflight(lease_rec["token"])
        result["resources_immediately_before_load"]=resources
        if resources["blockers"]: raise RuntimeError(json.dumps(resources))
        if torch.cuda.mem_get_info()[0] / 2**30 < 6: raise RuntimeError("GPU changed before load")
        result["model_load_attempts"]=1
        # Recheck bindings at the last boundary before the native load.
        bound.verify_current()
        model = AutoModelForCausalLM.from_pretrained(str(geometry.SOURCES["BASE"][0].parent),
                    torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True,
                    low_cpu_mem_usage=True, device_map={"": "cuda"}).eval()
        result["model_loads"] = 1
        tok = AutoTokenizer.from_pretrained(str(geometry.SOURCES["BASE"][0].parent), local_files_only=True)
        device = next(model.parameters()).device
        b = torch.as_tensor(data["basis"], dtype=torch.float32, device=device)
        random_b = torch.as_tensor(data["random_basis"], dtype=torch.float32, device=device)
        r = torch.as_tensor(data["rotation"], dtype=torch.float32, device=device)
        state = {"basis": b, "change": torch.zeros_like(r), "energy": 0., "displacement": 0., "valid": None}

        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            coordinates = hidden @ state["basis"]
            delta = (coordinates @ state["change"].T) @ state["basis"].T
            valid = state["valid"].unsqueeze(-1)
            state["energy"] += float((hidden.square()*valid).sum().item())
            state["displacement"] += float((delta.square()*valid).sum().item())
            changed = hidden + delta
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        handle = model.model.layers[geometry.LAYER].register_forward_hook(hook)

        def decode(system, user, padded=True, slots=OUTPUT_SLOTS):
            # Fixed padding and slots match attention/decode compute across arms.
            ids = tok(render(system,user), add_special_tokens=False)["input_ids"]
            if len(ids) > INPUT_SLOTS: raise RuntimeError("STOP_INPUT_TOKEN_BUDGET; no truncation")
            pad = INPUT_SLOTS-len(ids) if padded else 0
            input_ids = torch.tensor([[tok.eos_token_id]*pad+ids], device=device)
            mask = torch.tensor([[0]*pad+[1]*len(ids)], device=device)
            positions = (mask.cumsum(-1)-1).clamp(min=0)
            cache = None
            generated = []
            semantic_end = False
            raw_tokens = []
            for step in range(slots):
                if torch.cuda.mem_get_info()[0] / 2**30 < .5: raise RuntimeError("STOP_GPU_ACTIVE_RESERVE")
                import psutil
                if psutil.virtual_memory().available / 2**30 < 1.5: raise RuntimeError("STOP_HOST_ACTIVE_RESERVE")
                state["valid"] = mask if cache is None else torch.ones_like(input_ids)
                budget.forward(int(input_ids.numel()))  # Count actual invocation attempts only.
                with torch.inference_mode():
                    output = model(input_ids=input_ids, attention_mask=mask, position_ids=positions,
                                   past_key_values=cache, use_cache=True)
                budget.completed_forwards += 1
                cache = output.past_key_values
                token = int(output.logits[0,-1].argmax().item())
                raw_tokens.append(token)
                if not semantic_end and token != tok.eos_token_id: generated.append(token)
                if token == tok.eos_token_id: semantic_end = True
                # EOS after termination is a compute-only suffix, never scored.
                input_ids = torch.tensor([[tok.eos_token_id if semantic_end else token]],device=device)
                mask = torch.cat((mask,torch.ones((1,1),dtype=mask.dtype,device=device)),dim=1)
                positions = torch.tensor([[len(ids)+step]],device=device)
            return tok.decode(generated,skip_special_tokens=True), raw_tokens

        try:
            # Padding transport is qualified on separate development fixtures only.
            parity = []
            for row in build_development()[:2]:
                unpadded = decode(row.system,row.user,False,4)[1]
                padded = decode(row.system,row.user,True,4)[1]
                parity.append(unpadded == padded)
            result["development_padding_parity"] = parity
            if not all(parity): raise RuntimeError("STOP_NATIVE_MECHANICS_PADDING")
            schedule=screen_schedule(rows)
            result["screen_allocation"]=[{"arm":arm,"row_ids":[row.row_id for row in unit_rows]} for arm,unit_rows in schedule]
            for unit_index,(arm,unit_rows) in enumerate(schedule):
                budget.check()
                load_arm_weights(model,arm,torch,geometry)
                state["basis"] = random_b if arm == "RANDOM" else b
                selected_rotation=directional_rotation(r,arm)
                state["change"] = selected_rotation-torch.eye(len(r),device=device) if selected_rotation is not None else torch.zeros_like(r)
                state["energy"] = state["displacement"] = 0.
                prior=result["arms"].get(arm,{})
                outputs, scores = dict(prior.get("outputs",{})), dict(prior.get("scores",{}))
                for row in unit_rows:
                    budget.generation()
                    system = prompt_system(row,arm)
                    outputs[row.row_id] = decode(system,row.user)[0]
                    budget.completed_generations += 1
                    scores[row.row_id] = score_response(row,outputs[row.row_id])
                energy=state["energy"]+prior.get("activation_energy",0.)
                displacement=state["displacement"]+prior.get("activation_displacement_energy",0.)
                result["arms"][arm] = {"outputs": outputs, "scores": scores,
                    "activation_energy":energy,"activation_displacement_energy":displacement,
                    "relative_activation_displacement": (displacement/max(energy,1e-30))**.5}
                if unit_index == 2:
                    # Do not attempt construction if its source manifestations do not transport.
                    for parent, kind in (("T","transition_parent"),("H","answerability_parent")):
                        relevant = [row for row in rows if row.task_kind == kind]
                        base_n = sum(result["arms"]["BASE"]["scores"][row.row_id]["strict_correct"] for row in relevant)
                        parent_n = sum(scores2["strict_correct"] for key,scores2 in result["arms"][parent]["scores"].items()
                                       if any(row.row_id == key for row in relevant))
                        damage=any(result["arms"]["BASE"]["scores"][row.row_id]["strict_correct"] and
                                   not result["arms"][parent]["scores"][row.row_id]["strict_correct"] for row in relevant)
                        if parent_n < 1 or parent_n <= base_n or damage:
                            raise RuntimeError(f"STOP_SCREEN_NO_PARENT_SIGNAL:{parent}")
                if arm == "IDENTITY" and any(text != result["arms"]["TH"]["outputs"].get(row_id) for row_id,text in outputs.items()):
                    raise RuntimeError("STOP_IDENTITY_CANARY")
                if arm == "IDENTITY":
                    result["identity_canary_row_ids"]=list(outputs)
                    result["identity_parent_retention"]="unmeasured"
            result["comparisons"] = {
                control: compare_outputs(rows,result["arms"][control]["outputs"],result["arms"]["ROTATION"]["outputs"])
                for control in ("BASE","T","H","TH","IDENTITY","RANDOM","PROMPT","REVERSE")}
            result["status"] = "SCREEN_COMPLETE_NOT_CONFIRMATION"
        except Exception as exc:
            result["status"] = str(exc)
        finally:
            handle.remove()
    # execute() has returned/unwound before CUDA cleanup and token-checked release.
    with exclusive_lease() as lease_rec:
        try:
            execute(lease_rec)
        except Exception as exc:
            result["status"]=str(exc)
        finally:
            gc.collect()
            try:
                if torch.cuda.is_initialized():
                    result["peak_cuda_allocated_bytes"]=torch.cuda.max_memory_allocated()
                    result["peak_cuda_reserved_bytes"]=torch.cuda.max_memory_reserved()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                lease_rec["cleanup_complete"]=True
            except Exception as cleanup_error:
                result["cleanup_error"]=str(cleanup_error)
                result["lease_retained_for_supervisor_cleanup"]=True
            result["budget"] = budget.receipt()
            with out.open("x",encoding="utf-8") as stream:
                json.dump(result,stream,indent=2)
                stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight","run"), default="preflight", nargs="?")
    parser.add_argument("--out",type=Path,default=HERE/"native_screen.json")
    parser.add_argument("--containment-receipt",type=Path)
    parser.add_argument("--containment-sha256")
    parser.add_argument("--freeze",type=Path,default=HERE/"FREEZE_GAPFIX.json")
    parser.add_argument("--freeze-sha256")
    args=parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(),indent=2))
    elif not args.containment_receipt or not args.containment_sha256:
        parser.error("run requires the owner lane's qualified live-worker containment handoff")
    else:
        print(json.dumps(run_screen(args.out,args.containment_receipt,args.containment_sha256,args.freeze,args.freeze_sha256),indent=2))
