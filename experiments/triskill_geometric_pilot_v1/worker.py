"""Bounded section H screen worker; launched only by supervise.py.

The supervisor holds the shared GPU lease and enforces the external hard wall; this
worker refuses to run unless the lease names its parent process and token. The default
command is harmless preflight, with no torch import or model load. Native internals are
module-level functions so CPU tests can drive them with a tiny model (test_native_tiny.py).
See AMENDMENT_SECTIONH.md.
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
LEASE_LABEL = "triskill-geo001-sectionh"
ARMS = ("BASE", "T", "H", "TH", "IDENTITY", "ROTATION", "REVERSE", "RANDOM", "PROMPT")
TH_HOSTED = ("TH", "IDENTITY", "ROTATION", "REVERSE", "RANDOM", "PROMPT")
READOUT_ARMS = ("TH", "ROTATION", "REVERSE", "RANDOM")
LIVENESS_ARMS = ("ROTATION", "REVERSE", "RANDOM")
INPUT_SLOTS = 288
OUTPUT_SLOTS = 32
READOUT_SLOTS = INPUT_SLOTS + OUTPUT_SLOTS - 1   # prompt plus TH's tokens but the last
EOS_TOKEN_IDS = (151645, 151643)   # generation_config.json eos_token_id, checked at load
MAX_GENERATIONS = 154
MAX_FORWARDS = 5100
MAX_TOKEN_POSITIONS = 76000
WALL_SECONDS = 540                 # internal; supervise.py enforces the 600 s hard wall
GPU_MEMORY_FRACTION = 0.78         # Windows pages CUDA allocations past VRAM into host RAM
PRELOAD_RAM_GIB, PRELOAD_DISK_GIB, PRELOAD_GPU_GIB = 5.0, 6.0, 6.0
ACTIVE_RAM_GIB, ACTIVE_GPU_GIB = 1.5, 0.5
FOREIGN_RSS_BYTES = 512 * 1024 ** 2
RESEARCH_PROCESS_WORDS = ("python", "ollama", "llama", "vmmem")
LIVENESS_RTOL = 1e-3
GEOMETRY_NPZ, GEOMETRY_JSON = "geometry_sectionh.npz", "geometry_sectionh.json"
SCREEN_JSON, DEVELOPMENT_JSON = "screen_sectionh.json", "development_sectionh.json"
FREEZE_NAME = "FREEZE_SECTIONH.json"
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


def _gpu_query(run=subprocess.run):
    gpu = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu",
               "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    holders = run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                  check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    return gpu, holders


def ancestor_pids():
    """This process's ancestors. Windows venv/uv python.exe is a launcher that runs the
    interpreter as a child, so the supervisor is an ancestor, not necessarily the parent."""
    import psutil
    try:
        return {parent.pid for parent in psutil.Process(os.getpid()).parents()}
    except (psutil.Error, OSError):
        return {os.getppid()}


def lease_owned(lease, owner_token, ancestors=None) -> bool:
    """A supervisor among this process's ancestors holds the lease under our token and label."""
    ancestors = ancestor_pids() if ancestors is None else ancestors
    return (isinstance(lease, dict) and owner_token is not None and lease.get("token") == owner_token
            and lease.get("pid") in ancestors and lease.get("label") == LEASE_LABEL)


def preflight(owner_token=None, *, lock=LOCK, processes=None, gpu_query=_gpu_query, ancestors=None):
    """Gate on measured resources and the shared lease, never on GPU process attribution.

    Under WDDM, nvidia-smi always lists the compositor and shell, often as
    "[Insufficient Permissions]"; those records are kept as information only. The
    lease (honoured by every research lane) plus measured free VRAM/RAM and the
    programme's foreign-research-process rule decide.
    """
    import psutil
    rec = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "host_ram_free_gib": psutil.virtual_memory().available / 2**30,
           "disk_free_gib": shutil.disk_usage(HERE).free / 2**30,
           "lease": None, "research_processes": [], "model_forwards": 0, "model_loads": 0}
    if lock.exists():
        try:
            rec["lease"] = json.loads(lock.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rec["lease"] = "unreadable; fail closed"
    ancestors = ancestor_pids() if ancestors is None else set(ancestors)
    mine = {os.getpid(), *ancestors}
    for proc in (processes if processes is not None else psutil.process_iter(["pid", "name", "memory_info"])):
        try:
            info = proc.info
            name = (info.get("name") or "").lower()
            if not any(word in name for word in RESEARCH_PROCESS_WORDS):
                continue
            memory = info.get("memory_info")
            if memory is None:
                if info.get("pid") not in mine:
                    rec["research_process_visibility_incomplete"] = True
                continue
            rec["research_processes"].append({"pid": info.get("pid"), "name": name, "rss_bytes": memory.rss})
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    try:
        gpu, holders = gpu_query()
        rec["gpu_raw"], rec["gpu_holders_raw_information_only"] = gpu, holders
        fields = [field.strip() for field in gpu.split(",")]
        if len(fields) != 4:
            raise ValueError("expected exactly one GPU")
        rec["gpu_free_gib"] = float(fields[2]) / 1024
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        rec["gpu_error"] = str(exc)
    reasons = []
    if rec["host_ram_free_gib"] < PRELOAD_RAM_GIB: reasons.append("RAM below 5 GiB preload reserve")
    if rec["disk_free_gib"] < PRELOAD_DISK_GIB: reasons.append("disk below 6 GiB reserve")
    if rec.get("gpu_error"): reasons.append("GPU query failed")
    elif rec["gpu_free_gib"] < PRELOAD_GPU_GIB: reasons.append("GPU below 6 GiB preload threshold")
    owned = lease_owned(rec["lease"], owner_token, ancestors)
    rec["lease_owned_by_supervisor"] = owned
    if owner_token is not None and not owned:
        reasons.append("shared GPU lease not held by this worker's supervisor")
    if owner_token is None and rec["lease"] is not None:
        reasons.append("shared GPU lease held/unreadable")
    if any(proc["pid"] not in mine and proc["rss_bytes"] >= FOREIGN_RSS_BYTES for proc in rec["research_processes"]):
        reasons.append("foreign research/model process has >=512MiB RSS")
    if rec.get("research_process_visibility_incomplete"):
        reasons.append("research process memory not visible")
    rec["blockers"] = reasons
    rec["status"] = "BLOCKED" if reasons else "RESOURCES_CLEAR"
    return rec


def prompt_system(row, arm):
    return PROMPT_COMPOSITION+row.system if arm == "PROMPT" and "rules" in row.metadata else row.system


def directional_rotation(rotation, arm):
    """Reversing parent order inverts the operator, not the behavioral answer."""
    if arm == "REVERSE":
        return rotation.T
    if arm in ("ROTATION", "RANDOM"):
        return rotation
    return None


def weights_target(arm):
    return "TH" if arm in TH_HOSTED else arm


def screen_schedule(rows):
    """154 generations: every arm covers every row except PROMPT (chain rows only).

    Parent units run first so the registered parent gate can stop before construction.
    The order after the gate minimizes weight reloads (H weights are already loaded).
    """
    parents = tuple(row for row in rows if row.task_kind in ("transition_parent", "answerability_parent"))
    chains = tuple(row for row in rows if "rules" in row.metadata)
    if len(parents) != 8 or len(chains) != 10 or len(rows) != 18:
        raise ValueError("frozen screening allocation changed")
    schedule = [(arm, parents) for arm in ("BASE", "T", "H")]
    schedule += [(arm, chains) for arm in ("H", "T", "BASE")]
    schedule += [(arm, tuple(rows)) for arm in ("TH", "IDENTITY", "ROTATION", "REVERSE", "RANDOM")]
    schedule += [("PROMPT", chains)]
    if sum(len(unit_rows) for _, unit_rows in schedule) != MAX_GENERATIONS:
        raise ValueError("allocation does not match the registered generation count")
    return tuple(schedule)


def native_allocation(rows, development_rows):
    """Upper bounds on forwards and token positions for the whole native run."""
    generations = sum(len(unit_rows) for _, unit_rows in screen_schedule(rows))
    mechanics = development_rows[:2]
    parity_forwards = len(mechanics) * 2 * 4
    parity_positions = len(mechanics) * 2 * (INPUT_SLOTS + 3)
    liveness_forwards = len(mechanics) * (1 + len(LIVENESS_ARMS))
    liveness_positions = liveness_forwards * INPUT_SLOTS
    generation_forwards = generations * OUTPUT_SLOTS
    generation_positions = generations * (INPUT_SLOTS + OUTPUT_SLOTS - 1)
    readout_forwards = len(rows) * len(READOUT_ARMS)
    readout_positions = readout_forwards * READOUT_SLOTS
    forwards = parity_forwards + liveness_forwards + generation_forwards + readout_forwards
    positions = parity_positions + liveness_positions + generation_positions + readout_positions
    return {"generations": generations, "forwards": forwards, "token_positions": positions,
            "parts": {"padding_parity": [parity_forwards, parity_positions],
                      "hook_liveness": [liveness_forwards, liveness_positions],
                      "generation": [generation_forwards, generation_positions],
                      "teacher_forced_readout": [readout_forwards, readout_positions]},
            "fits": generations <= MAX_GENERATIONS and forwards <= MAX_FORWARDS and positions <= MAX_TOKEN_POSITIONS}


class HookState:
    """Mutable state read by the block-14 forward hook."""

    def __init__(self, torch, basis):
        self.torch = torch
        self.basis = basis
        self.change = torch.zeros((basis.shape[1], basis.shape[1]), dtype=basis.dtype, device=basis.device)
        self.valid = None
        self.capture = None
        self.reset()

    def reset(self):
        self.energy = 0.0
        self.displacement = 0.0

    def set(self, basis, change):
        self.basis, self.change = basis, change
        self.reset()


def make_hook(state):
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        coordinates = hidden @ state.basis
        delta = (coordinates @ state.change.T) @ state.basis.T
        if state.capture is not None:
            state.capture.append((hidden.detach().to("cpu").double(), delta.detach().to("cpu").double()))
        valid = state.valid.unsqueeze(-1).to(hidden.dtype)
        state.energy += float((hidden.square() * valid).sum().item())
        state.displacement += float((delta.square() * valid).sum().item())
        changed = hidden + delta
        return (changed, *output[1:]) if isinstance(output, tuple) else changed
    return hook


def _padded(torch, device, ids, pad_id, padded=True, slots=INPUT_SLOTS):
    if len(ids) > slots:
        raise RuntimeError("STOP_INPUT_TOKEN_BUDGET; no truncation")
    pad = slots - len(ids) if padded else 0
    input_ids = torch.tensor([[pad_id] * pad + list(ids)], device=device)
    mask = torch.tensor([[0] * pad + [1] * len(ids)], device=device)
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    return input_ids, mask, positions


def decode(model, torch, device, ids, *, pad_id, eos_ids, state, budget, slots=OUTPUT_SLOTS,
           padded=True, reserve=None):
    """Fixed-slot greedy decode. Text ends at the first token in eos_ids; the remaining
    slots are a compute-only suffix fed eos_ids[0], so every arm spends equal compute."""
    input_ids, mask, positions = _padded(torch, device, ids, pad_id, padded)
    cache, generated, raw, margins, ended = None, [], [], [], False
    for step in range(slots):
        if reserve is not None:
            reserve()
        state.valid = mask if cache is None else torch.ones_like(input_ids)
        budget.forward(int(input_ids.numel()))
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=mask, position_ids=positions,
                           past_key_values=cache, use_cache=True)
        budget.completed_forwards += 1
        cache = output.past_key_values
        logits = output.logits[0, -1]
        token = int(logits.argmax().item())
        top = torch.topk(logits.float(), 2).values
        raw.append(token)
        margins.append(float((top[0] - top[1]).item()))
        if not ended and token not in eos_ids:
            generated.append(token)
        if token in eos_ids:
            ended = True
        input_ids = torch.tensor([[eos_ids[0] if ended else token]], device=device)
        mask = torch.cat((mask, torch.ones((1, 1), dtype=mask.dtype, device=device)), dim=1)
        positions = torch.tensor([[len(ids) + step]], device=device)
    first_eos = next((index for index, token in enumerate(raw) if token in eos_ids), None)
    scored = raw[:first_eos + 1] if first_eos is not None else list(raw)
    return {"tokens": generated, "scored_tokens": scored, "raw_tokens": raw, "margins": margins, "ended": ended}


def prefill_logits(model, torch, device, ids, *, pad_id, state, budget):
    input_ids, mask, positions = _padded(torch, device, ids, pad_id)
    state.valid = mask
    budget.forward(int(input_ids.numel()))
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=mask, position_ids=positions, use_cache=False)
    budget.completed_forwards += 1
    return output.logits[0], mask


def hook_liveness(model, torch, device, rows_ids, *, pad_id, state, budget, basis, random_basis, rotation):
    """Registered mechanics canary on DEVELOPMENT prompts, before any construction read.

    IDENTITY must displace exactly nothing. For ROTATION/REVERSE/RANDOM the displaced
    vectors themselves must match h B (R - I)^T B^T computed in float64 from the registered
    arrays (not from hook state), element-wise; an energy-only check cannot see direction
    errors (R and R^T, or a column-reversed union basis, displace equal energy). Their
    energy must be positive, and ROTATION must change the logits. A hook that never fires
    fails the check rather than raising.
    """
    import numpy as np
    b64, rb64, r64 = (np.asarray(array, dtype=np.float64) for array in (basis, random_basis, rotation))
    torch_basis = torch.as_tensor(b64, dtype=torch.float32, device=device)
    torch_random = torch.as_tensor(rb64, dtype=torch.float32, device=device)
    eye = np.eye(len(r64))

    def captured(arm_state_basis, change):
        state.set(arm_state_basis, change)
        state.capture = []
        logits, mask = prefill_logits(model, torch, device, ids, pad_id=pad_id, state=state, budget=budget)
        record, state.capture = (state.capture[0] if state.capture else None), None
        return logits, mask, record

    records, ok = [], True
    try:
        for ids in rows_ids:
            base_logits, mask, record0 = captured(torch_basis, torch.zeros((len(r64), len(r64)), dtype=torch.float32, device=device))
            if record0 is None:
                records.append({"hook_fired": False})
                ok = False
                continue
            valid = mask[0].to("cpu").numpy().astype(bool)
            row = {"hook_fired": True, "identity_displacement": state.displacement,
                   "identity_delta_nonzero": bool(record0[1].abs().max().item() > 0), "arms": {}}
            ok = ok and state.displacement == 0.0 and not row["identity_delta_nonzero"]
            for arm in LIVENESS_ARMS:
                arm_basis = rb64 if arm == "RANDOM" else b64
                arm_rotation = r64.T if arm == "REVERSE" else r64
                logits, _, record = captured(torch_random if arm == "RANDOM" else torch_basis,
                                             torch.as_tensor(arm_rotation - eye, dtype=torch.float32, device=device))
                if record is None:
                    row["arms"][arm] = {"hook_fired": False, "ok": False}
                    ok = False
                    continue
                hidden, measured_delta = record[0][0].numpy()[valid], record[1][0].numpy()[valid]
                analytic_delta = ((hidden @ arm_basis) @ (arm_rotation - eye).T) @ arm_basis.T
                analytic = float(np.square(analytic_delta).sum())
                scale = float(np.abs(analytic_delta).max())
                vector_error = float(np.abs(measured_delta - analytic_delta).max() / scale) if scale > 0 else float("inf")
                measured = state.displacement
                max_delta = float((logits - base_logits).abs().max().item())
                arm_ok = (analytic > 0 and abs(measured - analytic) <= LIVENESS_RTOL * analytic
                          and vector_error <= LIVENESS_RTOL)
                if arm == "ROTATION":
                    arm_ok = arm_ok and max_delta > 0
                row["arms"][arm] = {"analytic": analytic, "measured": measured, "max_relative_vector_error": vector_error,
                                    "max_abs_logit_change": max_delta, "ok": arm_ok}
                ok = ok and arm_ok
            records.append(row)
    finally:
        state.capture = None
    return {"ok": ok, "rows": records, "rtol": LIVENESS_RTOL}


def teacher_forced_logits(model, torch, device, ids, target, *, pad_id, state, budget):
    """Logits predicting each target token, from one full-sequence forward."""
    if not target:
        return None
    sequence = list(ids) + list(target[:-1])
    input_ids, mask, positions = _padded(torch, device, sequence, pad_id, slots=READOUT_SLOTS)
    state.valid = mask
    budget.forward(int(input_ids.numel()))
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=mask, position_ids=positions, use_cache=False)
    budget.completed_forwards += 1
    start = READOUT_SLOTS - len(sequence) + len(ids) - 1   # last prompt position predicts target[0]
    return output.logits[0, start:start + len(target)]


def readout_metrics(torch, reference, candidate, target):
    """Continuous sensitivity of `candidate` relative to `reference` along TH's tokens."""
    target_index = torch.tensor(target, device=reference.device)
    ref, cand = reference.double(), candidate.double()
    log_p, log_q = torch.log_softmax(ref, -1), torch.log_softmax(cand, -1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1)
    top_ref = torch.topk(ref, 2, dim=-1).values
    def target_margin(logits):
        chosen = logits.gather(1, target_index[:, None])[:, 0]
        masked = logits.clone()
        masked.scatter_(1, target_index[:, None], float("-inf"))
        return chosen - masked.max(-1).values
    return {"steps": len(target),
            "argmax_agrees_with_TH_token": int((cand.argmax(-1) == target_index).sum().item()),
            "max_abs_logit_change": float((cand - ref).abs().max().item()),
            "mean_kl_TH_to_arm": float(kl.mean().item()), "max_kl_TH_to_arm": float(kl.max().item()),
            "TH_min_top2_margin": float((top_ref[:, 0] - top_ref[:, 1]).min().item()),
            "min_target_margin_change": float((target_margin(cand) - target_margin(ref)).min().item())}


def load_arm_weights(model, arm, torch, geometry, budget=None, reserve_bytes=None):
    """Replace only in-memory parameters; H uses the realized compressed update.

    Before each tensor the wall budget is checked and, when reserve_bytes is given,
    host RAM must cover this tensor's float64 working set plus the active reserve.
    """
    target = weights_target(arm)
    files = {name: geometry.TensorFile(path) for name, (path, _) in geometry.SOURCES.items()}
    try:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if budget is not None:
                    budget.check()
                if reserve_bytes is not None:
                    reserve_bytes(param.numel() * 8 * (5 if target == "H" else 2))
                if target == "H":
                    value = files["BASE"].tensor(name) + files["TH"].tensor(name) - files["T"].tensor(name)
                else:
                    value = files[target].tensor(name)
                if tuple(value.shape) != tuple(param.shape):
                    raise ValueError(f"parameter shape mismatch {name}")
                param.copy_(torch.from_numpy(value.astype("float32")).to(param.device))
                del value
    finally:
        for stream in files.values(): stream.close()


def _validate_freeze(path: Path, expected_sha: str):
    _reject_bootstrap_path_substitution(path)
    freeze_source=path.read_bytes()
    if not expected_sha or hashlib.sha256(freeze_source).hexdigest() != expected_sha:
        raise ValueError("supervisor-pinned prospective freeze digest required")
    freeze=json.loads(freeze_source)
    required={"worker.py","geometry.py","benchmark.py","dependency_binding.py","supervise.py","launch_wmi.py",
              GEOMETRY_NPZ,GEOMETRY_JSON,SCREEN_JSON,DEVELOPMENT_JSON,"AMENDMENT_SECTIONH.md",
              "PREREG.md","BENCHMARK_MATRIX.md","FOLLOWUP_PREREG.md","FUTURE_PROTOCOL.md","SOURCES_VERIFIED.md"}
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


def run_screen(out: Path, freeze_path: Path, freeze_sha: str, lease_token: str):
    """Narrow bootstrap: native functions come only from pinned source bytes.

    supervise.py pins this bootstrap's bytes (via the freeze) before executing it.
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
    return worker._run_bound_screen(out,freeze_path,freeze_sha,lease_token,_started=started)


def _run_bound_screen(out: Path, freeze_path: Path, freeze_sha: str, lease_token: str, _started=None):
    """Single model load; all arms share padded greedy fixed-slot inference."""
    module=sys.modules.get(__name__)
    if module is None or getattr(module,"_triskill_source_binding",None) != (str(HERE/"worker.py"),sha256(HERE/"worker.py")):
        raise RuntimeError("STOP_UNBOUND_NATIVE_WORKER")
    budget = Budget()
    if _started is not None: budget.started=_started
    budget.check()
    freeze,bound,geometry = _validate_freeze(freeze_path,freeze_sha)
    bound.verify_worker_execution(module)
    benchmark=bound.load_local("benchmark")
    import numpy as np
    import psutil
    snapshot = preflight(lease_token)
    if snapshot["blockers"]: raise RuntimeError(json.dumps(snapshot))
    if out.exists(): raise FileExistsError("create-only screen output exists")
    geo_receipt = json.loads((HERE / GEOMETRY_JSON).read_text(encoding="utf-8"))
    if geo_receipt["status"] != "PREPARED" or sha256(HERE / GEOMETRY_NPZ) != geo_receipt["geometry_sha256"]:
        raise ValueError("geometry is absent/commuting/unverified")
    for path, expected in geometry.SOURCES.values():
        budget.check()
        if sha256(path) != expected: raise ValueError("source weight digest changed")
    data = np.load(HERE / GEOMETRY_NPZ, allow_pickle=False)
    renderer=bound.load_renderer()
    renderer.template_selftest()  # Pinned original DEV bytes; no protected bank.
    render=renderer.render
    rows, development = benchmark.build_screen(), benchmark.build_development()
    if [row.to_dict() for row in rows] != json.loads((HERE/SCREEN_JSON).read_text(encoding="utf-8")):
        raise ValueError("screen generator differs from frozen membership")
    if [row.to_dict() for row in development] != json.loads((HERE/DEVELOPMENT_JSON).read_text(encoding="utf-8")):
        raise ValueError("development generator differs from frozen membership")
    allocation = native_allocation(rows, development)
    if not allocation["fits"]: raise ValueError("registered allocation exceeds the budget")
    by_id = {row.row_id: row for row in rows}

    def host_reserve(extra_bytes=0):
        if (psutil.virtual_memory().available - extra_bytes) / 2**30 < ACTIVE_RAM_GIB:
            raise RuntimeError("STOP_HOST_ACTIVE_RESERVE")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    result = {"experiment": "TRISKILL-GEO-001-SECTIONH", "status": "STARTED", "confirmation": False,
              "study_mode": "mechanical_exploratory", "natural_causal_mechanism_claim": False,
              "construction_uses_C_labels": False, "direction_selection_adapted_from_C": False,
              "quality_claim": False, "training_examples": 0, "model_loads": 0, "weight_loads": 0,
              "resources_before": snapshot, "arms": {}, "allocation": allocation,
              "freeze_sha256": sha256(freeze_path), "geometry_sha256": geo_receipt["geometry_sha256"],
              "dependency_binding": bound.receipt(),
              "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                          "dtype": "float32", "attention": "eager", "fixed_input_slots": INPUT_SLOTS,
                          "fixed_output_slots": OUTPUT_SLOTS, "eos_token_ids": list(EOS_TOKEN_IDS),
                          "greedy": True, "tf32": False, "gpu_memory_fraction": GPU_MEMORY_FRACTION}}
    outputs, mechanics = {}, {}
    try:
        budget.check()
        resources = preflight(lease_token)
        result["resources_immediately_before_load"] = resources
        if resources["blockers"]: raise RuntimeError(json.dumps(resources))
        torch.cuda.init()
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
        if torch.cuda.mem_get_info()[0] / 2**30 < PRELOAD_GPU_GIB: raise RuntimeError("GPU changed before load")

        def reserve():
            if torch.cuda.mem_get_info()[0] / 2**30 < ACTIVE_GPU_GIB: raise RuntimeError("STOP_GPU_ACTIVE_RESERVE")
            host_reserve()

        result["model_load_attempts"] = 1
        bound.verify_current()  # last boundary before the native load
        model = AutoModelForCausalLM.from_pretrained(str(geometry.SOURCES["BASE"][0].parent),
                    torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True,
                    low_cpu_mem_usage=True, device_map={"": "cuda"}).eval()
        result["model_loads"] = 1
        budget.check()
        tok = AutoTokenizer.from_pretrained(str(geometry.SOURCES["BASE"][0].parent), local_files_only=True)
        configured = model.generation_config.eos_token_id
        configured = set(configured if isinstance(configured, list) else [configured])
        if configured != set(EOS_TOKEN_IDS) or tok.eos_token_id != EOS_TOKEN_IDS[0]:
            raise RuntimeError("STOP_EOS_CONFIGURATION_CHANGED")
        pad_id = EOS_TOKEN_IDS[0]
        device = next(model.parameters()).device
        b = torch.as_tensor(data["basis"], dtype=torch.float32, device=device)
        random_b = torch.as_tensor(data["random_basis"], dtype=torch.float32, device=device)
        r = torch.as_tensor(data["rotation"], dtype=torch.float32, device=device)
        eye = torch.eye(len(r), device=device)
        state = HookState(torch, b)
        handle = model.model.layers[geometry.LAYER].register_forward_hook(make_hook(state))

        def ids_for(system, user):
            return tok(render(system, user), add_special_tokens=False)["input_ids"]

        def set_arm(arm):
            selected = directional_rotation(r, arm)
            state.set(random_b if arm == "RANDOM" else b,
                      selected - eye if selected is not None else torch.zeros_like(r))

        loaded = {"target": "BASE"}

        def ensure_weights(arm):
            if loaded["target"] != weights_target(arm):
                load_arm_weights(model, arm, torch, geometry, budget, host_reserve)
                loaded["target"] = weights_target(arm)
                result["weight_loads"] += 1

        try:
            mechanics_ids = [ids_for(row.system, row.user) for row in development[:2]]
            set_arm("IDENTITY")
            parity = []
            for ids in mechanics_ids:
                unpadded = decode(model, torch, device, ids, pad_id=pad_id, eos_ids=EOS_TOKEN_IDS, state=state,
                                  budget=budget, slots=4, padded=False, reserve=reserve)["raw_tokens"]
                padded = decode(model, torch, device, ids, pad_id=pad_id, eos_ids=EOS_TOKEN_IDS, state=state,
                                budget=budget, slots=4, padded=True, reserve=reserve)["raw_tokens"]
                parity.append(unpadded == padded)
            result["development_padding_parity"] = parity
            mechanics["padding_parity_ok"] = all(parity)
            if not all(parity): raise RuntimeError("STOP_NATIVE_MECHANICS_PADDING")
            schedule = screen_schedule(rows)
            result["screen_allocation"] = [{"arm": arm, "row_ids": [row.row_id for row in unit_rows]}
                                           for arm, unit_rows in schedule]
            for unit_index, (arm, unit_rows) in enumerate(schedule):
                budget.check()
                ensure_weights(arm)
                if arm == "TH" and "liveness_ok" not in mechanics:
                    liveness = hook_liveness(model, torch, device, mechanics_ids, pad_id=pad_id, state=state,
                                             budget=budget, basis=data["basis"], random_basis=data["random_basis"],
                                             rotation=data["rotation"])
                    result["hook_liveness"] = liveness
                    mechanics["liveness_ok"] = liveness["ok"]
                    if not liveness["ok"]: raise RuntimeError("STOP_MECHANICS_HOOK_LIVENESS")
                set_arm(arm)
                prior = result["arms"].get(arm, {})
                arm_outputs = dict(prior.get("outputs", {}))
                tokens, margins = dict(prior.get("scored_tokens", {})), dict(prior.get("margins", {}))
                for row in unit_rows:
                    budget.generation()
                    decoded = decode(model, torch, device, ids_for(prompt_system(row, arm), row.user), pad_id=pad_id,
                                     eos_ids=EOS_TOKEN_IDS, state=state, budget=budget, reserve=reserve)
                    arm_outputs[row.row_id] = tok.decode(decoded["tokens"], skip_special_tokens=True)
                    tokens[row.row_id], margins[row.row_id] = decoded["scored_tokens"], decoded["margins"]
                    budget.completed_generations += 1
                result["arms"][arm] = {
                    "outputs": arm_outputs,
                    "scores": {row_id: benchmark.score_response(by_id[row_id], text) for row_id, text in arm_outputs.items()},
                    "scored_tokens": tokens, "margins": margins,
                    "activation_energy": state.energy + prior.get("activation_energy", 0.),
                    "activation_displacement_energy": state.displacement + prior.get("activation_displacement_energy", 0.)}
                result["arms"][arm]["relative_activation_displacement"] = (
                    result["arms"][arm]["activation_displacement_energy"] / max(result["arms"][arm]["activation_energy"], 1e-30)) ** .5
                outputs[arm] = arm_outputs
                if unit_index == 2:
                    # Do not attempt construction if its source manifestations do not transport.
                    gate = benchmark.parent_gate(rows, {name: result["arms"][name]["scores"] for name in ("BASE", "T", "H")})
                    result["parent_gate"] = gate
                    if not gate["passed"]: raise RuntimeError(gate["stop"])
                if arm == "IDENTITY" and any(text != outputs["TH"].get(row_id) for row_id, text in arm_outputs.items()):
                    raise RuntimeError("STOP_IDENTITY_CANARY")
            # Continuous sensitivity along TH's own greedy tokens (TH weights still loaded).
            readout = {}
            for row in rows:
                target = result["arms"]["TH"]["scored_tokens"][row.row_id]
                ids = ids_for(prompt_system(row, "TH"), row.user)
                logits = {}
                for arm in READOUT_ARMS:
                    reserve()
                    set_arm(arm)
                    logits[arm] = teacher_forced_logits(model, torch, device, ids, target, pad_id=pad_id,
                                                        state=state, budget=budget)
                readout[row.row_id] = None if logits["TH"] is None else {
                    arm: readout_metrics(torch, logits["TH"], logits[arm], target) for arm in READOUT_ARMS}
                del logits
            result["teacher_forced_readout"] = readout
            result["completed_schedule"] = True
        except Exception as exc:
            result["status"] = str(exc)
        finally:
            handle.remove()
    except Exception as exc:
        result["status"] = str(exc)
    finally:
        gc.collect()
        try:
            if torch.cuda.is_initialized():
                result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
                result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception as cleanup_error:
            result["cleanup_error"] = str(cleanup_error)
        mechanics.setdefault("padding_parity_ok", None)
        mechanics.setdefault("liveness_ok", None)
        verdict = benchmark.decide_screen(rows, outputs, mechanics)
        result["verdict"] = verdict
        if result.get("completed_schedule"):
            result["status"] = verdict["status"]
        result["budget"] = budget.receipt()
        with out.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight","run"), default="preflight", nargs="?")
    parser.add_argument("--out",type=Path,default=HERE/"native_screen_sectionh.json")
    parser.add_argument("--freeze",type=Path,default=HERE/FREEZE_NAME)
    parser.add_argument("--freeze-sha256")
    parser.add_argument("--lease-token")
    args=parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(),indent=2))
    elif not args.lease_token or not args.freeze_sha256:
        parser.error("run requires --freeze-sha256 and the supervisor's --lease-token; launch via supervise.py")
    else:
        print(json.dumps(run_screen(args.out,args.freeze,args.freeze_sha256,args.lease_token),indent=2))
