"""PARENTQ native worker; launched only by parentq_supervise.py.

It measures the GEO screen's parents (BASE, T, H) on fresh qualification rows, before the
one-shot GEO screen is spent. It runs no geometric arm, generates on no J or K row and
opens no protected bank (PREREG_PARENTQ.md).

Native internals come from the FROZEN GEO section H code, never from copies. The worker
binds GEO through GEO's own dependency binding and validates the whole GEO freeze
(FREEZE_SECTIONH.json, registered digest), including its CPU validation receipt, source
sidecars, weight digests and runtime packages. It then uses GEO's resource preflight, model
load call, block-14 hook (held at IDENTITY, as in the screen's parent units), padded input
layout, padding-parity procedure, in-memory arm loading and scorer. Its own code is the
schedule, the budget caps, the lease rule for this lane's label, and `decode_until_eos`.
That function is GEO's fixed-slot greedy decode stopped at the first EOS, and a CPU test
shows it returns GEO's tokens exactly.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
GEO_DIR = HERE.parent / "triskill_geometric_pilot_v1"
GEO_FREEZE_NAME = "FREEZE_SECTIONH.json"
GEO_FREEZE_SHA256 = "f18d48670e8928afb72b7182756f8f7bb6c8e241fb5988ae1daa91ebc534b687"
LEASE_LABEL = "triskill-geo001-parentq"
FREEZE_NAME = "FREEZE_PARENTQ.json"
BANK_JSON, CANARY_JSON = "parentq_bank.json", "parentq_canary.json"
START_NAME = "SUPERVISOR_START_PARENTQ.json"
WORKER_PID_NAME = "WORKER_PID_PARENTQ.json"
PROGRESS_NAME = "native_progress_parentq.jsonl"
BLOCKED_NAME = "blocked_result_parentq.json"
INCOMPLETE_NAME = "incomplete_result_parentq.json"
OUT_NAME = "native_parentq.json"
MIN_GRANT_CHARS = 12
WALL_SECONDS = 540                 # internal; parentq_supervise.py enforces the 600 s hard wall
PARITY_SLOTS = 4
REQUIRED_FILES = ("parentq_worker.py", "parentq_rows.py", "parentq_decide.py", "parentq_supervise.py",
                  BANK_JSON, CANARY_JSON, "PREREG_PARENTQ.md")
_NO_LOCK = HERE / "__resources_only_no_lock__"   # GEO preflight measures resources; the lease is checked here
# The schedule: each arm's canary first, then its qualification blocks; weight loads BASE -> H -> T.
SCHEDULE = (("BASE", "CANARY"), ("BASE", "T"), ("BASE", "H_ON"), ("BASE", "H_OFF"),
            ("H", "CANARY"), ("H", "H_ON"), ("H", "H_OFF"),
            ("T", "CANARY"), ("T", "T"))
TERMINAL_PREFIXES = ("PARENTS_", "STOP_ARM_CANARY:", "STOP_NATIVE_MECHANICS_PADDING")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _exec_pinned(path: Path, expected: str, name: str):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError(f"STOP_PINNED_SOURCE_MISMATCH:{path.name}")
    spec = importlib.util.spec_from_file_location(f"_parentq_{name}_{expected[:12]}", path)
    module = importlib.util.module_from_spec(spec)
    module._parentq_source_binding = (str(path), expected)
    sys.modules[spec.name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def bind_geo(geo_dir: Path = GEO_DIR, geo_freeze_sha: str = GEO_FREEZE_SHA256):
    """The frozen GEO worker, then GEO's own full freeze validation through it."""
    freeze_path = geo_dir / GEO_FREEZE_NAME
    geo_freeze = json.loads(freeze_path.read_bytes())
    if sha256(freeze_path) != geo_freeze_sha:
        raise ValueError("STOP_GEO_FREEZE_DIGEST_MISMATCH")
    entry = geo_freeze["dependency_binding"]["roles"]["binding"]
    binding = _exec_pinned(geo_dir / "dependency_binding.py", entry["sha256"], "geo_binding")
    geo = binding.validate_dependencies(geo_freeze["dependency_binding"]).load_worker()
    _, bound, geometry = geo._validate_freeze(freeze_path, geo_freeze_sha)
    return geo, bound, geometry


def allocation(n_generations: int, input_slots: int, output_slots: int) -> dict:
    """Upper bounds as if every generation used all output slots (decoding stops at EOS)."""
    parity_forwards, parity_positions = 2 * 2 * PARITY_SLOTS, 2 * 2 * (input_slots + PARITY_SLOTS - 1)
    forwards = parity_forwards + n_generations * output_slots
    positions = parity_positions + n_generations * (input_slots + output_slots - 1)
    return {"generations": n_generations, "forwards": forwards, "token_positions": positions,
            "parts": {"padding_parity": [parity_forwards, parity_positions],
                      "generation": [n_generations * output_slots, n_generations * (input_slots + output_slots - 1)]}}


def schedule_units(bank: dict, canary: list) -> list:
    blocks = {"CANARY": [row["row_id"] for row in canary], **{name: [row.row_id for row in rows]
                                                              for name, rows in bank.items()}}
    return [(arm, block, blocks[block]) for arm, block in SCHEDULE]


class Budget:
    def __init__(self, caps: dict, now=time.monotonic):
        self.caps, self.now = caps, now
        self.started = now()
        self.forwards = self.completed_forwards = self.token_positions = 0
        self.generations = self.completed_generations = 0

    def check(self):
        if self.now() - self.started >= WALL_SECONDS:
            raise RuntimeError("STOP_WALL_BUDGET")

    def forward(self, positions: int):
        self.check()
        if self.forwards + 1 > self.caps["forwards"] or self.token_positions + positions > self.caps["token_positions"]:
            raise RuntimeError("STOP_FORWARD_OR_TOKEN_BUDGET")
        self.forwards += 1
        self.token_positions += positions

    def generation(self):
        self.check()
        if self.generations >= self.caps["generations"]:
            raise RuntimeError("STOP_GENERATION_BUDGET")
        self.generations += 1

    def receipt(self):
        return {"model_forwards": self.forwards, "token_positions_including_padding": self.token_positions,
                "completed_model_forwards": self.completed_forwards, "generations_requested": self.generations,
                "completed_generations": self.completed_generations, "elapsed_seconds": self.now() - self.started}


def decode_until_eos(geo, model, torch, device, ids, *, pad_id, eos_ids, state, budget, reserve=None):
    """GEO's `decode` (padded, fixed input slots, greedy) stopped at the first EOS.

    Every forward up to and including the EOS step is the same call GEO's fixed-slot
    decode makes, so the text tokens are identical; GEO's compute-only suffix is skipped.
    """
    input_ids, mask, positions = geo._padded(torch, device, ids, pad_id, True)
    cache, generated, raw, margins = None, [], [], []
    for step in range(geo.OUTPUT_SLOTS):
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
        if token in eos_ids:
            break
        generated.append(token)
        input_ids = torch.tensor([[token]], device=device)
        mask = torch.cat((mask, torch.ones((1, 1), dtype=mask.dtype, device=device)), dim=1)
        positions = torch.tensor([[len(ids) + step]], device=device)
    return {"tokens": generated, "scored_tokens": raw, "margins": margins, "ended": bool(raw) and raw[-1] in eos_ids}


def lease_owned(lease, token, ancestors) -> bool:
    return (isinstance(lease, dict) and token is not None and lease.get("token") == token
            and lease.get("pid") in ancestors and lease.get("label") == LEASE_LABEL)


def preflight(geo, token, *, lock=None, **kwargs) -> dict:
    """GEO's measured-resource preflight (its thresholds unchanged), plus this lane's lease rule."""
    if _NO_LOCK.exists():
        raise RuntimeError("STOP_PREFLIGHT_SENTINEL_EXISTS")
    lock = geo.LOCK if lock is None else lock
    ancestors = set(kwargs.pop("ancestors", None) or geo.ancestor_pids())
    record = geo.preflight(None, lock=_NO_LOCK, ancestors=ancestors, **kwargs)
    lease = None
    if Path(lock).exists():
        try:
            lease = json.loads(Path(lock).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            lease = "unreadable; fail closed"
    record["lease"] = lease
    record["lease_owned_by_supervisor"] = owned = lease_owned(lease, token, ancestors)
    if not owned:
        record["blockers"].append("shared GPU lease not held by this worker's supervisor")
    record["status"] = "BLOCKED" if record["blockers"] else "RESOURCES_CLEAR"
    return record


def verify_supervisor_start(attempt_dir: Path, lease_token: str) -> dict:
    start_path = Path(attempt_dir) / START_NAME
    if not start_path.exists():
        raise RuntimeError("STOP_NO_SUPERVISOR_START_RECEIPT")
    start = json.loads(start_path.read_text(encoding="utf-8"))
    grant = start.get("operator_grant")
    if start.get("lease_token") != lease_token or not isinstance(grant, str) or len(grant.strip()) < MIN_GRANT_CHARS:
        raise RuntimeError("STOP_NO_SUPERVISOR_GRANT")
    return start


def validate_freeze(freeze_path: Path, freeze_sha: str, here: Path = HERE) -> dict:
    data = freeze_path.read_bytes()
    if not freeze_sha or hashlib.sha256(data).hexdigest() != freeze_sha:
        raise ValueError("STOP_FREEZE_DIGEST_MISMATCH")
    freeze = json.loads(data)
    files = freeze.get("files", {})
    if not set(REQUIRED_FILES) <= set(files):
        raise ValueError("STOP_INCOMPLETE_FREEZE_MEMBERSHIP")
    for name, expected in files.items():
        if Path(name).name != name or sha256(here / name) != expected:
            raise ValueError(f"STOP_FREEZE_FILE_MISMATCH:{name}")
    if freeze.get("geo_freeze", {}).get("sha256") != GEO_FREEZE_SHA256:
        raise ValueError("STOP_GEO_FREEZE_NOT_REGISTERED")
    validation = freeze.get("cpu_validation", {})
    receipt = here / str(validation.get("file", ""))
    if validation.get("passed") is not True or receipt.parent != here or not receipt.is_file() or sha256(receipt) != validation.get("sha256"):
        raise ValueError("STOP_NO_PASSING_PINNED_CPU_VALIDATION")
    return freeze


def _write_json(path: Path, payload) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def run(out: Path, freeze_path: Path, freeze_sha: str, lease_token: str, attempt_dir: Path):
    """Bootstrap: execute this worker's own pinned bytes, then run from that module."""
    started = time.monotonic()
    freeze = validate_freeze(freeze_path, freeze_sha)
    module = _exec_pinned(HERE / "parentq_worker.py", freeze["files"]["parentq_worker.py"], "worker")
    return module._run_bound(out, freeze_path, freeze_sha, lease_token, attempt_dir, _started=started)


def _run_bound(out: Path, freeze_path: Path, freeze_sha: str, lease_token: str, attempt_dir: Path, _started=None,
               _bind_geo=bind_geo):
    """One model load; BASE, H and T in that weight order; per-generation checkpoints.

    Output policy (PREREG_PARENTQ.md section 7): an attempt that never loads the model
    writes BLOCKED_NAME in its attempt directory. A loaded attempt with a terminal verdict
    writes the create-only OUT_NAME; a loaded attempt that stops short writes
    INCOMPLETE_NAME in its attempt directory, and a later granted attempt may rerun it.
    """
    if getattr(sys.modules.get(__name__), "_parentq_source_binding", None) != (str(HERE / "parentq_worker.py"), sha256(HERE / "parentq_worker.py")):
        raise RuntimeError("STOP_UNBOUND_PARENTQ_WORKER")
    attempt_dir = Path(attempt_dir)
    freeze = validate_freeze(freeze_path, freeze_sha)
    rows_module = _exec_pinned(HERE / "parentq_rows.py", freeze["files"]["parentq_rows.py"], "rows")
    decide = _exec_pinned(HERE / "parentq_decide.py", freeze["files"]["parentq_decide.py"], "decide")
    geo, bound, geometry = _bind_geo()
    benchmark = bound.load_local("benchmark")
    bank = rows_module.build_bank(benchmark)
    if [row.to_dict() for row in rows_module.all_rows(bank)] != json.loads((HERE / BANK_JSON).read_text(encoding="utf-8")):
        raise ValueError("STOP_BANK_GENERATOR_DIFFERS_FROM_FROZEN_MEMBERSHIP")
    canary = json.loads((HERE / CANARY_JSON).read_text(encoding="utf-8"))["rows"]
    units = schedule_units(bank, canary)
    caps = allocation(sum(len(ids) for _, _, ids in units), geo.INPUT_SLOTS, geo.OUTPUT_SLOTS)
    if caps != freeze["native_allocation"]:
        raise ValueError("STOP_ALLOCATION_CHANGED")
    budget = Budget(caps)
    if _started is not None:
        budget.started = _started
    start_receipt = verify_supervisor_start(attempt_dir, lease_token)
    import numpy as np
    import psutil
    _write_json(attempt_dir / WORKER_PID_NAME, {"pid": os.getpid(), "ancestors": sorted(geo.ancestor_pids())})
    snapshot = preflight(geo, lease_token)
    if snapshot["blockers"]:
        raise RuntimeError(json.dumps(snapshot))
    supervisor_pid = snapshot["lease"]["pid"]
    if out.exists():
        raise FileExistsError("create-only PARENTQ output exists")
    geo_receipt = json.loads((GEO_DIR / geo.GEOMETRY_JSON).read_text(encoding="utf-8"))
    if geo_receipt["status"] != "PREPARED" or sha256(GEO_DIR / geo.GEOMETRY_NPZ) != geo_receipt["geometry_sha256"]:
        raise ValueError("STOP_GEO_GEOMETRY_UNVERIFIED")
    for path, expected in geometry.SOURCES.values():
        budget.check()
        if sha256(path) != expected:
            raise ValueError("STOP_SOURCE_WEIGHT_DIGEST_CHANGED")
    with np.load(GEO_DIR / geo.GEOMETRY_NPZ, allow_pickle=False) as archive:
        basis = archive["basis"]
    renderer = bound.load_renderer()
    renderer.template_selftest()
    rows_by_id = {row.row_id: row for row in rows_module.all_rows(bank)}
    canary_by_id = {row["row_id"]: row for row in canary}
    development = benchmark.build_development()

    def supervisor_alive():
        if not psutil.pid_exists(supervisor_pid):
            raise RuntimeError("STOP_SUPERVISOR_LOST")

    def host_reserve(extra_bytes=0):
        supervisor_alive()
        if (psutil.virtual_memory().available - extra_bytes) / 2**30 < geo.ACTIVE_RAM_GIB:
            raise RuntimeError("STOP_HOST_ACTIVE_RESERVE")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    result = {"experiment": "TRISKILL-GEO-001-PARENTQ", "status": "STARTED", "confirmation": False,
              "geometric_arms_run": 0, "j_or_k_rows_generated": 0, "protected_banks_opened": [],
              "training_examples": 0, "model_loads": 0, "weight_loads": 0, "attempt_dir": str(attempt_dir),
              "operator_grant": start_receipt["operator_grant"], "worker_pid": os.getpid(),
              "supervisor_pid": supervisor_pid, "resources_before": snapshot, "allocation": caps,
              "freeze_sha256": freeze_sha, "geo_freeze_sha256": GEO_FREEZE_SHA256, "outputs": {}, "records": {},
              "dependency_binding": bound.receipt(),
              "runtime": {"torch": torch.__version__, "transformers": transformers.__version__, "dtype": "float32",
                          "attention": "eager", "fixed_input_slots": geo.INPUT_SLOTS,
                          "max_output_slots": geo.OUTPUT_SLOTS, "decoding": "greedy, stops at first EOS",
                          "eos_token_ids": list(geo.EOS_TOKEN_IDS), "tf32": False,
                          "gpu_memory_fraction": geo.GPU_MEMORY_FRACTION, "hook": "block 14 at IDENTITY"}}
    outputs, mechanics = result["outputs"], {}
    progress = (attempt_dir / PROGRESS_NAME).open("x", encoding="utf-8", newline="\n")

    def record_progress(entry):
        progress.write(json.dumps(entry) + "\n")
        progress.flush()

    try:
        budget.check()
        resources = preflight(geo, lease_token)
        result["resources_immediately_before_load"] = resources
        if resources["blockers"]:
            raise RuntimeError(json.dumps(resources))
        torch.cuda.init()
        torch.cuda.set_per_process_memory_fraction(geo.GPU_MEMORY_FRACTION, 0)
        if torch.cuda.mem_get_info()[0] / 2**30 < geo.PRELOAD_GPU_GIB:
            raise RuntimeError("STOP_GPU_CHANGED_BEFORE_LOAD")

        def reserve():
            if torch.cuda.mem_get_info()[0] / 2**30 < geo.ACTIVE_GPU_GIB:
                raise RuntimeError("STOP_GPU_ACTIVE_RESERVE")
            host_reserve()

        result["model_load_attempts"] = 1
        bound.verify_current()
        model = AutoModelForCausalLM.from_pretrained(str(geometry.SOURCES["BASE"][0].parent),
                    torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True,
                    low_cpu_mem_usage=True, device_map={"": "cuda"}).eval()
        result["model_loads"] = 1
        record_progress({"event": "model_loaded"})
        tok = AutoTokenizer.from_pretrained(str(geometry.SOURCES["BASE"][0].parent), local_files_only=True)
        configured = model.generation_config.eos_token_id
        configured = set(configured if isinstance(configured, list) else [configured])
        if configured != set(geo.EOS_TOKEN_IDS) or tok.eos_token_id != geo.EOS_TOKEN_IDS[0]:
            raise RuntimeError("STOP_EOS_CONFIGURATION_CHANGED")
        pad_id = geo.EOS_TOKEN_IDS[0]
        device = next(model.parameters()).device
        b = torch.as_tensor(basis, dtype=torch.float32, device=device)
        state = geo.HookState(torch, b)   # change = 0: IDENTITY, as in the screen's parent units
        handle = model.model.layers[geometry.LAYER].register_forward_hook(geo.make_hook(state))
        loaded = {"target": "BASE"}

        def ids_for(system, user):
            return tok(renderer.render(system, user), add_special_tokens=False)["input_ids"]

        try:
            parity = []
            for row in development[:2]:
                ids = ids_for(row.system, row.user)
                unpadded = geo.decode(model, torch, device, ids, pad_id=pad_id, eos_ids=geo.EOS_TOKEN_IDS, state=state,
                                      budget=budget, slots=PARITY_SLOTS, padded=False, reserve=reserve)["raw_tokens"]
                padded = geo.decode(model, torch, device, ids, pad_id=pad_id, eos_ids=geo.EOS_TOKEN_IDS, state=state,
                                    budget=budget, slots=PARITY_SLOTS, padded=True, reserve=reserve)["raw_tokens"]
                parity.append(unpadded == padded)
            result["development_padding_parity"] = parity
            mechanics["padding_parity_ok"] = all(parity)
            record_progress({"event": "padding_parity", "ok": all(parity)})
            if not all(parity):
                raise RuntimeError("STOP_NATIVE_MECHANICS_PADDING")
            for arm, block, row_ids in units:
                budget.check()
                supervisor_alive()
                if loaded["target"] != arm:
                    geo.load_arm_weights(model, arm, torch, geometry, budget, host_reserve)
                    loaded["target"] = arm
                    result["weight_loads"] += 1
                    record_progress({"event": "weights_loaded", "arm": arm})
                if state.change.abs().max().item() != 0:
                    raise RuntimeError("STOP_HOOK_NOT_IDENTITY")
                texts = outputs.setdefault(arm, {})
                records = result["records"].setdefault(arm, {})
                for row_id in row_ids:
                    budget.generation()
                    if block == "CANARY":
                        system, user = canary_by_id[row_id]["system"], canary_by_id[row_id]["user"]
                    else:
                        system, user = rows_by_id[row_id].system, rows_by_id[row_id].user
                    decoded = decode_until_eos(geo, model, torch, device, ids_for(system, user), pad_id=pad_id,
                                               eos_ids=geo.EOS_TOKEN_IDS, state=state, budget=budget, reserve=reserve)
                    text = tok.decode(decoded["tokens"], skip_special_tokens=True)
                    texts[row_id] = text
                    records[row_id] = {"scored_tokens": decoded["scored_tokens"], "margins": decoded["margins"],
                                       "ended": decoded["ended"]}
                    budget.completed_generations += 1
                    record_progress({"event": "generation", "arm": arm, "block": block, "row_id": row_id,
                                     "text": text, "scored_tokens": decoded["scored_tokens"],
                                     "margins": decoded["margins"]})
                if block == "CANARY":
                    check = decide.canary_check(canary, {arm: texts})[arm]
                    record_progress({"event": "canary", "arm": arm, **check})
                    if not check["passed"]:
                        raise RuntimeError(f"STOP_ARM_CANARY:{arm}")
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
        completed = bool(result.get("completed_schedule"))
        verdict = decide.decide(benchmark, bank, canary, outputs, mechanics,
                                stopped_by=None if completed else result["status"])
        result["verdict"] = verdict
        if completed:
            result["status"] = verdict["status"]
        result["budget"] = budget.receipt()
        result["utc_end"] = dt.datetime.now(dt.timezone.utc).isoformat()
        progress.close()
        if result["model_loads"] < 1:
            target = attempt_dir / BLOCKED_NAME
        elif verdict["status"].startswith(TERMINAL_PREFIXES):
            target = out
        else:
            target = attempt_dir / INCOMPLETE_NAME
        _write_json(target, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--out", type=Path, default=HERE / OUT_NAME)
    parser.add_argument("--freeze", type=Path, default=HERE / FREEZE_NAME)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--lease-token", required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.freeze, args.freeze_sha256, args.lease_token, args.attempt_dir), indent=2))
