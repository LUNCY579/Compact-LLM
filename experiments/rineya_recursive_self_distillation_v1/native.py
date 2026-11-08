"""Explicit, Windows-supervised exact-Titan host path. Never invoked on import.

E0 is always required. The optional ordinary-SFT parent stage creates a candidate;
an operator-reviewed, hash-bound SFT parent is required by every RSD treatment. Native qualification uses three disposable updates and
restores the parent. It cannot authorize an efficacy claim.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .contracts import BASE_SHA256, Plan, digest, file_hash, read_json, require
from .data import load_catalog, require_training_roles, validate_catalog, verify
from .evidence import write_new

PACKAGE = "experiments.rineya_recursive_self_distillation_v1"
CENTRAL = "experiments/post_training_mimo_centralized_v1"
PACKAGES = {"torch": "2.10.0+cu126", "transformers": "4.57.1",
            "safetensors": "0.7.0", "tokenizers": "0.22.2"}


def source_identity(root: Path):
    own = Path(__file__).parent
    paths = list(own.glob("*.py"))
    paths += [root / CENTRAL / "scripts/qualify_native.py",
              root / CENTRAL / "scripts/supervise_native.py",
              root / "experiments/maxrl_comp002_pretraining_v1/mechanics.py",
              root / "experiments/opallo_native_collector_v1/supervisor.py",
              root / "experiments/opallo_native_collector_v1/contract.py"]
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(paths)}


def checked_file(spec, base: Path):
    require(type(spec) is dict and set(spec) == {"path", "sha256"}, "file binding fields")
    p = Path(spec["path"])
    p = p if p.is_absolute() else base / p
    p = p.resolve()
    require(file_hash(p) == spec["sha256"], "bound file drift: " + str(p))
    return p


def inputs(binding_path: Path, plan: Plan, root: Path, *, parent_stage=False):
    binding_path = binding_path.resolve()
    cfg = read_json(binding_path)
    require(set(cfg) == {"schema", "e0_config", "e0_result", "parent", "catalog", "exclusions", "catalog_review"}
            and cfg["schema"] == 2, "native binding schema2 with reviewed catalog required")
    require(not parent_stage or cfg["parent"] is None, "parent stage starts with zero LoRA, not a hidden warm start")
    require(parent_stage or cfg["parent"] is not None, "RSD treatment needs reviewed SFT parent")
    files = {k: checked_file(cfg[k], binding_path.parent) for k in cfg if k != "schema" and cfg[k] is not None}
    e0 = read_json(files["e0_config"])
    from .e0_qualification import validate_e0_chain
    e0_evidence = validate_e0_chain(files["e0_config"], files["e0_result"], root)
    adapter = source_receipt = None
    if not parent_stage:
        from .parent_qualification import validate_parent
        adapter, source_receipt, _ = validate_parent(files["parent"])
    tasks, _ = load_catalog(files["catalog"])
    audit = validate_catalog(tasks, read_json(files["exclusions"]))
    require_training_roles(tasks)
    from .catalog_qualification import validate_catalog_review
    catalog_review = validate_catalog_review(files["catalog"], files["exclusions"], files["catalog_review"])
    require(len({t.parent for t in tasks if t.split == "train_selection"}) >= 8,
            "need at least 8 independent selection parents for bounded qualification")
    sources = source_identity(root)
    identity = {"plan": plan.fingerprint, "catalog": digest([asdict(t) for t in tasks]),
                "files": {k: file_hash(p) for k, p in files.items()},
                "adapter": file_hash(adapter) if adapter else "ZERO_LORA_ON_EXACT_TITAN",
                "parent_evidence": file_hash(source_receipt) if source_receipt else "parent_stage",
                "source": digest(sources), "sources": sources, "base": BASE_SHA256,
                "e0_evidence": e0_evidence, "catalog_review": catalog_review,
                "evidence": "native_exact_titan", "semantic_decontamination": False}
    return e0, tasks, adapter, identity, audit


def launch(args):
    """A cooperative GPU lock and independent process-tree watchdog surround all work."""
    require(os.name == "nt", "registered native path is Windows-only; other hosts require qualification")
    require(args.allow_model_load, "explicit --allow-model-load required")
    if args.mode in ("parent", "qualify", "train"):
        require(args.allow_training, "--allow-training required, including disposable qualification updates")
    require(args.mode in ("parent", "train") or not args.resume, "only training can resume")
    require(args.mode != "parent" or args.arm == "sft", "parent stage must explicitly select --arm sft")
    plan = Plan.load(args.plan)
    root = Path(__file__).resolve().parents[2]
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=root, text=True).strip(), "commit source and config before native work")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    # Read-only checks run before a worker, torch import, or weight allocation.
    _, _, _, identity, _ = inputs(Path(args.binding), plan, root, parent_stage=args.mode == "parent")
    out = Path(args.output).resolve()
    if args.resume:
        require(out.is_dir() and not (out / "INFLIGHT.json").exists(), "unsafe/missing resume directory")
    else:
        require(not out.exists(), "output exists; use a new directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    supervision = out.parent / (out.name + "-supervisor-" + uuid.uuid4().hex)
    common = Path(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=root, text=True).strip())
    common = (root / common).resolve() if not common.is_absolute() else common
    lock = common / "rineya-central-gpu0.lock"
    with lock.open("x", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    try:
        import psutil
        for _ in range(2):
            require(psutil.virtual_memory().available / 2**30 >= plan.host_free_gib, "HOST_RAM_ADMISSION")
            time.sleep(.2)
        require(psutil.disk_usage(str(out.parent)).free / 2**30 >= plan.disk_free_gib, "DISK_ADMISSION")
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name",
                         "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
        require(not any(any(term in line.lower() for term in
                    ("python", "torch", "llama", "ollama", "vllm", "train", "comfy"))
                    for line in active.splitlines()), "competing GPU runtime")
        supervision.mkdir()
        payload = {"mode": args.mode, "plan": str(Path(args.plan).resolve()),
                   "binding": str(Path(args.binding).resolve()), "output": str(out),
                   "arm": args.arm, "max_updates": args.max_updates, "resume": args.resume,
                   "qualification": str(Path(args.qualification).resolve()) if args.qualification else None,
                   "identity": identity, "source_commit": head,
                   "training_permission": args.allow_training, "parent_pid": os.getpid()}
        write_new(supervision / "REQUEST.json", payload)
        command = [sys.executable, "-B", "-m", PACKAGE, "_worker", "--request", str(supervision / "REQUEST.json")]
        write_new(supervision / "LAUNCH.json", {"command": command, "gpu_processes": active.splitlines()})
        from experiments.opallo_native_collector_v1.supervisor import run_bounded
        receipt = run_bounded(command, root, supervision, seconds=plan.wall_seconds,
                           min_free_gib=plan.worker_free_gib, max_rss_gib=4.0,
                           go={"request_sha256": file_hash(supervision / "REQUEST.json")})
        return dict(receipt, result_path=str(supervision / "RESULT.json"),
                    native_status=read_json(supervision / "RESULT.json")["status"])
    except BaseException as exc:
        supervision.mkdir(parents=True, exist_ok=True)
        write_new(supervision / "LAUNCH_FAILURE.json", {"status": "BLOCKED", "reason": str(exc)})
        raise
    finally:
        lock.unlink()


def build_backend(e0, plan, adapter, out_parent):
    # The supervisor sets offline mode before process start; repeat defensively.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    import psutil
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import install_lora
    from .backend import TorchBackend, tensor_hash
    require({k: importlib.metadata.version(k) for k in PACKAGES} == PACKAGES, "unregistered native packages")
    require(sys.version_info[:3] == (3, 11, 15), "registered native Python is 3.11.15")
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA required")
    torch.set_num_threads(2)
    torch.manual_seed(plan.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started, samples = time.monotonic(), []
    def guard(kind="guard", tokens=0, generated=0, admission=False):
        free, _ = torch.cuda.mem_get_info()
        snap = {"seconds": time.monotonic() - started,
                "host_free_gib": psutil.virtual_memory().available / 2**30,
                "gpu_free_gib": free / 2**30, "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        require(snap["host_free_gib"] >= (plan.host_free_gib if admission else plan.worker_free_gib), "HOST_RAM_GUARD")
        require(snap["gpu_free_gib"] >= (plan.gpu_free_gib if admission else plan.gpu_worker_free_gib), "GPU_GUARD")
        require(psutil.disk_usage(str(out_parent)).free / 2**30 >= plan.disk_free_gib, "DISK_GUARD")
        require(snap["seconds"] <= plan.wall_seconds, "WORKER_WALL_GUARD")
        if not samples or snap["seconds"] - samples[-1]["seconds"] >= 1:
            samples.append(snap)
    guard(admission=True)
    directory = Path(e0["model_dir"])
    require(file_hash(directory / "model.safetensors") == BASE_SHA256, "Titan weight drift")
    for name, expected in e0["asset_sha256"].items():
        require(file_hash(directory / name) == expected, "model asset drift: " + name)
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    cfg = Qwen3Config.from_pretrained(str(directory), local_files_only=True)
    cfg._attn_implementation = "eager"
    require(cfg.num_hidden_layers == 28 and cfg.rope_theta == 1000000, "Titan configuration drift")
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda:0"):
            model = Qwen3ForCausalLM(cfg)
    finally:
        torch.set_default_dtype(old_dtype)
    model.eval().requires_grad_(False)
    states = model.state_dict()
    with safe_open(str(directory / "model.safetensors"), framework="pt", device="cpu") as weights:
        require(set(states) - set(weights.keys()) == {"lm_head.weight"}
                and not set(weights.keys()) - set(states), "strict Titan checkpoint keys")
        with torch.no_grad():
            for name in sorted(weights.keys()):
                value = weights.get_tensor(name)
                require(value.shape == states[name].shape and value.dtype == states[name].dtype, "tensor shape/dtype")
                states[name].copy_(value)
                require(tensor_hash({name: states[name]}) == tensor_hash({name: value}), "streamed load mismatch")
                guard()
    del states, value
    require(model.lm_head.weight is model.model.embed_tokens.weight
            and sum(p.numel() for p in model.parameters()) == 596049920, "Titan tie/parameter-count drift")
    install_lora(model, torch)
    backend = TorchBackend(model, tokenizer, plan, meter=guard)
    if adapter is not None:
        backend.restore(load_file(str(adapter), device="cpu"))
    backend.initial = backend.snapshot()
    backend.initial_digest = tensor_hash(backend.initial)
    guard()
    runtime = {"python": sys.version, "platform": platform.platform(), "packages": PACKAGES,
               "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
               "base_dtype": "bfloat16", "adapter_dtype": "float32", "attention": "eager",
               "sampler": "full-prefix-no-cache-temp1", "evaluation": "greedy-fixed-budget"}
    return backend, runtime, samples


def token_preflight(backend, tasks, plan):
    """Every enrolled student/teacher/target is checked before any model forward."""
    maxima = {"student": 0, "teacher": 0, "target": 0}
    for task in tasks:
        prefix = backend.views.student(task.public())
        backend.views.fit(prefix, plan.max_new_tokens)
        maxima["student"] = max(maxima["student"], len(prefix))
        target = backend.views.target(task.solution)
        backend.views.fit(prefix, len(target))
        require(len(target) <= plan.max_new_tokens, "solution exceeds the frozen response budget")
        maxima["target"] = max(maxima["target"], len(target))
        if task.split in ("train", "train_selection"):
            teacher = backend.views.privileged(task, selection=task.split == "train_selection")
            backend.views.fit(teacher, plan.max_new_tokens)
            maxima["teacher"] = max(maxima["teacher"], len(teacher))
    return maxima


def probe(backend, tasks, plan):
    """Bounded unaided/privileged continuation pairs on explicit selection parents."""
    from .contracts import purpose_seed
    selected, seen = [], set()
    for task in sorted(tasks, key=lambda t: t.key):
        if task.split == "train_selection" and task.parent not in seen:
            seen.add(task.parent)
            selected.append(task)
    rows = []
    for task in selected[:8]:
        original = backend.sample(task.public(), seed=purpose_seed(plan.seed, 0, task.key, "probe-original"))
        backend.audit_rollout(task, original)
        prefix_ids = original.response_ids[:max(0, (len(original.response_ids) - 1) // 2)]
        outcomes = {}
        for role, context in (("unaided", backend.views.student(task.public())),
                              ("privileged", backend.views.privileged(task, selection=True))):
            continuation = backend.sample(task.public(), prefix=context + prefix_ids,
                 max_new=plan.max_new_tokens - len(prefix_ids),
                 seed=purpose_seed(plan.seed, 0, task.key, "paired-continuation"))
            tail = continuation.response_ids[:-1] if continuation.terminated else continuation.response_ids
            text = backend.views.tokenizer.decode(prefix_ids + tail, skip_special_tokens=False)
            ok, reason = verify(task, text, continuation.terminated)
            outcomes[role] = {"correct": ok, "reason": reason, "terminated": continuation.terminated,
                              "tokens": len(prefix_ids) + len(continuation.response_ids)}
        rows.append({"key": task.key, "parent": task.parent, "prefix_hash": digest(prefix_ids), **outcomes})
    return {"status": "PROBE_ONLY_NO_UPDATES", "rows": rows,
            "privileged_minus_unaided": sum(int(r["privileged"]["correct"]) - int(r["unaided"]["correct"]) for r in rows) / len(rows),
            "interpretation": "8-parent feasibility diagnostic, not a capability or causal efficacy estimate"}


def qualify(backend, tasks, plan, arm, checkpoint_path):
    """Disposable mechanics on TRAIN only; no persistent learned checkpoint leaves this gate."""
    import torch
    from .backend import tensor_hash
    from .engine import scheduled, update
    from .objectives import guidance_kl
    # Independent projection/gradient reference at exact native dtype and full vocabulary.
    task = next(t for t in tasks if t.split == "train")
    response = backend.views.target(task.solution)
    with torch.no_grad():
        p = backend.hidden(backend.views.student(task.public()), response, grad=False)[:2].detach()
        q = backend.hidden(backend.views.privileged(task), response, grad=False)[:2].detach()
    h = p.clone().requires_grad_(True)
    weight = backend.model.lm_head.weight
    custom = guidance_kl(h, q, weight, plan.chunk_tokens)
    grad = torch.autograd.grad(custom, h)[0]
    h2 = p.clone().requires_grad_(True)
    lp = torch.nn.functional.linear(h2, weight).float().log_softmax(-1)
    lq = torch.nn.functional.linear(q, weight).float().log_softmax(-1)
    reference = (lq.exp() * (lq - lp)).sum(-1).mean()
    reference_grad = torch.autograd.grad(reference, h2)[0]
    require(torch.allclose(custom, reference, atol=2e-5, rtol=2e-5)
            and torch.allclose(grad.float(), reference_grad.float(), atol=1e-5, rtol=.02), "native KL parity failed")
    initial = backend.snapshot()
    frozen_before = tensor_hash({n: p for n, p in backend.model.named_parameters() if not p.requires_grad})
    optimizer = torch.optim.AdamW(list(backend.params.values()), lr=plan.learning_rate,
                                 betas=(.9, .999), eps=1e-8, weight_decay=0)
    def one_step():
        acquisition = scheduled(tasks, "train", plan.batch_tasks, plan.seed, backend.version)
        replay = scheduled(tasks, "retention_train", plan.replay_tasks, plan.seed, backend.version)
        result = update(backend, optimizer, acquisition, replay, arm, plan)
        require(result["optimizer_step"], "no usable update for this arm; native mechanics not qualified")
        return result
    try:
        first = one_step()
        with Path(checkpoint_path).open("xb") as f:
            torch.save({"parameters": backend.snapshot(), "optimizer": optimizer.state_dict(),
                        "version": backend.version, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all()}, f)
            f.flush()
            os.fsync(f.fileno())
        second = one_step()
        expected = tensor_hash(backend.snapshot())
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        backend.restore(saved["parameters"])
        backend.version = saved["version"]
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        repeated = one_step()
        require(expected == tensor_hash(backend.snapshot()) and second == repeated,
                "native disk-resume update disagreement")
        require(tensor_hash(initial) != expected, "disposable optimizer made no change")
        with backend.teacher_role(True):
            require(tensor_hash(backend.snapshot()) == backend.initial_digest, "frozen teacher role drift")
        require(tensor_hash({n: p for n, p in backend.model.named_parameters() if not p.requires_grad}) == frozen_before,
                "frozen model changed")
    finally:
        backend.restore(initial)
        backend.version = 0
        for parameter in backend.params.values():
            parameter.grad = None
    return {"status": "PASS_RSD_MECHANICS_ONLY", "qualified_arm": arm,
            "disposable_optimizer_steps": 3,
            "checks": {"full_vocab_kl_parity": True, "native_disk_resume_bitwise": True,
                       "frozen_base_unchanged": True, "parent_restored": True},
            "unqualified": ["capability gain", "SRCL acceptance quality", "other arms",
                            "longer contexts", "new runtime/plan/parent"], "results": [first, second, repeated]}


def export_parent(backend, result, out, identity, runtime, *, source_commit=None):
    """Export a completed SFT candidate, never auto-approve its capability quality."""
    from safetensors.torch import save_file
    require(result["status"] == "COMPLETE" and result["steps"] > 0, "unfinished parent cannot be exported")
    adapter = out / "SFT_PARENT_ADAPTER.safetensors"
    require(not adapter.exists(), "parent adapter already exists")
    save_file(backend.snapshot(), str(adapter))
    receipt = out / "PARENT_TRAINING_RECEIPT.json"
    write_new(receipt, {"schema": 2, "experiment": "RSD-001", "mode": "parent", "arm": "sft",
                      "source_commit": source_commit,
                      "exported_adapter": {"path": str(adapter.resolve()), "sha256": file_hash(adapter)},
                      "status": "COMPLETED_SFT_CANDIDATE_REVIEW_REQUIRED", "identity": identity,
                      "runtime": runtime, "result": result, "native_efficacy_claim": False})
    write_new(out / "PARENT_CANDIDATE.json", {"schema": 1, "base_weight_sha256": BASE_SHA256,
              "native_sft_completed": True, "parent_evaluation_reviewed": False,
              "optimizer_steps": result["steps"], "reviewed_by": "",
              "adapter": {"path": str(adapter.resolve()), "sha256": file_hash(adapter)},
              "source_run_receipt": {"path": str(receipt.resolve()), "sha256": file_hash(receipt)}})


def worker(request_path):
    request_path = Path(request_path).resolve()
    go = request_path.parent / "GO.json"
    start = time.monotonic()
    while not go.exists() and time.monotonic() - start < 30:
        time.sleep(.05)
    require(go.exists() and read_json(go)["request_sha256"] == file_hash(request_path), "supervisor GO mismatch")
    request = read_json(request_path)
    require(os.name == "nt" and os.getppid() == request["parent_pid"], "native worker containment/parent mismatch")
    root = Path(__file__).resolve().parents[2]
    out = Path(request["output"])
    result = {"status": "BLOCKED", "native_efficacy_claim": False, "pretrained_loads": 0}
    try:
        require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
                == request["source_commit"], "worker source commit drift")
        require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=root, text=True).strip(), "worker source became dirty")
        plan = Plan.load(request["plan"])
        e0, tasks, adapter, identity, audit = inputs(Path(request["binding"]), plan, root, parent_stage=request["mode"] == "parent")
        require(identity == request["identity"], "preflight/worker identity drift")
        if request["mode"] in ("parent", "qualify", "train"):
            require(request["training_permission"], "native training permission absent")
        if request["mode"] == "train":
            receipt = read_json(request["qualification"])
            require(receipt["status"] == "PASS_RSD_MECHANICS_ONLY" and receipt["identity"] == identity
                    and receipt["qualified_arm"] == request["arm"],
                    "matching RSD mechanics receipt required")
        backend, runtime, samples = build_backend(e0, plan, adapter, out.parent)
        result["pretrained_loads"] = 1
        if request["mode"] == "train":
            require(runtime == receipt["runtime"], "qualification/training runtime drift")
        maxima = token_preflight(backend, tasks, plan)
        if request["mode"] in ("parent", "train"):
            if request["mode"] == "parent" and not request["resume"]:
                counts = {"processed_tokens": 0, "generated_tokens": 0, "model_calls": 0}
                previous = backend.meter
                def prelude_meter(kind, tokens, generated):
                    for key, delta, limit in (("processed_tokens", tokens, plan.processed_token_limit),
                            ("generated_tokens", generated, plan.generated_token_limit),
                            ("model_calls", int(kind == "hidden"), plan.model_call_limit)):
                        require(counts[key] + delta <= limit, "parent mechanics budget exhausted")
                        counts[key] += delta
                    previous(kind, tokens, generated)
                backend.meter = prelude_meter
                result["parent_mechanics"] = qualify(backend, tasks, plan, "sft",
                    request_path.parent / "DISPOSABLE_PARENT_MECHANICS.pt")
                result["parent_mechanics_costs"] = counts
                backend.meter = previous
            from .engine import train
            result.update(train(backend, tasks, plan, request["arm"], out,
                          dict(identity, runtime=runtime, arm=request["arm"]),
                          max_updates=request["max_updates"], resume=request["resume"]))
            if request["mode"] == "parent" and result["status"] == "COMPLETE":
                export_parent(backend, result, out, identity, runtime, source_commit=request["source_commit"])
        else:
            # Bound diagnostic work too; not only optimizer work has a cost.
            counts = {"processed_tokens": 0, "generated_tokens": 0, "model_calls": 0}
            resource_meter = backend.meter
            def meter(kind, tokens, generated):
                for key, delta, limit in (("processed_tokens", tokens, plan.processed_token_limit),
                        ("generated_tokens", generated, plan.generated_token_limit),
                        ("model_calls", int(kind == "hidden"), plan.model_call_limit)):
                    require(counts[key] + delta <= limit, "diagnostic budget exhausted")
                    counts[key] += delta
                resource_meter(kind, tokens, generated)
            backend.meter = meter
            result.update(qualify(backend, tasks, plan, request["arm"],
                                request_path.parent / "DISPOSABLE_QUALIFICATION_STATE.pt") if request["mode"] == "qualify" else probe(backend, tasks, plan))
            result["costs"] = counts
            out.mkdir(parents=True, exist_ok=False)
        result.update(identity=identity, runtime=runtime, input_audit=audit, token_maxima=maxima, resources=samples)
    except BaseException as exc:
        result.update(status="BLOCKED", error=str(exc), exception=type(exc).__name__)
        raise
    finally:
        result["worker_wall_seconds"] = time.monotonic() - start
        # Supervisor-owned results are immutable even when the training run is quarantined.
        write_new(request_path.parent / "RESULT.json", result)
        if out.is_dir():
            write_new(out / ("NATIVE_RECEIPT-" + uuid.uuid4().hex + ".json"), result)
    return result
