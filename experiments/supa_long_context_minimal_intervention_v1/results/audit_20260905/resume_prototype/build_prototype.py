"""Create the isolated review copy and unified patch. Never writes live sources."""
from pathlib import Path
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
LIVE = HERE.parents[2] / "src" / "slcmi" / "training.py"
source = LIVE.read_text(encoding="utf-8")
original = source


def replace(old, new):
    global source
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one replacement: {old[:80]!r}")
    source = source.replace(old, new)


replace("import contextlib\n", "import contextlib\nimport hashlib\nfrom pathlib import Path\nimport platform\n")
replace("import torch\n", "import torch\n\nfrom .checkpoint_resume import validate_identity, save_latest, load_latest, restore_rng\n")
replace("    progress_callback: Callable[[dict], None] | None = None,\n", "    progress_callback: Callable[[dict], None] | None = None,\n    checkpoint_dir: str | Path | None = None,\n    checkpoint_fingerprints: dict[str, str] | None = None,\n    resume_from_checkpoint: bool = False,\n")
replace("    The budget is never silently extended to compensate for skipped updates.\n", "    The budget is never silently extended to compensate for skipped updates.\n    Checkpoints commit whole attempts and scheduled-validation state atomically.\n    Resume uses the original total max_steps, never a fresh segment budget.\n")
replace("    torch.manual_seed(seed)\n", "    if resume_from_checkpoint and checkpoint_dir is None:\n        raise ValueError('resume requires checkpoint_dir')\n    if checkpoint_dir is None and checkpoint_fingerprints is not None:\n        raise ValueError('checkpoint fingerprints require checkpoint_dir')\n    if checkpoint_dir is not None and not resume_from_checkpoint and (Path(checkpoint_dir) / 'latest.pt').exists():\n        raise FileExistsError('latest checkpoint exists; explicitly resume or choose a new directory')\n    torch.manual_seed(seed)\n")
replace("    trajectory: list[dict] = []\n", '''    identity = None
    if checkpoint_dir is not None:
        effective_recipe = {
            "max_steps": max_steps, "grad_accum": grad_accum, "lr": lr,
            "warmup_ratio": warmup_ratio, "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm, "seed": seed,
            "gradient_checkpointing": bool(gradient_checkpointing),
            "eval_every": eval_every, "device": device,
            "amp_dtype": str(amp_dtype), "scaler_enabled": scaler.is_enabled(),
            "gate_bias": model.config.abstain_logit_bias,
            "trainables": [[name, list(p.shape), str(p.dtype)] for name, p in named_params],
            "torch_version": str(torch.__version__), "python_version": platform.python_version(),
            "cuda_build": torch.version.cuda,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if device == "cuda" else [],
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "num_threads": torch.get_num_threads(),
            "training_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "checkpoint_code_sha256": hashlib.sha256(Path(__file__).with_name("checkpoint_resume.py").read_bytes()).hexdigest(),
        }
        identity = validate_identity(checkpoint_fingerprints, train_set, eval_set, effective_recipe)

    trajectory: list[dict] = []
''')
replace("    t0 = time.perf_counter()\n", '''    completed_attempts = 0
    pending_validation = None
    elapsed_before_segment = 0.0
    resumed_rng = None

    def eval_due(attempt):
        return bool(eval_set) and (attempt == max_steps or bool(eval_every and attempt % eval_every == 0))

    if resume_from_checkpoint:
        saved, resumed_rng = load_latest(
            checkpoint_dir, identity=identity, named_params=named_params,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            max_steps=max_steps, grad_accum=grad_accum, eval_due=eval_due, train_set=train_set, device=device,
        )
        completed_attempts = saved["completed_attempts"]
        cursor = saved["cursor"]
        consumed = saved["consumed"]
        skipped_steps = saved["skipped_steps"]
        safety_skipped_steps = saved["safety_skipped_steps"]
        applied_steps = saved["applied_steps"]
        finite_nonzero_steps = saved["finite_nonzero_steps"]
        applied_examples = saved["applied_examples"]
        applied_answer_tokens = saved["applied_answer_tokens"]
        trajectory = saved["trajectory"]
        evals = saved["evals"]
        pending_validation = saved["pending_validation"]
        elapsed_before_segment = saved["elapsed_seconds"]

    t0 = time.perf_counter()

    def elapsed():
        return elapsed_before_segment + time.perf_counter() - t0

    def commit():
        if checkpoint_dir is not None:
            save_latest(
                checkpoint_dir, identity=identity, named_params=named_params,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler, device=device,
                progress={
                    "completed_attempts": completed_attempts, "cursor": cursor,
                    "consumed": consumed, "skipped_steps": skipped_steps,
                    "safety_skipped_steps": safety_skipped_steps,
                    "applied_steps": applied_steps, "finite_nonzero_steps": finite_nonzero_steps,
                    "applied_examples": applied_examples, "applied_answer_tokens": applied_answer_tokens,
                    "trajectory": trajectory, "evals": evals,
                    "pending_validation": pending_validation, "elapsed_seconds": elapsed(),
                },
            )

    def run_pending_validation(original_bias):
        nonlocal pending_validation
        if pending_validation is not None:
            result = {"step": pending_validation,
                      **_validation_with_gate(model, eval_set, device, amp_dtype, original_bias)}
            evals.append(result)
            pending_validation = None
            commit()
            if progress_callback:
                progress_callback({"event": "validation", **result})
''')
replace("        for step in range(max_steps):\n", '''        if resumed_rng is not None:
            # Restore after constructing the model/optimizer/checkpointing context.
            restore_rng(resumed_rng, device)
        else:
            commit()  # Attempt-zero recovery, including a death inside accumulation.
        run_pending_validation(original_bias)
        for step in range(completed_attempts, max_steps):
''')
replace('"elapsed_seconds": round(time.perf_counter() - t0, 2),', '"elapsed_seconds": round(elapsed(), 2),')
replace("            trajectory.append(record)\n", '''            trajectory.append(record)
            completed_attempts = step + 1
            pending_validation = step + 1 if eval_due(step + 1) else None
            commit()  # Commit the update before callbacks and before validation.
''')
replace('''            if eval_every and eval_set and (step + 1) % eval_every == 0:
                evals.append({"step": step + 1,
                              **_validation_with_gate(model, eval_set, device, amp_dtype, original_bias)})
                if progress_callback:
                    progress_callback({"event": "validation", **evals[-1]})
''', '''            if pending_validation is not None:
                run_pending_validation(original_bias)
''')
replace('''        if eval_set and (not evals or evals[-1]["step"] != max_steps):
            evals.append({"step": max_steps,
                          **_validation_with_gate(model, eval_set, device, amp_dtype, original_bias)})
            if progress_callback:
                progress_callback({"event": "validation", **evals[-1]})

''', '')
replace("    wall = time.perf_counter() - t0\n", "    wall = elapsed()\n")

target = HERE / "src" / "slcmi" / "training.py"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(source, encoding="utf-8")
(HERE / "baseline_training.py").write_text(original, encoding="utf-8")
target.with_name("__init__.py").write_text("", encoding="utf-8")
(HERE / "training.patch").write_text("".join(difflib.unified_diff(original.splitlines(True), source.splitlines(True), fromfile="src/slcmi/training.py", tofile="src/slcmi/training.py")), encoding="utf-8")
helper = HERE / "src" / "slcmi" / "checkpoint_resume.py"
with (HERE / "training.patch").open("a", encoding="utf-8") as out:
    out.write("".join(difflib.unified_diff([], helper.read_text(encoding="utf-8").splitlines(True), fromfile="/dev/null", tofile="src/slcmi/checkpoint_resume.py")))
(HERE / "source_manifest.json").write_text(json.dumps({"live_source_path": str(LIVE), "live_source_sha256": hashlib.sha256(LIVE.read_bytes()).hexdigest(), "prototype_source_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "canonical_sources_modified": False}, indent=2) + "\n", encoding="utf-8")
print(target)
