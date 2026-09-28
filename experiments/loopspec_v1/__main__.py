"""Offline CPU experiment and read-only native metadata preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from .core import Plan, Sampling, decode

BASE_SHA = "7d9b4fd450bc56405f6cec41459a7e83df7958ae"


def source_manifest():
    root = Path(__file__).parent
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and "evidence" not in p.relative_to(root).parts
        and "__pycache__" not in p.parts and p.suffix in {".py", ".md", ".txt", ".json"}
    }


def write_new(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def preflight(config_path: Path):
    """Inspect config bytes only; NEVER imports checkpoint code or loads weights.

    There is deliberately no native launch path in this revision. A metadata
    declaration is not enough to turn an arbitrary checkpoint into a backend.
    """
    result = dict(schema="loopspec.native_metadata_preflight.v1", status="BLOCKED",
                  native_execution="NOT_RUN", weights_loaded=False,
                  config_sha256=None, model_type=None, reasons=[],
                  base_sha=BASE_SHA)
    if not config_path.is_file():
        result["reasons"].append("CONFIG_MISSING")
    else:
        try:
            if config_path.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("config exceeds 2 MiB metadata limit")
            raw = config_path.read_bytes()
            result["config_sha256"] = hashlib.sha256(raw).hexdigest()
            config = json.loads(raw)
            if not isinstance(config, dict):
                raise ValueError("config must be an object")
            result["model_type"] = config.get("model_type")
            if config.get("model_type") == "qwen3":
                result["reasons"].append("STANDARD_QWEN3_IS_NOT_A_QUALIFIED_LOOPED_BACKEND")
        except (OSError, ValueError) as exc:
            result["reasons"].append("CONFIG_INVALID:" + type(exc).__name__)
    result["reasons"].append("NATIVE_LOOP_SLICED_BACKEND_NOT_IMPLEMENTED_OR_QUALIFIED")
    result["required_before_native_execution"] = [
        "Bind exact checkpoint, tokenizer, runtime, dtype, processors and fixed recurrent depth.",
        "Provide reviewed prefill/start/advance/readout/cache_bytes backend without changing target semantics.",
        "Qualify independent full-prefix, incremental and mixed-depth cache/coda/position parity.",
        "Freeze calibration-only depth/cap settings before untouched evaluation prompts.",
        "Obtain exact-host compute authorization and enforce memory/time limits.",
    ]
    return result


def smoke():
    import torch
    from .tiny import TinyLoopBackend

    torch.set_num_threads(1)
    rows, headroom, sampled_rows = [], [], []
    arms = {"ar": (None, None, True), "single": (1, None, True),
            "residual_gated": (1, 2, True), "residual_ungated": (1, 2, False)}
    prompts = [(1, 4, 2), (3, 6, 1), (2, 1, 7, 4)]
    max_drift = 0.0
    for seed in (0, 1, 2):
        model = TinyLoopBackend(seed=seed)
        digest = model.weight_digest()
        # Warm the local kernels; not included in measured arm rows.
        decode(model, prompts[0], Plan(first=None, second=None, max_new_tokens=4))
        for request_id, prompt in enumerate(prompts):
            outputs = []
            for _ in range(16):
                z = model.full_logits(prompt + tuple(outputs))[4]
                outputs.append(max(range(len(z)), key=lambda i: z[i]))
            # Synthetic apparatus headroom, not real language calibration.
            for n in range(6):
                prefix = prompt + tuple(outputs[:n])
                oracle = model.full_logits(prefix)
                state, cached = model.prefill(prefix, (1, 2, 3, 4))
                max_drift = max(max_drift, max(abs(a-b) for a, b in zip(cached, oracle[4])))
                p = Sampling(temperature=1, top_p=0.9).distribution(oracle[4])
                target_top = max(range(len(p)), key=lambda i: oracle[4][i])
                for depth in (1, 2, 3):
                    q = Sampling(temperature=1, top_p=0.9).distribution(oracle[depth])
                    headroom.append(dict(seed=seed, request=request_id, position=n, depth=depth,
                        tv=0.5*sum(abs(a-b) for a, b in zip(p, q)),
                        top1_agreement=max(range(len(q)), key=lambda i: oracle[depth][i]) == target_top))
                del state
            # Counterbalance order across seeds/requests. One measured run per
            # cell: timing is diagnostic only, never a significance claim.
            names = list(arms)
            shift = (seed + request_id) % len(names)
            for name in names[shift:] + names[:shift]:
                first, second, gated = arms[name]
                plan = Plan(first=first, second=second, gated=gated, max_new_tokens=16,
                            max_branches=8, seed=20260926)
                mixed_before = model.mixed_depth_batches
                result = decode(model, prompt, plan)
                equal = result.tokens == outputs and result.stop_reason == "length"
                if not equal:
                    raise AssertionError("greedy/full-prefix mismatch; no successful receipt written")
                rows.append(dict(seed=seed, request=request_id, arm=name, prompt=list(prompt),
                    plan=asdict(plan), greedy_oracle_equal=equal, mixed_depth_batches=model.mixed_depth_batches-mixed_before,
                    **asdict(result)))
        for gated in (True, False):
            plan = Plan(first=1, second=2, gated=gated, max_new_tokens=16,
                        sampling=Sampling(temperature=1, top_p=0.9), seed=1299)
            result = decode(model, prompts[0], plan)
            sampled_rows.append(dict(seed=seed, plan=asdict(plan), **asdict(result)))
        if model.weight_digest() != digest:
            raise AssertionError("apparatus weights changed")
    aggregates = {}
    for name in arms:
        selected = [r for r in rows if r["arm"] == name]
        aggregates[name] = dict(
            runs=len(selected), greedy_matches=sum(r["greedy_oracle_equal"] for r in selected),
            mean_recurrent_batches=mean(r["stats"]["recurrent_batches"] for r in selected),
            mean_branch_steps=mean(r["stats"]["branch_steps"] for r in selected),
            mean_cpu_decode_seconds=mean(r["stats"]["decode_seconds"] for r in selected),
            max_active_branches=max(r["stats"]["peak_active"] for r in selected),
            max_logical_kv_bytes=max(r["stats"]["peak_logical_kv_bytes"] for r in selected))
    return dict(schema="loopspec.cpu_apparatus_smoke.v1", status="PASS",
        created_utc=datetime.now(timezone.utc).isoformat(), base_sha=BASE_SHA,
        environment=dict(python=sys.version, torch=torch.__version__, platform=platform.platform(),
                         cuda_available=torch.cuda.is_available(), execution_device="cpu", dtype="float64", threads=1),
        source_sha256=source_manifest(), apparatus="random_frozen_prelude_shared_GQA_RoPE_SwiGLU_core_coda",
        total_depth=4, vocab_size=17, weights_unchanged=True, training_steps=0,
        native_rineya_execution="NOT_RUN", native_backend_status="NOT_IMPLEMENTED_OR_QUALIFIED",
        native_speedup=None, capability_gain=None, inherited_repository_tests="NOT_RUN",
        timing_scope="one diagnostic CPU run per cell; prefill includes first token; decode excludes prefill",
        memory_scope="unique logical KV tensor bytes only; excludes hidden/logit/Python/allocator memory",
        oracle_max_abs_logit_drift=max_drift, aggregates=aggregates, runs=rows,
        synthetic_headroom=headroom, sampled_gate_ablation=sampled_rows,
        sampled_interpretation="Law tested separately; sampled sequences need not match AR or each other.",
        claim_ceiling="CPU apparatus correctness and work accounting only; not language quality or GPU speedup")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "native-preflight"):
        command = sub.add_parser(name)
        command.add_argument("--out", type=Path, required=True, help="new JSON path; existing files refused")
        if name == "native-preflight":
            command.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output already exists; receipts are append-only")
    value = smoke() if args.command == "smoke" else preflight(args.config)
    write_new(args.out, value)
    print(json.dumps({"status": value["status"], "output": str(args.out)}))
    return 0 if value["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
