"""Explicit CPU diagnostics and guarded, local-only native training/evaluation."""
import argparse
import json
from pathlib import Path
import random
import time
import torch
from torch.nn import functional as F
from .core import MemoryConfig
from .sidecar import build_arm, parameter_count
from .native import FrozenQwenSidecar, greedy_decode
from .data import generate_rows, with_retrieval_probes, write_jsonl, load_rows
from .analysis import paired_report
from .provenance import (canonical_hash, environment, create_checkpoint_lock,
                         verify_checkpoint_lock, sha256_file, save_json,
                         source_hash, source_commit, check_device, check_peak)

ROOT = Path(__file__).parent
ARMS = ("baseline", "stateless", "gdn_e1", "triadic_e4", "matrix_state_matched")


def read_job(path):
    job = json.loads(Path(path).read_text())
    if job.get("schema") != "twm-pilot-v1":
        raise ValueError("unsupported job schema")
    MemoryConfig(**job["memory"])
    for key in ("steps", "max_seq_len", "max_new_tokens"):
        if type(job[key]) is not int or job[key] <= 0:
            raise ValueError(f"positive integer required for {key}")
    if not 0 < job["learning_rate"] <= 0.1 or job["clip_grad_norm"] <= 0 or job["teacher_kl"] < 0:
        raise ValueError("invalid optimization hyperparameters")
    if job["memory"]["slices"] != 4:
        raise ValueError("first campaign fixes E=4 against the E=1 control")
    return job


def load_native(checkpoint, lock, job, arm, seed, device):
    root = verify_checkpoint_lock(checkpoint, lock)
    check_device(device, job["min_free_gib"])
    import transformers
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("native contract requires transformers==4.57.1; requalify before changing versions")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    geometry = json.loads((root / "config.json").read_text())
    expected = dict(model_type="qwen3", hidden_size=1024, num_hidden_layers=28,
                    num_attention_heads=16, num_key_value_heads=8, head_dim=128)
    if any(geometry.get(k) != v for k, v in expected.items()) or geometry.get("quantization_config"):
        raise ValueError("pilot requires unquantized 28-layer Qwen3-derived Rineya 0.6B geometry")
    if job["memory"]["d_model"] != 1024 or not 0 <= job["layer"] < 28:
        raise ValueError("native sidecar width/layer mismatch")
    torch.manual_seed(seed)
    random.seed(seed)
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(root, local_files_only=True, trust_remote_code=False,
                                               torch_dtype=dtype, attn_implementation="eager").to(device)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    sidecar = build_arm(arm, MemoryConfig(**job["memory"]))
    if sidecar is not None:
        sidecar.to(device=device, dtype=torch.float32)
    model = FrozenQwenSidecar(base, sidecar, job["layer"]).eval()
    return model, tokenizer


def encode_prompt(tokenizer, text):
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("a checkpoint-bound chat template is required")
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                         add_generation_prompt=True, enable_thinking=False)


def encode_training_row(tokenizer, row, max_length, device):
    prompt = encode_prompt(tokenizer, row["prompt"])
    answer = tokenizer.encode(row["answer"], add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        answer.append(tokenizer.eos_token_id)
    if not prompt or not answer or len(prompt)+len(answer) > max_length:
        raise ValueError(f"row {row['id']} empty or exceeds max_seq_len; no silent truncation")
    ids = torch.tensor([prompt+answer], device=device, dtype=torch.long)
    labels = ids.clone()
    labels[:, :len(prompt)] = -100
    return ids, labels, len(prompt)


def answer_loss(logits, labels):
    targets = labels[:, 1:]
    selected = targets != -100
    if not bool(selected.any()):
        raise ValueError("no answer tokens available for supervision")
    return F.cross_entropy(logits[:, :-1][selected].float(), targets[selected])


def verify_qualification(path, job, lock, device):
    receipt = json.loads(Path(path).read_text())
    expected = dict(status="NATIVE_MECHANICAL_PASS", checkpoint_lock_sha256=lock["lock_sha256"],
                    source_sha256=source_hash(), job_sha256=canonical_hash(job), device=device)
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("native qualification missing or not bound to this source/config/checkpoint/device")
    current = environment()
    if any(receipt.get("environment", {}).get(k) != current[k] for k in ("torch", "transformers")):
        raise ValueError("native runtime changed since qualification")
    required = ("zero_gate_exact_identity", "active_cache_equivalence", "greedy_token_equality",
                "disabled_after_activation_exact_identity")
    if not all(receipt.get("results", {}).get(k) is True for k in required):
        raise ValueError("incomplete qualification results")


def train(args, job, lock):
    rows = load_rows(args.data, "train")
    model, tokenizer = load_native(args.checkpoint, lock, job, args.arm, args.seed, args.device)
    if model.sidecar is None:
        model.close()
        raise ValueError("baseline is frozen and is evaluated, not trained")
    model.train()
    opt = torch.optim.AdamW(model.sidecar.parameters(), lr=job["learning_rate"], weight_decay=0.0)
    rng = random.Random(args.seed)
    order, log = list(range(len(rows))), []
    versions = {n: p._version for n, p in model.backbone.named_parameters()}
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    try:
        for step in range(job["steps"]):
            if step % len(rows) == 0:
                rng.shuffle(order)
            row = rows[order[step % len(rows)]]
            ids, labels, prefix = encode_training_row(tokenizer, row, job["max_seq_len"], args.device)
            opt.zero_grad(set_to_none=True)
            output = model(ids)
            supervised = answer_loss(output.logits, labels)
            kl = supervised.new_zeros(())
            if job["teacher_kl"]:
                # Same-input preservation aid, NOT proof of broad retention.
                selected = slice(max(0, prefix-16), prefix)
                with torch.no_grad():
                    teacher = model(ids, enabled=False).logits[:, selected].float().softmax(-1)
                student = output.logits[:, selected].float().log_softmax(-1)
                kl = F.kl_div(student.reshape(-1, student.shape[-1]),
                              teacher.reshape(-1, teacher.shape[-1]), reduction="batchmean")
            loss = supervised + job["teacher_kl"] * kl
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite native training loss")
            loss.backward()
            model.assert_frozen()
            norm = torch.nn.utils.clip_grad_norm_(model.sidecar.parameters(), job["clip_grad_norm"], error_if_nonfinite=True)
            opt.step()
            peak = check_peak(args.device, job["peak_budget_gib"])
            log.append(dict(step=step+1, row_id=row["id"], tokens=ids.numel(),
                            answer_tokens=int((labels != -100).sum()), loss=float(loss.detach()),
                            answer_loss=float(supervised.detach()), teacher_kl=float(kl.detach()),
                            grad_norm=float(norm), peak_allocated_bytes=peak))
        if any(p._version != versions[n] for n, p in model.backbone.named_parameters()):
            raise RuntimeError("a backbone parameter changed in-place")
        model.assert_frozen()
        checkpoint = dict(schema="twm-adapter-v1", arm=args.arm, seed=args.seed,
                          job=job, checkpoint_lock_sha256=lock["lock_sha256"],
                          source_sha256=source_hash(),
                          state_dict={k: v.detach().cpu() for k, v in model.sidecar.state_dict().items()})
        torch.save(checkpoint, args.out / "adapter.pt")
        write_jsonl(args.out / "training_steps.jsonl", log)
        save_json(args.out / "training_receipt.json", dict(status="NATIVE_TRAINING_COMPLETED_NO_EFFICACY_CLAIM",
                  elapsed_seconds=time.perf_counter()-started, arm=args.arm, seed=args.seed,
                  train_data_sha256=sha256_file(Path(args.data)), steps=len(log),
                  total_input_tokens=sum(x["tokens"] for x in log),
                  trainable_parameters=parameter_count(model.sidecar),
                  backbone_frozen=True, adapter_sha256=sha256_file(args.out / "adapter.pt"),
                  checkpoint_lock_sha256=lock["lock_sha256"], source_sha256=source_hash()))
    finally:
        model.close()


def load_adapter(model, path, arm, seed, job, lock):
    obj = torch.load(path, weights_only=True, map_location="cpu")
    expected = dict(schema="twm-adapter-v1", arm=arm, seed=seed, job=job,
                    checkpoint_lock_sha256=lock["lock_sha256"], source_sha256=source_hash())
    if any(obj.get(k) != v for k, v in expected.items()):
        raise ValueError("adapter source/config/seed/checkpoint identity mismatch")
    model.sidecar.load_state_dict(obj["state_dict"], strict=True)


def evaluate(args, job, lock):
    rows = load_rows(args.data, "eval", args.allow_holdout)
    if args.allow_holdout and args.oracle:
        raise ValueError("oracle-facts analysis is development-only in this first campaign")
    model, tokenizer = load_native(args.checkpoint, lock, job, args.arm, args.seed, args.device)
    try:
        if args.arm != "baseline":
            if not args.adapter:
                raise ValueError("candidate evaluation requires a trained, source-bound adapter")
            load_adapter(model, args.adapter, args.arm, args.seed, job, lock)
        if torch.device(args.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(args.device)
        comparison = canonical_hash(dict(data_sha256=sha256_file(Path(args.data)),
                         source=source_hash(), checkpoint=lock["lock_sha256"], job=job,
                         oracle=args.oracle, seed=args.seed, decoding="greedy_cached",
                         device=args.device))
        predictions = []
        for row in rows:
            text = row["oracle_prompt"] if args.oracle else row["prompt"]
            prompt = encode_prompt(tokenizer, text)
            if not prompt or len(prompt)+job["max_new_tokens"] > job["max_seq_len"]:
                raise ValueError(f"row {row['id']} exceeds input+output budget; no truncation")
            ids = torch.tensor([prompt], device=args.device, dtype=torch.long)
            if torch.device(args.device).type == "cuda":
                torch.cuda.synchronize(args.device)
            start = time.perf_counter()
            tokens, stop, profile = greedy_decode(model, ids, max_new_tokens=job["max_new_tokens"],
                                         eos_token_id=tokenizer.eos_token_id, ablation=args.ablation, return_profile=True)
            peak = check_peak(args.device, job["peak_budget_gib"])
            elapsed = time.perf_counter()-start
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            correct = " ".join(text.split()) == " ".join(row["answer"].split())
            predictions.append(dict(id=row["id"], group_id=row["group_id"], family=row["family"],
                           split=row["split"], answer=row["answer"], prediction=text, correct=int(correct),
                           comparison_id=comparison, seed=args.seed, arm=args.arm, ablation=args.ablation,
                           oracle=args.oracle, stop_reason=stop, generated_token_ids=tokens,
                           input_tokens=len(prompt), output_tokens=len(tokens),
                           prefill_plus_decode_seconds=elapsed, timing=profile, peak_allocated_bytes=peak))
        write_jsonl(args.out / "predictions.jsonl", predictions)
        by_family = {}
        for family in sorted({r["family"] for r in predictions}):
            subset = [r for r in predictions if r["family"] == family]
            by_family[family] = dict(n=len(subset), accuracy=sum(r["correct"] for r in subset)/len(subset))
        save_json(args.out / "evaluation_receipt.json", dict(status="NATIVE_SINGLE_SEED_DIAGNOSTIC",
                 arm=args.arm, seed=args.seed, n=len(predictions), by_family=by_family,
                 accuracy=sum(r["correct"] for r in predictions)/len(predictions),
                 comparison_id=comparison, checkpoint_lock_sha256=lock["lock_sha256"],
                 source_sha256=source_hash(), prospective_holdout_opened=args.allow_holdout,
                 production_promotion=False, ordinary_KV_cache_retained=True))
    finally:
        model.close()


def qualify(args, job, lock):
    model, tokenizer = load_native(args.checkpoint, lock, job, "triadic_e4", args.seed, args.device)
    ids = torch.tensor([encode_prompt(tokenizer, "The parcel is in box A. Reply with the box name only.")], device=args.device)
    tolerance = 2e-2 if torch.device(args.device).type == "cuda" else 1e-5
    results = {}
    try:
        with torch.no_grad():
            baseline = model(ids, enabled=False).logits
            neutral = model(ids).logits
            results["zero_gate_exact_identity"] = torch.equal(baseline, neutral)
            model.sidecar.gate.fill_(0.1)
            full = model(ids).logits
            cache, chunks = None, []
            for i in range(ids.shape[1]):
                output = model(ids[:, i:i+1], cache=cache, use_cache=True)
                chunks.append(output.logits)
                cache = output.cache
            streamed = torch.cat(chunks, 1)
            results["active_cache_max_abs_error"] = float((full-streamed).abs().max())
            results["active_cache_equivalence"] = bool(torch.allclose(full, streamed, atol=tolerance, rtol=tolerance))
            a, _ = greedy_decode(model, ids, max_new_tokens=8, cached=True)
            b, _ = greedy_decode(model, ids, max_new_tokens=8, cached=False)
            results["greedy_token_equality"] = a == b
            results["disabled_after_activation_exact_identity"] = torch.equal(baseline, model(ids, enabled=False).logits)
        required = ("zero_gate_exact_identity", "active_cache_equivalence", "greedy_token_equality",
                    "disabled_after_activation_exact_identity")
        passed = all(results[k] for k in required)
        save_json(args.out / "qualification.json", dict(status="NATIVE_MECHANICAL_PASS" if passed else "NATIVE_MECHANICAL_FAIL",
                    results=results, tolerance=tolerance, checkpoint_lock_sha256=lock["lock_sha256"],
                    source_sha256=source_hash(), job_sha256=canonical_hash(job), device=args.device,
                    environment=environment(), efficacy_claim=False))
        if not passed:
            raise RuntimeError("native mechanical qualification failed; do not start efficacy campaign")
    finally:
        model.close()


def smoke(out):
    """Tiny fixed-tensor optimization diagnostic. Not a language/reasoning benchmark."""
    c = MemoryConfig(d_model=16, heads=2, key_dim=4, value_dim=4, slices=4, conv_width=4)
    torch.manual_seed(9)
    hidden = torch.randn(2, 12, 16)
    target = hidden + 0.1*hidden.cumsum(1)/torch.arange(1, 13)[None, :, None]
    records = []
    for arm in ARMS:
        torch.manual_seed(17)
        module = build_arm(arm, c)
        if module is None:
            losses = [float(F.mse_loss(hidden, target))]
        else:
            opt = torch.optim.AdamW(module.parameters(), lr=0.01, weight_decay=0.0)
            losses = []
            for _ in range(12):
                opt.zero_grad(set_to_none=True)
                output, _ = module(hidden)
                loss = F.mse_loss(output, target)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            if not all(torch.isfinite(torch.tensor(losses))):
                raise FloatingPointError("nonfinite smoke loss")
        records.append(dict(arm=arm, trainable_parameters=parameter_count(module) if module else 0,
                            initial_loss=losses[0], last_preupdate_loss=losses[-1], optimizer_steps=12 if module else 0))
    receipt = dict(status="REFERENCE_OPTIMIZATION_SMOKE_COMPLETED", environment=environment(),
                   task="fixed_tensor_prefix_mean_regression", real_checkpoint_loaded=False,
                   efficacy_claim=False, arms=records)
    save_json(out, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("smoke")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("generate")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=("train", "dev", "holdout", "retention_demo"), required=True)
    p.add_argument("--count", type=int, default=320)
    p.add_argument("--seed", type=int, default=20261005)
    p.add_argument("--distractors", type=int, default=4)
    p.add_argument("--retrieval-probes", action="store_true")
    p = sub.add_parser("lock")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-id", required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--families", nargs="+")
    p.add_argument("--out", type=Path, required=True)
    for command in ("train", "eval", "qualify"):
        p = sub.add_parser(command)
        p.add_argument("--config", type=Path, default=ROOT / "configs" / "pilot.json")
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--lock", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--seed", type=int, default=17)
        p.add_argument("--execute-native", action="store_true")
        if command != "qualify":
            p.add_argument("--arm", choices=ARMS, required=True)
            p.add_argument("--data", type=Path, required=True)
            p.add_argument("--qualification", type=Path, required=True)
        if command == "eval":
            p.add_argument("--adapter", type=Path)
            p.add_argument("--allow-holdout", action="store_true")
            p.add_argument("--oracle", action="store_true")
            p.add_argument("--ablation", choices=("none", "constant_second_keys", "erase_reads", "read_only"), default="none")
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out)
    elif args.command == "generate":
        rows = generate_rows(args.split, args.count, args.seed, args.distractors)
        write_jsonl(args.out, with_retrieval_probes(rows) if args.retrieval_probes else rows)
    elif args.command == "lock":
        save_json(args.out, create_checkpoint_lock(args.checkpoint, args.checkpoint_id))
    elif args.command == "compare":
        a = [json.loads(x) for x in args.baseline.read_text().splitlines() if x.strip()]
        b = [json.loads(x) for x in args.candidate.read_text().splitlines() if x.strip()]
        save_json(args.out, paired_report(a, b, families=args.families))
    else:
        if not args.execute_native:
            parser.error("native commands require --execute-native; default behavior does not start training")
        job, lock = read_job(args.config), json.loads(args.lock.read_text())
        if args.seed not in job["seeds"]:
            raise ValueError("seed is outside the registered campaign")
        args.out.mkdir(parents=True, exist_ok=False)
        save_json(args.out / "run_manifest.json", dict(command=args.command, environment=environment(),
                  job=job, job_sha256=canonical_hash(job), checkpoint_lock=lock, git_commit=source_commit()))
        try:
            if args.command != "qualify":
                verify_qualification(args.qualification, job, lock, args.device)
            {"train": train, "eval": evaluate, "qualify": qualify}[args.command](args, job, lock)
        except Exception as error:
            save_json(args.out / "failure.json", dict(status="BLOCKED_OR_FAILED_NO_EFFICACY_CLAIM",
                      exception_type=type(error).__name__, reason=str(error), environment=environment()))
            raise
    print(f"Completed {args.command}: {args.out}")


if __name__ == "__main__":
    main()
