"""QWEN-FT-001: train a LoRA adapter on Qwen3-4B using the RRP-015 control's EXACT rows.

Data equality is exact, not approximate: the Supa tokenizer and Qwen3-4B's tokenizer are
byte-identical on all four files, so `control-prepared/microfit-train.jsonl`'s precomputed
`ids` and `labels` transfer directly. Same 384 rows, same label masks, same row order, same
28 update groups as the 596M control.

What is NOT the same, and is recorded rather than glossed:
  - parameterization: LoRA r=16 on attention+MLP projections, not FP32 tail-20 blocks.
    The MQ recipe is arithmetically impossible here -- tail-20 of Qwen3-4B is 2.02B params,
    30.08 GiB of weights+grads+Adam in FP32, against 15.22 GiB of host RAM.
  - base precision: 4-bit NF4, because BF16 weights are 7.49 GiB against ~7.2 GiB of free
    GPU. A 4-bit zero-shot control is measured separately to isolate the quantization cost.
  - learning rate: LoRA needs a higher rate than full fine-tuning. Declared below.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--groups", type=int, help="limit update groups, for smoke tests")
    ap.add_argument("--four-bit", action="store_true")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit("create-only output")

    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(17)

    rows = {r["id"]: r for r in
            (json.loads(l) for l in (a.prepared / "microfit-train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    plan = json.loads(a.plan.read_text(encoding="utf-8"))
    # update_groups hold POSITIONAL indices into row_order_per_exposure, which is the
    # control's exact training order. Resolving through it reproduces that order too.
    order = plan["row_order_per_exposure"]
    if order and isinstance(order[0], list):
        order = [rid for ex in order for rid in ex]
    groups = [[order[i] for i in g] for g in plan["update_groups"]]
    if a.groups:
        groups = groups[:a.groups]
    missing = [rid for g in groups for rid in g if rid not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} planned rows absent from prepared data, e.g. {missing[:3]}")

    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if a.four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
    else:
        kw["device_map"] = "auto"
        kw["max_memory"] = {0: "6GiB", "cpu": "6GiB"}
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    if a.four_bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    lcfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    load_s = time.time() - t0
    print(f"loaded in {load_s:.1f}s | trainable {n_train/1e6:.2f}M params", flush=True)

    opt = torch.optim.AdamW(trainable, lr=a.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    # WSD 20/60/20 decaying to 0.1x peak, matching the MQ schedule's SHAPE.
    n = len(groups)
    def lr_at(step):
        w, d = max(1, int(0.2 * n)), max(1, int(0.2 * n))
        if step < w: return (step + 1) / w
        if step >= n - d: return max(0.1, 1.0 - 0.9 * (step - (n - d) + 1) / d)
        return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    a.out.mkdir(parents=True)
    log, proc, sup = [], 0, 0
    started = time.time()
    for gi, g in enumerate(groups):
        opt.zero_grad(set_to_none=True)
        gl, gt = 0.0, 0
        for rid in g:
            r = rows[rid]
            ids = torch.tensor([r["ids"]], device=0)
            lab = torch.tensor([r["labels"]], device=0)
            out = model(input_ids=ids, labels=lab)
            ntok = int((lab != -100).sum())
            # weight each row by its supervised tokens so the group's loss is a token mean
            (out.loss * ntok).backward()
            gl += float(out.loss.detach()) * ntok; gt += ntok
            proc += len(r["ids"]); sup += ntok
        for p in trainable:
            if p.grad is not None: p.grad /= max(gt, 1)
        gn = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        opt.step(); sched.step()
        rec = {"step": gi + 1, "rows": len(g), "supervised_tokens": gt,
               "loss": gl / max(gt, 1), "grad_norm": gn, "lr": opt.param_groups[0]["lr"]}
        log.append(rec)
        print(f"  step {gi+1}/{n} loss {rec['loss']:.4f} gn {gn:.3f} "
              f"lr {rec['lr']:.2e} | {time.time()-started:.0f}s", flush=True)

    model.save_pretrained(str(a.out / "adapter"))
    record = {
        "experiment": "COMPOSE4B-001", "base_model": str(a.model),
        "data": "compose001-20260918/H/H-prepared, byte-identical rows to the 596M H (abstention/boundary) skill",
        "rows": sum(len(g) for g in groups), "update_groups": len(groups),
        "processed_tokens": proc, "supervised_tokens": sup,
        "supervised_fraction": sup / proc,
        "parameterization": f"LoRA r={a.rank} alpha={2*a.rank} on q,k,v,o,gate,up,down",
        "trainable_params": n_train,
        "base_precision": "nf4 double-quant" if a.four_bit else "bf16 with cpu offload",
        "optimizer": "AdamW", "peak_lr": a.lr, "betas": [0.9, 0.95], "eps": 1e-8,
        "weight_decay": 0.0, "grad_clip": 1.0,
        "schedule": "WSD 20/60/20 to 0.1x peak (shape matched to MQ)",
        "seed": 17, "gradient_checkpointing": True,
        "deviations_from_mq_recipe": [
            "LoRA instead of FP32 tail-20 blocks: tail-20 of Qwen3-4B is 2.02B params = 30.08 GiB in FP32, impossible on 15.22 GiB host RAM",
            f"peak lr {a.lr} instead of 1e-5: LoRA requires a higher rate than full fine-tuning",
            "4-bit NF4 base instead of FP32: BF16 base is 7.49 GiB against ~7.2 GiB free GPU" if a.four_bit else "bf16 base with CPU offload instead of FP32",
        ],
        "final_loss": log[-1]["loss"] if log else None,
        "first_loss": log[0]["loss"] if log else None,
        "wall_seconds": round(time.time() - started, 1), "load_seconds": round(load_s, 1),
        "steps": log, "quality_claim": False, "release_permitted": False,
    }
    (a.out / "TRAIN.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in
                      ("rows", "update_groups", "processed_tokens", "supervised_tokens",
                       "trainable_params", "first_loss", "final_loss", "wall_seconds")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
