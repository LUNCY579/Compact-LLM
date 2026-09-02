"""Mechanism liveness diagnostic for the P2 adapters.

The P2 A/B/C/D run concluded NO-GO on the basis that arms B, C and D emitted
identical predictions. That is only informative if the fast-weight pathway was
actually contributing something. This script measures how much it contributed.

For each trained adapter it records, at the instrumented layer:

* alpha (the merge gate) as trained;
* ||alpha * P(m)|| relative to ||attn_out||, per token -- the fraction of the
  attention output that the fast memory is responsible for;
* the recurrent state Frobenius norm as a function of t (arm C/D only), which
  shows whether the memory accumulates or saturates;
* the trained plasticity/forgetting rates eta and decay, and the implied
  per-token memory time constant;
* final-token logit deltas between arms on identical inputs, so we can say
  whether B, C and D were numerically distinguishable at all.

Read-only with respect to the checkpoint. Writes under artifacts/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rineya_fwa.arms import apply_arm  # noqa: E402
from rineya_fwa.eval_position import build_position_suite  # noqa: E402
from rineya_fwa.falcon2 import beta_from_raw, lambda_from_raw  # noqa: E402
from rineya_fwa.loading import load_supa, load_tokenizer  # noqa: E402
from rineya_fwa.paths import ARTIFACTS  # noqa: E402
from rineya_fwa.provenance import write_json  # noqa: E402
from rineya_fwa.train import load_adapter  # noqa: E402

P2 = ARTIFACTS / "p2_abcd"


def encode(tokenizer: Any, messages: list[dict[str, str]], device: str) -> torch.Tensor:
    try:
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", enable_thinking=False,
        )
    except TypeError:
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
    ids = enc.input_ids if hasattr(enc, "input_ids") else enc
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    if not torch.is_tensor(ids):
        ids = torch.tensor(ids, dtype=torch.long)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return ids.to(device)


def rate_table(block: Any) -> dict[str, Any]:
    """Trained plasticity/forgetting, evaluated at the observed feature scale.

    ||x||^2 is taken as rank * mean(k_norm.weight^2), which is what an RMSNorm
    with those weights produces on average.
    """
    cfg = block.cfg
    beta = beta_from_raw(block.beta_raw.detach().float())
    lam = lambda_from_raw(block.lambda_raw.detach().float())
    knw = block.k_norm.weight.detach().float()
    x_sq = float(cfg.rank * (knw.pow(2).mean()))
    eta = (beta / (x_sq + lam + cfg.eps)).clamp(0.0, 1.5)
    decay = (1.0 - lam * eta).clamp(0.0, 1.0)
    d_mean = float(decay.mean())
    half_life = math.log(0.5) / math.log(d_mean) if 0.0 < d_mean < 1.0 else float("inf")
    return {
        "beta_mean": float(beta.mean()),
        "lambda_mean": float(lam.mean()),
        "x_norm_sq_expected": x_sq,
        "eta_mean": float(eta.mean()),
        "decay_mean": d_mean,
        "decay_min": float(decay.min()),
        "decay_max": float(decay.max()),
        "memory_half_life_tokens": half_life,
        "note": "arm D overrides beta/lambda from the significance gate; this row is the static fallback",
    }


def probe_arm(
    model: Any,
    tokenizer: Any,
    arm: str,
    adapter: Path | None,
    items: list[dict[str, Any]],
    layers: list[int],
    rank: int,
    device: str,
) -> dict[str, Any]:
    handle = apply_arm(model, arm, layers=layers, rank=rank, init_alpha=0.0)
    captured: dict[str, torch.Tensor] = {}
    hooks = []
    try:
        if handle is not None:
            if adapter is not None:
                load_adapter(handle, adapter)
            idx = handle.layers[0]
            wrapper = handle.wrappers[idx]

            def cap_inner(_m, _i, out):
                captured["attn"] = (out[0] if isinstance(out, tuple) else out).detach().float()

            def cap_block(_m, _i, out):
                captured["delta"] = (out[0] if isinstance(out, tuple) else out).detach().float()

            hooks.append(wrapper.inner.register_forward_hook(cap_inner))
            hooks.append(wrapper.block.register_forward_hook(cap_block))

        per_item = []
        logits_last = []
        for item in items:
            captured.clear()
            ids = encode(tokenizer, item["messages"], device)
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
            logits_last.append(out.logits[0, -1].detach().float().cpu())
            row: dict[str, Any] = {"id": item["id"], "n_tokens": int(ids.shape[1])}
            if handle is not None and "delta" in captured and "attn" in captured:
                alpha = float(handle.wrappers[handle.layers[0]].block.alpha.detach())
                delta = captured["delta"][0]  # (T, H)
                attn = captured["attn"][0]  # (T, H)
                contrib = (alpha * delta).norm(dim=-1)  # (T,)
                base = attn.norm(dim=-1).clamp_min(1e-9)  # (T,)
                ratio = contrib / base
                row.update(
                    {
                        "alpha": alpha,
                        "delta_norm_mean": float(delta.norm(dim=-1).mean()),
                        "attn_norm_mean": float(base.mean()),
                        "contrib_over_attn_mean": float(ratio.mean()),
                        "contrib_over_attn_max": float(ratio.max()),
                        "contrib_over_attn_last": float(ratio[-1]),
                    }
                )
            per_item.append(row)

        result: dict[str, Any] = {
            "arm": arm,
            "adapter": str(adapter) if adapter else None,
            "items": per_item,
            "logits_last": torch.stack(logits_last),
        }
        if handle is not None:
            block = handle.wrappers[handle.layers[0]].block
            result["alpha"] = float(block.alpha.detach())
            result["rates"] = rate_table(block)
            result["P_weight_l2"] = float(block.P.weight.detach().float().norm())
            result["P_weight_absmax"] = float(block.P.weight.detach().float().abs().max())
            result["state_bytes"] = block.state_bytes(1)
        return result
    finally:
        for h in hooks:
            h.remove()
        if handle is not None:
            handle.uninstall()


def state_trace(
    model: Any, tokenizer: Any, adapter: Path, item: dict[str, Any], layers: list[int], rank: int, device: str
) -> dict[str, Any]:
    """Frobenius norm of the recurrent state S_t as a function of t, arm C."""
    handle = apply_arm(model, "C", layers=layers, rank=rank, init_alpha=0.0)
    assert handle is not None
    try:
        load_adapter(handle, adapter)
        block = handle.wrappers[handle.layers[0]].block
        hidden_cap: dict[str, torch.Tensor] = {}

        def cap_hidden(_m, args, _out):
            hidden_cap["h"] = args[0].detach().float()

        h = handle.wrappers[handle.layers[0]].inner.register_forward_hook(cap_hidden)
        ids = encode(tokenizer, item["messages"], device)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=torch.ones_like(ids))
        h.remove()
        hidden = hidden_cap["h"]
        with torch.no_grad():
            q, k, v = block._features(hidden)
            beta, lam, _ = block.rates(hidden)
            valid = torch.ones(hidden.shape[0], hidden.shape[1], dtype=torch.bool, device=hidden.device)
            # replay the scan, recording ||S_t||
            from rineya_fwa.falcon2 import falcon2_scan

            norms = []
            step = max(1, hidden.shape[1] // 64)
            for t_end in range(step, hidden.shape[1] + 1, step):
                _, state = falcon2_scan(
                    q[:, :t_end], k[:, :t_end], v[:, :t_end], valid[:, :t_end],
                    beta, lam, block.cfg.eps, recurrent=True,
                )
                norms.append({"t": t_end, "state_fro": float(state.float().norm())})
        return {"id": item["id"], "n_tokens": int(ids.shape[1]), "trace": norms}
    finally:
        handle.uninstall()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n-items", type=int, default=4)
    ap.add_argument("--out", type=str, default=str(ARTIFACTS / "diagnostics" / "mechanism.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [args.layer]
    items = build_position_suite(lengths=[512], n_per_cell=1, seed=11)[: args.n_items]

    tokenizer = load_tokenizer()
    model = load_supa(device=device, eval_mode=True)

    probes: dict[str, Any] = {}
    plan = [("A", None)]
    for arm in ("B", "C", "D"):
        plan.append((arm, P2 / f"arm_{arm}" / f"seed_{args.seed}" / "fwa_adapter.pt"))

    for arm, adapter in plan:
        if adapter is not None and not adapter.is_file():
            print(f"[skip] {arm}: no adapter at {adapter}")
            continue
        print(f"[probe] arm {arm}")
        probes[arm] = probe_arm(model, tokenizer, arm, adapter, items, layers, args.rank, device)

    # cross-arm logit deltas on identical inputs
    deltas: dict[str, Any] = {}
    arms = [a for a in ("A", "B", "C", "D") if a in probes]
    for i, a1 in enumerate(arms):
        for a2 in arms[i + 1 :]:
            l1 = probes[a1]["logits_last"]
            l2 = probes[a2]["logits_last"]
            diff = (l1 - l2).abs()
            same_argmax = int((l1.argmax(-1) == l2.argmax(-1)).sum())
            p1 = torch.log_softmax(l1, dim=-1)
            p2 = torch.log_softmax(l2, dim=-1)
            kl = float((p1.exp() * (p1 - p2)).sum(-1).mean())
            deltas[f"{a1}_vs_{a2}"] = {
                "max_abs_logit_delta": float(diff.max()),
                "mean_abs_logit_delta": float(diff.mean()),
                "argmax_agree": f"{same_argmax}/{l1.shape[0]}",
                "mean_kl_nats": kl,
            }

    trace = None
    cadapter = P2 / "arm_C" / f"seed_{args.seed}" / "fwa_adapter.pt"
    if cadapter.is_file():
        print("[probe] arm C recurrent-state trace")
        trace = state_trace(model, tokenizer, cadapter, items[0], layers, args.rank, device)

    payload = {
        "device": device,
        "layer": args.layer,
        "rank": args.rank,
        "seed": args.seed,
        "n_items": len(items),
        "arms": {
            a: {k: v for k, v in probes[a].items() if k != "logits_last"} for a in probes
        },
        "cross_arm_logit_deltas": deltas,
        "arm_C_state_trace": trace,
    }
    out = Path(args.out)
    write_json(out, payload, config=vars(args))
    print(json.dumps(payload["cross_arm_logit_deltas"], indent=2))
    for a in probes:
        if "alpha" in probes[a]:
            rows = [r for r in probes[a]["items"] if "contrib_over_attn_mean" in r]
            if rows:
                m = sum(r["contrib_over_attn_mean"] for r in rows) / len(rows)
                print(f"arm {a}: alpha={probes[a]['alpha']:.6f}  mean ||a*P(m)||/||attn|| = {m:.6e}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
