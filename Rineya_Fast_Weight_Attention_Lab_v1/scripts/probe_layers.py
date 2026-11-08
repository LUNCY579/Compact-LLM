"""Zero-training layer probe: where could a fast-weight pathway matter at all?

Replaces the cancelled trained layer sweep (PREREGISTRATION Amendment 10). That
sweep would have spent ~9 GPU-hours to return four near-certain nulls, because it
reused a suite on which the trained control already scores 159/168.

This asks the cheaper prior question: **is there any placement where the pathway
has more leverage than at layer 14?** For each of the 28 decoder layers it
installs an untrained block at a fixed alpha and measures, on real eval items:

  contribution   ||alpha * P(m)|| / ||attn_out||, the share of the attention
                 output the pathway would contribute
  state_norm     ||S_T||_F, how much the recurrent state accumulates
  attn_norm      ||attn_out||, the scale it is competing against

P is zero-initialised in the real block, so a fresh block outputs exactly zero
and measures nothing. The probe therefore gives P a fixed random projection with
a common seed across layers, so every layer is measured with the SAME projection
and only the layer's own hidden statistics differ. This measures *leverage*, not
*benefit*: a layer where the pathway cannot make itself heard cannot help, but a
layer where it can is not thereby useful.

Decision rule, fixed before running (Amendment 10):
  flat profile   -> a trained sweep is not justified; the layer question is
                    answered cheaply in the negative
  peaked profile -> trained arms go where the evidence points, and the suite is
                    rebuilt against a TRAINED control first so the contrast has
                    headroom
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rineya_fwa.arms import apply_arm  # noqa: E402
from rineya_fwa.data_longctx import build_position_suite_v2  # noqa: E402
from rineya_fwa.eval_longctx import encode_prompt  # noqa: E402
from rineya_fwa.loading import load_supa, load_tokenizer  # noqa: E402
from rineya_fwa.paths import ARTIFACTS  # noqa: E402
from rineya_fwa.provenance import write_json  # noqa: E402

SHORT_HORIZON = {"beta_raw_init": 0.0, "lambda_raw_init": -2.0}


def probe_layer(
    model: Any, tok: Any, layer: int, items: list[dict[str, Any]], rank: int,
    alpha: float, proj_seed: int, recurrent: bool,
) -> dict[str, Any]:
    handle = apply_arm(
        model, "C" if recurrent else "B", layers=[layer], rank=rank,
        init_alpha=alpha, **SHORT_HORIZON,
    )
    assert handle is not None
    block = handle.wrappers[layer].block
    # identical random projection at every layer, so only the layer differs
    g = torch.Generator(device="cpu").manual_seed(proj_seed)
    w = torch.randn(block.P.weight.shape, generator=g) * 0.05
    block.P.weight.data.copy_(w.to(block.P.weight.device))

    cap: dict[str, torch.Tensor] = {}
    hooks = [
        handle.wrappers[layer].inner.register_forward_hook(
            lambda _m, _i, o: cap.__setitem__("attn", (o[0] if isinstance(o, tuple) else o).detach().float())
        ),
        block.register_forward_hook(
            lambda _m, _i, o: cap.__setitem__("delta", (o[0] if isinstance(o, tuple) else o).detach().float())
        ),
    ]
    try:
        ratios, attn_norms, state_norms = [], [], []
        for item in items:
            cap.clear()
            ids = encode_prompt(tok, item["messages"]).to(next(model.parameters()).device)
            with torch.no_grad():
                model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=1)
            if "delta" not in cap or "attn" not in cap:
                continue
            d, a = cap["delta"][0], cap["attn"][0]
            base = a.norm(dim=-1).clamp_min(1e-9)
            ratios.append(float(((alpha * d).norm(dim=-1) / base).mean()))
            attn_norms.append(float(base.mean()))
            if recurrent:
                with torch.no_grad():
                    hid = cap["attn"]  # not the block input, but same scale; state needs a replay
                    q, k, v = block._features(hid)
                    beta, lam, _ = block.rates(hid)
                    from rineya_fwa.falcon2 import falcon2_scan

                    _, st = falcon2_scan(
                        q, k, v, torch.ones(hid.shape[0], hid.shape[1], dtype=torch.bool, device=hid.device),
                        beta, lam, block.cfg.eps, recurrent=True,
                    )
                    state_norms.append(float(st.norm()))
        return {
            "layer": layer,
            "contribution_mean": sum(ratios) / len(ratios) if ratios else float("nan"),
            "attn_norm_mean": sum(attn_norms) / len(attn_norms) if attn_norms else float("nan"),
            "state_norm_mean": (sum(state_norms) / len(state_norms)) if state_norms else None,
            "n_items": len(ratios),
        }
    finally:
        for h in hooks:
            h.remove()
        handle.uninstall()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--proj-seed", type=int, default=7)
    ap.add_argument("--n-items", type=int, default=4)
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--out", type=str, default=str(ARTIFACTS / "diagnostics" / "layer_probe.json"))
    args = ap.parse_args()

    tok = load_tokenizer()
    items = build_position_suite_v2(
        tok, lengths=[args.length], positions=(0.05, 0.50), n_per_cell=args.n_items // 2,
        n_decoys=24, n_candidates=32, seed=101, split="eval", key_mode="confusable", hops=2,
    )
    model = load_supa(device="cuda" if torch.cuda.is_available() else "cpu", eval_mode=True)
    n_layers = int(model.config.num_hidden_layers)
    print(f"probing {n_layers} layers on {len(items)} items at {args.length} tokens", flush=True)

    rows = []
    for layer in range(n_layers):
        r = probe_layer(model, tok, layer, items, args.rank, args.alpha, args.proj_seed, recurrent=True)
        rows.append(r)
        sn = f"{r['state_norm_mean']:8.3f}" if r["state_norm_mean"] is not None else "     n/a"
        print(f"  layer {layer:2d}  contribution={r['contribution_mean']:.5f}  "
              f"||attn||={r['attn_norm_mean']:8.3f}  ||S_T||={sn}", flush=True)

    good = [r for r in rows if r["contribution_mean"] == r["contribution_mean"]]
    best = max(good, key=lambda r: r["contribution_mean"])
    worst = min(good, key=lambda r: r["contribution_mean"])
    l14 = next((r for r in good if r["layer"] == 14), None)
    spread = best["contribution_mean"] / max(worst["contribution_mean"], 1e-12)
    payload = {
        "config": vars(args), "n_layers": n_layers, "rows": rows,
        "best_layer": best["layer"], "worst_layer": worst["layer"],
        "max_over_min_ratio": spread,
        "layer_14_contribution": l14["contribution_mean"] if l14 else None,
        "layer_14_rank_from_top": (
            sorted(good, key=lambda r: -r["contribution_mean"]).index(l14) + 1 if l14 else None
        ),
    }
    write_json(Path(args.out), payload, config=vars(args))
    print(f"\nbest layer {best['layer']} ({best['contribution_mean']:.5f}), "
          f"worst layer {worst['layer']} ({worst['contribution_mean']:.5f}), "
          f"max/min = {spread:.2f}x")
    if l14:
        print(f"layer 14 = {l14['contribution_mean']:.5f}, "
              f"rank {payload['layer_14_rank_from_top']} of {len(good)} by leverage")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
