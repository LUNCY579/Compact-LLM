"""Layer-location pilot (brief section 6), run after P2b rather than before it.

P2b returned NO-GO at layer 14. Layer 14 was inherited from P1 and never
justified, so that NO-GO is a statement about one arbitrary placement. This
sweeps the placement and asks whether the C-vs-B null is a property of the
architecture or of where it was attached.

Design decisions and why:

* **Arms B and C only.** The decisive preregistered contrast is C vs B
  (recurrence against matched parameters). Arms A and D add cost without
  addressing the question; arm A is placement-independent and already scored.

* **Short horizon everywhere** (`beta_raw = 0.0`, `lambda_raw = -2.0`). P2b found
  the long horizon actively harmful: arm C-short was the best arm in the run and
  the interference cost tracked the horizon, not recurrence. Sweeping placement
  with a configuration already known to be worse would confound the two, and
  would give recurrence less than its best shot. This is the one deviation from
  the P2b main-run settings and it is deliberate.

* **A matched B-short at layer 14 is required.** P2b never trained one: the
  layer-14 arm B carries the LONG horizon, so the tempting
  "C-short 163 vs B 159" comparison differs in horizon *and* recurrence at once.
  Layer 14 is re-run for arm B so every placement has an internally clean pair.

* **Identical suite.** Same generator arguments as P2b, so the suite fingerprint
  matches and every number here is directly comparable to the P2b tables.

* **One seed (11).** P2b showed the position metric is stable across seeds
  (arm B scored 0.9464 at both). Interference is not stable across seeds and no
  interference claim is made from this pilot.

Placements, over a 28-layer backbone: early 4, middle 14, late 24, and
distributed [4, 14, 24]. Distributed triples the trainable parameters, so it is
matched only against its own arm B, never against the single-layer arms.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rineya_fwa.arms import apply_arm  # noqa: E402
from rineya_fwa.data_longctx import (  # noqa: E402
    build_interference_battery_v2,
    build_position_suite_v2,
    build_train_rows,
)
from rineya_fwa.eval_longctx import mcnemar_exact, score_interference, score_position_suite  # noqa: E402
from rineya_fwa.loading import load_supa, load_tokenizer  # noqa: E402
from rineya_fwa.paths import ARTIFACTS  # noqa: E402
from rineya_fwa.provenance import write_json  # noqa: E402
from rineya_fwa.train import load_adapter, train_arm  # noqa: E402

OUT = ARTIFACTS / "p2b_layer_pilot"
SHORT_HORIZON = {"beta_raw_init": 0.0, "lambda_raw_init": -2.0}

PLACEMENTS: list[tuple[str, list[int]]] = [
    ("early", [4]),
    ("mid", [14]),
    ("late", [24]),
    ("distributed", [4, 14, 24]),
]


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def adapter_matches(out_dir: Path, args: Any, layers: list[int]) -> bool:
    """Reuse an adapter only if it was trained with exactly this config.

    Checks ``layers`` explicitly: placement is the swept variable here, so an
    adapter trained at a different layer must never be silently adopted. That is
    the failure a 2-step smoke adapter already caused once in P2b.
    """
    if not (out_dir / "fwa_adapter.pt").is_file() or not (out_dir / "training_manifest.json").is_file():
        return False
    try:
        man = json.loads((out_dir / "training_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "layers": list(layers),
        "optimizer_steps": args.max_steps,
        "learning_rate": args.lr,
        "grad_accum": args.grad_accum,
        "rank": args.rank,
        "init_alpha": args.init_alpha,
        "beta_raw_init": SHORT_HORIZON["beta_raw_init"],
        "lambda_raw_init": SHORT_HORIZON["lambda_raw_init"],
        "n_train_rows": args.train_rows,
    }
    for k, want in expected.items():
        got = man.get(k)
        if isinstance(want, float):
            ok = got is not None and abs(float(got) - want) < 1e-9
        else:
            ok = got == want
        if not ok:
            print(f"[train] {out_dir}: manifest {k}={got!r} != {want!r}; retraining", flush=True)
            return False
    return man.get("aborted") is None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--eval-lengths", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--n-per-cell", type=int, default=8)
    ap.add_argument("--n-decoys", type=int, default=24)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--interference-length", type=int, default=2048)
    ap.add_argument("--n-per-category", type=int, default=8)
    ap.add_argument("--train-rows", type=int, default=512)
    ap.add_argument("--train-lengths", type=int, nargs="+", default=[1024])
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init-alpha", type=float, default=0.05)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer()

    train_rows = build_train_rows(
        tok, n_rows=args.train_rows, lengths=args.train_lengths,
        n_decoys=args.n_decoys, seed=909, hops=args.hops,
    )
    pos_items = build_position_suite_v2(
        tok, lengths=args.eval_lengths, n_per_cell=args.n_per_cell,
        n_decoys=args.n_decoys, n_candidates=32, seed=101, split="eval",
        key_mode="confusable", hops=args.hops,
    )
    int_items = build_interference_battery_v2(
        tok, target_tokens=args.interference_length, n_per_category=args.n_per_category, seed=202
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            [[i["id"], i["messages"][1]["content"], i["answer"], i["candidate_token_ids"]]
             for i in list(pos_items) + list(int_items)],
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    print(f"[data] position n={len(pos_items)} interference n={len(int_items)} fingerprint {fingerprint}", flush=True)

    plan = [(name, layers, arm) for name, layers in PLACEMENTS for arm in ("B", "C")]

    if not args.skip_train:
        for name, layers, arm in plan:
            out_dir = OUT / name / f"arm_{arm}"
            if adapter_matches(out_dir, args, layers):
                print(f"[train] {name}/{arm} already trained with this config, skipping", flush=True)
                continue
            print(f"[train] {name}/{arm} layers={layers}", flush=True)
            t0 = time.perf_counter()
            man = train_arm(
                arm=arm, layers=layers, train_file=None, output_dir=out_dir,
                seed=args.seed, rank=args.rank, max_steps=args.max_steps,
                max_sequence_length=max(args.train_lengths) + 128,
                learning_rate=args.lr, grad_accum=args.grad_accum,
                init_alpha=args.init_alpha, train_rows=train_rows, **SHORT_HORIZON,
            )
            print(
                f"[train] {name}/{arm} steps={man['optimizer_steps']} "
                f"alpha={man['final_alpha']:.5f} trainable={man['trainable_parameters']} "
                f"gpu_h={man['gpu_hours']:.3f} ({time.perf_counter()-t0:.0f}s)",
                flush=True,
            )
            free_cuda()

    print("[eval] loading model once", flush=True)
    model = load_supa(device=device, eval_mode=True)
    abst = int(getattr(model.config, "abstain_token_id"))
    results: dict[str, Any] = {}

    for name, layers, arm in plan:
        out_dir = OUT / name / f"arm_{arm}"
        adapter = out_dir / "fwa_adapter.pt"
        if not adapter.is_file():
            print(f"[eval] {name}/{arm}: no adapter, skipping", flush=True)
            continue
        cached = out_dir / "eval.json"
        if cached.is_file():
            try:
                rec = json.loads(cached.read_text(encoding="utf-8"))
                if rec.get("suite_fingerprint") == fingerprint and rec.get("layers") == list(layers):
                    results[f"{name}/{arm}"] = rec
                    print(f"[eval] {name}/{arm} cached exact={rec['position']['exact']:.4f}", flush=True)
                    continue
            except (OSError, json.JSONDecodeError, KeyError):
                pass
        handle = apply_arm(model, arm, layers=layers, rank=args.rank, init_alpha=0.0, **SHORT_HORIZON)
        try:
            load_adapter(handle, adapter)
            t0 = time.perf_counter()
            pos = score_position_suite(model, tok, pos_items, abstain_token_id=abst)
            inter = score_interference(model, tok, int_items)
            rec = {
                "placement": name, "layers": list(layers), "arm": arm, "seed": args.seed,
                "alpha": float(handle.blocks()[0].alpha.detach()),
                "trainable_parameters": sum(p.numel() for p in handle.trainable_parameters()),
                "position": {k: v for k, v in pos.items() if k != "rows"},
                "position_rows": pos["rows"],
                "interference": {k: v for k, v in inter.items() if k != "rows"},
                "interference_rows": inter["rows"],
                "suite_fingerprint": fingerprint,
                "eval_seconds": time.perf_counter() - t0,
            }
            results[f"{name}/{arm}"] = rec
            write_json(cached, rec, config=vars(args))
            print(
                f"[eval] {name}/{arm} position={pos['exact']:.4f} "
                f"({int(round(pos['exact']*len(pos_items)))}/{len(pos_items)}) "
                f"interference={inter['exact']:.4f} ({time.perf_counter()-t0:.0f}s)",
                flush=True,
            )
        finally:
            if handle is not None:
                handle.uninstall()
            free_cuda()

    contrasts = {}
    for name, layers in PLACEMENTS:
        b, c = results.get(f"{name}/B"), results.get(f"{name}/C")
        if b and c:
            contrasts[f"C_vs_B_{name}"] = {
                "layers": list(layers),
                "position": mcnemar_exact(b["position_rows"], c["position_rows"]),
                "interference": mcnemar_exact(b["interference_rows"], c["interference_rows"]),
                "b_total": int(round(b["position"]["exact"] * len(pos_items))),
                "c_total": int(round(c["position"]["exact"] * len(pos_items))),
            }

    summary = {
        "config": vars(args),
        "horizon": SHORT_HORIZON,
        "suite_fingerprint": fingerprint,
        "placements": {k: v["layers"] for k, v in
                       {f"{n}": {"layers": l} for n, l in PLACEMENTS}.items()},
        "results": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_rows")}
                    for k, v in results.items()},
        "contrasts": contrasts,
    }
    write_json(OUT / "summary.json", summary, config=vars(args))
    print(json.dumps(contrasts, indent=2, default=str))
    print(f"wrote {OUT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
