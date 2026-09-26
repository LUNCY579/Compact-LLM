#!/usr/bin/env python3
"""Phase 1, step 0: adapter mechanics receipt. CPU only, no training.

The brief asks for minimal checks that confirm expected column normalization,
that ordinary LoRA is left unchanged, that a zero adapter reproduces frozen
Supa, that adapter serialization works, and that merge behaviour stays valid.

Run on CPU deliberately. The merge check reloads and deep-copies a 596M-parameter
model, and the shared 8 GB GPU on this machine routinely sits above 7 GB from
other work; doing it on the GPU risks OOM for a check that needs no accelerator.

    python scripts/phase1_verify_adapters.py

Writes ``results/phase1_adapter_verification/receipt.json``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from slcmi.adapters import (  # noqa: E402
    ATTENTION_TARGETS,
    DEFAULT_ALPHA,
    DEFAULT_DROPOUT,
    DEFAULT_LAYERS,
    DEFAULT_RANK,
    EXPANDED_LAYERS,
    PROTECTED_LAYERS,
    build_adapter,
    build_lora_config,
    verify_adapter,
    verify_serialisation_and_merge,
)
from slcmi.evaluation import SEED, load_model  # noqa: E402
from slcmi.provenance import CANONICAL_CORE, Run, write_json  # noqa: E402


def main() -> None:
    config = {
        "stage": "phase1_adapter_verification",
        "device": "cpu",
        "layers": list(DEFAULT_LAYERS),
        "protected_layers": list(PROTECTED_LAYERS),
        "targets": list(ATTENTION_TARGETS),
        "rank": DEFAULT_RANK,
        "alpha": DEFAULT_ALPHA,
        "dropout": DEFAULT_DROPOUT,
        "seed": SEED,
    }

    with Run("phase1_adapter_verification", config) as run:
        receipt: dict = {"config": config, "arms": {}}

        # The protected stem must be refused at config-build time, not merely
        # avoided by convention.
        try:
            build_lora_config(layers=[0, 17])
            receipt["protected_stem_refused"] = False
        except ValueError as exc:
            receipt["protected_stem_refused"] = True
            receipt["protected_stem_refusal_message"] = str(exc)

        tokenizer, base = load_model(CANONICAL_CORE, "cpu")
        run.model_loads = 1
        probe = tokenizer(
            "CONTEXT\nA short probe document.\n\nQUESTION\nWhat is it?\n\nANSWER",
            return_tensors="pt",
            add_special_tokens=False,
        )
        probe = {k: v for k, v in probe.items() if k in ("input_ids", "attention_mask")}

        with torch.no_grad():
            reference_logits = base(**probe).logits.float().clone()
        receipt["frozen_reference_logits_sha_shape"] = list(reference_logits.shape)

        for arm, kind in (("B", "lora"), ("C", "nora")):
            model, info = build_adapter(
                base, kind, DEFAULT_LAYERS, DEFAULT_RANK, DEFAULT_ALPHA, DEFAULT_DROPOUT, SEED
            )
            checks = verify_adapter(model, kind, DEFAULT_LAYERS, reference_logits, probe)
            with tempfile.TemporaryDirectory() as tmp:
                checks["serialisation"] = verify_serialisation_and_merge(model, tmp, probe)
            receipt["arms"][arm] = {
                "kind": kind,
                "trainable_params": info["trainable_params"],
                "trainable_fraction": info["trainable_fraction"],
                "trainable_tensor_count": info["trainable_tensor_count"],
                "nora_init": info.get("nora_init"),
                "checks": checks,
            }
            print(f"  arm {arm} ({kind}): trainable {info['trainable_params']:,} "
                  f"({info['trainable_fraction'] * 100:.4f}%)  "
                  f"invariants {'PASS' if checks['all_passed'] else 'FAIL'}  "
                  f"serialisation {'PASS' if checks['serialisation']['all_passed'] else 'FAIL'}")
            base = model.get_base_model()
            model.unload()
            del model

        # Phase 2's expanded set is only sized here, never trained unless licensed.
        expanded, expanded_info = build_adapter(
            base, "lora", EXPANDED_LAYERS, DEFAULT_RANK, DEFAULT_ALPHA, DEFAULT_DROPOUT, SEED
        )
        receipt["phase2_expanded_sizing"] = {
            "layers": list(EXPANDED_LAYERS),
            "trainable_params": expanded_info["trainable_params"],
            "trainable_fraction": expanded_info["trainable_fraction"],
            "note": "sizing only; Phase 2 runs only if Phase 1 is clearly positive",
        }
        base = expanded.get_base_model()
        expanded.unload()
        del expanded

        arms_ok = all(
            r["checks"]["all_passed"] and r["checks"]["serialisation"]["all_passed"]
            for r in receipt["arms"].values()
        )
        receipt["status"] = "PASS" if (arms_ok and receipt["protected_stem_refused"]) else "FAIL"

        write_json(run.out / "receipt.json", receipt, overwrite=True)
        run.finish(receipt["status"], {"receipt_status": receipt["status"]})

    print(f"\nstatus: {receipt['status']}")
    print(f"L17-19 trainable: {receipt['arms']['B']['trainable_params']:,}")
    print(f"L17,18,19,25,27 trainable (sizing only): "
          f"{receipt['phase2_expanded_sizing']['trainable_params']:,}")
    print(f"wrote {run.out / 'receipt.json'}")


if __name__ == "__main__":
    main()
