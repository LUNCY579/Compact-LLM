"""SMALLVD-001 deliverable 13/14 -- A0/A1/A2/A3 architecture manifests with EXACT counts.

Everything here is computed, not estimated: unique parameters (analytic, which param_accounting
has shown equals the header count on the teacher), exact BF16 safetensors bytes (the layout
predictor that reproduces the teacher's header byte-for-byte), weight memory at bf16/fp32, and KV
cache bytes per token (2 x layers x kv_heads x head_dim x dtype bytes). A2/A3 take their block set
from the registered selection in LAYER_REPORT.json; before that exists they are written with
`kept_layers: null` and `status: PENDING_LAYER_SELECTION` rather than with a guessed set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import param_accounting as pa  # noqa: E402


def manifest(arm, teacher_cfg, header, vocab_rows, kept, vocab_dir, note):
    n = len(kept) if kept is not None else None
    base = {"schema": 1, "record": "SMALLVD_ARCH_MANIFEST", "experiment": "SMALLVD-001", "arm": arm,
            "note": note, "vocab_rows": vocab_rows, "vocab_dir": vocab_dir,
            "preserved": {k: teacher_cfg[k] for k in ("hidden_size", "intermediate_size", "num_attention_heads",
                                                      "num_key_value_heads", "head_dim", "rope_theta", "rope_scaling",
                                                      "rms_norm_eps", "hidden_act", "tie_word_embeddings",
                                                      "max_position_embeddings", "attention_bias")},
            "quality_claim": False, "release_permitted": False}
    if kept is None:
        return {**base, "status": "PENDING_LAYER_SELECTION", "kept_layers": None}
    acc = pa.analytic(teacher_cfg, vocab_rows, n)
    _, size = pa.safetensors_layout(pa.student_shapes(header, vocab_rows, kept), header.get("__metadata__"))
    kv = 2 * n * teacher_cfg["num_key_value_heads"] * teacher_cfg["head_dim"]
    return {**base, "status": "DEFINED", "n_layers": n, "kept_layers_old_index": kept,
            "old_to_new_layer": {str(o): i for i, o in enumerate(kept)},
            "unique_parameters": acc["unique_physical_parameters"], "by_family": acc["by_family"],
            "embedding_parameters": acc["embedding_and_head"],
            "embedding_share": acc["embedding_and_head"] / acc["unique_physical_parameters"],
            "bf16_safetensors_bytes_exact": size,
            "weight_bytes": {"bf16": acc["unique_physical_parameters"] * 2, "fp32": acc["unique_physical_parameters"] * 4},
            "kv_cache_bytes_per_token": {"bf16": kv * 2, "fp32": kv * 4},
            "kv_cache_bytes_at_4096_tokens": {"bf16": kv * 2 * 4096, "fp32": kv * 4 * 4096}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, type=Path)
    ap.add_argument("--vocab-dir", required=True, type=Path)
    ap.add_argument("--layer-report", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    cfg = json.loads((a.teacher / "config.json").read_text(encoding="utf-8"))
    header, _ = pa.read_header(a.teacher / "model.safetensors")
    v2 = json.loads((a.vocab_dir / "config.json").read_text(encoding="utf-8"))["vocab_size"]
    kept = None
    if a.layer_report and a.layer_report.exists():
        kept = json.loads(a.layer_report.read_text(encoding="utf-8"))["selected_layer_set"]["kept"]
    L = cfg["num_hidden_layers"]
    arms = {
        "A0": manifest("A0", cfg, header, cfg["vocab_size"], list(range(L)), None, "untouched teacher"),
        "A1": manifest("A1", cfg, header, v2, list(range(L)), str(a.vocab_dir), "vocabulary only"),
        "A2": manifest("A2", cfg, header, cfg["vocab_size"], kept, None, "depth only (registered 18-block set)"),
        "A3": manifest("A3", cfg, header, v2, kept, str(a.vocab_dir), "vocabulary + the SAME 18-block set"),
    }
    a.out.mkdir(parents=True, exist_ok=True)
    for k, m in arms.items():
        with open(a.out / f"ARCH_{k}.json", "x", encoding="utf-8", newline="\n") as f:
            json.dump(m, f, indent=1, allow_nan=False)
        print(k, m["status"], m.get("unique_parameters"), m.get("bf16_safetensors_bytes_exact"),
              m.get("kv_cache_bytes_per_token"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
