"""SMALLVD-001 -- vocabulary, depth and combined checkpoint surgery, with invariants asserted on the
BUILT file, not on the intention.

    A1  vocabulary only   rows re-indexed through token_id_map.json, all 28 blocks
    A2  depth only        original 151,936 rows untouched, kept blocks renumbered 0..L-1
    A3  both              the A1 map AND the A2 layer set, composed -- never re-selected

Nothing is averaged, interpolated, re-initialised or trained. Every output tensor is an EXACT copy
of a teacher tensor or an exact row gather of one, and `verify_built` proves it tensor by tensor.

Tying. The teacher stores no lm_head.weight (tie_word_embeddings=true). The student is written the
same way, so the ONE gathered embedding matrix serves as both input embedding and output head,
and the tie survives save/reload because there is only one tensor to save. Checked after reload
by storage identity, not by value.

Environment: .venv-rab001-cu126 (torch 2.10.0+cu126, transformers 4.51.3, safetensors 0.7.0),
run with -B so the pinned venv gets no __pycache__. CPU only; the GPU is not needed to cut
tensors. The caller holds the lease anyway, because builds compete for the same host RAM.
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import struct
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import param_accounting as pa  # noqa: E402
import vocab_prune as vp  # noqa: E402

TOKENIZER_SIDECARS = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def data_section_sha256(p: Path) -> str:
    """Identity of record for weights: the tensor bytes only. Whole-file sha is unreliable because
    `__metadata__` key order can vary between processes (programme memory)."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + n)
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def runtime() -> dict:
    import safetensors
    import transformers
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__, "safetensors": safetensors.__version__,
            "platform": platform.platform()}


def load_id_map(tok_dir: Path) -> dict[int, int]:
    raw = json.loads((tok_dir / "token_id_map.json").read_text(encoding="utf-8"))
    m = {int(k): int(v) for k, v in raw["map"].items()}
    if sorted(m.values()) != list(range(len(m))):
        raise ValueError("token_id_map is not a bijection onto 0..n-1")
    olds = sorted(m)
    if [m[o] for o in olds] != list(range(len(m))):
        raise ValueError("token_id_map is not monotone")
    return m


def gather_order(id_map: dict[int, int]) -> torch.Tensor:
    """Old row index for each new row, in new-id order."""
    inv = sorted(id_map.items(), key=lambda kv: kv[1])
    return torch.tensor([old for old, _ in inv], dtype=torch.long)


def build_student(teacher_dir: Path, out_dir: Path, *, kept_layers: list[int] | None,
                  tok_dir: Path | None, arm: str, label: str) -> dict:
    """Write model.safetensors + sidecars + SURGERY.json into a NEW directory (create-only)."""
    teacher_dir, out_dir = Path(teacher_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((teacher_dir / "config.json").read_text(encoding="utf-8"))
    gen = json.loads((teacher_dir / "generation_config.json").read_text(encoding="utf-8"))
    n_teacher = cfg["num_hidden_layers"]
    kept = list(range(n_teacher)) if kept_layers is None else sorted(int(x) for x in kept_layers)
    if len(set(kept)) != len(kept) or not all(0 <= k < n_teacher for k in kept):
        raise ValueError(f"invalid layer set {kept}")
    pos = {old: new for new, old in enumerate(kept)}
    id_map = load_id_map(tok_dir) if tok_dir is not None else None
    order = gather_order(id_map) if id_map is not None else None
    rows = len(id_map) if id_map is not None else cfg["vocab_size"]

    t0 = time.time()
    out = {}
    with safe_open(str(teacher_dir / "model.safetensors"), framework="pt", device="cpu") as S:
        for k in S.keys():
            m = pa.LAYER_RE.match(k)
            if m:
                old = int(m.group(1))
                if old not in pos:
                    continue
                out[f"model.layers.{pos[old]}.{m.group(2)}"] = S.get_tensor(k).contiguous()
            elif k in ("model.embed_tokens.weight", "lm_head.weight"):
                t = S.get_tensor(k)
                out[k] = (t.index_select(0, order) if order is not None else t).contiguous()
            else:
                out[k] = S.get_tensor(k).contiguous()
    wpath = out_dir / "model.safetensors"
    save_file(out, str(wpath), metadata={"format": "pt"})
    del out

    # config / generation config / tokenizer from ONE id map
    if id_map is not None:
        new_cfg = vp.remap_model_config(cfg, id_map, rows, len(kept))
        new_gen = vp.remap_generation_config(gen, id_map)
        for name in TOKENIZER_SIDECARS:
            shutil.copyfile(tok_dir / name, out_dir / name)
        shutil.copyfile(tok_dir / "token_id_map.json", out_dir / "token_id_map.json")
    else:
        new_cfg = dict(cfg)
        new_cfg["num_hidden_layers"] = len(kept)
        if new_cfg.get("max_window_layers") is not None:
            new_cfg["max_window_layers"] = min(new_cfg["max_window_layers"], len(kept))
        new_gen = gen
        for name in TOKENIZER_SIDECARS:
            shutil.copyfile(teacher_dir / name, out_dir / name)
    for name, obj in (("config.json", new_cfg), ("generation_config.json", new_gen)):
        with open(out_dir / name, "xb") as f:
            f.write(vp.canonical_json_bytes(obj))

    layer_map = {str(old): new for old, new in pos.items()}
    manifest = {
        "schema": 1, "record": "SMALLVD_SURGERY", "experiment": "SMALLVD-001", "arm": arm,
        "label": label, "quality_claim": False, "release_permitted": False,
        "teacher_dir": str(teacher_dir),
        "teacher_weight_sha256": sha256_file(teacher_dir / "model.safetensors"),
        "teacher_sidecar_sha256": {n: sha256_file(teacher_dir / n) for n in
                                   ("config.json", "generation_config.json") + TOKENIZER_SIDECARS},
        "builder_sha256": sha256_file(Path(__file__)),
        "kept_layers_old_index": kept, "old_to_new_layer": layer_map,
        "vocab_rows": rows, "n_layers": len(kept),
        "token_id_map_sha256": sha256_file(tok_dir / "token_id_map.json") if tok_dir else None,
        "tokenizer_dir": str(tok_dir) if tok_dir else None,
        "out_weight_file_sha256": sha256_file(wpath),
        "out_DATA_SECTION_sha256": data_section_sha256(wpath),
        "out_file_bytes": wpath.stat().st_size,
        "out_sidecar_sha256": {p.name: sha256_file(p) for p in sorted(out_dir.iterdir())
                               if p.name != "model.safetensors"},
        "model_calls": 0, "optimizer_steps": 0, "build_seconds": round(time.time() - t0, 2),
        "runtime": runtime(),
    }
    with open(out_dir / "SURGERY.json", "x", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, allow_nan=False)
    return manifest


def verify_built(out_dir: Path, teacher_dir: Path) -> dict:
    """Structural invariants on the BUILT artifact. Raises on the first violation."""
    out_dir, teacher_dir = Path(out_dir), Path(teacher_dir)
    man = json.loads((out_dir / "SURGERY.json").read_text(encoding="utf-8"))
    cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    tcfg = json.loads((teacher_dir / "config.json").read_text(encoding="utf-8"))
    kept = man["kept_layers_old_index"]
    theader, _ = pa.read_header(teacher_dir / "model.safetensors")
    header, _ = pa.read_header(out_dir / "model.safetensors")
    checks = {}

    # 1. architecture: only vocab rows and depth may differ
    for k in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads",
              "head_dim", "rope_theta", "rope_scaling", "rms_norm_eps", "hidden_act",
              "tie_word_embeddings", "attention_bias", "max_position_embeddings", "model_type"):
        if cfg.get(k) != tcfg.get(k):
            raise AssertionError(f"config {k} changed: {tcfg.get(k)} -> {cfg.get(k)}")
    assert cfg["num_hidden_layers"] == len(kept) == man["n_layers"]
    assert cfg["vocab_size"] == man["vocab_rows"]
    checks["architecture_preserved"] = True

    # 2. header shapes and exact file size match the accounting predictor
    pred = pa.student_shapes(theader, cfg["vocab_size"], kept)
    got = {k: (v["dtype"], v["shape"]) for k, v in header.items() if k != "__metadata__"}
    assert got == pred, "built tensor set/shapes differ from the accounting prediction"
    _, pred_size = pa.safetensors_layout(pred, header.get("__metadata__"))
    assert pred_size == (out_dir / "model.safetensors").stat().st_size, "file size != predicted layout"
    acc = pa.account_header(header, cfg)
    ana = pa.analytic(tcfg, cfg["vocab_size"], len(kept))
    assert acc["unique_physical_parameters"] == ana["unique_physical_parameters"]
    checks["unique_parameters"] = acc["unique_physical_parameters"]
    checks["file_bytes_equal_prediction"] = pred_size

    # 3. exact copies: every block tensor bitwise equal to its teacher source, every embedding row
    #    bitwise equal to its mapped teacher row; all finite
    id_map = load_id_map(out_dir) if (out_dir / "token_id_map.json").exists() else None
    with safe_open(str(teacher_dir / "model.safetensors"), framework="pt") as T, \
         safe_open(str(out_dir / "model.safetensors"), framework="pt") as O:
        n_equal = 0
        for new, old in enumerate(kept):
            for k in T.keys():
                m = pa.LAYER_RE.match(k)
                if m and int(m.group(1)) == old:
                    a = T.get_tensor(k)
                    b = O.get_tensor(f"model.layers.{new}.{m.group(2)}")
                    if not torch.equal(a, b):
                        raise AssertionError(f"block tensor {k} -> layer {new} not an exact copy")
                    n_equal += 1
        for k in ("model.norm.weight",):
            assert torch.equal(T.get_tensor(k), O.get_tensor(k))
        E_t, E_o = T.get_tensor("model.embed_tokens.weight"), O.get_tensor("model.embed_tokens.weight")
        if id_map is None:
            assert torch.equal(E_t, E_o)
        else:
            assert torch.equal(E_t.index_select(0, gather_order(id_map)), E_o)
        finite = all(torch.isfinite(O.get_tensor(k).float()).all().item() for k in O.keys())
        assert finite, "non-finite tensor in the built student"
    checks["exact_block_tensor_copies"] = n_equal
    checks["exact_embedding_rows"] = cfg["vocab_size"]
    checks["all_finite"] = True
    checks["data_section_sha256_reproduces"] = (
        data_section_sha256(out_dir / "model.safetensors") == man["out_DATA_SECTION_sha256"])
    assert checks["data_section_sha256_reproduces"]
    return checks


def unique_params_torch(module: torch.nn.Module) -> int:
    """Unique physical parameters of a LIVE module: dedupe by storage pointer + view geometry.
    `sum(p.numel() for p in state_dict().values())` counts a tied head twice.

    PARAMETERS ONLY. The first version also walked `module.buffers()`, which counts the rotary
    `inv_freq` (head_dim / 2 floats, recomputed from config and never saved) as if it were a
    weight: +8 on the tiny dry-run model, +64 on the real one. The tiny dry run caught it as a
    G3 failure before any real arm was built. Buffers are reported separately by reload_checks."""
    seen, n = set(), 0
    for t in module.parameters():
        key = (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()))
        if key not in seen:
            seen.add(key)
            n += t.numel()
    return n


def reload_checks(out_dir: Path, dtype=torch.bfloat16) -> dict:
    """Load through transformers exactly as a downstream user would, and check the tie survived.
    bf16 by default: the tie and the parameter counts do not depend on dtype, and host RAM does."""
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    model = Qwen3ForCausalLM.from_pretrained(str(out_dir), torch_dtype=dtype,
                                             attn_implementation="eager", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(str(out_dir), local_files_only=True, trust_remote_code=False)
    cfg = model.config
    emb, head = model.model.embed_tokens.weight, model.lm_head.weight
    res = {
        "tie_by_storage": emb.untyped_storage().data_ptr() == head.untyped_storage().data_ptr(),
        "unique_params_torch": sum(p.numel() for p in model.parameters()),  # parameters() dedupes ties
        "unique_params_by_storage": unique_params_torch(model),
        "float_buffer_elements_not_parameters": sum(b.numel() for b in model.buffers() if b.is_floating_point()),
        "naive_state_dict_sum": sum(v.numel() for v in model.state_dict().values()),
        "embedding_rows": emb.shape[0], "n_layers": len(model.model.layers),
        "tokenizer_len": len(tok), "config_vocab_size": cfg.vocab_size,
        "eos_ids": cfg.eos_token_id, "eos_decodes": tok.convert_ids_to_tokens(
            cfg.eos_token_id if isinstance(cfg.eos_token_id, list) else [cfg.eos_token_id]),
    }
    assert res["tie_by_storage"], "tie did not survive reload"
    assert res["tokenizer_len"] <= res["embedding_rows"], "tokenizer can emit ids beyond the matrix"
    del model
    return res
