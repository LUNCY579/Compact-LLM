"""Qualified official 4B donor extraction and hash-bound activation bundles."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time

import torch
from transformers import AutoConfig, AutoTokenizer

from shared.hashing import canonical_sha256, sha256_file, source_files
from shared.legacy import legacy_graft
from shared.guards import guard_non_sealed_path
from .delta_alignment import deadline, frozen

REPO = Path(__file__).resolve().parents[4]
DONOR_RECORD = REPO / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/cross_scale_evidence/02_architecture/qwen4b.json"


def donor_identity(config):
    from armr.provenance import checkpoint_files
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
    import inspect
    record = json.loads(DONOR_RECORD.read_text(encoding="utf-8"))
    if record["revision"] != config["donor_revision"] or record["source_id"] != "Qwen/Qwen3-4B":
        raise ValueError("only the preregistered Qwen3-4B revision is eligible")
    path = guard_non_sealed_path(Path(record["resolved_path"]))
    entries = record["weights"]
    entries = entries if isinstance(entries, list) else entries["files"]
    expected = {r["name"]: r["sha256"] for r in entries}
    expected["config.json"] = record["config"]["sha256"]
    actual = {name: sha256_file(path/name) for name in expected}
    if actual != expected:
        raise ValueError("donor checkpoint hash mismatch")
    return path, {"revision": record["revision"], "files": actual,
                  "record_sha256": sha256_file(DONOR_RECORD), "checkpoint": str(path),
                  "all_checkpoint_files": checkpoint_files(path),
                  "qwen_implementation_sha256": sha256_file(Path(inspect.getfile(Qwen3DecoderLayer)))}


def load_donor_layer(config, index, path=None):
    from rineya_cross_scale.streaming_forward import StreamingWeights
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    if index not in config["donor_layers"]:
        raise ValueError("unregistered donor layer")
    if path is None:
        path, _ = donor_identity(config)
    architecture = AutoConfig.from_pretrained(path, local_files_only=True)
    if architecture.num_hidden_layers != 36 or architecture.hidden_size != 2560:
        raise ValueError("unexpected donor architecture")
    architecture._attn_implementation = "eager"
    with StreamingWeights(path) as weights:
        layer = legacy_graft().load_layer(weights, architecture, index)
    if layer.self_attn.layer_idx != index:
        raise ValueError("loaded donor layer index mismatch")
    frozen(layer)
    return layer, Qwen3RotaryEmbedding(architecture).eval().requires_grad_(False), architecture


def strip_calibration_rows(rows):
    # Answer/scorer fields never enter activation cache or fitting APIs.
    retained = []
    for row in rows:
        if row.get("track") != "cross_scale" or row.get("split") != "calibration":
            raise ValueError("Track B capture requires its own calibration role")
        retained.append({key: row[key] for key in ("example_id", "scenario_id", "family", "prompt")})
    if len({r["example_id"] for r in retained}) != len(retained):
        raise ValueError("duplicate calibration IDs")
    return retained


def capture_bundle(rows, config):
    from shared.model_loading import load_supa
    from shared.hidden_capture import capture_native
    from armr.provenance import state_hash, checkpoint_files
    legacy = legacy_graft()
    started = time.perf_counter()
    source_before = source_files("cross_scale")
    if any(row.get("alignment_partition") not in ("fit", "holdout") for row in rows):
        raise ValueError("calibration manifest must freeze fit/holdout partitions")
    train = [row["alignment_partition"] == "fit" for row in rows]
    rows = strip_calibration_rows(rows)
    if not rows or not any(train) or all(train):
        raise ValueError("capture needs both training and held-out calibration prompts")
    heldout = sorted({row["scenario_id"] for row, keep in zip(rows, train) if not keep})
    if set(heldout) & {row["scenario_id"] for row, keep in zip(rows, train) if keep}:
        raise ValueError("scenario leakage across alignment partitions")
    donor_path, donor_before = donor_identity(config)
    model, tokenizer, identity = load_supa(config)
    # TorchVersion is a str subclass that weights_only=True deliberately rejects
    # when pickled. Restrict cache metadata to plain JSON primitives.
    identity = json.loads(json.dumps(identity, allow_nan=False))
    supa_path = Path(identity["model_path"])
    if not supa_path.is_dir():
        from shared.model_loading import resolve_model
        supa_path = resolve_model()
    supa_before = checkpoint_files(supa_path)
    state_before = state_hash(model)
    donor_tokenizer = AutoTokenizer.from_pretrained(donor_path, local_files_only=True)
    if tokenizer.get_vocab() != donor_tokenizer.get_vocab():
        raise ValueError("donor and Supa token vocabularies differ")
    encoded, lengths = [], []
    for row in rows:
        text = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
             tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        if not torch.equal(ids, donor_tokenizer(text, return_tensors="pt")["input_ids"]):
            raise ValueError("token alignment mismatch")
        if ids.numel() > config["calibration_max_tokens"]:
            raise ValueError("calibration prompt exceeds frozen token budget; truncation would change final-prompt metrics")
        lengths.append({"example_id": row["example_id"], "full_tokens": ids.numel(),
                        "retained_tokens": min(ids.numel(), config["calibration_max_tokens"])})
        encoded.append(ids)
    special_ids = sorted(set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values()))
    # Qualify raw streamed Supa states against canonical custom Supa at every requested site.
    fixture = {}
    def capture_fixture(module, args, out, *, index):
        if not isinstance(out, torch.Tensor) or out.ndim != 3 or out.shape[0] != 1:
            raise TypeError("only qualified tensor-returning Qwen3 decoders are supported")
        fixture[index] = out[0].clone()
    handles = [model.model.layers[i].register_forward_hook(
        lambda module, args, out, index=i: capture_fixture(module, args, out, index=index))
        for i in config["supa_layers"]]
    try:
        fixture_started = time.perf_counter()
        with torch.inference_mode():
            model(encoded[0], use_cache=False, logits_to_keep=1)
        deadline(fixture_started, config["max_prompt_seconds"])
    finally:
        for handle in handles:
            handle.remove()
    if state_hash(model) != state_before or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError("canonical Supa mutation or unexpected gradient")
    del model, donor_tokenizer
    gc.collect()
    def capture_progress(index, final):
        deadline(started, config["max_capture_seconds"])
        legacy.memory_guard(config["max_rss_gib"])
        print(f"Capture layer {index}/{final} complete", flush=True)
    capture_progress(-1, max(config["supa_layers"]))
    small = capture_native(supa_path, encoded, tuple(config["supa_layers"]), config["max_rss_gib"],
                           progress=capture_progress)
    for index in config["supa_layers"]:
        if not torch.equal(fixture[index], small[index][0]):
            raise RuntimeError("streamed/canonical Supa raw-state qualification failed")
    donor_depths = sorted({depth for j in config["donor_layers"] for depth in (j-1, j)})
    big = capture_native(donor_path, encoded, tuple(donor_depths), config["max_rss_gib"],
                         progress=capture_progress)
    _, donor_after = donor_identity(config)
    if donor_before != donor_after or supa_before != checkpoint_files(supa_path):
        raise RuntimeError("checkpoint changed during capture")
    source_after = source_files("cross_scale")
    if source_before != source_after:
        raise RuntimeError("capture source changed during execution")
    deadline(started, config["max_capture_seconds"])
    return {"schema_version": 1, "track": "cross_scale", "role": "calibration",
            "rows": rows, "input_ids": encoded, "special_ids": special_ids,
            "lengths": lengths, "train_prompt_mask": train, "heldout_scenarios": heldout,
            "small": small, "donor": big, "supa_identity": identity,
            "supa_files": supa_before, "donor_identity": donor_before,
            "config_sha256": canonical_sha256(config),
            "source_sha256": canonical_sha256(source_before), "sources": source_before,
            "canonical_fixture_bit_identical": True}
