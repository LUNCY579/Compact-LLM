"""Prospective label-free scalar calibration for the unchanged frozen AM map.

This module never loads a full causal LM, scores an answer or generates a token.
The v2.1/v2.2 operators remain unchanged. Registration and output writes are
exclusive; downstream use requires a committed, hash-bound completed artifact.
"""
from __future__ import annotations

from collections import Counter
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import time

from shared.guards import require_within
from shared.hashing import canonical_sha256, sha256_file, source_files, write_json
from shared.integrity import checkpoint_files, frozen_inventory, state_hash
from shared.legacy import ROOT, REPO, ARMR, GRAFT
from shared.manifests import _require_committed, git_identity, new_run, utc_now


SCHEMA = "native_control_calibration_v2_3"
DEFAULT_CONFIG = ROOT / "native_armr/configs/control_calibration_v2_3.json"
DEFAULT_MANIFEST = ROOT / "data/generated/v2/native_armr/manifest.json"
CONTROL = {"insertion_layer": 13, "seed": 7301, "std": .02, "intermediate_size": 10240}
FORMULA = "sqrt(mean_scenario_mean_ordinary_token_relative_A3_energy / mean_scenario_mean_ordinary_token_relative_M_energy)"
PROMPT_FIELDS = ("example_id", "scenario_id", "family", "template_id", "prompt",
                 "prompt_sha256", "alignment_partition")


def load_config(path=DEFAULT_CONFIG):
    path = require_within(path, ROOT / "native_armr/configs")
    if path != DEFAULT_CONFIG.resolve():
        raise ValueError("only the prospective v2.3 calibration config is supported")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": SCHEMA, "track": "native_armr", "device": "cpu",
        "dtype": "float32", "threads": 4, "seed": 17, "use_cache": False,
        "attention": "eager", "capture_layer": 13, "repeated_layers": [12, 13],
        "fit_scenarios": 96, "holdout_scenarios": 32, "max_input_tokens": 256,
        "energy_floor": 1e-8, "formula": FORMULA, "control": CONTROL,
        "label_free": True, "alpha_search": False, "alpha_clipping": False,
        "holdout_selects_alpha": False, "max_rss_gib": 6.0,
        "min_available_gib": 1.0, "max_prompt_seconds": 900,
        "max_stage_seconds": 172800,
        "checkpoint_sha256": "a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026",
    }
    if value != expected:
        raise ValueError("unregistered control calibration settings")
    return value


def prompt_projection(row):
    """Read only public prompt identity fields; never validate/consult labels."""
    if row["track"] != "native_armr" or row["split"] != "calibration" or row["variant"] != "original":
        raise ValueError("only original Track A calibration prompts are allowed")
    value = {key: row[key] for key in PROMPT_FIELDS}
    partition = "holdout" if int(value["scenario_id"].rsplit("_", 1)[1]) % 4 == 3 else "fit"
    if value["alignment_partition"] != partition:
        raise ValueError("scenario violates frozen fit/holdout boundary")
    if not value["prompt"] or canonical_sha256(value["prompt"]) != value["prompt_sha256"]:
        raise ValueError("prompt hash mismatch")
    if not value["example_id"].startswith("native_armr_calibration_"):
        raise ValueError("noncalibration identity")
    return value


def calibration_prompts(manifest_path=DEFAULT_MANIFEST):
    manifest_path = require_within(manifest_path, ROOT / "data/generated/v2/native_armr")
    if manifest_path != DEFAULT_MANIFEST.resolve():
        raise ValueError("only the existing frozen Track A manifest is allowed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest.pop("manifest_sha256", None)
    if canonical_sha256(manifest) != digest:
        raise ValueError("manifest digest mismatch")
    if (manifest["track"], manifest["status"], manifest["sealed_access"]) != ("native_armr", "FROZEN_UNSEALED", False):
        raise ValueError("wrong track or unfrozen data")
    info = manifest["datasets"]["calibration"]
    path = require_within(manifest_path.parent / info["path"], manifest_path.parent)
    if path != manifest_path.parent / "calibration.jsonl" or sha256_file(path) != info["sha256"]:
        raise ValueError("frozen calibration file mismatch")
    # Labels share the immutable JSONL container. Bytes are integrity-hashed;
    # answer/scorer/facts/oracle fields are discarded immediately after parsing.
    with path.open(encoding="utf-8") as stream:
        rows = sorted((prompt_projection(json.loads(line)) for line in stream if line.strip()),
                      key=lambda row: row["example_id"])
    if len(rows) != 128 or len({r["example_id"] for r in rows}) != 128 or len({r["scenario_id"] for r in rows}) != 128:
        raise ValueError("128 unique original calibration scenarios required")
    if Counter(r["alignment_partition"] for r in rows) != {"fit": 96, "holdout": 32}:
        raise ValueError("frozen 96/32 scenario split required")
    for key in ("family", "template_id"):
        expected = info["families" if key == "family" else "templates"]
        if dict(Counter(r[key] for r in rows)) != expected:
            raise ValueError("calibration manifest counts mismatch")
    return rows, {"dataset_manifest_path": manifest_path.relative_to(REPO).as_posix(),
                  "dataset_manifest_sha256": sha256_file(manifest_path),
                  "calibration_data_path": path.relative_to(REPO).as_posix(),
                  "calibration_data_sha256": info["sha256"]}


def calibration_sources():
    values = source_files("native_armr")
    # Native's shared inventory does not include this lazily imported helper.
    # Bind it explicitly without changing the historical shared implementation.
    for path in (GRAFT / "graft.py", ROOT / "native_armr/tests/test_control_calibration.py"):
        values[path.relative_to(REPO).as_posix()] = sha256_file(path)
    return dict(sorted(values.items()))


def _contract(config_path, manifest_path):
    config = load_config(config_path)
    rows, data = calibration_prompts(manifest_path)
    return {"schema_version": SCHEMA, "track": "native_armr", "config": config,
            "calibration_config_path": Path(config_path).resolve().relative_to(REPO).as_posix(),
            "calibration_config_sha256": sha256_file(config_path), "source_files": calibration_sources(),
            **data, "fit_example_ids": [r["example_id"] for r in rows if r["alignment_partition"] == "fit"],
            "holdout_example_ids": [r["example_id"] for r in rows if r["alignment_partition"] == "holdout"]}


def freeze_calibration(config_path=DEFAULT_CONFIG, manifest_path=DEFAULT_MANIFEST):
    contract = _contract(config_path, manifest_path)
    digest = canonical_sha256(contract)
    path = ROOT / "native_armr/evidence/00_control_calibration_registration" / digest / "registration.json"
    write_json(path, {"contract": contract, "registration_sha256": digest, "status": "FROZEN",
                      "registered_at_utc": utc_now(), "registered_against": git_identity()})
    return path


def verify_calibration_registration(config_path=DEFAULT_CONFIG, manifest_path=DEFAULT_MANIFEST):
    contract = _contract(config_path, manifest_path)
    digest = canonical_sha256(contract)
    path = ROOT / "native_armr/evidence/00_control_calibration_registration" / digest / "registration.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["status"] != "FROZEN" or value["registration_sha256"] != digest or value["contract"] != contract:
        raise RuntimeError("calibration registration mismatch")
    for relative in (*contract["source_files"], contract["calibration_config_path"],
                     contract["dataset_manifest_path"], contract["calibration_data_path"]):
        candidate = REPO / relative
        if candidate.is_relative_to(ROOT):
            _require_committed(candidate)
    _require_committed(path)
    return {**value, "registration_path": path.relative_to(REPO).as_posix()}


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((value.dtype, tuple(value.shape))).encode())
    digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def ordinary_mask(token_ids, special_ids):
    import torch
    if token_ids.ndim != 1:
        raise ValueError("one complete unpadded prompt required")
    mask = torch.ones_like(token_ids, dtype=torch.bool)
    for index in special_ids:
        mask &= token_ids != int(index)
    if not mask.any():
        raise ValueError("prompt has no ordinary tokens")
    return mask


def scenario_energies(hidden, native_delta, control_delta, ordinary):
    """Equal ordinary-token means within one scenario; diagnostics in FP64."""
    import torch
    if hidden.ndim != 2 or hidden.shape != native_delta.shape or hidden.shape != control_delta.shape:
        raise ValueError("unaligned full-prompt transformation tensors")
    if ordinary.shape != hidden.shape[:1] or ordinary.dtype != torch.bool or not ordinary.any():
        raise ValueError("ordinary token mask must be nonempty and aligned")
    if any(v.dtype != torch.float32 or v.device.type != "cpu" or not torch.isfinite(v).all()
           for v in (hidden, native_delta, control_delta)):
        raise ValueError("finite CPU FP32 inference tensors required")
    h, native, control = (v.double() for v in (hidden, native_delta, control_delta))
    denominator = h.square().mean(-1).clamp_min(1e-8)
    native_relative, control_relative = native.square().mean(-1) / denominator, control.square().mean(-1) / denominator
    metrics = {}
    masks = {"all": torch.ones_like(ordinary), "ordinary": ordinary, "special_control": ~ordinary,
             "final_prompt": torch.arange(len(ordinary)) == len(ordinary) - 1}
    for name, mask in masks.items():
        metrics[name] = None if not mask.any() else {
            "token_count": int(mask.sum()), "native_energy": float(native_relative[mask].mean()),
            "control_energy": float(control_relative[mask].mean()), "hidden_rms": float(h[mask].square().mean().sqrt()),
            "native_delta_rms": float(native[mask].square().mean().sqrt()),
            "control_delta_rms": float(control[mask].square().mean().sqrt())}
    return metrics


def aggregate_partition(records):
    if not records or len({r["scenario_id"] for r in records}) != len(records):
        raise ValueError("one original per unique scenario required")
    def mean(key):
        values = [r["metrics"]["ordinary"][key] for r in records]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("invalid calibration statistic")
        return math.fsum(values) / len(values)
    return {"scenario_ids": sorted(r["scenario_id"] for r in records),
            "example_ids": sorted(r["example_id"] for r in records), "scenarios": len(records),
            "ordinary_tokens": sum(r["metrics"]["ordinary"]["token_count"] for r in records),
            **{key: mean(key) for key in ("native_energy", "control_energy", "hidden_rms", "native_delta_rms", "control_delta_rms")}}


def fit_alpha(fit):
    native, control = fit["native_energy"], fit["control_energy"]
    if any(not math.isfinite(v) or v <= 0 for v in (native, control)):
        raise ValueError("native and control fit energies must be finite positive")
    alpha = math.sqrt(native / control)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("finite positive alpha required; no clipping or fallback")
    return alpha


def replay_native_delta(hidden, leading, trailing, *, attention_mask, position_ids, position_embeddings):
    """Reuse the original post-L13 mask/positions/RoPE for extra L12 then L13."""
    import torch
    kwargs = {"attention_mask": attention_mask, "position_ids": position_ids,
              "position_embeddings": position_embeddings, "use_cache": False}
    with torch.inference_mode():
        return trailing(leading(hidden, **kwargs), **kwargs) - hidden


def _buffer_hashes(module):
    # state_dict omits nonpersistent RoPE buffers; bind those explicitly too.
    return {name: tensor_sha256(value) for name, value in module.named_buffers()}


def _module_receipt(module, before, buffers_before=None):
    inventory = frozen_inventory(module)
    after = state_hash(module)
    buffers_after = _buffer_hashes(module)
    if buffers_before is None:
        buffers_before = buffers_after
    if before != after or buffers_before != buffers_after:
        raise RuntimeError("streamed module state mutated")
    return {**inventory, "state_sha256_before": before, "state_sha256_after": after,
            "buffer_sha256_before": buffers_before, "buffer_sha256_after": buffers_after}


def _stream_prefix(path, encoded, model_config, rope, boundary, layer_receipts):
    import torch
    from shared.hidden_capture import load_layer, positions_mask
    from rineya_cross_scale.streaming_forward import StreamingWeights
    with torch.inference_mode(), StreamingWeights(path) as weights:
        states = [weights.rows("model.embed_tokens.weight", ids[0]).unsqueeze(0) for ids in encoded]
        embeddings = [tensor_sha256(h) for h in states]
        weights.close()
        for index in range(14):
            layer = load_layer(weights, model_config, index)
            frozen_inventory(layer)
            before = state_hash(layer)
            buffers_before = _buffer_hashes(layer)
            for row, hidden in enumerate(states):
                started = time.perf_counter()
                positions, mask = positions_mask(hidden)
                value = layer(hidden, attention_mask=mask, position_ids=positions,
                              position_embeddings=rope(hidden, positions), use_cache=False)
                if value.dtype != torch.float32 or value.device.type != "cpu" or not torch.isfinite(value).all():
                    raise RuntimeError("invalid streamed raw prefix state")
                states[row] = value
                boundary(started)
            layer_receipts.append({"stage": "ordinary_prefix", "layer": index, **_module_receipt(layer, before, buffers_before)})
            del layer, hidden, value, positions, mask
            weights.close()
            gc.collect()
            boundary()
            print(f"CALIBRATION_PREFIX_LAYER={index}/13", flush=True)
    return states, embeddings


def run_calibration(config_path=DEFAULT_CONFIG, manifest_path=DEFAULT_MANIFEST, model_path=None):
    registration = verify_calibration_registration(config_path, manifest_path)
    config = registration["contract"]["config"]
    rows, data = calibration_prompts(manifest_path)
    from shared.timing import machine_lease
    with machine_lease():
        output, manifest = new_run("native_armr", "00_control_calibration", {
            "schema_version": SCHEMA, "registration_sha256": registration["registration_sha256"],
            "registration_path": registration["registration_path"], **registration["contract"]})
        report, started, peak_rss = {}, time.perf_counter(), 0
        def boundary(prompt_start=None):
            nonlocal peak_rss
            import psutil
            peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
            if peak_rss > config["max_rss_gib"] * 2**30:
                raise MemoryError("calibration RSS budget exceeded")
            now = time.perf_counter()
            if now - started > config["max_stage_seconds"] or (prompt_start is not None and now - prompt_start > config["max_prompt_seconds"]):
                raise TimeoutError("calibration elapsed budget exceeded")
        try:
            import torch
            import transformers
            from transformers import AutoConfig, AutoTokenizer
            from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding
            from shared.model_loading import resolve_model, require_available_memory
            from shared.runtime import configure_runtime, runtime_identity
            from shared.hidden_capture import load_layer, positions_mask
            from rineya_cross_scale.streaming_forward import StreamingWeights
            from armr.evaluation import encode
            from .operators import FrozenSwiGLU, matched_control_width
            if str(torch.__version__) != "2.12.1+cu126" or transformers.__version__ != "5.14.1":
                raise RuntimeError("unqualified runtime; use historical Chat-Bubble/.venv Python")
            configure_runtime(config)
            report.update(require_available_memory(config))
            path = resolve_model(model_path)
            before = checkpoint_files(path)
            reference = json.loads((ARMR / "evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/run_manifest.json").read_text())
            implementation = sha256_file(Path(inspect.getfile(Qwen3DecoderLayer)))
            if before != reference["checkpoint_files_before"] or before["model.safetensors"] != config["checkpoint_sha256"]:
                raise RuntimeError("canonical checkpoint/tokenizer/config hashes differ from reference")
            if implementation != reference["qwen_implementation_sha256"]:
                raise RuntimeError("actual HF decoder implementation changed")
            model_config = AutoConfig.from_pretrained(path, local_files_only=True)
            model_config._attn_implementation = "eager"
            if model_config.hidden_size != 1024 or model_config.num_hidden_layers != 28 or matched_control_width(model_config) != 10240:
                raise RuntimeError("unregistered Supa geometry")
            if any(model_config.layer_types[i] != "full_attention" for i in range(14)):
                raise RuntimeError("only registered full attention prefix allowed")
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
            special = sorted(set(map(int, tokenizer.all_special_ids)) | set(map(int, tokenizer.get_added_vocab().values())))
            encoded = [encode(tokenizer, row) for row in rows]
            for ids in encoded:
                if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < ids.shape[1] <= config["max_input_tokens"]:
                    raise RuntimeError("full prompt exceeds cap; truncation forbidden")
                ordinary_mask(ids[0], special)
            rope = Qwen3RotaryEmbedding(model_config).eval().requires_grad_(False)
            rope_before = state_hash(rope)
            rope_buffers_before = _buffer_hashes(rope)
            layer_receipts = []
            report.update(model_path=str(path), checkpoint_files_before=before, qwen_implementation_sha256=implementation,
                          runtime_identity=runtime_identity(config), special_control_token_ids=special,
                          layer_state_receipts=layer_receipts, model_config_sha256=sha256_file(path / "config.json"))
            boundary()
            states, embeddings = _stream_prefix(path, encoded, model_config, rope, boundary, layer_receipts)
            transform = FrozenSwiGLU(1024, 10240, seed=7301, std=.02, rms_epsilon=model_config.rms_norm_eps)
            transform_before = state_hash(transform)
            transform_buffers_before = _buffer_hashes(transform)
            frozen_inventory(transform)
            with StreamingWeights(path) as weights:
                leading, trailing = load_layer(weights, model_config, 12), load_layer(weights, model_config, 13)
            leading_before, trailing_before = state_hash(leading), state_hash(trailing)
            leading_buffers_before, trailing_buffers_before = _buffer_hashes(leading), _buffer_hashes(trailing)
            frozen_inventory(leading); frozen_inventory(trailing)
            cases = []
            for partition in ("fit", "holdout"):
                for index, row in enumerate(rows):
                    if row["alignment_partition"] != partition:
                        continue
                    prompt_start = time.perf_counter()
                    hidden, ids = states[index], encoded[index][0]
                    positions, mask = positions_mask(hidden)
                    with torch.inference_mode():
                        pos_emb = rope(hidden, positions)
                        native = replay_native_delta(hidden, leading, trailing, attention_mask=mask,
                                                     position_ids=positions, position_embeddings=pos_emb)
                        control = transform(hidden)
                    metrics = scenario_energies(hidden[0], native[0], control[0], ordinary_mask(ids, special))
                    case = {key: row[key] for key in PROMPT_FIELDS if key != "prompt"}
                    case.update(token_ids=ids.tolist(), full_prompt_tokens=len(ids), retained_prompt_tokens=len(ids),
                                embedding_rows_sha256=embeddings[index], hidden_sha256=tensor_sha256(hidden),
                                native_delta_sha256=tensor_sha256(native), control_delta_sha256=tensor_sha256(control),
                                original_positions_sha256=tensor_sha256(positions), original_mask_sha256=tensor_sha256(mask),
                                original_rope_sha256=[tensor_sha256(v) for v in pos_emb], metrics=metrics)
                    write_json(output / "cases" / (canonical_sha256(row["example_id"]) + ".json"), case)
                    cases.append(case)
                    boundary(prompt_start)
                aggregate = aggregate_partition([c for c in cases if c["alignment_partition"] == partition])
                report[partition] = aggregate
                if partition == "fit":
                    report["alpha"] = fit_alpha(aggregate)
                    # Alpha is fixed and written before evaluating held-out deltas.
                    write_json(output / "fit_scale.json", {"alpha": report["alpha"], "fit": aggregate, "formula": FORMULA})
                report[partition]["scaled_control_energy"] = report["alpha"] ** 2 * aggregate["control_energy"]
            layer_receipts.extend([{"stage": "replay", "layer": 12, **_module_receipt(leading, leading_before, leading_buffers_before)},
                                   {"stage": "replay", "layer": 13, **_module_receipt(trailing, trailing_before, trailing_buffers_before)}])
            report["rotary_state_receipt"] = _module_receipt(rope, rope_before, rope_buffers_before)
            report["control_state_receipt"] = _module_receipt(transform, transform_before, transform_buffers_before)
            report["control_state_sha256"] = transform_before
            report["checkpoint_files_after"] = checkpoint_files(path)
            if before != report["checkpoint_files_after"] or calibration_sources() != registration["contract"]["source_files"]:
                raise RuntimeError("checkpoint or registered source mutation during calibration")
            if _contract(config_path, manifest_path) != registration["contract"]:
                raise RuntimeError("calibration config/data/source contract changed")
            boundary()
            report.update(schema_version=SCHEMA, status="COMPLETE", track="native_armr", control=CONTROL,
                          formula=FORMULA, label_free=True, answer_labels_used=False, generated_tokens=0,
                          holdout_selects_alpha=False, alpha_search=False, alpha_clipping=False,
                          trainable_parameters=0, backbone_gradients=0, canonical_unchanged=True,
                          device="cpu", dtype="float32", threads=4, use_cache=False,
                          registration_path=registration["registration_path"],
                          registration_sha256=registration["registration_sha256"],
                          calibration_config_path=registration["contract"]["calibration_config_path"],
                          calibration_config_sha256=registration["contract"]["calibration_config_sha256"],
                          source_files=registration["contract"]["source_files"], **data,
                          fit_scale_sha256=sha256_file(output / "fit_scale.json"),
                          case_files={p.name: sha256_file(p) for p in sorted((output / "cases").glob("*.json"))},
                          peak_rss_bytes=peak_rss, wall_seconds=time.perf_counter() - started)
            write_json(output / "report.json", report)
            manifest.update(status="COMPLETE", report_sha256=sha256_file(output / "report.json"),
                            alpha=report["alpha"], backbone_gradients=0, canonical_unchanged=True)
        except BaseException as exc:
            manifest.update(status="FAILED", failure_receipt={"type": type(exc).__name__, "message": str(exc)})
            write_json(output / "failure_details.json", report)
            raise
        finally:
            manifest.update(end_utc=utc_now(), wall_seconds=time.perf_counter() - started, peak_rss_bytes=peak_rss)
            write_json(output / "run_manifest.json", manifest)
        return output


def verify_calibration_artifact(path, expected_sha256, *, require_committed=True):
    """Fail closed on provenance, scalar arithmetic, coverage or state mutation."""
    path = require_within(path, ROOT / "native_armr/evidence/00_control_calibration")
    if path.name != "report.json" or sha256_file(path) != expected_sha256:
        raise RuntimeError("calibration artifact hash/path mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version": SCHEMA, "status": "COMPLETE", "track": "native_armr", "control": CONTROL,
                "formula": FORMULA, "label_free": True, "answer_labels_used": False, "generated_tokens": 0,
                "holdout_selects_alpha": False, "alpha_search": False, "alpha_clipping": False,
                "trainable_parameters": 0, "backbone_gradients": 0, "canonical_unchanged": True,
                "device": "cpu", "dtype": "float32", "threads": 4, "use_cache": False}
    if any(report.get(key) != value for key, value in required.items()):
        raise RuntimeError("invalid calibration science/freeze contract")
    if report["alpha"] != fit_alpha(report["fit"]):
        raise RuntimeError("alpha differs from registered fit-only formula")
    registration_path = require_within(REPO / report["registration_path"], ROOT / "native_armr/evidence/00_control_calibration_registration")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    contract = registration["contract"]
    if registration["status"] != "FROZEN" or canonical_sha256(contract) != report["registration_sha256"] or registration["registration_sha256"] != report["registration_sha256"]:
        raise RuntimeError("calibration registration digest mismatch")
    if report["source_files"] != calibration_sources():
        raise RuntimeError("calibration source inventory changed")
    for key in ("source_files", "calibration_config_path", "calibration_config_sha256", "dataset_manifest_path",
                "dataset_manifest_sha256", "calibration_data_path", "calibration_data_sha256"):
        if report[key] != contract[key]:
            raise RuntimeError("calibration artifact/registration binding mismatch")
    config_path = require_within(REPO / report["calibration_config_path"], ROOT / "native_armr/configs")
    if load_config(config_path) != contract["config"]:
        raise RuntimeError("calibration configuration changed")
    bound_files = dict(report["source_files"])
    bound_files.update({report[key + "_path"]: report[key + "_sha256"] for key in
                       ("calibration_config", "dataset_manifest", "calibration_data")})
    for relative, digest in bound_files.items():
        candidate = require_within(REPO / relative, REPO)
        if sha256_file(candidate) != digest:
            raise RuntimeError("registered calibration source/config/data changed")
        if require_committed and candidate.is_relative_to(ROOT):
            _require_committed(candidate)
    rows, _ = calibration_prompts(REPO / report["dataset_manifest_path"])
    cases = []
    if set(report["case_files"]) != {canonical_sha256(r["example_id"]) + ".json" for r in rows}:
        raise RuntimeError("calibration case coverage mismatch")
    for filename, digest in report["case_files"].items():
        case_path = require_within(path.parent / "cases" / filename, path.parent / "cases")
        if sha256_file(case_path) != digest:
            raise RuntimeError("calibration case hash mismatch")
        cases.append(json.loads(case_path.read_text(encoding="utf-8")))
        if require_committed:
            _require_committed(case_path)
    by_id = {case["example_id"]: case for case in cases}
    for row in rows:
        case = by_id[row["example_id"]]
        if any(case[key] != row[key] for key in PROMPT_FIELDS if key != "prompt"):
            raise RuntimeError("case differs from frozen calibration identity")
        tokens = case["token_ids"]
        count = sum(i not in report["special_control_token_ids"] for i in tokens)
        if not 0 < len(tokens) <= 256 or count <= 0 or case["full_prompt_tokens"] != len(tokens) or case["retained_prompt_tokens"] != len(tokens) or case["metrics"]["ordinary"]["token_count"] != count:
            raise RuntimeError("calibration token exclusion/length mismatch")
    for partition, count in (("fit", 96), ("holdout", 32)):
        reconstructed = aggregate_partition([c for c in cases if c["alignment_partition"] == partition])
        reconstructed["scaled_control_energy"] = report["alpha"] ** 2 * reconstructed["control_energy"]
        if reconstructed != report[partition] or reconstructed["scenarios"] != count or reconstructed["example_ids"] != contract[partition + "_example_ids"]:
            raise RuntimeError("calibration scenario aggregation mismatch")
    if set(report["fit"]["scenario_ids"]) & set(report["holdout"]["scenario_ids"]):
        raise RuntimeError("fit/holdout leakage")
    states = report["layer_state_receipts"] + [report["rotary_state_receipt"], report["control_state_receipt"]]
    if [(r["stage"], r["layer"]) for r in report["layer_state_receipts"]] != [("ordinary_prefix", i) for i in range(14)] + [("replay", 12), ("replay", 13)]:
        raise RuntimeError("streamed layer coverage mismatch")
    for item in states:
        if item["trainable_parameters"] != 0 or item["backbone_gradients"] != 0 or not item["state_sha256_before"] or item["state_sha256_before"] != item["state_sha256_after"] or item["buffer_sha256_before"] != item["buffer_sha256_after"]:
            raise RuntimeError("streamed/control state mutated or has gradients")
    for index in (12, 13):
        ordinary = report["layer_state_receipts"][index]
        replay = report["layer_state_receipts"][index + 2]
        if ordinary["state_sha256_before"] != replay["state_sha256_before"] or ordinary["buffer_sha256_before"] != replay["buffer_sha256_before"]:
            raise RuntimeError("replayed layer differs from ordinary checkpoint layer")
    if report["control_state_sha256"] != report["control_state_receipt"]["state_sha256_before"] or report["control_state_receipt"]["parameters"] != 31457280:
        raise RuntimeError("control state identity mismatch")
    if report["checkpoint_files_before"] != report["checkpoint_files_after"] or report["checkpoint_files_before"].get("model.safetensors") != contract["config"]["checkpoint_sha256"]:
        raise RuntimeError("canonical checkpoint mutation")
    fit_path, run_path = path.parent / "fit_scale.json", path.parent / "run_manifest.json"
    if sha256_file(fit_path) != report["fit_scale_sha256"]:
        raise RuntimeError("prospective fit scale receipt mismatch")
    scale = json.loads(fit_path.read_text(encoding="utf-8"))
    if scale != {"alpha": report["alpha"], "formula": FORMULA,
                 "fit": {key: value for key, value in report["fit"].items() if key != "scaled_control_energy"}}:
        raise RuntimeError("fit scale was not frozen at registered value")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run["status"] != "COMPLETE" or run["report_sha256"] != expected_sha256 or run["registration_sha256"] != report["registration_sha256"]:
        raise RuntimeError("completed calibration run manifest required")
    if require_committed:
        for candidate in (path, registration_path, fit_path, run_path, path.parent / "started_manifest.json"):
            _require_committed(candidate)
    return report
