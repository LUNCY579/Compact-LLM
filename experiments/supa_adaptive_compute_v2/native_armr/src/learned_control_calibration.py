"""Prospective prompt-only energy calibration of the v2.4 pretrained FFN control.

The earlier calibration's prompt boundary, FP64 energy arithmetic, streamed
prefix and frozen-state helpers are reused unchanged. No full LM/head, generated
token, answer, scorer, efficacy data or optimizer participates in this stage.
"""
from __future__ import annotations

import gc
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
from .control_calibration import (
    DEFAULT_MANIFEST, FORMULA, PROMPT_FIELDS, prompt_projection,
    calibration_prompts, tensor_sha256, ordinary_mask, scenario_energies,
    aggregate_partition, fit_alpha, replay_native_delta, _buffer_hashes,
    _module_receipt, _stream_prefix,
)


SCHEMA = "native_learned_control_calibration_v2_4"
DEFAULT_CONFIG = ROOT / "native_armr/configs/learned_control_calibration_v2_4.json"
STAGE = "00_learned_control_calibration"
REGISTRATION_STAGE = "00_learned_control_calibration_registration"
CONTROL = {"kind": "pretrained_parallel_ffn", "insertion_layer": 13,
           "source_layers": [12, 13, 14], "summation_order": "(F12+F13)+F14",
           "local_frozen_copies": True}
CHECKPOINT_SHA256 = "a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026"
MODEL_CONFIG_SHA256 = "89c4ae7398449895d044750c858614584b07db8d67130074920ef524c6eb5705"
HF_SHA256 = "fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab"
PARAMETER_COUNTS = {"matrix_parameters": 28311552, "norm_parameters": 3072,
                    "resident_copied_parameters": 28314624,
                    "active_copied_parameters": 28314624,
                    "resident_bytes": 113258496, "newly_learned_parameters": 0,
                    "extra_matmul_flops_per_token": 56623104}
RUNTIME = {"device": "cpu", "dtype": "float32", "threads": 4, "seed": 17,
           "torch": "2.12.1+cu126", "transformers": "5.14.1",
           "attention": "eager", "use_cache": False}


def expected_config():
    return {"schema_version": SCHEMA, "track": "native_armr", "device": "cpu",
            "dtype": "float32", "threads": 4, "seed": 17, "use_cache": False,
            "attention": "eager", "capture_layer": 13, "repeated_layers": [12, 13],
            "fit_scenarios": 96, "holdout_scenarios": 32, "max_input_tokens": 256,
            "energy_floor": 1e-8, "formula": FORMULA, "control": CONTROL,
            "parameter_counts": PARAMETER_COUNTS, "label_free": True,
            "reused_engineering_calibration": True, "alpha_search": False,
            "alpha_clipping": False, "holdout_selects_alpha": False,
            "max_rss_gib": 6.0, "min_available_gib": 1.0,
            "max_prompt_seconds": 900, "max_stage_seconds": 172800,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "model_config_sha256": MODEL_CONFIG_SHA256,
            "qwen_implementation_sha256": HF_SHA256}


def load_config(path=DEFAULT_CONFIG):
    path = require_within(path, ROOT / "native_armr/configs")
    if path != DEFAULT_CONFIG.resolve():
        raise ValueError("only the prospective v2.4 learned-control calibration config is supported")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value != expected_config():
        raise ValueError("unregistered learned-control calibration settings")
    return value


def calibration_sources():
    values = source_files("native_armr")
    # The preserved graft helper supplies streamed actual HF layers. Tests are
    # explicit dependencies because source_files intentionally omits test trees.
    for path in (GRAFT / "graft.py", ROOT / "native_armr/tests/test_learned_control_calibration.py"):
        values[path.relative_to(REPO).as_posix()] = sha256_file(path)
    return dict(sorted(values.items()))


def _contract(config_path, manifest_path):
    config = load_config(config_path)
    rows, data = calibration_prompts(manifest_path)
    return {"schema_version": SCHEMA, "track": "native_armr", "config": config,
            "calibration_config_path": Path(config_path).resolve().relative_to(REPO).as_posix(),
            "calibration_config_sha256": sha256_file(config_path), "source_files": calibration_sources(),
            **data, **{partition + "_example_ids": [r["example_id"] for r in rows
                    if r["alignment_partition"] == partition] for partition in ("fit", "holdout")}}


def freeze_calibration(config_path=DEFAULT_CONFIG, manifest_path=DEFAULT_MANIFEST):
    contract = _contract(config_path, manifest_path)
    digest = canonical_sha256(contract)
    path = ROOT / "native_armr/evidence" / REGISTRATION_STAGE / digest / "registration.json"
    write_json(path, {"contract": contract, "registration_sha256": digest, "status": "FROZEN",
                      "registered_at_utc": utc_now(), "registered_against": git_identity()})
    return path


def verify_calibration_registration(config_path=DEFAULT_CONFIG, manifest_path=DEFAULT_MANIFEST):
    contract = _contract(config_path, manifest_path)
    digest = canonical_sha256(contract)
    path = ROOT / "native_armr/evidence" / REGISTRATION_STAGE / digest / "registration.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["status"] != "FROZEN" or value["registration_sha256"] != digest or value["contract"] != contract:
        raise RuntimeError("learned-control calibration registration mismatch")
    for relative in (*contract["source_files"], contract["calibration_config_path"],
                     contract["dataset_manifest_path"], contract["calibration_data_path"]):
        candidate = require_within(REPO / relative, REPO)
        if candidate.is_relative_to(ROOT):
            _require_committed(candidate)
    _require_committed(path)
    return {**value, "registration_path": path.relative_to(REPO).as_posix()}


def _branch_before(branch):
    return {name: (state_hash(getattr(branch, name)), _buffer_hashes(getattr(branch, name)))
            for name in ("mlp", "post_attention_layernorm")}


def _branch_receipt(index, branch, before):
    return {"source_layer": index, **{name: _module_receipt(getattr(branch, name), *before[name])
                                    for name in ("mlp", "post_attention_layernorm")}}


def _required_report():
    return {"schema_version": SCHEMA, "status": "COMPLETE", "track": "native_armr",
            "control": CONTROL, "formula": FORMULA, "label_free": True,
            "answer_labels_used": False, "generated_tokens": 0,
            "reused_engineering_calibration": True, "holdout_selects_alpha": False,
            "alpha_search": False, "alpha_clipping": False, "trainable_parameters": 0,
            "backbone_gradients": 0, "canonical_unchanged": True,
            "device": "cpu", "dtype": "float32", "threads": 4, "seed": 17,
            "attention": "eager", "use_cache": False}


def run_calibration(config_path=DEFAULT_CONFIG, manifest_path=DEFAULT_MANIFEST, model_path=None):
    registration = verify_calibration_registration(config_path, manifest_path)
    config = registration["contract"]["config"]
    rows, data = calibration_prompts(manifest_path)
    from shared.timing import machine_lease
    with machine_lease():
        output, manifest = new_run("native_armr", STAGE, {
            **registration["contract"], "registration_sha256": registration["registration_sha256"],
            "registration_path": registration["registration_path"]})
        report, started, peak_rss = {}, time.perf_counter(), 0

        def boundary(prompt_start=None):
            nonlocal peak_rss
            import psutil
            peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
            if peak_rss > config["max_rss_gib"] * 2**30:
                raise MemoryError("learned-control calibration RSS budget exceeded")
            now = time.perf_counter()
            if now - started > config["max_stage_seconds"] or (prompt_start is not None and now - prompt_start > config["max_prompt_seconds"]):
                raise TimeoutError("learned-control calibration elapsed budget exceeded")

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
            from .learned_control import FrozenParallelFFN
            if str(torch.__version__) != RUNTIME["torch"] or transformers.__version__ != RUNTIME["transformers"]:
                raise RuntimeError("unqualified runtime; use historical Chat-Bubble/.venv Python")
            configure_runtime(config)
            report.update(require_available_memory(config))
            path = resolve_model(model_path)
            before = checkpoint_files(path)
            reference = json.loads((ARMR / "evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/run_manifest.json").read_text())
            implementation = sha256_file(Path(inspect.getfile(Qwen3DecoderLayer)))
            if before != reference["checkpoint_files_before"] or before.get("model.safetensors") != CHECKPOINT_SHA256:
                raise RuntimeError("canonical checkpoint/tokenizer/config hashes differ from reference")
            if implementation != HF_SHA256 or sha256_file(path / "config.json") != MODEL_CONFIG_SHA256:
                raise RuntimeError("canonical model configuration or actual HF implementation changed")
            model_config = AutoConfig.from_pretrained(path, local_files_only=True)
            model_config._attn_implementation = "eager"
            if (model_config.hidden_size, model_config.intermediate_size, model_config.num_hidden_layers) != (1024, 3072, 28):
                raise RuntimeError("unregistered Supa geometry")
            if any(model_config.layer_types[i] != "full_attention" for i in range(15)):
                raise RuntimeError("only registered full attention layers allowed")
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
            special = sorted(set(map(int, tokenizer.all_special_ids)) | set(map(int, tokenizer.get_added_vocab().values())))
            encoded = [encode(tokenizer, row) for row in rows]
            for ids in encoded:
                if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < ids.shape[1] <= config["max_input_tokens"]:
                    raise RuntimeError("full prompt exceeds cap; truncation forbidden")
                ordinary_mask(ids[0], special)
            rope = Qwen3RotaryEmbedding(model_config).eval().requires_grad_(False)
            rope_before, rope_buffers_before = state_hash(rope), _buffer_hashes(rope)
            layer_receipts = []
            report.update(model_path=str(path), checkpoint_files_before=before, qwen_implementation_sha256=implementation,
                          runtime_identity=runtime_identity(config), special_control_token_ids=special,
                          layer_state_receipts=layer_receipts, model_config_sha256=MODEL_CONFIG_SHA256,
                          norm_epsilon=float(model_config.rms_norm_eps))
            boundary()
            states, embeddings = _stream_prefix(path, encoded, model_config, rope, boundary, layer_receipts)
            with StreamingWeights(path) as weights:
                sources = {i: load_layer(weights, model_config, i) for i in (12, 13, 14)}
            source_before = {i: _branch_before(layer) for i, layer in sources.items()}
            decoder_before = {i: (state_hash(layer), _buffer_hashes(layer)) for i, layer in sources.items()}
            for layer in sources.values():
                frozen_inventory(layer)
            del layer
            transform = FrozenParallelFFN.from_layers(sources)
            transform_before, transform_buffers_before = state_hash(transform), _buffer_hashes(transform)
            if transform_before != transform.state_sha256:
                raise RuntimeError("control composite creation identity mismatch")
            for key, value in PARAMETER_COUNTS.items():
                if transform.parameter_inventory.get(key) != value:
                    raise RuntimeError("unregistered control parameter accounting")
            copies_before = {i: _branch_before(transform.branches[str(i)]) for i in (12, 13, 14)}
            report.update(branch_provenance=transform.branch_provenance, source_hashes=transform.source_hashes,
                          copy_hashes=transform.copy_hashes, parameter_inventory=transform.parameter_inventory)
            # L14 supplies copied pretrained features only; it is never applied
            # as a decoder to calibration states. Release this extra source layer.
            source_receipts = {"14": _branch_receipt(14, sources[14], source_before[14])}
            source14_receipt = {"stage": "control_source", "layer": 14,
                                **_module_receipt(sources[14], *decoder_before[14])}
            del sources[14]
            gc.collect()
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
                        native = replay_native_delta(hidden, sources[12], sources[13], attention_mask=mask,
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
                    write_json(output / "fit_scale.json", {"alpha": report["alpha"], "fit": aggregate, "formula": FORMULA})
                    print("LEARNED_CONTROL_FIT_SCALE_FROZEN", flush=True)
                report[partition]["scaled_control_energy"] = report["alpha"] ** 2 * aggregate["control_energy"]
            layer_receipts.extend([{"stage": "replay", "layer": i, **_module_receipt(sources[i], *decoder_before[i])}
                                   for i in (12, 13)])
            layer_receipts.append(source14_receipt)
            source_receipts.update({str(i): _branch_receipt(i, sources[i], source_before[i]) for i in (12, 13)})
            report["source_branch_receipts"] = source_receipts
            report["copy_branch_receipts"] = {str(i): _branch_receipt(i, transform.branches[str(i)], copies_before[i])
                                              for i in (12, 13, 14)}
            transform.assert_unchanged()
            report["rotary_state_receipt"] = _module_receipt(rope, rope_before, rope_buffers_before)
            report["control_state_receipt"] = _module_receipt(transform, transform_before, transform_buffers_before)
            report["control_state_sha256"] = transform_before
            report["checkpoint_files_after"] = checkpoint_files(path)
            if before != report["checkpoint_files_after"] or _contract(config_path, manifest_path) != registration["contract"]:
                raise RuntimeError("checkpoint or registered calibration contract mutated")
            boundary()
            report.update(**_required_report(), registration_path=registration["registration_path"],
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


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _verify_state(item):
    if (item["trainable_parameters"] != 0 or item["backbone_gradients"] != 0
            or not _is_sha256(item["state_sha256_before"])
            or item["state_sha256_before"] != item["state_sha256_after"]
            or item["buffer_sha256_before"] != item["buffer_sha256_after"]
            or any(not _is_sha256(v) for v in item["buffer_sha256_before"].values())):
        raise RuntimeError("streamed/control state mutated or has gradients")


def _verify_case(case, row, special):
    if any(case[key] != row[key] for key in PROMPT_FIELDS if key != "prompt"):
        raise RuntimeError("case differs from frozen calibration identity")
    tokens = case["token_ids"]
    if any(type(i) is not int or i < 0 for i in tokens):
        raise RuntimeError("invalid recorded token IDs")
    ordinary = sum(i not in special for i in tokens)
    if (not 0 < len(tokens) <= 256 or ordinary <= 0 or case["full_prompt_tokens"] != len(tokens)
            or case["retained_prompt_tokens"] != len(tokens)):
        raise RuntimeError("calibration token exclusion/length mismatch")
    counts = {"all": len(tokens), "ordinary": ordinary, "special_control": len(tokens) - ordinary,
              "final_prompt": 1}
    if set(case["metrics"]) != set(counts):
        raise RuntimeError("calibration token-stratum coverage mismatch")
    for name, count in counts.items():
        metrics = case["metrics"][name]
        if count == 0:
            if metrics is not None:
                raise RuntimeError("empty token stratum must be explicit")
        elif (metrics is None or metrics["token_count"] != count
              or set(metrics) != {"token_count", "native_energy", "control_energy", "hidden_rms", "native_delta_rms", "control_delta_rms"}
              or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in metrics.values())):
            raise RuntimeError("invalid calibration token-stratum metrics")
    for key in ("embedding_rows_sha256", "hidden_sha256", "native_delta_sha256", "control_delta_sha256",
                "original_positions_sha256", "original_mask_sha256"):
        if not _is_sha256(case[key]):
            raise RuntimeError("calibration tensor identity missing")
    if len(case["original_rope_sha256"]) != 2 or any(not _is_sha256(v) for v in case["original_rope_sha256"]):
        raise RuntimeError("original rotary identity missing")


def _verify_branches(report):
    keys = {"12", "13", "14"}
    for name in ("branch_provenance", "source_hashes", "copy_hashes", "source_branch_receipts", "copy_branch_receipts"):
        if set(report[name]) != keys:
            raise RuntimeError("control branch provenance coverage mismatch")
    epsilon = report["norm_epsilon"]
    if epsilon != 1e-6:
        raise RuntimeError("invalid learned norm epsilon")
    for key in sorted(keys):
        provenance = report["branch_provenance"][key]
        expected_fields = {"source_layer", "source_mlp_sha256", "copy_mlp_sha256", "source_norm_sha256",
                           "copy_norm_sha256", "norm_epsilon", "hidden_size", "intermediate_size",
                           "mlp_class", "norm_class", "activation_class"}
        if (set(provenance) != expected_fields or provenance["source_layer"] != int(key)
                or provenance["hidden_size"] != 1024 or provenance["intermediate_size"] != 3072
                or provenance["norm_epsilon"] != epsilon or provenance["mlp_class"] != "Qwen3MLP"
                or provenance["norm_class"] != "Qwen3RMSNorm" or provenance["activation_class"] != "SiLUActivation"):
            raise RuntimeError("unregistered control branch geometry or norm")
        for receipt_kind in ("source_branch_receipts", "copy_branch_receipts"):
            if report[receipt_kind][key]["source_layer"] != int(key):
                raise RuntimeError("control source/copy layer identity mismatch")
        for module_name, short, parameters in (("mlp", "mlp", 9437184), ("post_attention_layernorm", "norm", 1024)):
            source = report["source_branch_receipts"][key][module_name]
            copied = report["copy_branch_receipts"][key][module_name]
            _verify_state(source)
            _verify_state(copied)
            hashes = [source["state_sha256_before"], copied["state_sha256_before"],
                      report["source_hashes"][key][module_name], report["copy_hashes"][key][module_name],
                      provenance["source_" + short + "_sha256"], provenance["copy_" + short + "_sha256"]]
            if (len(set(hashes)) != 1 or source["parameters"] != parameters or copied["parameters"] != parameters
                    or source["buffer_sha256_before"] != copied["buffer_sha256_before"]):
                raise RuntimeError("control source/copy weights or buffers differ")
    if report["parameter_inventory"] != PARAMETER_COUNTS:
        raise RuntimeError("control parameter/FLOP inventory mismatch")


def verify_calibration_artifact(path, expected_sha256, *, require_committed=True):
    """Reconstruct fit-only scalar and validate complete, immutable provenance.

    This audits recorded computations; it does not rerun the model or claim an
    independent numerical reproduction from hashes alone.
    """
    path = require_within(path, ROOT / "native_armr/evidence" / STAGE)
    if path.name != "report.json" or sha256_file(path) != expected_sha256:
        raise RuntimeError("calibration artifact hash/path mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    if any(report.get(key) != value for key, value in _required_report().items()):
        raise RuntimeError("invalid learned-control calibration science/freeze contract")
    if report["alpha"] != fit_alpha(report["fit"]):
        raise RuntimeError("alpha differs from registered fit-only formula")
    registration_path = require_within(REPO / report["registration_path"], ROOT / "native_armr/evidence" / REGISTRATION_STAGE)
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    contract = registration["contract"]
    if (registration["status"] != "FROZEN" or canonical_sha256(contract) != report["registration_sha256"]
            or registration["registration_sha256"] != report["registration_sha256"]):
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
    if contract != _contract(config_path, REPO / report["dataset_manifest_path"]):
        raise RuntimeError("calibration registration differs from current exact contract")
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
    expected_cases = {canonical_sha256(r["example_id"]) + ".json" for r in rows}
    if set(report["case_files"]) != expected_cases or {p.name for p in (path.parent / "cases").glob("*.json")} != expected_cases:
        raise RuntimeError("calibration case coverage mismatch")
    cases = []
    for filename, digest in report["case_files"].items():
        case_path = require_within(path.parent / "cases" / filename, path.parent / "cases")
        if sha256_file(case_path) != digest:
            raise RuntimeError("calibration case hash mismatch")
        case = json.loads(case_path.read_text(encoding="utf-8"))
        if filename != canonical_sha256(case["example_id"]) + ".json":
            raise RuntimeError("calibration case filename/identity mismatch")
        cases.append(case)
        if require_committed:
            _require_committed(case_path)
    special = report["special_control_token_ids"]
    if special != sorted(set(special)) or any(type(i) is not int or i < 0 for i in special):
        raise RuntimeError("invalid special/control token inventory")
    by_id = {case["example_id"]: case for case in cases}
    for row in rows:
        _verify_case(by_id[row["example_id"]], row, special)
    for partition, count in (("fit", 96), ("holdout", 32)):
        reconstructed = aggregate_partition([c for c in cases if c["alignment_partition"] == partition])
        reconstructed["scaled_control_energy"] = report["alpha"] ** 2 * reconstructed["control_energy"]
        if reconstructed != report[partition] or reconstructed["scenarios"] != count or reconstructed["example_ids"] != contract[partition + "_example_ids"]:
            raise RuntimeError("calibration scenario aggregation mismatch")
    if set(report["fit"]["scenario_ids"]) & set(report["holdout"]["scenario_ids"]):
        raise RuntimeError("fit/holdout leakage")
    layers = report["layer_state_receipts"]
    if [(r["stage"], r["layer"]) for r in layers] != ([('ordinary_prefix', i) for i in range(14)]
            + [("replay", 12), ("replay", 13), ("control_source", 14)]):
        raise RuntimeError("streamed layer coverage mismatch")
    for item in layers + [report["rotary_state_receipt"], report["control_state_receipt"]]:
        _verify_state(item)
    for index in (12, 13):
        ordinary, replay = layers[index], layers[index + 2]
        if (ordinary["state_sha256_before"] != replay["state_sha256_before"]
                or ordinary["buffer_sha256_before"] != replay["buffer_sha256_before"]):
            raise RuntimeError("replayed layer differs from ordinary checkpoint layer")
    _verify_branches(report)
    if (report["control_state_sha256"] != report["control_state_receipt"]["state_sha256_before"]
            or report["control_state_receipt"]["parameters"] != PARAMETER_COUNTS["resident_copied_parameters"]):
        raise RuntimeError("control composite state identity mismatch")
    reference = json.loads((ARMR / "evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/run_manifest.json").read_text())
    if (report["checkpoint_files_before"] != report["checkpoint_files_after"]
            or report["checkpoint_files_before"] != reference["checkpoint_files_before"]
            or report["checkpoint_files_before"].get("model.safetensors") != CHECKPOINT_SHA256):
        raise RuntimeError("canonical checkpoint mutation or identity mismatch")
    if (report["qwen_implementation_sha256"] != HF_SHA256 or report["model_config_sha256"] != MODEL_CONFIG_SHA256
            or report["runtime_identity"] != RUNTIME):
        raise RuntimeError("unqualified calibration runtime/model configuration")
    fit_path, run_path, started_path = (path.parent / name for name in ("fit_scale.json", "run_manifest.json", "started_manifest.json"))
    if sha256_file(fit_path) != report["fit_scale_sha256"]:
        raise RuntimeError("prospective fit scale receipt mismatch")
    scale = json.loads(fit_path.read_text(encoding="utf-8"))
    if scale != {"alpha": report["alpha"], "formula": FORMULA,
                 "fit": {key: value for key, value in report["fit"].items() if key != "scaled_control_energy"}}:
        raise RuntimeError("fit scale was not frozen at registered value")
    run, start = [json.loads(p.read_text(encoding="utf-8")) for p in (run_path, started_path)]
    if (run["status"] != "COMPLETE" or run["report_sha256"] != expected_sha256
            or run["registration_sha256"] != report["registration_sha256"]
            or start["status"] != "STARTED" or start["registration_sha256"] != report["registration_sha256"]):
        raise RuntimeError("completed calibration run manifest required")
    for receipt in (run, start):
        if (any(receipt.get(k) != v for k, v in contract.items())
                or receipt["registration_path"] != report["registration_path"]
                or receipt["track"] != "native_armr" or receipt["stage"] != STAGE
                or not receipt["branch"] or not isinstance(receipt["commit_sha"], str) or len(receipt["commit_sha"]) != 40):
            raise RuntimeError("calibration run/source/branch binding mismatch")
    if any(run[key] != start[key] for key in ("branch", "commit_sha", "start_utc")) or run["failure_receipt"] is not None:
        raise RuntimeError("calibration execution identity or completion mismatch")
    if require_committed:
        for candidate in (path, registration_path, fit_path, run_path, started_path):
            _require_committed(candidate)
    return report
