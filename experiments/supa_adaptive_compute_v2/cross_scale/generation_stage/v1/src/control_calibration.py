"""B-only, label-free scalar calibration against the selected frozen XG delta.

Reads only the already qualified S19 activation slice through a read-only mmap.
No complete Supa model, native-donor activation slice, label, scorer or efficacy
bank is used. The fit scalar is persisted before any held-out transformation.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
import uuid

from shared.guards import require_within
from shared.hashing import canonical_sha256, sha256_file, write_json
from shared.integrity import checkpoint_files, frozen_inventory, state_hash
from shared.legacy import ROOT, REPO, legacy_graft
from shared.manifests import git_identity, utc_now
from .compute_control import CONTROL, PARAMETERS, MATMUL_FLOPS_PER_TOKEN, new_transform
from .operators import buffer_hashes, validate_bridge_state


STAGE = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = STAGE / "configs/control_calibration_v1.json"
DEFAULT_MANIFEST = ROOT / "data/generated/v2/cross_scale/manifest.json"
SCHEMA = "cross_scale_control_calibration_v1"
FORMULA = "sqrt(mean_scenario_mean_ordinary_token_relative_XG_energy / mean_scenario_mean_ordinary_token_relative_XM_energy)"
PROMPT_FIELDS = ("example_id", "scenario_id", "family", "template_id", "prompt",
                 "prompt_sha256", "alignment_partition")
REGISTRATION_ROOT = STAGE / "evidence/00_control_calibration_registration"
ARTIFACT_ROOT = STAGE / "evidence/00_control_calibration"
MANIFEST_ID = "6857158ee7a360bdf2f72a21c5ee78041ab0a5ecda264c0387e1e5facf638a46"


def _read(path):
    from .upstream import _read as strict_read
    return strict_read(path)


def _relative(path):
    return Path(path).resolve().relative_to(REPO).as_posix()


def load_config(path=DEFAULT_CONFIG):
    path = require_within(path, STAGE / "configs")
    if path != DEFAULT_CONFIG.resolve():
        raise ValueError("only the prospective B-control-v1 calibration config is supported")
    value = _read(path)
    expected = {"schema_version": SCHEMA, "track": "cross_scale", "device": "cpu",
                "dtype": "float32", "threads": 4, "seed": 2701, "use_cache": False,
                "attention": "eager", "supa_layer": 19, "donor_layer": 24,
                "selected_objective": "B3", "fit_scenarios": 96, "holdout_scenarios": 32,
                "max_input_tokens": 256, "energy_floor": 1e-8, "formula": FORMULA,
                "control": CONTROL, "label_free": True, "alpha_search": False,
                "alpha_clipping": False, "holdout_selects_alpha": False,
                "directional_fit": False, "max_rss_gib": 6.0, "min_available_gib": 2.0,
                "max_prompt_seconds": 900, "max_stage_seconds": 172800,
                "donor_output_input_rms_bounds": [.05, 10.],
                "checkpoint_sha256": "a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026",
                "donor_revision": "1cfa9a7208912126459214e8b04321603b3df60c"}
    if canonical_sha256(value) != canonical_sha256(expected):
        raise ValueError("unregistered B control calibration settings")
    return value


def prompt_projection(row):
    """Discard label-bearing fields immediately; never call a scorer/oracle."""
    if (row.get("track"), row.get("split"), row.get("variant")) != ("cross_scale", "calibration", "original"):
        raise ValueError("only original Track B calibration prompts allowed")
    value = {key: row[key] for key in PROMPT_FIELDS}
    partition = "holdout" if int(value["scenario_id"].rsplit("_", 1)[1]) % 4 == 3 else "fit"
    if value["alignment_partition"] != partition:
        raise ValueError("scenario violates frozen 96/32 partition")
    if (not value["prompt"] or canonical_sha256(value["prompt"]) != value["prompt_sha256"]
            or not value["example_id"].startswith("cross_scale_calibration_")):
        raise ValueError("calibration prompt/identity mismatch")
    return value


def calibration_prompts(manifest_path=DEFAULT_MANIFEST):
    path = require_within(manifest_path, ROOT / "data/generated/v2/cross_scale")
    if path != DEFAULT_MANIFEST.resolve():
        raise ValueError("only the frozen Track B data manifest is allowed")
    manifest = _read(path)
    digest = manifest.pop("manifest_sha256", None)
    if canonical_sha256(manifest) != digest or digest != MANIFEST_ID:
        raise ValueError("frozen Track B manifest mismatch")
    if (manifest["track"], manifest["status"], manifest["sealed_access"]) != ("cross_scale", "FROZEN_UNSEALED", False):
        raise ValueError("wrong track or data boundary")
    info = manifest["datasets"]["calibration"]
    data_path = require_within(path.parent / info["path"], path.parent)
    if data_path != path.parent / "calibration.jsonl" or sha256_file(data_path) != info["sha256"]:
        raise ValueError("frozen calibration bytes differ")
    with data_path.open(encoding="utf-8") as stream:
        rows = sorted((prompt_projection(json.loads(line)) for line in stream if line.strip()),
                      key=lambda r: r["example_id"])
    if (len(rows) != 128 or len({r["example_id"] for r in rows}) != 128
            or len({r["scenario_id"] for r in rows}) != 128
            or Counter(r["alignment_partition"] for r in rows) != {"fit": 96, "holdout": 32}):
        raise ValueError("128 unique scenarios in the frozen 96/32 split required")
    for key, count_key in (("family", "families"), ("template_id", "templates")):
        if dict(Counter(r[key] for r in rows)) != info[count_key]:
            raise ValueError("calibration strata differ from manifest")
    return rows, {"dataset_manifest_path": _relative(path), "dataset_manifest_sha256": sha256_file(path),
                  "calibration_data_path": _relative(data_path), "calibration_data_sha256": info["sha256"]}


def calibration_sources():
    from .provenance import stage_sources
    return stage_sources()


def _verified_export(export_manifest_path, *, verify_upstream_now=False):
    from .provenance import verify_export
    return verify_export(export_manifest_path, verify_upstream_now=verify_upstream_now)


def _export_bindings(export_manifest_path):
    _verified_export(export_manifest_path)
    path = require_within(export_manifest_path, STAGE / "evidence/00_upstream_exports")
    manifest = _read(path)
    context = manifest["context"]
    if (context["supa_layer"], context["donor_layer"], context["selected_objective"]) != (19, 24, "B3"):
        raise ValueError("B calibration requires the frozen B3 S19/D24 winner")
    capture_path = require_within(Path(manifest["upstream"]["capture_directory"]) / "capture.json",
                                  ROOT / "cross_scale/evidence/01_alignment_matrix")
    if sha256_file(capture_path) != manifest["upstream"]["capture_sha256"]:
        raise RuntimeError("verified capture receipt changed")
    capture = _read(capture_path)
    activation = require_within(ROOT / "cross_scale/local" / manifest["upstream"]["run_id"] / "activations.pt",
                               ROOT / "cross_scale/local")
    if sha256_file(activation) != capture["activation_sha256"]:
        raise RuntimeError("upstream activation cache changed")
    return {"export_manifest_path": _relative(path), "export_manifest_sha256": sha256_file(path),
            "export_tensor_path": _relative(path.parent / manifest["tensor_file"]),
            "export_tensor_sha256": manifest["tensor_file_sha256"],
            "capture_path": _relative(capture_path), "capture_sha256": sha256_file(capture_path),
            "activation_path": _relative(activation), "activation_sha256": capture["activation_sha256"],
            "upstream_context": context}


def _contract(config_path, manifest_path, export_manifest_path):
    config = load_config(config_path)
    rows, data = calibration_prompts(manifest_path)
    export = _export_bindings(export_manifest_path)
    if (export["upstream_context"]["checkpoint_sha256"] != config["checkpoint_sha256"]
            or export["upstream_context"]["donor_revision"] != config["donor_revision"]
            or _read(REPO / export["capture_path"])["dataset_manifest_sha256"] != data["dataset_manifest_sha256"]):
        raise ValueError("calibration/export checkpoint or data binding mismatch")
    return {"schema_version": SCHEMA, "track": "cross_scale", "config": config,
            "calibration_config_path": _relative(config_path), "calibration_config_sha256": sha256_file(config_path),
            "source_files": calibration_sources(), **data, **export,
            "fit_example_ids": [r["example_id"] for r in rows if r["alignment_partition"] == "fit"],
            "holdout_example_ids": [r["example_id"] for r in rows if r["alignment_partition"] == "holdout"]}


def freeze_calibration_registration(config_path=DEFAULT_CONFIG, manifest_path=DEFAULT_MANIFEST, *, export_manifest_path):
    # A fresh full matrix/audit admission precedes the new calibration freeze.
    _verified_export(export_manifest_path, verify_upstream_now=True)
    contract = _contract(config_path, manifest_path, export_manifest_path)
    digest = canonical_sha256(contract)
    path = REGISTRATION_ROOT / digest / "registration.json"
    write_json(path, {"status": "FROZEN", "registration_sha256": digest, "contract": contract,
                      "registered_at_utc": utc_now(), "registered_against": git_identity()})
    return path


def verify_calibration_registration(config_path=DEFAULT_CONFIG, manifest_path=DEFAULT_MANIFEST, *, export_manifest_path):
    from .provenance import require_committed
    contract = _contract(config_path, manifest_path, export_manifest_path)
    digest = canonical_sha256(contract)
    path = REGISTRATION_ROOT / digest / "registration.json"
    value = _read(path)
    if value.get("status") != "FROZEN" or value.get("registration_sha256") != digest or value.get("contract") != contract:
        raise RuntimeError("B calibration registration mismatch")
    for relative in (*contract["source_files"], contract["calibration_config_path"],
                     contract["dataset_manifest_path"], contract["calibration_data_path"],
                     contract["export_manifest_path"], contract["capture_path"]):
        candidate = require_within(REPO / relative, REPO)
        if candidate.is_relative_to(ROOT):
            require_committed(candidate)
    require_committed(path)
    return {**value, "registration_path": _relative(path)}


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((value.dtype, tuple(value.shape))).encode())
    flat = value.view(-1)
    step = max(1, 8 * 1024 * 1024 // value.element_size())
    for start in range(0, flat.numel(), step):
        digest.update(memoryview(flat[start:start + step].numpy()).cast("B"))
    return digest.hexdigest()


def ordinary_mask(token_ids, special_ids):
    import torch
    if token_ids.ndim != 1 or token_ids.dtype != torch.int64 or token_ids.device.type != "cpu":
        raise ValueError("one complete unpadded CPU int64 prompt required")
    mask = torch.ones_like(token_ids, dtype=torch.bool)
    for token in special_ids:
        if type(token) is not int or token < 0:
            raise ValueError("invalid special-token ID")
        mask &= token_ids != token
    if not mask.any():
        raise ValueError("prompt has no ordinary tokens")
    return mask


def scenario_energies(hidden, teacher_delta, control_delta, ordinary):
    """FP64 reductions, equal ordinary-token weighting within each scenario."""
    import torch
    if hidden.ndim != 2 or hidden.shape != teacher_delta.shape or hidden.shape != control_delta.shape:
        raise ValueError("unaligned full-prompt transformation tensors")
    if ordinary.shape != hidden.shape[:1] or ordinary.dtype != torch.bool or not ordinary.any():
        raise ValueError("ordinary-token mask must be aligned and nonempty")
    if any(v.dtype != torch.float32 or v.device.type != "cpu" or not torch.isfinite(v).all()
           for v in (hidden, teacher_delta, control_delta)):
        raise ValueError("finite CPU FP32 inference tensors required")
    h, teacher, control = (v.double() for v in (hidden, teacher_delta, control_delta))
    denominator = h.square().mean(-1).clamp_min(1e-8)
    teacher_energy = teacher.square().mean(-1) / denominator
    control_energy = control.square().mean(-1) / denominator
    masks = {"all": torch.ones_like(ordinary), "ordinary": ordinary, "special_control": ~ordinary,
             "final_prompt": torch.arange(len(ordinary)) == len(ordinary) - 1}
    return {name: None if not mask.any() else {
        "token_count": int(mask.sum()), "teacher_energy": float(teacher_energy[mask].mean()),
        "control_energy": float(control_energy[mask].mean()), "hidden_rms": float(h[mask].square().mean().sqrt()),
        "teacher_delta_rms": float(teacher[mask].square().mean().sqrt()),
        "control_delta_rms": float(control[mask].square().mean().sqrt())}
        for name, mask in masks.items()}


def aggregate_partition(records):
    if not records or len({r["scenario_id"] for r in records}) != len(records):
        raise ValueError("one original per independent calibration scenario required")
    def mean(key):
        values = [r["metrics"]["ordinary"][key] for r in records]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("invalid calibration statistic")
        return math.fsum(values) / len(values)
    return {"scenario_ids": sorted(r["scenario_id"] for r in records),
            "example_ids": sorted(r["example_id"] for r in records), "scenarios": len(records),
            "ordinary_tokens": sum(r["metrics"]["ordinary"]["token_count"] for r in records),
            **{key: mean(key) for key in ("teacher_energy", "control_energy", "hidden_rms",
                                        "teacher_delta_rms", "control_delta_rms")}}


def fit_alpha(fit):
    teacher, control = fit["teacher_energy"], fit["control_energy"]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (teacher, control)):
        raise ValueError("positive finite teacher and control fit energies required")
    alpha = math.sqrt(teacher / control)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("finite positive alpha required; no clipping or fallback")
    return alpha


def teacher_delta(hidden, layer, rope, bridges, *, bounds=(.05, 10.)):
    """One actual full-prompt L24 on projected h; downward affine bias cancels."""
    import torch
    positions, mask = legacy_graft().positions_mask(hidden)
    with torch.inference_mode():
        z = bridges.project(hidden)
        embeddings = rope(z, positions)
        out = layer(z, attention_mask=mask, position_ids=positions,
                    position_embeddings=embeddings, use_cache=False, past_key_values=None)
        if not isinstance(out, torch.Tensor) or out.shape != z.shape:
            raise RuntimeError("donor must return the aligned full-prompt tensor")
        delta = bridges.delta(out - z)
    if any(not torch.isfinite(v).all() for v in (z, out, delta)):
        raise RuntimeError("nonfinite projected donor transformation")
    ratio = float(out.double().square().mean().sqrt() / z.double().square().mean().sqrt().clamp_min(1e-12))
    if not bounds[0] <= ratio <= bounds[1]:
        raise RuntimeError("donor output/input RMS outside calibration guard")
    return delta, {"original_positions_sha256": tensor_sha256(positions),
                   "original_mask_sha256": tensor_sha256(mask),
                   "donor_rope_sha256": [tensor_sha256(v) for v in embeddings],
                   "projected_hidden_sha256": tensor_sha256(z),
                   "donor_output_input_rms_ratio": ratio}


def load_used_capture(contract, rows, tokenizer):
    """Mmap a verified immutable cache; never touch its unused donor/Supa slices."""
    import torch
    path = require_within(REPO / contract["activation_path"], ROOT / "cross_scale/local")
    if sha256_file(path) != contract["activation_sha256"]:
        raise RuntimeError("activation cache hash changed before mmap")
    bundle = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    context = contract["upstream_context"]
    if (bundle.get("schema_version"), bundle.get("track"), bundle.get("role")) != (1, "cross_scale", "calibration"):
        raise ValueError("wrong activation cache schema/track/role")
    if (bundle.get("canonical_fixture_bit_identical") is not True
            or bundle.get("supa_identity") != context["supa_identity"]
            or bundle.get("supa_files") != context["supa_files"]
            or bundle.get("donor_identity") != context["donor_identity"]
            or bundle.get("sources") != context["stage1_sources"]
            or bundle.get("source_sha256") != canonical_sha256(context["stage1_sources"])
            or bundle.get("config_sha256") != context["stage1_config_sha256"]):
        raise ValueError("capture identity/source/config qualification mismatch")
    expected = {r["example_id"]: r for r in rows}
    cache_rows = bundle["rows"]
    if len(cache_rows) != len(expected) or len({r["example_id"] for r in cache_rows}) != len(expected):
        raise ValueError("cache calibration scenario coverage mismatch")
    if set(r["example_id"] for r in cache_rows) != set(expected):
        raise ValueError("cache contains noncalibration or wrong IDs")
    special = sorted(set(map(int, tokenizer.all_special_ids)) | set(map(int, tokenizer.get_added_vocab().values())))
    if bundle["special_ids"] != special:
        raise ValueError("special/control token inventory changed")
    states, ids, train, lengths = bundle["small"][19], bundle["input_ids"], bundle["train_prompt_mask"], bundle["lengths"]
    if any(len(values) != len(rows) for values in (states, ids, train, lengths)):
        raise ValueError("S19/token/partition/length coverage mismatch")
    if bundle["heldout_scenarios"] != sorted(r["scenario_id"] for r in rows if r["alignment_partition"] == "holdout"):
        raise ValueError("held-out scenario inventory changed")
    selected = {}
    for index, cached in enumerate(cache_rows):
        row = expected[cached["example_id"]]
        if cached != {key: row[key] for key in ("example_id", "scenario_id", "family", "prompt")}:
            raise ValueError("cache prompt identity differs from frozen calibration row")
        if type(train[index]) is not bool or train[index] != (row["alignment_partition"] == "fit"):
            raise ValueError("cache fit/holdout partition differs")
        token_ids, hidden = ids[index], states[index]
        if (token_ids.dtype != torch.int64 or token_ids.device.type != "cpu" or token_ids.ndim != 2
                or token_ids.shape[0] != 1 or not 0 < token_ids.shape[1] <= contract["config"]["max_input_tokens"]):
            raise ValueError("full untruncated token sequence required")
        if lengths[index] != {"example_id": row["example_id"], "full_tokens": token_ids.numel(), "retained_tokens": token_ids.numel()}:
            raise ValueError("cache full/retained length mismatch")
        text = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = tokenizer(text, return_tensors="pt")["input_ids"]
        if not torch.equal(encoded, token_ids):
            raise ValueError("cached token IDs differ from original complete prompt")
        if (hidden.shape != (token_ids.numel(), CONTROL["hidden_size"]) or hidden.dtype != torch.float32
                or hidden.device.type != "cpu" or hidden.requires_grad or hidden.grad is not None
                or not torch.isfinite(hidden).all()):
            raise ValueError("invalid qualified post-L19 activation slice")
        ordinary_mask(token_ids[0], special)
        selected[row["example_id"]] = (hidden, token_ids[0])
    # Deleting the metadata releases unused tensor wrappers without paging in their bytes.
    del bundle
    return selected, special


def _module_receipt(module, before, buffers_before):
    inventory = frozen_inventory(module)
    after, buffers_after = state_hash(module), buffer_hashes(module)
    if before != after or buffers_before != buffers_after:
        raise RuntimeError("frozen module state/buffers changed during calibration")
    return {**inventory, "state_sha256_before": before, "state_sha256_after": after,
            "buffer_sha256_before": buffers_before, "buffer_sha256_after": buffers_after}


def run_calibration(config_path=DEFAULT_CONFIG, manifest_path=DEFAULT_MANIFEST, *, export_manifest_path):
    registration = verify_calibration_registration(config_path, manifest_path, export_manifest_path=export_manifest_path)
    contract, started = registration["contract"], time.perf_counter()
    config = contract["config"]
    rows, data = calibration_prompts(manifest_path)
    from shared.timing import machine_lease
    with machine_lease():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
        output = ARTIFACT_ROOT / stamp
        output.mkdir(parents=True, exist_ok=False)
        manifest = {**git_identity(), "track": "cross_scale", "stage": SCHEMA,
                    "arm": "XM_calibration", "status": "STARTED", "start_utc": utc_now(),
                    "registration_sha256": registration["registration_sha256"],
                    "registration_path": registration["registration_path"], **contract,
                    "failure_receipt": None}
        write_json(output / "started_manifest.json", manifest)
        report, peak_rss = {}, 0
        def boundary(prompt_start=None):
            nonlocal peak_rss
            import psutil
            peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
            if peak_rss > config["max_rss_gib"] * 2**30:
                raise MemoryError("B control calibration RSS budget exceeded")
            now = time.perf_counter()
            if now - started > config["max_stage_seconds"] or (prompt_start is not None and now - prompt_start > config["max_prompt_seconds"]):
                raise TimeoutError("B control calibration elapsed budget exceeded")
        try:
            import torch
            import transformers
            from transformers import AutoTokenizer
            from shared.model_loading import require_available_memory
            from shared.runtime import configure_runtime, runtime_identity
            from cross_scale.src.donor_layers import donor_identity, load_donor_layer
            if str(torch.__version__) != "2.12.1+cu126" or transformers.__version__ != "5.14.1":
                raise RuntimeError("unqualified runtime; use historical Chat-Bubble/.venv Python")
            configure_runtime(config)
            report.update(require_available_memory(config))
            context = contract["upstream_context"]
            supa_path = Path(context["supa_identity"]["model_path"])
            supa_before = checkpoint_files(supa_path)
            if supa_before != context["supa_files"] or supa_before.get("model.safetensors") != config["checkpoint_sha256"]:
                raise RuntimeError("canonical checkpoint/config/tokenizer identity changed")
            donor_path, donor_before = donor_identity(context["stage1_config"])
            if donor_before != context["donor_identity"]:
                raise RuntimeError("donor checkpoint/config/implementation identity changed")
            tokenizer = AutoTokenizer.from_pretrained(supa_path, local_files_only=True, trust_remote_code=True)
            selected, special = load_used_capture(contract, rows, tokenizer)
            layer, rope, architecture = load_donor_layer(context["stage1_config"], 24, donor_path)
            if layer.self_attn.layer_idx != 24 or architecture.hidden_size != 2560:
                raise RuntimeError("wrong donor decoder identity")
            if sha256_file(REPO / contract["export_tensor_path"]) != contract["export_tensor_sha256"]:
                raise RuntimeError("exported bridges changed before deserialization")
            exported = torch.load(REPO / contract["export_tensor_path"], weights_only=True, map_location="cpu", mmap=True)
            if set(exported) != {"XG", "XS", "XR"}:
                raise RuntimeError("export must preserve selected, shuffled and random bridges")
            for state in exported.values():
                validate_bridge_state(state, 1024, 2560)
            bridges = legacy_graft().Bridges(*(exported["XG"][key] for key in ("up", "up_bias", "down", "down_bias"))).eval().requires_grad_(False)
            bridge_source_hashes = {key: tensor_sha256(value) for key, value in exported["XG"].items()}
            bridge_copy_hashes = {key: tensor_sha256(value) for key, value in bridges.named_buffers()}
            export_manifest = _read(REPO / contract["export_manifest_path"])
            if bridge_source_hashes != bridge_copy_hashes or bridge_source_hashes != export_manifest["tensor_sha256"]["XG"]:
                raise RuntimeError("copied XG bridge differs from verified export")
            transform = new_transform()
            modules = {"donor": layer, "rotary": rope, "bridges": bridges, "control": transform}
            module_before = {name: (state_hash(module), buffer_hashes(module)) for name, module in modules.items()}
            inventories = {name: frozen_inventory(module) for name, module in modules.items()}
            if inventories["control"]["parameters"] != PARAMETERS:
                raise RuntimeError("control parameter count differs from registered work")
            report.update(runtime_identity=runtime_identity(config), supa_checkpoint_files_before=supa_before,
                          donor_identity_before=donor_before, bridge_source_hashes=bridge_source_hashes,
                          bridge_copy_hashes=bridge_copy_hashes, special_control_token_ids=special,
                          activation_mmap=True, accessed_supa_layers=[19], accessed_donor_activation_layers=[],
                          full_supa_model_loaded=False, donor_layer_index=24, donor_hidden_size=2560)
            boundary()
            cases = []
            for partition in ("fit", "holdout"):
                for row in rows:
                    if row["alignment_partition"] != partition:
                        continue
                    prompt_start = time.perf_counter()
                    boundary(prompt_start)
                    hidden, ids = selected[row["example_id"]]
                    hidden_before = tensor_sha256(hidden)
                    with torch.inference_mode():
                        teacher, diagnostics = teacher_delta(hidden.unsqueeze(0), layer, rope, bridges,
                                                            bounds=tuple(config["donor_output_input_rms_bounds"]))
                        control = transform(hidden.unsqueeze(0))
                    metrics = scenario_energies(hidden, teacher[0], control[0], ordinary_mask(ids, special))
                    if tensor_sha256(hidden) != hidden_before:
                        raise RuntimeError("immutable S19 activation slice mutated")
                    case = {key: row[key] for key in PROMPT_FIELDS if key != "prompt"}
                    case.update(token_ids=ids.tolist(), full_prompt_tokens=ids.numel(), retained_prompt_tokens=ids.numel(),
                                hidden_sha256_before=hidden_before, hidden_sha256_after=tensor_sha256(hidden),
                                teacher_delta_sha256=tensor_sha256(teacher), control_delta_sha256=tensor_sha256(control),
                                metrics=metrics, **diagnostics)
                    write_json(output / "cases" / (canonical_sha256(row["example_id"]) + ".json"), case)
                    cases.append(case)
                    boundary(prompt_start)
                    print(f"B_CONTROL_CALIBRATION={partition}:{len(cases)}/128", flush=True)
                aggregate = aggregate_partition([case for case in cases if case["alignment_partition"] == partition])
                if partition == "fit":
                    report["alpha"] = fit_alpha(aggregate)
                    write_json(output / "fit_scale.json", {"alpha": report["alpha"], "formula": FORMULA,
                                                           "fit": aggregate, "heldout_transformations_at_write": 0})
                report[partition] = {**aggregate, "scaled_control_energy": report["alpha"] ** 2 * aggregate["control_energy"]}
            report["module_state_receipts"] = {name: _module_receipt(module, *module_before[name]) for name, module in modules.items()}
            report["control_state_sha256"] = module_before["control"][0]
            report["supa_checkpoint_files_after"] = checkpoint_files(supa_path)
            _, report["donor_identity_after"] = donor_identity(context["stage1_config"])
            if (supa_before != report["supa_checkpoint_files_after"] or donor_before != report["donor_identity_after"]
                    or _contract(config_path, manifest_path, export_manifest_path) != contract):
                raise RuntimeError("registered source/data/export/cache/checkpoint mutated during calibration")
            boundary()
            report.update(schema_version=SCHEMA, status="COMPLETE", track="cross_scale", control=CONTROL,
                          formula=FORMULA, label_free=True, answer_labels_used=False, generated_tokens=0,
                          holdout_selects_alpha=False, alpha_search=False, alpha_clipping=False, directional_fit=False,
                          trainable_parameters=0, backbone_gradients=0, canonical_unchanged=True,
                          device="cpu", dtype="float32", threads=4, use_cache=False,
                          registration_path=registration["registration_path"], registration_sha256=registration["registration_sha256"],
                          **{key: value for key, value in contract.items() if key not in ("schema_version", "track", "config")},
                          additional_parameters=PARAMETERS, additional_active_parameters=PARAMETERS,
                          additional_matmul_flops_per_token=MATMUL_FLOPS_PER_TOKEN,
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


def verify_calibration_artifact(path, expected_sha256=None, *, require_committed=True):
    """Verify arithmetic, exact coverage and provenance; never rerun a model."""
    from .provenance import require_committed as committed
    path = require_within(path, ARTIFACT_ROOT)
    if path.name != "report.json":
        raise ValueError("completed stage-local calibration report required")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError("B control calibration artifact hash mismatch")
    report = _read(path)
    required = {"schema_version": SCHEMA, "status": "COMPLETE", "track": "cross_scale", "control": CONTROL,
                "formula": FORMULA, "label_free": True, "answer_labels_used": False, "generated_tokens": 0,
                "holdout_selects_alpha": False, "alpha_search": False, "alpha_clipping": False, "directional_fit": False,
                "trainable_parameters": 0, "backbone_gradients": 0, "canonical_unchanged": True,
                "device": "cpu", "dtype": "float32", "threads": 4, "use_cache": False,
                "activation_mmap": True, "accessed_supa_layers": [19], "accessed_donor_activation_layers": [],
                "full_supa_model_loaded": False, "donor_layer_index": 24, "donor_hidden_size": 2560,
                "additional_parameters": PARAMETERS,
                "additional_active_parameters": PARAMETERS, "additional_matmul_flops_per_token": MATMUL_FLOPS_PER_TOKEN}
    if any(type(report.get(key)) is not type(value) or report.get(key) != value for key, value in required.items()):
        raise RuntimeError("invalid B calibration science/freeze contract")
    runtime = {"device": "cpu", "dtype": "float32", "threads": 4, "seed": 2701,
               "torch": "2.12.1+cu126", "transformers": "5.14.1", "attention": "eager", "use_cache": False}
    actual_runtime = report.get("runtime_identity", {})
    if any(type(actual_runtime.get(key)) is not type(value) or actual_runtime.get(key) != value
           for key, value in runtime.items()):
        raise RuntimeError("unqualified B calibration runtime identity")
    registration_path = require_within(REPO / report["registration_path"], REGISTRATION_ROOT)
    registration = _read(registration_path)
    contract = registration["contract"]
    if (registration.get("status") != "FROZEN" or canonical_sha256(contract) != report["registration_sha256"]
            or registration.get("registration_sha256") != report["registration_sha256"]):
        raise RuntimeError("B calibration registration digest mismatch")
    current = _contract(REPO / contract["calibration_config_path"], REPO / contract["dataset_manifest_path"],
                        REPO / contract["export_manifest_path"])
    if current != contract or any(report.get(key) != value for key, value in contract.items()
                                  if key not in ("schema_version", "track", "config")):
        raise RuntimeError("calibration source/config/data/export/cache registration changed")
    if report["alpha"] != fit_alpha(report["fit"]):
        raise RuntimeError("alpha differs from fit-only formula")
    rows, _ = calibration_prompts(REPO / contract["dataset_manifest_path"])
    if set(report["case_files"]) != {canonical_sha256(r["example_id"]) + ".json" for r in rows}:
        raise RuntimeError("calibration case coverage mismatch")
    cases = []
    for filename, expected in report["case_files"].items():
        case_path = require_within(path.parent / "cases" / filename, path.parent / "cases")
        if sha256_file(case_path) != expected:
            raise RuntimeError("calibration case bytes changed")
        cases.append(_read(case_path))
        if require_committed:
            committed(case_path)
    by_id = {case["example_id"]: case for case in cases}
    if len(by_id) != len(rows) or set(by_id) != {r["example_id"] for r in rows}:
        raise RuntimeError("calibration case identities are duplicated or incomplete")
    special = report["special_control_token_ids"]
    if special != sorted(set(special)) or any(type(v) is not int or v < 0 for v in special):
        raise RuntimeError("special/control token exclusion inventory malformed")
    for row in rows:
        case = by_id[row["example_id"]]
        if any(case[key] != row[key] for key in PROMPT_FIELDS if key != "prompt"):
            raise RuntimeError("calibration case differs from frozen identity")
        tokens = case["token_ids"]
        count = sum(token not in special for token in tokens)
        if (not 0 < len(tokens) <= contract["config"]["max_input_tokens"] or count <= 0
                or any(type(token) is not int or token < 0 for token in tokens)
                or case["full_prompt_tokens"] != len(tokens) or case["retained_prompt_tokens"] != len(tokens)
                or case["metrics"]["ordinary"]["token_count"] != count
                or case["metrics"]["all"]["token_count"] != len(tokens)
                or case["metrics"]["final_prompt"]["token_count"] != 1
                or case["hidden_sha256_before"] != case["hidden_sha256_after"]):
            raise RuntimeError("calibration token exclusion/length/state mismatch")
        bounds = contract["config"]["donor_output_input_rms_bounds"]
        if not bounds[0] <= case["donor_output_input_rms_ratio"] <= bounds[1]:
            raise RuntimeError("donor output/input RMS guard failed")
        hash_fields = ("hidden_sha256_before", "hidden_sha256_after", "teacher_delta_sha256", "control_delta_sha256",
                       "original_positions_sha256", "original_mask_sha256", "projected_hidden_sha256")
        hashes = [case[key] for key in hash_fields] + case["donor_rope_sha256"]
        if len(case["donor_rope_sha256"]) != 2 or any(not isinstance(v, str) or not re.fullmatch("[0-9a-f]{64}", v) for v in hashes):
            raise RuntimeError("calibration input/transform/position hash inventory incomplete")
        if set(case["metrics"]) != {"all", "ordinary", "special_control", "final_prompt"}:
            raise RuntimeError("complete token-stratified calibration metrics required")
        special_count = len(tokens) - count
        special_metrics = case["metrics"]["special_control"]
        if (special_count == 0 and special_metrics is not None) or (special_count > 0 and
                (special_metrics is None or special_metrics["token_count"] != special_count)):
            raise RuntimeError("special/control token-stratum count differs from token IDs")
        for name, values in case["metrics"].items():
            if values is not None:
                if set(values) != {"token_count", "teacher_energy", "control_energy", "hidden_rms", "teacher_delta_rms", "control_delta_rms"}:
                    raise RuntimeError("incomplete calibration token-stratum metrics")
                if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values.values()):
                    raise RuntimeError("nonfinite or negative case metric")
                if type(values["token_count"]) is not int or values["token_count"] <= 0:
                    raise RuntimeError("positive integer token-stratum count required")
    for partition, count in (("fit", 96), ("holdout", 32)):
        reconstructed = aggregate_partition([case for case in cases if case["alignment_partition"] == partition])
        reconstructed["scaled_control_energy"] = report["alpha"] ** 2 * reconstructed["control_energy"]
        if (reconstructed != report[partition] or reconstructed["scenarios"] != count
                or reconstructed["example_ids"] != contract[partition + "_example_ids"]):
            raise RuntimeError("calibration scenario aggregation mismatch")
    if set(report["fit"]["scenario_ids"]) & set(report["holdout"]["scenario_ids"]):
        raise RuntimeError("fit/holdout scenario leakage")
    if set(report["module_state_receipts"]) != {"donor", "rotary", "bridges", "control"}:
        raise RuntimeError("complete donor/rotary/bridge/control state receipts required")
    for receipt in report["module_state_receipts"].values():
        if (receipt["trainable_parameters"] != 0 or receipt["backbone_gradients"] != 0
                or not receipt["state_sha256_before"] or receipt["state_sha256_before"] != receipt["state_sha256_after"]
                or receipt["buffer_sha256_before"] != receipt["buffer_sha256_after"]):
            raise RuntimeError("calibration module mutated or was not frozen")
    control = report["module_state_receipts"]["control"]
    if control["parameters"] != PARAMETERS or report["control_state_sha256"] != control["state_sha256_before"]:
        raise RuntimeError("calibrated control identity/parameter count mismatch")
    export = _read(REPO / report["export_manifest_path"])
    if report["bridge_source_hashes"] != export["tensor_sha256"]["XG"] or report["bridge_copy_hashes"] != report["bridge_source_hashes"]:
        raise RuntimeError("calibration teacher differs from copied selected XG bridge")
    if (report["supa_checkpoint_files_before"] != report["supa_checkpoint_files_after"]
            or report["supa_checkpoint_files_before"] != contract["upstream_context"]["supa_files"]
            or report["donor_identity_before"] != report["donor_identity_after"]
            or report["donor_identity_before"] != contract["upstream_context"]["donor_identity"]):
        raise RuntimeError("checkpoint identity mutated or differs from selected context")
    fit_path, run_path = path.parent / "fit_scale.json", path.parent / "run_manifest.json"
    fit = _read(fit_path)
    expected_fit = {"alpha": report["alpha"], "formula": FORMULA, "heldout_transformations_at_write": 0,
                    "fit": {key: value for key, value in report["fit"].items() if key != "scaled_control_energy"}}
    if sha256_file(fit_path) != report["fit_scale_sha256"] or fit != expected_fit:
        raise RuntimeError("fit scale was not frozen at registered value before holdout")
    run = _read(run_path)
    if run.get("status") != "COMPLETE" or run.get("report_sha256") != digest or run.get("registration_sha256") != report["registration_sha256"]:
        raise RuntimeError("completed B calibration run manifest required")
    started = _read(path.parent / "started_manifest.json")
    if (started.get("status") != "STARTED" or started.get("registration_sha256") != report["registration_sha256"]
            or any(run.get(key) != value or started.get(key) != value for key, value in contract.items())
            or run.get("failure_receipt") is not None or started.get("failure_receipt") is not None):
        raise RuntimeError("started/final calibration input contract mismatch")
    if (not math.isfinite(report["wall_seconds"]) or not 0 <= report["wall_seconds"] <= contract["config"]["max_stage_seconds"]
            or not 0 < report["peak_rss_bytes"] <= contract["config"]["max_rss_gib"] * 2**30):
        raise RuntimeError("calibration runtime/resource budget invalid")
    if require_committed:
        for candidate in (path, registration_path, fit_path, run_path, path.parent / "started_manifest.json"):
            committed(candidate)
        for relative in (*contract["source_files"], contract["calibration_config_path"],
                         contract["dataset_manifest_path"], contract["calibration_data_path"]):
            candidate = require_within(REPO / relative, REPO)
            if candidate.is_relative_to(ROOT):
                committed(candidate)
    return report
