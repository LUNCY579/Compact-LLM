"""Immutable standard PEFT saves and offline GD-001 handoff preparation.

Serialization and content hashes are mechanical evidence. Native runtime parity
requires actual independently reviewed wrapper/HF/PEFT logits and greedy outputs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .contracts import digest, file_hash, read_json, require, sealed, unseal, write_new
from .manifests import TOKENIZER_NAMES, public_row, validate_manifest
from .data import normalized


def peft_names(state):
    require(bool(state) and all(n.endswith((".lora_A", ".lora_B")) for n in state), "q/v LoRA state only")
    return {"base_model.model." + n + ".weight": t for n, t in state.items()}


def _checked_state(state, plan):
    import torch
    modules = {n.rsplit(".", 1)[0] for n in state}
    require(state and all(n.endswith((".q_proj", ".v_proj")) for n in modules), "only q/v adapter modules")
    require(set(state) == {n + suffix for n in modules for suffix in (".lora_A", ".lora_B")},
            "both LoRA factors required for every adapted module")
    for name in modules:
        a, b = state[name + ".lora_A"], state[name + ".lora_B"]
        require(a.ndim == b.ndim == 2 and a.shape[0] == b.shape[1] == plan.rank
                and a.dtype == b.dtype == torch.float32 and bool(torch.isfinite(a).all())
                and bool(torch.isfinite(b).all()), "invalid adapter factor shape/dtype/value")


def write_standard_peft_state(state, plan, output, base, *, identity, source_receipt,
                              artifact_role, extra_lineage=None):
    """Serialize an already saved state; no model construction or optimizer work."""
    from safetensors.torch import load_file, save_file
    from .backend import tensor_hash
    _checked_state(state, plan)
    require(source_receipt["tensor_sha256"] == tensor_hash(state)
            and source_receipt["rank"] == plan.rank and source_receipt["alpha"] == plan.alpha,
            "serialization source tensor/configuration drift")
    require(artifact_role in {"original_training_snapshot", "qualification_disposable", "derived_format_export", "mechanical_fixture"},
            "explicit snapshot role required")
    require(Path(base["path"]).is_absolute() and base["files"] and base["sha256"] == digest(base["files"]),
            "exact base snapshot required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    mapped = {n: t.detach().cpu().contiguous() for n, t in peft_names(state).items()}
    save_file(mapped, str(output / "adapter_model.safetensors"))
    reloaded = load_file(str(output / "adapter_model.safetensors"), device="cpu")
    require(tensor_hash(mapped) == tensor_hash(reloaded), "PEFT serialization tensor mismatch")
    write_new(output / "adapter_config.json", {"peft_type": "LORA", "task_type": "CAUSAL_LM",
        "inference_mode": True, "r": plan.rank, "lora_alpha": plan.alpha, "lora_dropout": 0.0,
        "bias": "none", "target_modules": ["q_proj", "v_proj"], "fan_in_fan_out": False,
        "use_rslora": False, "base_model_name_or_path": base["path"]})
    files = {n: file_hash(output / n) for n in ("adapter_config.json", "adapter_model.safetensors")}
    payload = {"schema": 2, "identity": identity, "artifact_role": artifact_role,
               "evidence_kind": source_receipt["evidence_kind"], "source_receipt": source_receipt,
               "base_snapshot": base, "tokenizer_content_sha256": digest({n: h for n, h in base["files"].items() if n in TOKENIZER_NAMES}),
               "export_files": files, "gd_content_sha256": digest(files),
               "export_tensor_sha256": tensor_hash(reloaded), "serialization_bitwise_verified": True,
               "runtime_parity_qualified": False, "requires_actual_native_peft_parity": True,
               "permits_model_promotion": False,
               "note": "Standard format and tensor serialization only; no HF/PEFT model was executed.",
               **(extra_lineage or {})}
    require(not (set(extra_lineage or {}) & {"identity", "artifact_role", "evidence_kind", "source_receipt",
            "base_snapshot", "runtime_parity_qualified", "requires_actual_native_peft_parity", "export_files",
            "gd_content_sha256", "export_tensor_sha256", "serialization_bitwise_verified", "permits_model_promotion"}),
            "lineage extension cannot override export evidence")
    write_new(output / "RDD_EXPORT.json", sealed(payload))
    return payload


def _manifest_ref(ref, identity, tasks):
    path = Path(ref["path"])
    require(path.is_absolute() and file_hash(path) == ref["sha256"], "consumed prompt manifest file drift")
    payload = unseal(read_json(path))
    validate_manifest(payload)
    require(ref["complete"] is True and ref["payload_sha256"] == digest(payload)
            and ref["event_count"] == payload["event_count"] and payload["identity"] == identity,
            "complete registered consumed prompt manifest required")
    rows = {t.key: public_row(t) for t in tasks}
    for session in payload["sessions"]:
        require(session["identity"] == identity, "manifest session registration drift")
        for event in session["events"]:
            require(event["key"] in rows and all(event[k] == v for k, v in rows[event["key"]].items()),
                    "consumed prompt is absent from the registered public catalog")
    return payload


def export_peft(checkpoint, registration, output):
    """Prepare a derived format export only after a complete native supervised run.

    GD's original trajectory uses the per-save ``checkpoint/peft`` directories.
    This standalone conversion cannot replace their original post-save receipts.
    """
    from safetensors.torch import load_file
    from .registry import check_completed_native_run, check_registration
    r, plan, tasks = check_registration(registration)
    checkpoint = Path(checkpoint).resolve()
    result_path = checkpoint.parents[1] / "RESULT.json"
    result = unseal(read_json(result_path))
    receipt = unseal(read_json(checkpoint / "receipt.json"))
    identity = digest(r)
    require(result["status"] == "COMPLETE" and result["identity"] == identity
            and result["evidence_kind"] == "qualified_native_hf", "complete registered native run required")
    check_completed_native_run(result_path, identity)
    matches = [c for c in result["checkpoints"] if c["step"] == receipt["step"] and c["receipt"] == receipt]
    require(len(matches) == 1 and receipt["adapter_sha256"] == file_hash(checkpoint / "adapter.safetensors"),
            "checkpoint provenance")
    full_manifest = _manifest_ref(result["consumed_prompt_manifest"], identity, tasks)
    checkpoint_manifest = _manifest_ref(matches[0]["consumed_prompt_manifest"], identity, tasks)
    tokenizer_hash = digest({n: h for n, h in r["assets"]["student"]["files"].items() if n in TOKENIZER_NAMES})
    require(all(e["tokenizer_content_sha256"] == tokenizer_hash for m in (full_manifest, checkpoint_manifest)
                for s in m["sessions"] for e in s["events"]), "consumed tokenizer snapshot drift")
    qualification_path = checkpoint.parents[2] / "qualification" / "RESULT.json"
    qualification = unseal(read_json(qualification_path))
    require(qualification["status"] == "PASS_SCOPED_MECHANICS_AND_TEACHER_SCREEN",
            "completed disposable native qualification required")
    state = load_file(str(checkpoint / "adapter.safetensors"), device="cpu")
    output = Path(output).resolve()
    payload = write_standard_peft_state(state, plan, output, r["assets"]["student"], identity=identity,
        source_receipt=receipt, artifact_role="derived_format_export", extra_lineage={
            "registration": {"path": str(Path(registration).resolve()), "sha256": file_hash(registration)},
            "run_result": {"path": str(result_path), "sha256": file_hash(result_path)},
            "checkpoint_receipt": {"path": str(checkpoint / "receipt.json"), "sha256": file_hash(checkpoint / "receipt.json")},
            "qualification_result": {"path": str(qualification_path), "sha256": file_hash(qualification_path)},
            "consumed_prompt_manifest": result["consumed_prompt_manifest"],
            "checkpoint_consumed_prompt_manifest": matches[0]["consumed_prompt_manifest"]})
    write_new(output / "CONSUMED_PROMPTS.json", sealed(full_manifest))
    return payload


def verify_runtime_parity(export_directory, parity_receipt):
    """Reject metadata/fixture parity; validate separately measured native receipt."""
    root = Path(export_directory).resolve()
    export = unseal(read_json(root / "RDD_EXPORT.json"))
    receipt = unseal(read_json(parity_receipt))
    require(export["evidence_kind"] == "qualified_native_hf", "fixture serialization cannot become actual native export parity")
    require(not (Path(parity_receipt).parent / "FAILED.json").exists()
            and not (Path(parity_receipt).parent / "CLEANUP_FAILED.json").exists(), "failed native parity remains failed")
    require(receipt.get("status") == "NATIVE_WRAPPER_HF_PEFT_PARITY_MEASURED"
            and receipt.get("evidence_kind") == "qualified_native_hf"
            and receipt.get("actual_native_model_execution") is True,
            "actual native HF/PEFT execution receipt required; metadata is not parity")
    require(receipt["identity"] == export["identity"]
            and receipt["export_sha256"] == file_hash(root / "RDD_EXPORT.json")
            and receipt["base_snapshot_sha256"] == export["base_snapshot"]["sha256"]
            and receipt["export_files"] == export["export_files"], "native parity export binding drift")
    require(all(file_hash(root / n) == h for n, h in export["export_files"].items()), "parity-qualified export drift")
    from .registry import inventory
    base = export["base_snapshot"]
    current_base = inventory(base["path"])
    require(current_base["path"] == str(Path(base["path"]).resolve())
            and current_base["files"] == base["files"] and current_base["sha256"] == base["sha256"],
            "parity-qualified base/tokenizer snapshot drift")
    require(receipt.get("registered_policy_sha256") and receipt.get("probes")
            and receipt.get("reviewed_by") and receipt.get("strict_peft_load") is True,
            "prospective parity policy, probes, strict load and independent review required")
    require(file_hash(receipt["logits_payload"]["path"]) == receipt["logits_payload"]["sha256"],
            "measured full-vocabulary logits payload missing/drifted")
    policy_ref = receipt["registered_policy"]
    require(file_hash(policy_ref["path"]) == policy_ref["sha256"] == receipt["registered_policy_sha256"],
            "registered parity policy missing/drifted")
    policy = unseal(read_json(policy_ref["path"]))
    require(policy["schema"] == "rdd-native-peft-parity-policy-v1"
            and policy["identity"] == export["identity"] and policy["registered_before_outcomes"] is True
            and policy["export_sha256"] == file_hash(root / "RDD_EXPORT.json")
            and policy["base_snapshot_sha256"] == export["base_snapshot"]["sha256"]
            and all(type(policy[k]) in (int, float) and math.isfinite(policy[k]) and 0 <= policy[k] <= .01
                    for k in ("abs_tolerance", "rel_tolerance"))
            and type(policy["vocab_size"]) is int and policy["vocab_size"] > 1
            and type(policy["max_new_tokens"]) is int and 0 < policy["max_new_tokens"] <= 32
            and type(policy["probes"]) is list and 0 < len(policy["probes"]) <= 8
            and type(policy["eos_ids"]) is list and policy["eos_ids"]
            and all(type(t) is int and 0 <= t < policy["vocab_size"] for t in policy["eos_ids"]),
            "registered measured parity identity drift")
    base_config = Path(export["base_snapshot"]["path"]) / "config.json"
    base_data = read_json(base_config)
    base_eos = base_data["eos_token_id"]
    base_eos = base_eos if type(base_eos) is list else [base_eos]
    require(file_hash(base_config) == export["base_snapshot"]["files"]["config.json"]
            and base_data["vocab_size"] == policy["vocab_size"]
            and set(base_eos) == set(policy["eos_ids"]), "full-vocabulary/EOS policy/base mismatch")
    load_ref = receipt["native_load_receipt"]
    require(file_hash(load_ref["path"]) == load_ref["sha256"], "strict native load receipt drift")
    load = unseal(read_json(load_ref["path"]))
    require(load["status"] == "STRICT_NATIVE_HF_PEFT_LOAD" and load["actual_native_model_execution"] is True
            and load["base_snapshot_sha256"] == export["base_snapshot"]["sha256"]
            and all(load.get(k) == [] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
            and load.get("local_files_only") is True and load.get("trust_remote_code") is False,
            "strict native load provenance")
    import torch
    from safetensors.torch import load_file
    logits = load_file(receipt["logits_payload"]["path"], device="cpu")
    required = set()
    for index, probe in enumerate(receipt["probes"]):
        require(all(type(probe[k]) is list and probe[k]
                    and all(type(t) is int and 0 <= t < policy["vocab_size"] for t in probe[k])
                    for k in ("prompt_token_ids", "generated_token_ids")), "native parity token identity/range")
        require(probe["key"] == policy["probes"][index]["key"]
                and probe["prompt_hash"] == policy["probes"][index]["prompt_hash"]
                and digest(probe["prompt_token_ids"]) == policy["probes"][index]["prompt_token_sha256"]
                and 0 < len(probe["generated_token_ids"]) <= policy["max_new_tokens"],
                "measured parity probe schedule drift")
        eos = policy["eos_ids"]
        generated = probe["generated_token_ids"]
        require(not any(t in eos for t in generated[:-1])
                and (generated[-1] in eos or len(generated) == policy["max_new_tokens"]), "incomplete parity greedy trace")
        for position, token in enumerate(probe["generated_token_ids"]):
            names = [f"probe{index}.token{position}.{role}" for role in ("wrapper", "peft")]
            required.update(names)
            left, right = (logits[n] for n in names)
            require(left.shape == right.shape == (1, policy["vocab_size"])
                    and bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all())
                    and torch.allclose(left, right, atol=policy["abs_tolerance"], rtol=policy["rel_tolerance"])
                    and int(left.argmax()) == int(right.argmax()) == token, "measured full-vocabulary parity mismatch")
    require(set(logits) == required and len(receipt["probes"]) == len(policy["probes"]),
            "measured parity payload/probe coverage drift")
    require(all(p.get("full_vocabulary_logits_compared") is True and p.get("greedy_tokens_equal") is True
                and p.get("finite_logits") is True and p.get("logits_within_registered_tolerance") is True
                and p.get("rendered_prompt_token_ids_equal") is True for p in receipt["probes"]),
            "incomplete native logits/greedy/rendered-prompt parity")
    return receipt


def measure_loaded_peft_parity(backend, peft_model, peft_tokenizer, *, export_directory,
                               policy_path, native_load_receipt, output, reviewed_by, allow_native_forward=False):
    """Future bounded native check on independently loaded models; never a loader.

    Only call inside an independently authorized supervised process. The immutable
    policy binds exact probes, tolerance and generation cap before outcomes. Full
    vocabulary logits are retained at every shared greedy prefix. Any greedy or
    registered tolerance mismatch fails with a preserved receipt. Preparation and
    fixture serialization do not call this function.
    """
    import torch
    from safetensors.torch import load_file, save_file
    from .backend import tensor_hash
    from .data import Problem, normalized
    require(allow_native_forward is True and backend.evidence_kind == "qualified_native_hf",
            "separate explicit supervised native parity authorization required")
    from peft import get_peft_model_state_dict
    require(type(peft_model).__name__ == "PeftModelForCausalLM"
            and not any(p.requires_grad for p in peft_model.parameters()), "strict frozen standard PEFT model required")
    require(not ({p.data_ptr() for p in backend.student.parameters()} & {p.data_ptr() for p in peft_model.parameters()}),
            "independently loaded wrapper and PEFT models required")
    root = Path(export_directory).resolve()
    export = unseal(read_json(root / "RDD_EXPORT.json"))
    policy = unseal(read_json(policy_path))
    load_receipt = unseal(read_json(native_load_receipt))
    require(load_receipt.get("status") == "STRICT_NATIVE_HF_PEFT_LOAD" and load_receipt.get("actual_native_model_execution") is True
            and load_receipt["base_snapshot_sha256"] == export["base_snapshot"]["sha256"]
            and all(load_receipt.get(k) == [] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
            and load_receipt.get("local_files_only") is True and load_receipt.get("trust_remote_code") is False,
            "independently measured strict native HF/PEFT load receipt required")
    require(policy.get("schema") == "rdd-native-peft-parity-policy-v1"
            and policy.get("registered_before_outcomes") is True
            and policy["identity"] == backend.prompt_identity == export["identity"]
            and policy["export_sha256"] == file_hash(root / "RDD_EXPORT.json")
            and policy["base_snapshot_sha256"] == export["base_snapshot"]["sha256"],
            "prospective exact native parity policy required")
    require(type(policy["max_new_tokens"]) is int and 0 < policy["max_new_tokens"] <= 32
            and type(policy["abs_tolerance"]) in (int, float) and math.isfinite(policy["abs_tolerance"]) and 0 <= policy["abs_tolerance"] <= .01
            and type(policy["rel_tolerance"]) in (int, float) and math.isfinite(policy["rel_tolerance"]) and 0 <= policy["rel_tolerance"] <= .01
            and type(policy["vocab_size"]) is int and policy["vocab_size"] == backend.vocab
            and policy["eos_ids"] == list(backend.eos_ids)
            and 0 < len(policy["probes"]) <= 8 and reviewed_by, "bounded reviewed parity policy required")
    require(tensor_hash(backend.snapshot()) == export["source_receipt"]["tensor_sha256"], "wrapper adapter snapshot mismatch")
    saved = load_file(str(root / "adapter_model.safetensors"), device="cpu")
    loaded = {n: t.detach().cpu() for n, t in get_peft_model_state_dict(peft_model).items()}
    require(set(saved) == set(loaded) and tensor_hash(saved) == tensor_hash(loaded), "strict PEFT tensor load mismatch")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "STARTED.json", sealed({"identity": backend.prompt_identity,
        "registered_policy_sha256": file_hash(policy_path), "native_execution_authorized": True}))
    prior_phase = backend.meter.phase
    backend.meter.phase = "export_parity"
    logits_payload, probes = {}, []
    try:
        for index, probe in enumerate(policy["probes"]):
            row = backend.prompt_catalog[probe["key"]]
            problem = Problem(row["key"], row["prompt"])
            require(probe["prompt_hash"] == digest(normalized(problem.prompt)), "parity public probe drift")
            prefix = backend.prompt(problem)
            peft_ids = tuple(peft_tokenizer.apply_chat_template([{"role": "user", "content": problem.prompt}],
                tokenize=True, add_generation_prompt=True))
            require(peft_ids == prefix and digest(prefix) == probe["prompt_token_sha256"], "parity prompt token mismatch")
            rendered = backend.tokenizer.apply_chat_template([{"role": "user", "content": problem.prompt}],
                tokenize=False, add_generation_prompt=True)
            require(rendered == peft_tokenizer.apply_chat_template([{"role": "user", "content": problem.prompt}],
                tokenize=False, add_generation_prompt=True), "parity rendered text mismatch")
            generated, max_abs = [], 0.0
            for token_index in range(policy["max_new_tokens"]):
                backend.guard()
                peft_model.to("cpu")
                # A separately admitted post-run parity action may release the
                # unused teacher before loading PEFT. No teacher is used here.
                student = backend.student.to(backend.device).eval()
                with torch.no_grad():
                    h = backend.hidden(student, prefix, tuple(generated) + (0,), False, "student",
                        problem=problem, purpose="peft_parity_wrapper_logits")[-1:]
                    reference = student.lm_head(h).float().detach().cpu()
                student.to("cpu")
                backend.guard()
                if backend.device.type == "cuda":
                    torch.cuda.empty_cache()
                peft_model.to(backend.device).eval()
                full = prefix + tuple(generated)
                backend.meter.add("student_forward_input_tokens", len(full))
                backend.record_prompt_consumption(problem, role="student", purpose="peft_parity_hf_logits",
                    kind="forward", input_tokens=len(full), rendered_prompt_ids=prefix, forward_input_ids=full)
                with torch.no_grad():
                    observed = peft_model(input_ids=torch.tensor([full], dtype=torch.long, device=backend.device),
                                          use_cache=False).logits[0, -1:].float().detach().cpu()
                require(reference.shape == observed.shape == (1, policy["vocab_size"]) and bool(torch.isfinite(reference).all())
                        and bool(torch.isfinite(observed).all()), "nonfinite/shape-mismatched parity logits")
                logits_payload[f"probe{index}.token{token_index}.wrapper"] = reference
                logits_payload[f"probe{index}.token{token_index}.peft"] = observed
                error = float((reference - observed).abs().max())
                max_abs = max(max_abs, error)
                require(torch.allclose(reference, observed, atol=policy["abs_tolerance"], rtol=policy["rel_tolerance"]),
                        "native PEFT logits exceed registered tolerance")
                token = int(reference.argmax())
                require(token == int(observed.argmax()), "native PEFT greedy token mismatch")
                generated.append(token)
                if token in backend.eos_ids:
                    break
            probes.append({"key": problem.key, "prompt_hash": row["prompt_hash"], "rendered_prompt": rendered,
                "prompt_token_ids": list(prefix), "generated_token_ids": generated,
                "full_vocabulary_logits_compared": True, "greedy_tokens_equal": True, "finite_logits": True,
                "logits_within_registered_tolerance": True, "rendered_prompt_token_ids_equal": True,
                "max_abs_logit_error": max_abs})
        save_file(logits_payload, str(output / "full-vocabulary-logits.safetensors"))
        payload = {"status": "NATIVE_WRAPPER_HF_PEFT_PARITY_MEASURED", "evidence_kind": "qualified_native_hf",
            "actual_native_model_execution": True, "identity": export["identity"],
            "export_sha256": file_hash(root / "RDD_EXPORT.json"), "base_snapshot_sha256": export["base_snapshot"]["sha256"],
            "export_files": export["export_files"], "registered_policy_sha256": file_hash(policy_path),
            "registered_policy": {"path": str(Path(policy_path).resolve()), "sha256": file_hash(policy_path)},
            "native_load_receipt": {"path": str(Path(native_load_receipt).resolve()), "sha256": file_hash(native_load_receipt)},
            "strict_peft_load": True, "reviewed_by": reviewed_by, "probes": probes,
            "logits_payload": {"path": str(output / "full-vocabulary-logits.safetensors"),
                               "sha256": file_hash(output / "full-vocabulary-logits.safetensors")},
            "consumed_prompt_manifest": backend.write_prompt_manifest(output / "CONSUMED_PROMPTS.json", complete=True),
            "authorizes_gd_native_scoring": False, "permits_model_promotion": False}
        write_new(output / "RESULT.json", sealed(payload))
        return payload
    except BaseException as exc:
        if logits_payload:
            save_file(logits_payload, str(output / "FAILED_LOGITS.safetensors"))
        manifest, manifest_error = None, None
        try:
            manifest = backend.write_prompt_manifest(output / "FAILED_CONSUMED_PROMPTS.json")
        except BaseException as manifest_exc:
            manifest_error = {"type": type(manifest_exc).__name__, "message": str(manifest_exc)}
        write_new(output / "FAILED.json", {"type": type(exc).__name__, "message": str(exc),
                                          "partial_probes": probes, "runtime_parity_qualified": False,
                                          "consumed_prompt_manifest": manifest, "manifest_error": manifest_error})
        raise
    finally:
        try:
            peft_model.to("cpu")
            backend.meter.phase = prior_phase
        except BaseException as exc:
            write_new(output / "CLEANUP_FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
            raise


def _write_jsonl(path, rows):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    return {"path": str(path), "sha256": file_hash(path)}


def prepare_gd_public_source_inventory(tasks, output, *, reviewed_catalog=None):
    """Freeze the public training/replay DTO projection before future training.

    Keep the reviewed full catalog/labels separately. This is the actual source
    projection that model prompts will consume, not a claimed original upstream
    raw data source. Its file reference must enter the prospective producer plan.
    """
    rows = [public_row(t) for t in sorted(tasks, key=lambda t: t.key)
            if t.role in {"train", "retention_train"}]
    require(rows and len({r["key"] for r in rows}) == len(rows), "unique public train/replay source required")
    reference = _write_jsonl(output, rows)
    if reviewed_catalog is not None:
        require(Path(reviewed_catalog["path"]).is_absolute()
                and file_hash(reviewed_catalog["path"]) == reviewed_catalog["sha256"], "reviewed catalog file reference drift")
    write_new(str(output) + ".projection.json", sealed({"schema": 1, "kind": "public_problem_dto_projection",
        "source_inventory": reference, "reviewed_full_catalog": reviewed_catalog,
        "upstream_raw_source_reconstructed": False, "contains_private_labels": False,
        "task_projection_sha256": digest(rows), "must_bind_before_training": True,
        "native_admission_ready": reviewed_catalog is not None}))
    return reference


def prepare_gd_consumed_provenance(*, consumed_manifest, source_inventory, tokenizer_sha256,
                                   final_step, output, producer_plan=None):
    """Offline projection to GD source/filter/consume/unused JSONL contracts.

    This alone is not an attachment: genuine prospective approval, completion,
    exact base/tokenizer and original post-save GD receipts remain required.
    Qualification, bank generation and DEV exposures stay in the separately
    retained complete RDD manifest; only gradient inputs enter GD's training file.
    """
    require(type(final_step) is int and final_step > 0, "completed final step required")
    require(file_hash(consumed_manifest["path"]) == consumed_manifest["sha256"], "consumed manifest drift")
    payload = unseal(read_json(consumed_manifest["path"]))
    validate_manifest(payload)
    require(file_hash(source_inventory["path"]) == source_inventory["sha256"], "public source projection drift")
    source_path = Path(source_inventory["path"])
    require(source_path.is_absolute(), "retained absolute source projection required")
    projection_path = Path(str(source_path) + ".projection.json")
    projection = unseal(read_json(projection_path))
    require(projection["kind"] == "public_problem_dto_projection" and projection["source_inventory"] == source_inventory,
            "retained public DTO source projection identity required")
    if payload["evidence_kind"] == "qualified_native_hf":
        require(projection["native_admission_ready"] is True and producer_plan is not None,
                "native GD provenance requires genuine prospective source/canonical catalog binding")
        require(file_hash(producer_plan["path"]) == producer_plan["sha256"], "prospective producer plan drift")
        plan = read_json(producer_plan["path"])
        require(plan["status"] == "APPROVED_PROSPECTIVE" and plan["purpose"] == "research_training"
                and plan["public_source_inventory"] == source_inventory
                and plan["rdd_registration_identity"] == payload["identity"],
                "public source projection must be bound in genuine prospective producer plan")
    rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(rows and len({r["key"] for r in rows}) == len(rows)
            and all(r["role"] in {"train", "retention_train"} for r in rows), "public training source projection")
    require(projection["task_projection_sha256"] == digest(rows), "public DTO projection content drift")
    events, sources, used = [], {}, set()
    for ordinal, row in enumerate(rows, 1):
        sid = "source:" + digest([source_inventory["sha256"], ordinal])
        fid = "filter:" + digest(sid)
        sources[row["key"]] = (sid, fid, row)
        events.append({"event_id": sid, "event_type": "source", "parent_id": row["parent_id"],
                       "source_row_id": row["key"], "source_row_number": ordinal,
                       "source_ref": source_inventory, "raw_record": row})
        events.append({"event_id": fid, "event_type": "filter", "source_event_id": sid,
                       "decision": "accepted", "reason": "registered public catalog admission; no rollout outcome filtering",
                       "filtering_history": [{"operation": "registered_public_catalog_admission", "decision": "keep",
                                              "reason": "all enrolled public train/replay source rows retained"}]})
    consumed, per_step = [], {str(step): 0 for step in range(1, final_step + 1)}
    last_step = 0
    for session in payload["sessions"]:
        for event in session["events"]:
            if event["phase"] != "training" or event.get("gradient") is not True:
                continue
            require(event["model"] == "student" and event["role"] in {"train", "retention_train"}
                    and event["key"] in sources, "GD projection encountered foreign gradient input")
            require(event["tokenizer_content_sha256"] == tokenizer_sha256, "GD consumed tokenizer drift")
            step = event["step"] + 1
            require(last_step <= step <= final_step, "GD consumption optimizer order drift")
            last_step = step
            sid, fid, row = sources[event["key"]]
            require(all(event[k] == v for k, v in row.items()), "public source versus consumed event drift")
            used.add(event["key"])
            per_step[str(step)] += 1
            item = {"event_id": "consume:" + digest([payload["identity"], len(consumed), event]),
                    "event_type": "consume", "source_event_id": sid, "filter_event_id": fid,
                    "parent_id": row["parent_id"], "source_row_id": row["key"], "step": step,
                    **{k: event[k] for k in ("prompt", "raw_rendered_prompt", "consumed_prompt", "input_ids",
                                              "tokenizer_content_sha256", "rendering_history", "truncation")},
                    "rdd_purpose": event["purpose"], "public_prompt_hash": event["prompt_hash"],
                    "response_tokens": event["response_tokens"], "response_token_sha256": event["response_token_sha256"]}
            events.append(item)
            consumed.append({"prompt": item["consumed_prompt"], "parent_id": item["parent_id"]})
    require(all(per_step.values()), "GD consumed provenance must cover every completed optimizer step")
    for key in sorted(sources.keys() - used):
        sid, fid, _ = sources[key]
        events.append({"event_id": "unused:" + digest(sid), "event_type": "unused",
                       "source_event_id": sid, "filter_event_id": fid,
                       "reason": "prespecified task schedule did not sample this accepted public row"})
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    refs = {"events": _write_jsonl(output / "events.jsonl", events),
            "training_manifest": _write_jsonl(output / "training_manifest.jsonl", consumed)}
    counts = {"source_count": len(rows), "rejected_count": 0, "consumed_count": len(consumed),
              "unused_accepted_count": len(sources) - len(used), "consumed_per_step": per_step}
    write_new(output / "RDD_PROVENANCE_PROJECTION.json", sealed({"schema": 1, "provenance": refs,
        **counts, "full_consumed_manifest": consumed_manifest, "source_inventory": source_inventory,
        "public_source_projection": {"path": str(projection_path), "sha256": file_hash(projection_path)},
        "reviewed_full_catalog": projection["reviewed_full_catalog"],
        "source_inventory_is_public_dto_projection_not_upstream_raw": True,
        "all_non_gradient_exposures_retained_separately": True,
        "public_prompt_hashes": payload["public_prompt_hashes"], "gd_attachment_complete": False,
        "native_parity_qualified": False, "authorizes_training": False}))
    return {"provenance": refs, **counts}


def record_gd_post_save(*, checkpoint, producer_plan, receipt_directory, gd_after_completed_save):
    """Explicit future owner hook, after the original standard PEFT save succeeds.

    The caller supplies GD's reviewed callback; RDD never imports another checkout
    implicitly. GD's callback checks genuine APPROVED_PROSPECTIVE plan/path/step.
    """
    checkpoint = Path(checkpoint).resolve()
    receipt = unseal(read_json(checkpoint / "receipt.json"))
    peft = checkpoint / "peft"
    export = unseal(read_json(peft / "RDD_EXPORT.json"))
    require(export["artifact_role"] == "original_training_snapshot"
            and export["source_receipt"] == receipt, "original immutable training PEFT save required")
    require(file_hash(producer_plan["path"]) == producer_plan["sha256"], "prospective producer plan drift")
    plan = read_json(producer_plan["path"])
    require(plan["status"] == "APPROVED_PROSPECTIVE" and plan["purpose"] == "research_training"
            and plan["producer"]["independently_approved"] is True,
            "genuine prospective research producer approval required")
    return gd_after_completed_save(approval_plan=producer_plan, receipt_directory=str(receipt_directory),
                                   checkpoint_directory=str(peft), step=receipt["step"], kind="peft")
