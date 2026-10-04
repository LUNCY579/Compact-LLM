"""Local-only standard HF export worker, launched only by the bounded supervisor.

Native compatibility is UNQUALIFIED at delivery. No checkpoint conversion,
remote teacher, custom readout, QLoRA, or stock-Qwen fallback is implemented.
"""
from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

from .contracts import digest, file_hash, matrix, read_json, require, sealed, unseal, write_new


def load_backend(registration, plan, tasks, seed, guard):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .backend import TorchBackend, install_lora
    models, tokenizers = {}, {}
    for role in ("student", "teacher"):
        guard()
        path = registration["assets"][role]["path"]
        tokenizers[role] = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        model, info = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            torch_dtype=torch.bfloat16, attn_implementation="eager", output_loading_info=True)
        require(not any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
                "incomplete native model load")
        require(type(model).__name__ == "Qwen3ForCausalLM", "unsupported native architecture")
        require(not any(p.is_meta for p in model.parameters()), "unmaterialized native parameters")
        require(sum(p.numel() for p in model.parameters()) == registration["binding"][role+"_parameters"],
                "native parameter identity mismatch")
        require(plan.max_context_tokens <= model.config.max_position_embeddings, "model context capacity")
        model.requires_grad_(False).eval()
        models[role] = model
    def tokenizer_identity(tok):
        return digest({"backend": tok.backend_tokenizer.to_str(), "vocab": tok.get_vocab(),
                       "special": tok.special_tokens_map, "chat_template": tok.chat_template})
    require(tokenizer_identity(tokenizers["student"]) == tokenizer_identity(tokenizers["teacher"]),
            "teacher/student tokenizer, vocabulary, or template mismatch")
    tok = tokenizers["student"]
    eos_sets = []
    for role in ("student", "teacher"):
        e = models[role].config.eos_token_id
        eos_sets.append(set(e if isinstance(e, list) else [e]))
    require(eos_sets[0] == eos_sets[1] and tok.eos_token_id in eos_sets[0], "native EOS mismatch")
    eos = tuple([tok.eos_token_id] + sorted(eos_sets[0] - {tok.eos_token_id}))
    # Prove the inserted zero-B adapter does not change the exact loaded export.
    problem = next(t for t in tasks if t.role == "teacher_dev").public()
    ids = tok.apply_chat_template([{"role": "user", "content": problem.prompt}],
                                  tokenize=True, add_generation_prompt=True)
    require(len(ids) + plan.max_new_tokens <= plan.max_context_tokens, "native probe context overflow")
    student = models["student"].to("cuda:0")
    with torch.no_grad():
        x = torch.tensor([ids], dtype=torch.long, device="cuda:0")
        before = student.lm_head(student.model(input_ids=x, use_cache=False).last_hidden_state[:, -1:]).detach().cpu()
    student.to("cpu")
    torch.cuda.empty_cache()
    install_lora(student, plan, seed)
    student.to("cuda:0")
    with torch.no_grad():
        after = student.lm_head(student.model(input_ids=x, use_cache=False).last_hidden_state[:, -1:]).detach().cpu()
    require(torch.equal(before, after), "zero-adapter native-logit equivalence failed")
    return TorchBackend(student, models["teacher"], tok, plan, eos_ids=eos, device="cuda:0",
                        guard=guard, evidence_kind="qualified_native_hf")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--token", required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--arm")
    mode.add_argument("--selection")
    p.add_argument("--confirmation")
    a = p.parse_args()
    require(bool(os.environ.get("RDD001_WORKER_TOKEN")) and a.token == os.environ.get("RDD001_WORKER_TOKEN"),
            "must be launched by the authorized supervisor")
    from .registry import check_registration
    from .host import runtime_check, worker_guard
    runtime_check()
    r, plan, tasks = check_registration(a.registration)
    identity = digest(r)
    out = Path(a.output)
    require(out.is_dir() and (out / "SUPERVISOR_STARTED.json").is_file(), "missing supervised output")
    started = time.monotonic()
    guard = lambda: worker_guard(out, started)
    if a.arm:
        from .engine import run_arm
        from .qualification import qualify
        arms = {x.name: x for x in matrix(plan)}
        require(a.arm in arms, "arm not registered")
        arm = arms[a.arm]
        backend = load_backend(r, plan, tasks, arm.seed, guard)
        qualify(backend, tasks, out / "qualification")
        run_arm(backend, tasks, arm, out / "arm", identity=identity,
                bank_path=out.parent / "teacher_banks" / identity / f"seed-{arm.seed}.json")
        write_new(out / "NATIVE_RESULT.json", sealed({"status": "COMPLETED_NOT_PROMOTED", "identity": identity,
                  "qualification_sha256": file_hash(out / "qualification" / "RESULT.json"),
                  "result_sha256": file_hash(out / "arm" / "RESULT.json"), "deployment_approved": False}))
    else:
        from .data import CONFIRM_ROLES, cross_audit, load_catalog
        from .metrics import decision_features, evaluate, paired
        selection_path = Path(a.selection)
        chosen = unseal(read_json(selection_path))
        marker = unseal(read_json(selection_path.parent / "CONFIRMATION_STARTED.json"))
        require(chosen["status"] == "DEVELOPMENT_SELECTED" and chosen["identity"] == identity
                and chosen["evidence_kind"] == "qualified_native_hf", "native locked selection required")
        require(marker["selection_sha256"] == file_hash(selection_path)
                and marker["identity"] == identity and marker["confirmation_sha256"] == r["binding"]["confirmation_sha256"],
                "confirmation marker drift")
        require(file_hash(a.confirmation) == r["binding"]["confirmation_sha256"], "unregistered confirmation bank")
        confirm, provenance, audit = load_catalog(a.confirmation, read_json(r["files"]["exclusions"]), confirmation=True)
        require(provenance == "reviewed_external" and {t.role for t in confirm} == CONFIRM_ROLES,
                "independent complete confirmation catalog required")
        cross_audit(tasks, confirm)
        results = {}
        require(set(chosen["selected_seeds"]) == {str(s) for s in plan.seeds}, "confirmation seed coverage")
        for seed in plan.seeds:
            picked = chosen["selected_seeds"][str(seed)]
            require(file_hash(picked["result_path"]) == picked["result_sha256"], "selected run drift")
            backend = load_backend(r, plan, tasks, seed, guard)
            backend.meter.phase = "confirmation"
            baseline = evaluate(backend, confirm, "student", seed)
            checkpoint = Path(picked["checkpoint"])
            require(unseal(read_json(checkpoint / "receipt.json")) == picked["receipt"], "selected checkpoint drift")
            backend.load(checkpoint)
            predictions = evaluate(backend, confirm, "student", seed)
            summary = paired(baseline, predictions)
            results[str(seed)] = {"summary": summary, "features": decision_features(summary, prefix="confirm"),
                                  "baseline": baseline, "predictions": predictions, "meter": backend.meter.snapshot()}
            del backend
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        passed = all(v["features"]["acquisition_gain"] >= plan.min_acquisition_gain
                     and v["features"]["worst_retention_drop"] <= plan.max_retention_drop for v in results.values())
        write_new(out / "CONFIRMATION_RESULT.json", sealed({"status": "SCOPED_CONFIRMATION_PASS" if passed else "SCOPED_CONFIRMATION_FAIL",
                  "identity": identity, "audit": audit, "seeds": results, "deployment_approved": False}))


if __name__ == "__main__":
    main()
