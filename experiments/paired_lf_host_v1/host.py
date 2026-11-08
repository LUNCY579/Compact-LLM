"""Pin and reuse the existing native loader and CED conversion, without editing them.

Only full-prefix frozen captures are supported. This does not qualify CED cached
execution, restore failed legacy receipts, or run optimizer steps on a backbone.
"""
from __future__ import annotations
from dataclasses import asdict, replace
import gc
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import torch
from safetensors.torch import save_file
from experiments.paired_lf_v1.bank import finish, save_feature
from experiments.paired_lf_v1.contracts import (HOSTS, CODES, check_rows, digest, file_hash,
                                              read_json, regular, require)
from experiments.paired_lf_v1.model import Feature

PARENT = "a36de96b30eb03265db3a18e464a2e2a23ee67bf"
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PACKAGE = "experiments/architecture_benchmark_001"


def git(*args: str) -> str:
    p = subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, timeout=30)
    return p.stdout.decode("utf-8")


def verify_parent() -> dict:
    require(git("rev-parse", PARENT + "^{commit}").strip() == PARENT, "pinned parent unavailable")
    paths = git("ls-tree", "-r", "--name-only", PARENT, PACKAGE).splitlines()
    relevant = [p for p in paths if p.endswith(".py") or
                p in {PACKAGE + "/protocol.json", PACKAGE + "/source_lock.json"}]
    require(len(relevant) >= 10, "parent source tree unexpectedly incomplete")
    identities = {}
    for relative in relevant:
        raw = subprocess.run(["git", "-C", str(REPO), "show", f"{PARENT}:{relative}"],
                             capture_output=True, check=True, timeout=30).stdout
        actual = regular(REPO / relative).read_bytes()
        require(raw.replace(b"\r\n", b"\n") == actual.replace(b"\r\n", b"\n"),
                f"inherited source differs from pinned parent: {relative}")
        identities[relative] = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    untracked = git("ls-files", "--others", "--exclude-standard", "--", PACKAGE).splitlines()
    require(not any(p.endswith(".py") for p in untracked), "untracked inherited Python can alter imports")
    return identities


def preflight(source: Path) -> dict:
    """Read metadata only, no torch checkpoint or tokenizer/model load."""
    blockers = []
    try:
        version = importlib.metadata.version("transformers")
        if version != "4.51.3":
            blockers.append(f"requires Transformers 4.51.3, found {version}")
    except importlib.metadata.PackageNotFoundError:
        blockers.append("Transformers is absent")
    for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (source / name).is_file():
            blockers.append(f"official source asset missing: {name}")
    try:
        parent = verify_parent()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parent = None
        blockers.append(f"pinned inherited source unavailable: {type(exc).__name__}")
    return {"schema": "plf.host-preflight/1", "status": "BLOCKED" if blockers else "PREFLIGHT_ONLY",
            "blockers": blockers, "parent": PARENT, "parent_source": parent,
            "native_weights_loaded": False, "optimizer_steps": 0,
            "note": "metadata availability is not source verification or qualification"}


def frozen_identity(model) -> str:
    require(not model.training and all(not p.requires_grad for p in model.parameters()), "backbone not frozen/eval")
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        require(value.device.type == "cpu" and torch.isfinite(value).all(), "nonfinite/non-CPU backbone")
        h.update(digest([name, str(value.dtype), list(value.shape)]).encode())
        h.update(memoryview(value.detach().contiguous().view(torch.uint8).numpy()).cast("B"))
    return h.hexdigest()


def capture_features(model, ids: torch.Tensor, candidate_ids: tuple[int, ...],
                     encoder_cut: int) -> Feature:
    """Problem-only input. Hook removed even when the host raises an exception."""
    require(not model.training and all(not p.requires_grad for p in model.parameters()), "host must be frozen")
    require(ids.dtype == torch.long and ids.device.type == "cpu" and ids.ndim == 2 and
            ids.shape[0] == 1 and 1 <= ids.shape[1] <= 512, "invalid native input")
    require(len(candidate_ids) == 7 and len(set(candidate_ids)) == 7 and
            all(type(i) is int and 0 <= i < model.config.vocab_size for i in candidate_ids), "invalid native candidate IDs")
    require(type(encoder_cut) is int and 1 <= encoder_cut <= len(model.encoder), "invalid memory extraction layer")
    captured = []
    def collect(module, args, output):
        require(isinstance(output, tuple) and isinstance(output[0], torch.Tensor), "unexpected inherited block output")
        captured.append(output[0].detach().clone())
    hook = model.encoder[encoder_cut - 1].register_forward_hook(collect)
    try:
        with torch.inference_mode():
            output = model(ids, use_cache=False, compute_logits=False, return_hidden=True)
            require(len(captured) == 1 and output.hidden.shape[:2] == ids.shape, "host changed hidden shape/hook execution")
            query = output.hidden[0, -1].detach().clone()
            prior = torch.nn.functional.linear(query, model.embed_tokens.weight[list(candidate_ids)])
    finally:
        hook.remove()
    feature = Feature(captured[0][0], query, prior.detach().clone())
    feature.validate()
    return feature


def _ids(tokenizer, row: dict) -> torch.Tensor:
    value = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
    require(isinstance(value, torch.Tensor) and value.dtype == torch.long and
            value.ndim == 2 and value.shape[0] == 1 and 1 <= value.shape[1] <= 512,
            "native prompt outside frozen context bound; no truncation")
    return value.cpu()


def capture(source: Path, rows: list, out: Path) -> dict:
    """Explicit new frozen diagnostic; never calls the inherited optimizer CLI."""
    check_rows(rows)
    before_code = verify_parent()
    # Only import inherited modules after verifying their actual checked-out bytes.
    sys.path.insert(0, str(REPO))
    from experiments.architecture_benchmark_001 import run
    from experiments.architecture_benchmark_001.source import audit_source
    from experiments.architecture_benchmark_001.transfer import to_ced
    require(not any(p.is_symlink() for p in (source.absolute(), *source.absolute().parents)), "source symlink ancestors forbidden")
    audit = audit_source(source)
    reference, tokenizer, native = run._load(source, "cpu", native=True)
    require(reference.config.hidden_size == 1024 and reference.config.total_layers == 28 and
            reference.config.intermediate_size == 3072, "unregistered donor shape")
    candidates = tuple(tokenizer.encode(code, add_special_tokens=False) for code in CODES)
    require(all(len(ids) == 1 for ids in candidates), "A-G must each be one native token")
    candidate_ids = tuple(ids[0] for ids in candidates)
    require(len(set(candidate_ids)) == 7 and all(tokenizer.decode([i]) == code
            for i, code in zip(candidate_ids, CODES)), "candidate roundtrip mismatch")
    # Qualification prompts use CALIBRATION only; never decision labels/results.
    check_prompts = [r for r in rows if r["split"] == "calibration"][:2]
    require(len(check_prompts) == 2, "two calibration prompts required")
    with torch.inference_mode():
        for row in check_prompts:
            ids = _ids(tokenizer, row)
            actual = reference(ids, last_position_only=True).logits[0, -1]
            expected = native(ids, use_cache=False).logits[0, -1]
            require(torch.isfinite(actual).all() and torch.isfinite(expected).all(), "nonfinite native/reference logits")
            require(bool((abs(actual - expected) <= 1e-4 + 1e-4 * abs(expected)).all()) and
                    int(actual.argmax()) == int(expected.argmax()), "strict native/reference parity failed")
    del native
    gc.collect()
    reference.requires_grad_(False).eval()
    hosts = {}
    model = reference
    for host in HOSTS:
        conversion = {"kind": "unchanged_ordinary_source"}
        if host == "CED_LAST4":
            config = replace(reference.config, architecture="ced", encoder_layers=24, decoder_layers=4)
            model, conversion = to_ced(reference, config, donor_layer=24)
            model.requires_grad_(False).eval()
            del reference
            gc.collect()
        initial = frozen_identity(model)
        (out / host).mkdir(exist_ok=False)
        entries = {}
        for row in rows:
            ids = _ids(tokenizer, row)
            feature = capture_features(model, ids, candidate_ids, 24)
            path = out / host / (row["id"] + ".safetensors")
            save_feature(path, feature)
            entries[row["id"]] = {"path": path.relative_to(out).as_posix(), "sha256": file_hash(path),
                                  "input_ids_sha256": digest(ids.tolist()), "tokens": ids.shape[1]}
        final = frozen_identity(model)
        require(initial == final, "frozen host mutated during capture")
        hosts[host] = {"architecture": host, "native_dim": 1024, "source_assets": audit["assets"],
                       "frozen_before": initial, "frozen_after": final, "weights_origin": "source_initialization",
                       "features": entries, "config": asdict(model.config), "conversion": conversion,
                       "candidate_ids": list(candidate_ids), "ordinary_loader_parity_passed": True,
                       "source_commit": PARENT, "memory_cut": 24, "feature_mode": "full_prefix_only",
                       "parent_source_sha256": digest(before_code)}
    require(verify_parent() == before_code and audit_source(source) == audit, "source/code drift during capture")
    for row in rows:
        require(hosts[HOSTS[0]]["features"][row["id"]]["input_ids_sha256"] ==
                hosts[HOSTS[1]]["features"][row["id"]]["input_ids_sha256"], "different host-visible tokens")
    return finish(out, rows, hosts, kind="native_frozen_diagnostic")
