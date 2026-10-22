"""Fixed scientific contract and exclusive BusinessProject output policy."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

BUSINESS = Path("C:/Users/clclc/BusinessProject").resolve()
SOURCE_ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("reasoning", "transition", "arithmetic", "grounding", "long_context", "format")
DOMAIN_ROLE = {"reasoning": "reasoning", "transition": "preservation", "arithmetic": "preservation",
               "grounding": "grounding", "long_context": "preservation", "format": "format"}
ROLE_SHARES = {"reasoning": .50, "preservation": .25, "grounding": .15, "format": .10}
TRAIN_COUNTS = {"reasoning": 1000, "transition": 166, "arithmetic": 166,
                "grounding": 300, "long_context": 168, "format": 200}
PARENT_SHA = "a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284"
SUPA_SHA = "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b"
OWNER_CHAT = "01a0dfa3-9cdf-7703-a362-32f559ec15a7"
CONTRACT = {"experiment": "TITAN-JOINT-QUALITY-001", "parent_sha256": PARENT_SHA,
    "native_authorized": False, "training_authorized": False, "confirmation_authorized": False,
    "native_seconds_cap": 21600, "finalization_reserve_seconds": 1800,
    "processed_tokens_cap_per_arm": 54172, "max_row_tokens": 1024, "max_update_tokens": 2048,
    "max_eval_total_tokens": 4096, "max_new_tokens": 48, "mixture_tolerance": .05,
    "peak_lr": 1e-5, "betas": [.9, .95], "epsilon": 1e-8, "weight_decay": 0., "clip_norm": 1.,
    "trainable_layers": list(range(8, 28)), "dtype": "float32", "seed": 17,
    "greedy": True, "do_sample": False, "thinking": False, "attention": "eager", "tf32": False,
    "generation_eos_ids": [151645, 151643], "training_eos_id": 151645,
    "qualified_primitives_threads": 1, "sparse_delta_remasking": False,
    "replication_draw_seeds": [17, 23, 29], "native_model_calls": 0,
    "reasoning_gain_margin": .05, "control_reasoning_deficit_limit": .02,
    "retention_new_loss_limit": 0, "confirmation_rows_planned": 768,
    "source_outputs": "BusinessProject outside source; no .codex", "automatic_retry": False}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def utc():
    return datetime.now(timezone.utc).isoformat()

def safe_output(path):
    p = Path(path).resolve()
    if not p.is_relative_to(BUSINESS) or p.is_relative_to(SOURCE_ROOT):
        raise ValueError("outputs must be under BusinessProject outside implementation source")
    if any(part.lower() == ".codex" for part in p.parts):
        raise ValueError(".codex outputs forbidden")
    for ancestor in [p, *p.parents]:
        if (ancestor / ".git").exists():
            raise ValueError("output is inside a Git checkout")
    return p

def new_directory(path):
    p = safe_output(path)
    if not p.parent.is_dir():
        raise ValueError("output parent must already exist")
    p.mkdir(exist_ok=False)
    return p

def write_json(path, value):
    p = safe_output(path)
    with p.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")

def write_jsonl(path, rows):
    p = safe_output(path)
    with p.open("xb") as f:
        for row in rows:
            f.write(canonical(row) + b"\n")

def confirmation_closed():
    raise PermissionError("confirmation generation/read/scoring is closed in offline v1")
