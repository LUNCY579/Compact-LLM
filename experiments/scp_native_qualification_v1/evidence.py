"""Create-only external fixture evidence and an independent no-native verifier."""
from __future__ import annotations

from pathlib import Path
from .contract import (canonical, digest, file_hash, loads, require, safe_path, write_new,
                       EXPERIMENT, PARENT, NATIVE_BLOCK, PACKAGE)


def implementation_identity():
    return digest({p.name: file_hash(p) for p in sorted(PACKAGE.glob("*.py"))})


def external_directory(path, checkout):
    path, checkout = safe_path(path), safe_path(checkout).resolve()
    resolved = path.resolve()
    require(resolved != checkout and checkout not in resolved.parents, "mutable evidence must be outside checkout")
    require(not path.exists() and path.parent.is_dir(), "new evidence directory and existing parent required")
    path.mkdir(exist_ok=False)
    return path


def write_fixture_bundle(out, checkout, events, results):
    result = {"schema": "SCP_NQ_FIXTURE_RECEIPT_V1", "experiment": EXPERIMENT, "parent_commit": PARENT,
              "classification": "SOFTWARE_FIXTURE_ONLY", "native_execution_permitted": False,
              "pretrained_calls": 0, "backward_passes": 0, "optimizer_steps": 0, "permit_consumed": False,
              "events_sha256": digest(events), "implementation_sha256": implementation_identity(), "results": results}
    payloads = {"EVENTS.json": events, "RESULT.json": result}
    require(sum(len(canonical(v)) + 1 for v in payloads.values()) < 64 * 2**20 - 4096, "fixture evidence byte ceiling")
    out = external_directory(out, checkout)
    hashes = {name: write_new(out / name, value) for name, value in payloads.items()}
    write_new(out / "MANIFEST.json", {"schema": "SCP_NQ_MANIFEST_V1", "files": hashes})
    return verify_bundle(out)


def verify_bundle(directory):
    directory = safe_path(directory)
    require({p.name for p in directory.iterdir()} == {"MANIFEST.json", "EVENTS.json", "RESULT.json"},
            "unexpected/missing evidence inventory")
    require(sum(safe_path(p).stat().st_size for p in directory.iterdir()) <= 64 * 2**20, "evidence byte ceiling")
    manifest = loads((directory / "MANIFEST.json").read_text("utf-8"))
    require(type(manifest) is dict and set(manifest) == {"schema", "files"}
            and manifest["schema"] == "SCP_NQ_MANIFEST_V1"
            and set(manifest["files"]) == {"EVENTS.json", "RESULT.json"}, "manifest schema differs")
    require(all(file_hash(directory / name) == sha for name, sha in manifest["files"].items()), "evidence hash mismatch")
    result = loads((directory / "RESULT.json").read_text("utf-8"))
    events = loads((directory / "EVENTS.json").read_text("utf-8"))
    expected = {"schema", "experiment", "parent_commit", "classification", "native_execution_permitted",
                "pretrained_calls", "backward_passes", "optimizer_steps", "permit_consumed", "events_sha256", "implementation_sha256", "results"}
    require(type(result) is dict and set(result) == expected, "receipt keys differ")
    require(result["schema"] == "SCP_NQ_FIXTURE_RECEIPT_V1" and result["experiment"] == EXPERIMENT
            and result["parent_commit"] == PARENT and result["classification"] == "SOFTWARE_FIXTURE_ONLY",
            "native/foreign receipt not issuable or accepted by this release")
    require(result["native_execution_permitted"] is False and result["permit_consumed"] is False
            and all(type(result[k]) is int and result[k] == 0 for k in ("pretrained_calls", "backward_passes", "optimizer_steps")),
            "fixture receipt widened authority/execution")
    require(result["implementation_sha256"] == implementation_identity(), "receipt implementation differs")
    require(type(events) is list and result["events_sha256"] == digest(events), "event identity mismatch")
    completed, active, tokens, forwards = [], None, [], 0
    requests = allowance = 0
    for event in events:
        require(type(event) is dict, "invalid event")
        kind = event.get("event")
        keys = {
            "RESERVED": {"event", "request_sha256", "messages", "cap", "classification"},
            "FORWARD_RESERVED": {"event", "request_sha256", "index"},
            "TOKEN": {"event", "request_sha256", "index", "token_id"},
            "COMPLETED": {"event", "text", "input_tokens", "output_tokens", "terminated",
                          "generated_ids_including_terminal", "actual_forward_calls", "full_vocab_checked",
                          "eos_ids", "request_sha256", "classification", "pretrained_calls"},
        }
        require(type(kind) is str and kind in keys and set(event) == keys[kind], "event schema differs")
        if kind == "RESERVED":
            require(active is None and event.get("classification") == "SOFTWARE_FIXTURE_ONLY", "overlapping/native reservation")
            require(event.get("cap") in (32, 64, 288) and type(event["cap"]) is int, "invalid request allowance")
            messages = event.get("messages")
            require(type(messages) is list and len(messages) == 2
                    and all(type(m) is dict and set(m) == {"role", "content"}
                            and m["role"] == role and type(m["content"]) is str and bool(m["content"].strip())
                            for role, m in zip(("system", "user"), messages)), "invalid public message receipt")
            requests += 1
            allowance += event["cap"]
            require(requests <= 12 and allowance <= 928, "receipt exceeds fixed request/allowance envelope")
            active = event
            require(event.get("request_sha256") == digest({"messages": event.get("messages"), "cap": event["cap"]}), "request hash mismatch")
            tokens, forwards = [], 0
        elif kind == "FORWARD_RESERVED":
            require(active is not None and event.get("request_sha256") == active["request_sha256"]
                    and type(event.get("index")) is int and event["index"] == len(tokens) and forwards == len(tokens) and forwards < active["cap"], "unreserved/duplicate forward")
            forwards += 1
        elif kind == "TOKEN":
            require(active is not None and event.get("request_sha256") == active["request_sha256"]
                    and type(event.get("index")) is int and event["index"] == len(tokens) and forwards == len(tokens) + 1
                    and type(event.get("token_id")) is int and event["token_id"] >= 0, "token/forward mismatch")
            tokens.append(event["token_id"])
        elif kind == "COMPLETED":
            require(active is not None and event.get("request_sha256") == active["request_sha256"]
                    and event.get("generated_ids_including_terminal") == tokens
                    and 0 < event.get("output_tokens", -1) == forwards == len(tokens) <= active["cap"]
                    and event.get("actual_forward_calls") == forwards
                    and event.get("classification") == "SOFTWARE_FIXTURE_ONLY"
                    and type(event.get("pretrained_calls")) is int and event["pretrained_calls"] == 0,
                    "completion/counter mismatch")
            eos = event.get("eos_ids")
            vocab = event.get("full_vocab_checked")
            require(type(vocab) is int and vocab > 0 and all(t < vocab for t in tokens)
                    and type(eos) is list and eos and all(type(t) is int and 0 <= t < vocab for t in eos)
                    and eos == sorted(set(eos)), "invalid vocabulary/EOS receipt")
            terminated = tokens[-1] in eos
            require(not any(t in eos for t in tokens[:-1]) and type(event.get("terminated")) is bool
                    and event["terminated"] == terminated and (terminated or len(tokens) == active["cap"]),
                    "termination receipt differs from raw tokens")
            require(type(event.get("input_tokens")) is int and event["input_tokens"] > 0
                    and event["input_tokens"] == 1 + sum(len(m["content"].split()) for m in active["messages"])
                    and event["input_tokens"] + active["cap"] <= 2048
                    and type(event.get("output_tokens")) is int and type(event.get("actual_forward_calls")) is int,
                    "token/forward types or context differ")
            public = tokens[:-1] if terminated else tokens
            require(event.get("text") == " ".join(f"<token:{i}>" for i in public), "fixture decode differs")
            completed.append({k: v for k, v in event.items() if k != "event"})
            active = None
        else:
            require(False, "failed/unknown event cannot support completed fixture PASS")
    require(active is None and completed and completed == result["results"], "unfinished/empty/different fixture results")
    return {"classification": "VERIFIED_SOFTWARE_FIXTURE_ONLY", "completed_requests": len(completed),
            "native_execution_permitted": False, "native_status": NATIVE_BLOCK,
            "not_a_native_pass": True, "receipt_sha256": file_hash(directory / "RESULT.json")}
