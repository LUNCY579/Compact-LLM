"""Public consumed-prompt evidence, including non-gradient model calls.

Hashes document exact inputs and coverage, not absence of semantic overlap or a
maliciously rewritten history. Public prompt fields never contain answer/solution
text. Separate forward_input_ids retain actual response-prefix conditioning,
including prior SFT/replay target tokens; they are never sent as public prompts.
"""
from __future__ import annotations

from .contracts import digest, require
from .data import normalized


FORMAT = "rdd-consumed-prompts-v1"
TOKENIZER_NAMES = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                   "vocab.json", "vocab.txt", "merges.txt", "tokenizer.model", "chat_template.jinja"}


def public_row(task):
    return {"key": task.key, "namespace": task.key.split(":", 1)[0],
            "parent_id": task.parent, "parent_namespace": task.parent.split(":", 1)[0],
            "source": task.source, "revision": task.revision, "role": task.role,
            "prompt": task.prompt, "prompt_hash": digest(normalized(task.prompt))}


def validate_manifest(payload, *, complete=True):
    require(payload.get("schema") == 1 and payload.get("format") == FORMAT,
            "consumed prompt manifest format")
    require(type(payload.get("sessions")) is list and payload["sessions"], "manifest sessions required")
    all_events = []
    for session in payload["sessions"]:
        require(type(session.get("events")) is list, "manifest events required")
        require(session["event_count"] == len(session["events"])
                and session["events_sha256"] == digest(session["events"]), "manifest event inventory drift")
        counters = {}
        for ordinal, event in enumerate(session["events"]):
            require(event["ordinal"] == ordinal, "manifest event order drift")
            all_events.append(event)
            require(event["model"] in {"student", "teacher"} and event["purpose"]
                    and type(event["step"]) is int and event["step"] >= 0, "manifest model/purpose/version")
            require(event["kind"] in {"generation", "forward", "external_forward", "teacher_bank_reuse", "rollout_consumption"},
                    "unknown model consumption kind")
            require(type(event["input_tokens"]) is int and event["input_tokens"] >= 0,
                    "manifest token count")
            if event["complete"]:
                require(all(type(event[k]) is str and event[k] for k in
                            ("key", "namespace", "parent_id", "parent_namespace", "source", "revision", "role", "prompt")),
                        "complete manifest task metadata")
                require(event["prompt_hash"] == digest(normalized(event["prompt"])), "public prompt hash drift")
                require(event["namespace"] == event["key"].split(":", 1)[0]
                        and event["parent_namespace"] == event["parent_id"].split(":", 1)[0], "namespace drift")
                require(type(event["input_ids"]) is list and event["input_ids"]
                        and all(type(t) is int and t >= 0 for t in event["input_ids"]), "rendered prompt IDs required")
                require(event["rendered_prompt_sha256"] == digest(event["input_ids"])
                        and event["prompt_token_sha256"] == digest(event["input_ids"]), "rendered token hash drift")
                require(type(event["raw_rendered_prompt"]) is str and event["raw_rendered_prompt"]
                        and event["raw_rendered_prompt"] == event["consumed_prompt"], "rendered public prompt required")
                require(event["truncation"] == {"policy": "none", "original_input_ids": event["input_ids"],
                                                "max_tokens": None, "history": []}, "manifest no-truncation policy drift")
            if event["complete"] and event["kind"] in {"forward", "external_forward"}:
                require("forward_input_ids" in event and "causal_input_token_sha256" in event,
                        "complete forward requires actual causal input IDs/hash")
            if "forward_input_ids" in event:
                require(type(event["forward_input_ids"]) is list and len(event["forward_input_ids"]) == event["input_tokens"]
                        and all(type(t) is int and t >= 0 for t in event["forward_input_ids"])
                        and digest(event["forward_input_ids"]) == event["causal_input_token_sha256"]
                        and (not event["complete"] or event["forward_input_ids"][:len(event["input_ids"])] == event["input_ids"]),
                        "actual causal input token receipt drift")
            if event["kind"] in {"generation", "forward", "external_forward"}:
                suffix = {"generation": "_generation_attempts", "forward": "_forward_input_tokens",
                          "external_forward": "_external_forward_input_tokens"}[event["kind"]]
                key = event["phase"] + "/" + event["model"] + suffix
                counters[key] = counters.get(key, 0) + (1 if event["kind"] == "generation" else event["input_tokens"])
        expected = {k: v for k, v in session["meter"].items()
                    if k.endswith(("_generation_attempts", "_forward_input_tokens"))}
        require(counters == expected, "manifest missing model consumption relative to meter")
    require(payload["event_count"] == len(all_events)
            and payload["public_prompt_hashes"] == sorted({e["prompt_hash"] for e in all_events}),
            "manifest coverage inventory drift")
    actual_complete = bool(all_events) and all(e["complete"] for e in all_events)
    require(payload["complete"] == actual_complete, "manifest completeness drift")
    if complete:
        require(actual_complete, "partial/missing consumed prompt manifest")
    return payload
