"""CPU prompt-only input binding; never native execution or scorer authority."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

from cpu_tokenizer import ASSISTANT_PREFIX, TEMPLATE_KWARGS, load_local_tokenizer
from rs_contract import canonical, digest, require

PACKAGE = Path(__file__).resolve().parent
CONTEXT_CAP, GENERATION_CAP = 512, 32
ROW_FIELDS = {"row_id", "messages", "prompt_sha256"}
PACKET_FIELDS = {"schema_version", "panel_sha256", "policy_sha256", "population", "rows",
                 "native_launch_authorized", "confirmation_eligible"}


def filehash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _cpu_only():
    require(not any(name.split(".", 1)[0] in {"torch", "transformers", "tensorflow", "flax", "jax"}
                    for name in sys.modules), "CPU prompt preparation imported model framework")


def prepare(packet, tokenizer=None):
    """Preserve every prompt; report the whole packet blocked if any exceeds480."""
    _cpu_only()
    require(type(packet) is dict and type(packet.get("rows")) is list and 0 < len(packet["rows"]) <= 4096,
            "bounded prompt-only packet required")
    require(set(packet) <= PACKET_FIELDS, "oracle or unregistered packet metadata")
    require(packet.get("native_launch_authorized") is False and packet.get("confirmation_eligible") is False,
            "closed CPU DEV packet required")
    tokenizer = load_local_tokenizer() if tokenizer is None else tokenizer
    binding_before = tokenizer.verify_binding()
    seen, arrays, overlong = set(), [], []
    for row in packet["rows"]:
        require(type(row) is dict and set(row) == ROW_FIELDS and type(row["row_id"]) is str
                and len(row["row_id"]) == 64 and all(c in "0123456789abcdef" for c in row["row_id"])
                and row["row_id"] not in seen, "prompt row schema or duplicate ID")
        messages = row["messages"]
        require(type(messages) is list and 1 <= len(messages) <= 8 and
                all(type(m) is dict and set(m) == {"role", "content"} and m["role"] in {"system", "user"}
                    and type(m["content"]) is str and len(m["content"].encode("utf-8")) <= 65536
                    for m in messages) and messages[-1]["role"] == "user", "prompt-only message boundary")
        require(row["prompt_sha256"] == digest(messages), "visible prompt identity changed")
        seen.add(row["row_id"])
        rendered = tokenizer._template.render(messages=copy.deepcopy(messages), tools=None, **TEMPLATE_KWARGS)
        require(type(rendered) is str and rendered.endswith(ASSISTANT_PREFIX), "non-thinking generation prefix")
        ids = tokenizer._engine.encode(rendered, add_special_tokens=False).ids
        require(type(ids) is list and bool(ids) and all(type(x) is int and 0 <= x < 151936 for x in ids),
                "typed tokenizer output")
        require(151643 not in ids, "padding token in unpadded prompt")
        if len(ids) + GENERATION_CAP > CONTEXT_CAP:
            overlong.append(row["row_id"])
        arrays.append({"row_id": row["row_id"], "prompt_sha256": row["prompt_sha256"],
                       "prompt_ids": ids, "prompt_tokens": len(ids), "attention_mask": [1] * len(ids)})
    require(tokenizer.verify_binding() == binding_before, "tokenizer binding drift during preparation")
    result = {"schema_version": 1, "scope": "CPU_PROMPT_BINDING_NOT_NATIVE_OR_GRADER_AUTHORITY",
              "status": "BLOCKED_PROMPT_ENVELOPE" if overlong else "CPU_INPUTS_PREPARED_NATIVE_UNADMITTED",
              "source_packet_sha256": digest(packet), "tokenizer_binding": binding_before,
              "implementation_sha256": filehash(Path(__file__)), "context_cap": CONTEXT_CAP,
              "maximum_generated_tokens": GENERATION_CAP, "rows": arrays, "overlong_row_ids": overlong,
              "rows_dropped": 0, "rows_truncated": 0, "padding": False, "packing": False,
              "native_launch_authorized": False, "confirmation_eligible": False,
              "oracle_inputs_read": False, "model_outputs_used": False}
    result["sha256"] = digest(result)
    _cpu_only()
    return result


def validate(inputs, packet, tokenizer=None):
    require(type(inputs) is dict and inputs.get("sha256") == digest({k: v for k, v in inputs.items() if k != "sha256"}),
            "input packet checksum")
    expected = prepare(packet, tokenizer)
    require(canonical(inputs) == canonical(expected), "prompt/token/order/source binding drift")
    require(inputs["status"] == "CPU_INPUTS_PREPARED_NATIVE_UNADMITTED", "prompt envelope blocked; no filtering")
    return {"rows": len(inputs["rows"]), "input_positions": sum(r["prompt_tokens"] for r in inputs["rows"]),
            "maximum_prompt_tokens": max(r["prompt_tokens"] for r in inputs["rows"]),
            "native_launch_authorized": False, "oracle_inputs_read": False}


def write_inputs(inputs, path):
    target = Path(path).absolute()
    require(target.parent.resolve().parent == (PACKAGE / "artifacts" / "evaluation_dev").resolve()
            and not target.is_symlink() and not target.exists(), "new owned evaluation bundle file required")
    raw = (canonical(inputs) + "\n").encode("ascii")
    require(len(raw) <= 16 * 1024 * 1024, "bounded token input packet")
    with target.open("xb", buffering=0) as stream:
        require(stream.write(raw) == len(raw), "short token input packet write")
    return {"path": str(target), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
