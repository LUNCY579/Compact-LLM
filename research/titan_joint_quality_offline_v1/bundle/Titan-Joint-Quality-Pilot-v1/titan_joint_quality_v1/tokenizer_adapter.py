"""Offline local tokenizer only. No Transformers or model imports, no downloads."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from .common import PARENT_SHA, file_sha
from .source import MODEL_SOURCES

class LocalTokenizerAdapter:
    def __init__(self, source_receipt):
        if source_receipt.get("status") != "PASS_METADATA_ONLY":
            raise ValueError("exact source metadata must pass first")
        source = source_receipt["models"]["sparse"]
        directory = Path(source["path"]).resolve()
        if directory != MODEL_SOURCES["sparse"][0].resolve() or source["actual_weight_sha256"] != PARENT_SHA:
            raise ValueError("wrong tokenizer parent")
        for name, digest in source["sidecars"].items():
            if file_sha(directory / name) != digest:
                raise ValueError("changed tokenizer sidecar:" + name)
        # These libraries deserialize tokenizer metadata, never weights or HF code.
        from tokenizers import Tokenizer
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
        model_config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        template = config.get("chat_template")
        if not isinstance(template, str) or not template:
            raise ValueError("exact string chat template required")
        self.eos_token_id = model_config["eos_token_id"]
        if type(self.eos_token_id) != int:
            raise ValueError("one canonical training EOS token required")
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        def reject(message):
            raise ValueError(str(message))
        env.globals.update(raise_exception=reject, strftime_now=lambda f: datetime(2026, 10, 5).strftime(f))
        env.filters["tojson"] = lambda x, **_: json.dumps(x, ensure_ascii=False)
        self.template = env.from_string(template)
        self.variables = {k: v for k, v in config.items() if k.endswith("_token") and isinstance(v, str)}
        self.manifest = {"kind": "exact_local_tokenizer_not_native_qualification", "model_loads": 0,
            "parent_sha256": PARENT_SHA, "sidecars": source["sidecars"], "eos_token_id": self.eos_token_id,
            "thinking": False, "implementation": "tokenizers + sandboxed pinned Jinja template"}

    def encode_case(self, messages, target):
        if not isinstance(messages, list) or not isinstance(target, str):
            raise ValueError("messages and final answer text required")
        if any(set(m) != {"role", "content"} or m["role"] not in ("system", "user") or not isinstance(m["content"], str) for m in messages):
            raise ValueError("prompt-only system/user messages required")
        if sum(m["role"] == "user" for m in messages) != 1 or messages[-1]["role"] != "user":
            raise ValueError("one final user task required")
        rendered = self.template.render(messages=messages, add_generation_prompt=True,
            enable_thinking=False, tools=None, **self.variables)
        prompt = self.tokenizer.encode(rendered, add_special_tokens=False).ids
        answer = self.tokenizer.encode(target, add_special_tokens=False).ids
        if not prompt or not answer or self.eos_token_id in answer:
            raise ValueError("invalid prompt/answer/EOS boundaries")
        # Refuse BPE boundary changes rather than silently shifting supervision.
        combined = self.tokenizer.encode(rendered + target, add_special_tokens=False).ids
        if combined != prompt + answer:
            raise ValueError("tokenizer prompt/response boundary requires explicit owner qualification")
        eos_text = self.tokenizer.id_to_token(self.eos_token_id)
        serialized = rendered + target + eos_text + "\n"
        full = self.tokenizer.encode(serialized, add_special_tokens=False).ids
        expected = prompt + answer + [self.eos_token_id] + self.tokenizer.encode("\n", add_special_tokens=False).ids
        if full != expected:
            raise ValueError("EOS/newline serialization boundary differs")
        end = len(prompt) + len(answer) + 1
        return {"ids": full, "response_start": len(prompt), "response_end": end, "eos_position": end - 1,
                "eos_token_id": self.eos_token_id, "prompt_tokens": len(prompt),
                "response_tokens": len(answer) + 1}
