"""Local Titan token measurement; no model/framework, bank, or launch access.

The optional Rust-tokenizer/Jinja dependencies are imported only by the checked
loader.  Unit tests can exercise the token boundary with injected CPU fixtures.
These arrays are prospective data, never native-training admission evidence.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any

try:
    from . import rs_contract as contract
    from . import rs_verifier as verifier
except ImportError:
    import rs_contract as contract
    import rs_verifier as verifier

VERSION = "synth-cpu-tokenizer-1"
EOS_ID = 151645
PAD_ID = 151643
CONTEXT_CAP = 512
ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
TEMPLATE_KWARGS = {"add_generation_prompt": True, "enable_thinking": False}
DEPENDENCIES = {"tokenizers": "0.22.2", "jinja2": "3.1.6"}
PACKAGE = Path(__file__).resolve().parent
ASSET_PINS = {
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
}
CHAT_TEMPLATE_PIN = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
_FRAMEWORK_ROOTS = {"transformers", "torch", "tensorflow", "jax", "flax"}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _no_frameworks() -> None:
    contract.require(not any(name.split(".", 1)[0] in _FRAMEWORK_ROOTS
                             for name in sys.modules),
                     "CPU tokenizer process must contain no model frameworks")


def _read_json(path: Path) -> tuple[dict, str]:
    try:
        raw = path.read_bytes()
        value = contract.strict_json(raw.decode("utf-8"))
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read local binding: {path}") from exc
    contract.require(type(value) is dict, "binding must be a JSON object")
    return value, _sha(raw)


def _asset_hashes(model_dir: Path, expected: dict[str, str]) -> dict:
    result = {}
    for name, pin in expected.items():
        path = model_dir / name
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"missing local tokenizer asset: {path}") from exc
        digest = _sha(raw)
        contract.require(digest == pin, f"tokenizer asset drift: {name}")
        result[name] = {"sha256": digest, "bytes": len(raw)}
    return result


def _ids(value: Any) -> list[int]:
    contract.require(type(value) is list and value and
                     all(type(i) is int and i >= 0 for i in value),
                     "invalid or empty token IDs")
    return value


class CpuTokenizer:
    """A checked local tokenizer; masks are determined by position, not token ID.

    Constructor dependencies are private injection points for CPU-only tests.
    Production callers must obtain the object through load_local_tokenizer().
    Call verify_binding() before and after each materialization/audit operation;
    encode_example() uses the already loaded immutable engine and template.
    """

    def __init__(self, *, _engine, _template, _binding: dict):
        self._engine = _engine
        self._template = _template
        self._binding = copy.deepcopy(_binding)

    @property
    def binding(self) -> dict:
        return copy.deepcopy(self._binding)

    def verify_binding(self) -> dict:
        """Recheck local source/asset hashes and dependency versions, no model IO."""
        _no_frameworks()
        for path, digest in self._binding["binding_sources"].items():
            try:
                raw = Path(path).read_bytes()
            except OSError as exc:
                raise ValueError(f"missing local binding source: {path}") from exc
            contract.require(_sha(raw) == digest, "binding source changed")
        expected = {name: value["sha256"] for name, value in self._binding["assets"].items()}
        assets = _asset_hashes(Path(self._binding["model_dir"]), expected)
        contract.require(assets == self._binding["assets"], "tokenizer asset identity changed")
        for name, version in self._binding["dependency_versions"].items():
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as exc:
                raise ValueError(f"missing CPU dependency: {name}") from exc
            contract.require(actual == version, f"CPU dependency drift: {name}")
        return self.binding

    def encode_example(self, messages: Any, target: str,
                       context_cap: int = CONTEXT_CAP) -> dict:
        _no_frameworks()
        contract.require(contract.isint(context_cap, 1, CONTEXT_CAP),
                         "context cap must be an integer within 1..512")
        spec = contract.parse_visible(messages)
        contract.require(contract.validate_spec(spec) in ("R", "S"),
                         "only atomic R/S supervised examples may be tokenized")
        contract.require(verifier.verify(messages, target)["accepted"] is True,
                         "target failed independent verifier")
        answer = contract.parse_answer(target, len(spec["initial"]))
        contract.require(target == contract.format_answer(answer),
                         "target must use canonical final-answer JSON")
        prompt = self._template.render(messages=copy.deepcopy(messages), tools=None,
                                       **TEMPLATE_KWARGS)
        contract.require(type(prompt) is str and prompt.endswith(ASSISTANT_PREFIX),
                         "non-thinking assistant generation prefix mismatch")
        prompt_ids = _ids(self._engine.encode(prompt, add_special_tokens=False).ids)
        target_ids = _ids(self._engine.encode(target, add_special_tokens=False).ids)
        contract.require(EOS_ID not in target_ids, "unexpected EOS in target")
        completion = target_ids + [EOS_ID]
        contract.require(self._engine.decode(completion, skip_special_tokens=False)
                         == target + "<|im_end|>", "completion roundtrip mismatch")
        full_messages = copy.deepcopy(messages) + [{"role": "assistant", "content": target}]
        full = self._template.render(messages=full_messages, tools=None,
                                     add_generation_prompt=False, enable_thinking=False)
        contract.require(full == prompt + target + "<|im_end|>\n",
                         "full template differs from prompt + target + EOS + newline")
        full_ids = _ids(self._engine.encode(full, add_special_tokens=False).ids)
        end = len(prompt_ids) + len(completion)
        contract.require(full_ids[:len(prompt_ids)] == prompt_ids,
                         "full conversation changed the prompt token prefix")
        contract.require(full_ids[len(prompt_ids):end] == completion,
                         "joint completion differs from separate target encoding")
        newline_ids = _ids(self._engine.encode("\n", add_special_tokens=False).ids)
        contract.require(full_ids[end:] == newline_ids,
                         "unexpected template tail after EOS")
        contract.require(end <= context_cap,
                         f"example exceeds {context_cap} tokens; truncation forbidden")
        input_ids = prompt_ids + completion
        _no_frameworks()
        return {"prompt_ids": prompt_ids, "completion_ids": completion,
                "input_ids": input_ids, "labels": [-100] * len(prompt_ids) + completion,
                "attention_mask": [1] * end, "prompt_tokens": len(prompt_ids),
                "supervised_tokens": len(completion), "total_tokens": end,
                "prompt_text_sha256": _sha(prompt.encode("utf-8")),
                "full_template_sha256": _sha(full.encode("utf-8"))}


def load_local_tokenizer(decisions_path: str | Path | None = None, *,
                         model_dir: str | Path | None = None) -> CpuTokenizer:
    """Use only pinned local Titan tokenizer assets, with no fallback/download.

    Explicit model_dir changes location only.  The pilot and historical receipt
    must still agree with the exact registered asset/template pins.  The historic
    checkpoint path is read only when no explicit local directory is supplied.
    No model config, generation config or weight file is opened by this module.
    """
    _no_frameworks()
    decision_file = Path(decisions_path) if decisions_path is not None else PACKAGE / "PILOT_DECISIONS.json"
    decision_file = decision_file.resolve()
    decisions, decision_digest = _read_json(decision_file)
    student = decisions.get("student")
    contract.require(type(student) is dict and
                     student.get("tokenizer_sha256") == ASSET_PINS["tokenizer.json"] and
                     student.get("tokenizer_config_sha256") == ASSET_PINS["tokenizer_config.json"] and
                     student.get("chat_template_sha256_recorded") == CHAT_TEMPLATE_PIN,
                     "pilot tokenizer/template pins missing or changed")
    receipt_file = (PACKAGE.parent / "post_training_mimo_centralized_v1" / "receipts" /
                    "titan_static_v1" / "TOKENIZER_RECEIPT.json").resolve()
    receipt, receipt_digest = _read_json(receipt_file)
    files = receipt.get("tokenizer_files")
    contract.require(type(files) is dict and set(files) == set(ASSET_PINS) and
                     all(type(files[name]) is dict and files[name].get("sha256") == pin
                         for name, pin in ASSET_PINS.items()) and
                     receipt.get("chat_template_sha256") == CHAT_TEMPLATE_PIN,
                     "historical tokenizer asset pins missing or changed")
    chosen_dir = model_dir if model_dir is not None else receipt.get("checkpoint_path")
    contract.require(isinstance(chosen_dir, (str, Path)) and bool(str(chosen_dir)),
                     "an explicit local tokenizer directory is required")
    local_dir = Path(chosen_dir)
    contract.require(local_dir.is_absolute() and local_dir.is_dir(),
                     "tokenizer directory must be an existing absolute local path")
    local_dir = local_dir.resolve()
    assets = _asset_hashes(local_dir, ASSET_PINS)
    config, _ = _read_json(local_dir / "tokenizer_config.json")
    template_text = config.get("chat_template")
    contract.require(type(template_text) is str and _sha(template_text.encode("utf-8")) == CHAT_TEMPLATE_PIN,
                     "configured chat template changed")
    contract.require(config.get("eos_token") == "<|im_end|>" and
                     config.get("pad_token") == "<|endoftext|>" and
                     config.get("add_bos_token") is False and
                     config.get("clean_up_tokenization_spaces") is False,
                     "tokenizer special-token configuration changed")
    for name, expected in DEPENDENCIES.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"missing CPU dependency: {name}; no automatic install") from exc
        contract.require(actual == expected, f"CPU dependency {name} must be {expected}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from tokenizers import Tokenizer
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    engine = Tokenizer.from_file(str(local_dir / "tokenizer.json"))
    # Stored tokenizer-side padding/truncation must never affect measured arrays.
    contract.require(engine.padding is None and engine.truncation is None,
                     "stored tokenizer enables padding or truncation")
    contract.require(engine.token_to_id("<|im_end|>") == EOS_ID and
                     engine.token_to_id("<|endoftext|>") == PAD_ID,
                     "Titan EOS/PAD identity changed")
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    environment.filters["tojson"] = lambda value, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    template = environment.from_string(template_text)
    _no_frameworks()
    binding = {"schema_version": 1, "implementation": VERSION,
               "backend": "local_tokenizers_json_and_exact_Jinja_template",
               "model_dir": str(local_dir), "assets": assets,
               "chat_template_sha256": CHAT_TEMPLATE_PIN,
               "dependency_versions": dict(DEPENDENCIES),
               "binding_sources": {str(decision_file): decision_digest,
                                   str(receipt_file): receipt_digest},
               "template_kwargs": dict(TEMPLATE_KWARGS), "eos_token_id": EOS_ID,
               "pad_token_id": PAD_ID, "context_cap": CONTEXT_CAP,
               "special_tokens_added_automatically": False,
               "template_trailing_newline_in_training_input": False,
               "framework_modules_imported": [], "local_only": True,
               "native_qualification": False, "training_admitted": False}
    tokenizer = CpuTokenizer(_engine=engine, _template=template, _binding=binding)
    tokenizer.verify_binding()
    return tokenizer
