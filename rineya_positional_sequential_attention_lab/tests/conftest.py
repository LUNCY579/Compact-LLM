"""Shared fixtures for the PSA lab suite.

Nine modules were written in parallel and each grew its own randomly-initialised
Qwen3. The configurations were nearly identical, so the duplication was a
maintenance hazard rather than a design choice: this file owns the canonical one
and the modules that used the same numbers now import it.

Two deliberate non-choices:

* **torch is imported inside the fixtures, not at module scope.** SPEC section 2 keeps
  most of the package importable on a torch-free interpreter, and a conftest that
  imports torch would pay that cost for every test in the suite, including the pure
  numpy ones.
* **``tiny_qwen3_model`` is module-scoped, not session-scoped.** The intervention,
  sequential and merging suites install hooks into, and in one case write weights
  into, the model they are handed. Sharing one instance across modules would couple
  files that are meant to be independent; sharing it across a file is safe because
  each of those suites restores what it changed and asserts that it did.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from rineya_psa.types import Arch

# The canonical toy architecture: Qwen3-0.6B's shape ratios (GQA group size 2, q_norm
# and k_norm over head_dim) at 1/16 the width, so head bookkeeping bugs still show up
# but a forward pass costs microseconds.
TINY_QWEN3_CONFIG: dict[str, Any] = {
    "vocab_size": 512,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "rope_theta": 1e6,
    "max_position_embeddings": 1024,
    "tie_word_embeddings": True,
}

TINY_SEED = 20260828

TINY_ARCH = Arch(
    num_layers=TINY_QWEN3_CONFIG["num_hidden_layers"],
    hidden_size=TINY_QWEN3_CONFIG["hidden_size"],
    num_q_heads=TINY_QWEN3_CONFIG["num_attention_heads"],
    num_kv_heads=TINY_QWEN3_CONFIG["num_key_value_heads"],
    head_dim=TINY_QWEN3_CONFIG["head_dim"],
    vocab_size=TINY_QWEN3_CONFIG["vocab_size"],
)


def make_tiny_qwen3(seed: int = TINY_SEED, **overrides: Any) -> Any:
    """A seeded, CPU, float32, eval-mode Qwen3 with :data:`TINY_QWEN3_CONFIG`'s shape.

    ``use_cache`` is off because the sequential-attention wrapper refuses an
    incremental cache and every teacher-forced path in the lab passes
    ``use_cache=False`` anyway.
    """
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = {**TINY_QWEN3_CONFIG, **overrides}
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(Qwen3Config(**cfg)).to(torch.float32).eval()
    model.config.use_cache = False
    return model


# --------------------------------------------------------------------------------------
# Tokenizer stub
# --------------------------------------------------------------------------------------

_WORD_RE = re.compile(r"[^\s]+")


class StubTokenizer:
    """A whitespace tokenizer that behaves like the fast Qwen3 one where it matters.

    The lab's tokenizer contract is narrow but specific: ``apply_chat_template`` must
    embed the user turn verbatim and emit the pre-closed empty think block
    (SPEC 13.5), ``__call__`` must return real ``offset_mapping`` spans so
    ``benchmark/tokenize.py`` can localise evidence by offsets rather than by
    incremental re-encoding (SPEC 11.4), and ids must fit the toy model's 512-token
    vocabulary.

    Whitespace is *not* emitted as tokens. That is not cosmetic: a token per space
    doubles the length of every prompt and coarsens ``assemble.fit_bank``'s budget
    search to the point where it cannot hit a 2% tolerance, which would make the
    benchmark builder look broken when it is the stub that is.
    """

    is_fast = True
    chat_template = "stub"
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = None
    padding_side = "left"
    model_max_length = 1 << 20

    def __init__(self, vocab_size: int = TINY_QWEN3_CONFIG["vocab_size"]) -> None:
        if vocab_size < 8:
            raise ValueError("vocab_size must leave room for the special ids")
        self.vocab_size = int(vocab_size)
        # id -> first surface form seen for it, so ``decode`` round-trips well enough
        # for the runner's prediction column. Collisions are possible and harmless:
        # nothing in the suite asserts on decoded text from this stub.
        self._surface: dict[int, str] = {}

    def _token_id(self, piece: str) -> int:
        h = 0
        for ch in piece:
            h = (h * 131 + ord(ch)) % 1_000_003
        token_id = 3 + h % (self.vocab_size - 3)
        self._surface.setdefault(token_id, piece)
        return token_id

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        pieces = [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(str(text))]
        out: dict[str, Any] = {"input_ids": [self._token_id(p) for p, _, _ in pieces]}
        out["attention_mask"] = [1] * len(out["input_ids"])
        if return_offsets_mapping:
            out["offset_mapping"] = [(a, b) for _, a, b in pieces]
        return out

    def encode(self, text: str, add_special_tokens: bool = False, **_kwargs: Any) -> list[int]:
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]

    def decode(self, ids: Any, skip_special_tokens: bool = True, **_kwargs: Any) -> str:
        parts: list[str] = []
        for raw in ids:
            i = int(raw)
            if skip_special_tokens and i < 3:
                continue
            parts.append(self._surface.get(i, "?"))
        return " ".join(parts)

    def apply_chat_template(
        self,
        messages: Any,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        **_kwargs: Any,
    ) -> str:
        if tokenize:
            raise NotImplementedError("the stub renders text only")
        body = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        if add_generation_prompt:
            body += "<|im_start|>assistant\n"
            if not enable_thinking:
                body += "<think>\n\n</think>\n\n"
        return body


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_config() -> dict[str, Any]:
    return dict(TINY_QWEN3_CONFIG)


@pytest.fixture(scope="session")
def tiny_arch() -> Arch:
    """``Arch`` for :data:`TINY_QWEN3_CONFIG`. Torch-free, so it costs nothing."""
    return TINY_ARCH


@pytest.fixture(scope="module")
def tiny_qwen3_model() -> Any:
    return make_tiny_qwen3()


@pytest.fixture(scope="module")
def tiny_qwen3(tiny_qwen3_model: Any) -> tuple[Any, Arch]:
    """``(model, arch)`` where ``arch`` is read back off the model's own config.

    Reading it back rather than handing over :data:`TINY_ARCH` is the point: it checks
    that ``loading.arch_from_config`` agrees with the constant.
    """
    from rineya_psa.loading import arch_from_config

    return tiny_qwen3_model, arch_from_config(tiny_qwen3_model.config)


@pytest.fixture(scope="module")
def stub_tokenizer() -> StubTokenizer:
    return StubTokenizer()


@pytest.fixture
def lab_dirs(tmp_path: Path) -> dict[str, Path]:
    """A throwaway mirror of the lab's output tree.

    Every write in the suite goes here rather than into the real ``results/`` and
    ``data/``, so a test run leaves the repository exactly as it found it.
    """
    names = (
        "configs",
        "data",
        "data/generated",
        "data/calibration",
        "data/validation",
        "data/sealed",
        "manifests",
        "results",
        "figures",
        "reports",
        "experiments",
        "experiments/merges",
    )
    made: dict[str, Path] = {"root": tmp_path}
    for name in names:
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        made[name] = path
    return made
