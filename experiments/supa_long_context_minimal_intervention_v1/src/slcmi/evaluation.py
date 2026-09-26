"""Shared, frozen evaluation path for every arm and condition of the pilot.

The decoding protocol is copied from the suite's own runner
(``Rineya_Long_Context_Demo_v1/scripts/run_local_hf.py``): greedy,
``max_new_tokens=24``, ``enable_thinking=False``, seed 17. The scorer is the
suite's own ``normalize`` + strict exact match, imported rather than
reimplemented so this lab cannot drift from the published numbers.

Validity gate: running condition A over all 256 rows through this module must
reproduce the published Supa-A1.1 result of 45/64 Showcase, 125/192 Stability,
170/256 combined.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .provenance import CANONICAL_CORE, REPO

# --- frozen decoding protocol ---------------------------------------------

MAX_NEW_TOKENS = 24
SEED = 17
DTYPE = torch.float16
SYSTEM_PROMPT = (
    "You are answering a closed-context reading test. Use only the supplied "
    "context. Reply with the shortest exact answer and no explanation. If the "
    "context does not determine the answer, reply UNKNOWN."
)
USER_TEMPLATE = "CONTEXT\n{context}\n\nQUESTION\n{question}\n\nANSWER"


def _load_suite_scorer() -> Callable[[str], str]:
    """Import ``normalize`` from the demo suite's official evaluator."""
    path = REPO / "Rineya_Long_Context_Demo_v1/scripts/evaluate_predictions.py"
    spec = importlib.util.spec_from_file_location("rlc_evaluate_predictions", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.normalize


normalize = _load_suite_scorer()

_WORD_RE = re.compile(r"\b[\w'-]+\b")


DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def load_model(
    path: Path = CANONICAL_CORE,
    device: str = "cuda",
    dtype: str | None = None,
    gate: bool = True,
):
    """Load an A-series core under its documented requirements.

    ``trust_remote_code=True`` is mandatory: without it the directory silently
    loads as plain ``Qwen3ForCausalLM`` with the abstention gate disabled and no
    warning at all.

    ``dtype`` defaults to the core's own stored dtype -- float16 for A1.1,
    bfloat16 for A1. Each core is loaded in the precision it was published in;
    re-rounding A1's bfloat16 into float16 (or the reverse) would confound a
    checkpoint comparison with a precision conversion.

    ``gate=False`` zeroes ``abstain_logit_bias`` on the in-memory config only.
    The offset lives in the config and is applied in ``forward``, never in the
    weights, so this is the exact ungated control the core's own evidence files
    use -- and nothing is written to disk.
    """
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    stored_config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    stored_dtype = getattr(stored_config, "dtype", None) or getattr(stored_config, "torch_dtype", None)
    resolved = DTYPES[dtype] if dtype else stored_dtype
    if isinstance(resolved, str):
        resolved = DTYPES[resolved.removeprefix("torch.")]
    if resolved is None:
        raise ValueError(f"checkpoint does not declare a dtype: {path}")
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=resolved, trust_remote_code=True, local_files_only=True
    )
    assert type(model).__name__ == "Qwen3AbstainForCausalLM", (
        f"abstention gate missing: loaded {type(model).__name__}. "
        "trust_remote_code=True is required."
    )
    if not gate:
        assert model.config.abstain_logit_bias, "expected a gated core before disabling"
        model.config.abstain_logit_bias = 0.0
    return tokenizer, model.eval().to(device).requires_grad_(False)


def render_prompt(tokenizer, context: str, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def count_tokens_factory(tokenizer) -> Callable[[str], int]:
    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    return count


def is_malformed(raw: str) -> bool:
    """Flag output that is not a bare short answer.

    Empty, leaked reasoning tags, or a response long enough that the 24-token
    budget was spent narrating rather than answering.
    """
    stripped = raw.strip()
    if not stripped:
        return True
    if "<think>" in stripped or "</think>" in stripped:
        return True
    return len(_WORD_RE.findall(stripped)) > 12


@torch.no_grad()
def generate_one(tokenizer, model, prompt_text: str, device: str = "cuda") -> tuple[str, int, float]:
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    t0 = time.perf_counter()
    out = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.perf_counter() - t0
    text = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    return text, int(enc.input_ids.shape[1]), elapsed


def score(raw_prediction: str, accepted_answers: Sequence[str]) -> bool:
    return normalize(raw_prediction) in {normalize(a) for a in accepted_answers}


def evaluate_rows(
    tokenizer,
    model,
    items: Sequence[dict[str, Any]],
    *,
    device: str = "cuda",
    progress_every: int = 64,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Run the frozen protocol over prepared items.

    Each item needs ``example_id``, ``context``, ``question``, ``accepted_answers``
    and is echoed back with the model's output and per-example measurements.
    """
    torch.manual_seed(SEED)
    records: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, item in enumerate(items, 1):
        prompt_text = render_prompt(tokenizer, item["context"], item["question"])
        raw, n_input_tokens, elapsed = generate_one(tokenizer, model, prompt_text, device)
        records.append(
            {
                **{k: v for k, v in item.items() if k not in ("context", "accepted_answers")},
                "prompt_tokens": n_input_tokens,
                "raw_prediction": raw,
                "normalized_prediction": normalize(raw),
                "correct": bool(score(raw, item["accepted_answers"])),
                "malformed": bool(is_malformed(raw)),
                "inference_seconds": round(elapsed, 4),
            }
        )
        if progress_every and i % progress_every == 0:
            log(f"    {i}/{len(items)}  {time.perf_counter() - t0:.0f}s")
    return records
