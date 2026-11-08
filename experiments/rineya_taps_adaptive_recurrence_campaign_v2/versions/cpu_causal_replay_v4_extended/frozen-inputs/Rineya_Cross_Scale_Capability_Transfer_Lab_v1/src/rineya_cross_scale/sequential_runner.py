"""One-model-at-a-time orchestration for cross-scale diagnostics.

Never holds more than one Transformers model. Prefer CPU. Optional CUDA only
for a single 0.6B short pass, and only when resource_trace.refuse_gpu_if_busy
permits it. Donor-scale work should use streaming_forward instead.
"""

from __future__ import annotations

import gc
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .resource_trace import ResourceTrace, refuse_gpu_if_busy


class SequentialRunnerError(RuntimeError):
    pass


def release(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


@contextmanager
def load_one_model(
    model_path: Path,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    trust_remote_code: bool = True,
    trace: ResourceTrace | None = None,
) -> Iterator[tuple[Any, Any]]:
    """Load exactly one tokenizer+model; always release on exit."""
    if device == "cuda":
        gate = refuse_gpu_if_busy()
        if gate.get("refuse") or not torch.cuda.is_available():
            raise SequentialRunnerError(
                f"GPU refused: {gate.get('refuse_reason', gate)}"
            )

    model_path = model_path.expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    if trace is not None:
        trace.record_model_load()
        trace.sample_rss()
    try:
        yield tokenizer, model
    finally:
        release(model, tokenizer)


def run_prompts_last_token(
    model_path: Path,
    prompts: list[str],
    *,
    device: str = "cpu",
    max_items: int | None = None,
    capture_layers: list[int] | None = None,
    trace: ResourceTrace | None = None,
) -> dict[str, Any]:
    """Sparse last-token capture for a single 0.6B-class model.

    Returns CPU float32 summaries only — no model residency after return.
    """
    selected = prompts if max_items is None else prompts[:max_items]
    layer_keys = capture_layers or []
    hidden_rows: dict[str, list[list[float]]] = {
        f"layer_{index}": [] for index in layer_keys
    }
    logits_rows: list[list[float]] = []

    with load_one_model(model_path, device=device, trace=trace) as (tokenizer, model):
        with torch.no_grad():
            for prompt in selected:
                encoded = tokenizer(prompt, return_tensors="pt")
                encoded = {key: value.to(device) for key, value in encoded.items()}
                output = model(**encoded, output_hidden_states=True)
                states = output.hidden_states
                for index in layer_keys:
                    # hidden_states[0] is embedding; layers are 1..N
                    tensor = states[index + 1][0, -1, :].float().cpu()
                    hidden_rows[f"layer_{index}"].append(tensor.tolist())
                logits_rows.append(output.logits[0, -1, :].float().cpu().tolist())
                del output, states, encoded
                if trace is not None:
                    trace.sample_rss()
                gc.collect()

    return {
        "model_path": str(model_path),
        "n_prompts": len(selected),
        "device": device,
        "hidden_last_token": hidden_rows,
        "logits_last_token_dim": len(logits_rows[0]) if logits_rows else 0,
        # Store only norms + a fixed projection seed elsewhere; full logits are huge.
        "logit_norms": [
            float(sum(value * value for value in row) ** 0.5) for row in logits_rows
        ],
    }
