"""Native Rineya (Qwen3ForCausalLM, 596M) adapter: BF16 STORAGE, FP32 COMPUTE.

Supa is published in BF16, so every stored weight is exactly representable in BF16 and its FP32
upcast is exact. Keeping weights in BF16 and upcasting per module at compute time therefore
computes the same FP32 function as an FP32 load, at half the resident weight memory. This is
what lets a dense AdamW state over the 188.7M-coordinate footprint fit on the 8 GB card without
host offload. Exactness is a qualification GATE (``parity_check``), not an assumption.

Mechanics (all verified by tests on a tiny random Qwen3 and by the native qualification):

* every ``nn.Linear`` computes ``F.linear(x_fp32, W.to(fp32) [+ delta])``;
* the tied embedding returns ``F.embedding(ids, W_bf16).to(fp32)``;
* the LM head uses ``UpcastHead``: a chunked FP32 matmul whose backward re-upcasts instead of
  saving an FP32 copy of the 151,936 x 1,024 weight;
* RMSNorm weights are cast to FP32 (exact) and RoPE ``inv_freq`` is recomputed in FP32 with the
  model's own ``rope_init_fn`` (a BF16 load could otherwise round it).

Runtime: TF32 is forced OFF (cuDNN defaults it on for this Ada card; it costs 5.4e-2 per logit),
matmul precision 'highest', deterministic algorithms, and ``set_per_process_memory_fraction(0.78)``
so Windows WDDM cannot silently spill CUDA allocations into host RAM.
"""
from __future__ import annotations

import os
import platform
import sys
import types
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

SUPA_DIR = Path("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze")
SUPA_WEIGHT_SHA256 = "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b"
SUPA_CONFIG_SHA256 = "660db3b7"  # prefix recorded by SOUP-MINI-001 (soup.ASSETS); full value checked there
EOS_IDS = (151645, 151643)
REGISTERED_RUNTIME = {"python": "3.11.15", "torch": "2.6.0+cu124", "transformers": "5.14.1"}
VRAM_FRACTION = 0.78
THINK_STUB = "<think>\n\n</think>\n\n"


# ---------------------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------------------

def configure_runtime(seed: int, device: str) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION, 0)
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 is still enabled")
    return runtime_receipt(device)


def runtime_receipt(device: str) -> dict:
    import transformers

    rec = {"python": platform.python_version(), "sys_executable": sys.executable,
           "torch": torch.__version__, "transformers": transformers.__version__,
           "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
           "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
           "float32_matmul_precision": torch.get_float32_matmul_precision(),
           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
           "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "device": device}
    if device.startswith("cuda") and torch.cuda.is_available():
        rec["gpu"] = torch.cuda.get_device_name(0)
        rec["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
        rec["vram_fraction"] = VRAM_FRACTION
    return rec


def require_registered_runtime() -> None:
    import transformers

    got = {"python": platform.python_version(), "torch": torch.__version__,
           "transformers": transformers.__version__}
    if got != REGISTERED_RUNTIME:
        raise RuntimeError(f"unregistered runtime {got}; registered {REGISTERED_RUNTIME}")


# ---------------------------------------------------------------------------------------
# BF16-storage / FP32-compute patches
# ---------------------------------------------------------------------------------------

class _UpcastHeadFn(torch.autograd.Function):
    """logits = h @ W.to(fp32).T in vocabulary chunks; backward re-upcasts W (never saved in FP32)."""

    @staticmethod
    def forward(ctx, h, weight, chunk):
        ctx.save_for_backward(h, weight)
        ctx.chunk = chunk
        out = h.new_empty(h.shape[:-1] + (weight.shape[0],))
        for lo in range(0, weight.shape[0], chunk):
            out[..., lo:lo + chunk] = h @ weight[lo:lo + chunk].to(h.dtype).T
        return out

    @staticmethod
    def backward(ctx, grad):
        h, weight = ctx.saved_tensors
        gh = torch.zeros_like(h)
        for lo in range(0, weight.shape[0], ctx.chunk):
            gh += grad[..., lo:lo + ctx.chunk] @ weight[lo:lo + ctx.chunk].to(h.dtype)
        return gh, None, None


def _upcast_linear_forward(self, x):
    cache = getattr(self, "_fp32_cache", None)
    w = cache if (cache is not None and not torch.is_grad_enabled()) else self.weight.to(x.dtype)
    b = None if self.bias is None else self.bias.to(x.dtype)
    return F.linear(x, w, b)


def _upcast_embedding_forward(self, ids):
    return F.embedding(ids, self.weight, self.padding_idx).to(torch.float32)


def _upcast_head_forward(self, h):
    if h.dtype != torch.float32:
        raise TypeError("the LM head expects FP32 hidden states")
    if not torch.is_grad_enabled() and getattr(self, "_fp32_cache", None) is not None:
        return F.linear(h, self._fp32_cache)
    if self.weight.dtype == torch.float32:
        return F.linear(h, self.weight)
    return _UpcastHeadFn.apply(h, self.weight, self._chunk)


def make_fp32_compute(model: nn.Module, head_chunk: int = 16384, skip: Sequence[str] = ()) -> dict:
    """Patch a (BF16-stored) Qwen3ForCausalLM to compute exactly in FP32. ``skip`` lists Linear
    module names that another patcher (DeltaHost) owns. Returns an inventory of what was patched."""
    skip = set(skip)
    n_lin = n_norm = 0
    head = model.get_output_embeddings()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod is not head and name not in skip:
            mod.forward = types.MethodType(_upcast_linear_forward, mod)
            n_lin += 1
        elif "RMSNorm" in type(mod).__name__:
            mod.float()
            n_norm += 1
    emb = model.get_input_embeddings()
    emb.forward = types.MethodType(_upcast_embedding_forward, emb)
    head._chunk = head_chunk
    head._fp32_cache = None
    head.forward = types.MethodType(_upcast_head_forward, head)
    rotary = model.model.rotary_emb
    if rotary.rope_type == "default":
        rope_init_fn = rotary.compute_default_rope_parameters
    else:  # pragma: no cover - Supa uses default RoPE
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        rope_init_fn = ROPE_INIT_FUNCTIONS[rotary.rope_type]
    # Same call the constructor makes, so the FP32 buffer equals an FP32 load's buffer exactly.
    inv_freq, scaling = rope_init_fn(rotary.config, rotary.inv_freq.device)
    rotary.register_buffer("inv_freq", inv_freq.to(torch.float32), persistent=False)
    rotary.register_buffer("original_inv_freq", inv_freq.to(torch.float32).clone(), persistent=False)
    rotary.attention_scaling = scaling
    tied = head.weight is emb.weight
    return {"patched_linears": n_lin, "fp32_norms": n_norm, "embedding": type(emb).__name__,
            "head_chunk": head_chunk, "tied_head": tied, "rope_inv_freq_dtype": str(rotary.inv_freq.dtype)}


def set_eval_head_cache(model: nn.Module, on: bool) -> None:
    """Evaluation only: cache the FP32 head weight (622 MB) to avoid re-upcasting every step."""
    head = model.get_output_embeddings()
    head._fp32_cache = head.weight.detach().to(torch.float32) if on else None


def set_eval_weight_cache(model: nn.Module, on: bool, skip: Sequence[str] = ()) -> int:
    """Evaluation only: cache FP32 copies of every patched non-footprint Linear weight (the exact
    same values the per-step upcast produces), used only when autograd is off. Returns the number of
    cached modules. ``skip`` = footprint module names (DeltaHost caches those itself)."""
    skip = set(skip)
    head = model.get_output_embeddings()
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod is not head and name not in skip:
            mod._fp32_cache = mod.weight.detach().to(torch.float32) if on else None
            n += 1
    set_eval_head_cache(model, on)
    return n


# ---------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------

def load_tokenizer(path: Path = SUPA_DIR):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    if tok.eos_token_id != EOS_IDS[0]:
        raise ValueError(f"unexpected EOS id {tok.eos_token_id}")
    return tok


def load_model(path: Path = SUPA_DIR, device: str = "cuda", *, verify_sha: bool = True,
               attn_implementation: str = "eager"):
    """Load Supa in BF16 storage. The caller then installs DeltaHost and make_fp32_compute."""
    from transformers import AutoModelForCausalLM

    from .io import sha256_file

    weights = Path(path) / "model.safetensors"
    if verify_sha:
        got = sha256_file(weights)
        if got != SUPA_WEIGHT_SHA256:
            raise ValueError(f"Supa weight sha256 {got} != pin {SUPA_WEIGHT_SHA256}")
    # device_map loads tensors straight onto the device: host RAM (the binding resource on this
    # machine) never holds the full model.
    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.bfloat16, local_files_only=True,
                                                 attn_implementation=attn_implementation,
                                                 low_cpu_mem_usage=True, device_map={"": device})
    if model.get_output_embeddings().weight is not model.get_input_embeddings().weight:
        raise ValueError("expected tied embeddings")
    model.config.use_cache = False
    return model


# ---------------------------------------------------------------------------------------
# Rows, loss, decoding
# ---------------------------------------------------------------------------------------

def compile_train_rows(rows, tokenizer, max_sequence: int = 2048, completion_reserve: int = 128) -> list[dict]:
    """data.Row (with oracle target) -> Titan-contract training dict (ids, labels, ...)."""
    from .recipe import compile_row

    out = []
    for r in rows:
        c = compile_row(list(r.messages), r.target, tokenizer, max_sequence=max_sequence,
                        completion_reserve=completion_reserve)
        c["id"] = r.id
        c["parent_id"] = r.parent_id
        c["signature"] = r.signature
        c["role"] = r.role
        out.append(c)
    return out


def prompt_ids(row, tokenizer) -> list[int]:
    prefix = tokenizer.apply_chat_template(list(row.messages), tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    if not prefix.endswith(THINK_STUB):
        raise ValueError("thinking must be disabled: the prefix must end with the closed think stub")
    return tokenizer.encode(prefix, add_special_tokens=False)


def make_loss_fn(model: nn.Module, device: str):
    """Summed next-token CE over supervised label positions (labels shifted once), as the native
    lane's ``losses.chunked_sum``; logits are computed only at supervised positions."""
    head = model.get_output_embeddings()

    def loss_fn(row: dict):
        ids = torch.tensor(row["ids"], device=device)[None]
        labels = torch.tensor(row["labels"], device=device)
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
        nxt = labels[1:]
        pos = (nxt != -100).nonzero().squeeze(1)
        logits = head(hidden[pos])
        loss = F.cross_entropy(logits, nxt[pos], reduction="sum")
        return loss, int(pos.numel())
    return loss_fn


@torch.no_grad()
def greedy_generate(model: nn.Module, ids: Sequence[int], device: str, max_new_tokens: int = 64,
                    eos_ids: Sequence[int] = EOS_IDS) -> dict:
    """Greedy, batch 1, KV-cached; stops at any EOS id (the EOS token is NOT part of the text)."""
    x = torch.tensor(list(ids), device=device)[None]
    out = model(input_ids=x, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    gen: list[int] = []
    terminated = False
    for _ in range(max_new_tokens):
        nxt = int(out.logits[0, -1].argmax())
        if nxt in eos_ids:
            terminated = True
            break
        gen.append(nxt)
        out = model(input_ids=torch.tensor([[nxt]], device=device), past_key_values=past, use_cache=True,
                    logits_to_keep=1)
        past = out.past_key_values
    return {"ids": gen, "terminated": terminated}


@torch.no_grad()
def teacher_forced(model: nn.Module, row: dict, device: str, keep_vectors: bool = False) -> dict:
    """Per-token log-probabilities of the supervised target (unclipped), optionally the full
    log-softmax vectors at those positions (for the vocabulary-level interaction)."""
    head = model.get_output_embeddings()
    ids = torch.tensor(row["ids"], device=device)[None]
    labels = torch.tensor(row["labels"], device=device)
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
    nxt = labels[1:]
    pos = (nxt != -100).nonzero().squeeze(1)
    logsm = torch.log_softmax(head(hidden[pos]), dim=-1)
    lp = logsm.gather(1, nxt[pos][:, None]).squeeze(1)
    out = {"token_logprobs": lp.double().cpu().numpy(), "sum_logprob": float(lp.double().sum())}
    if keep_vectors:
        out["logsm"] = logsm
    return out
