"""Exact Titan-Sparse instantiation and generation-only qualification.

Imported ONLY by a claimed MQ child. Helpers are also unit-tested with tiny CPU
fixtures; helper return values are not native receipts. No optimizer, backward,
training data, teacher, protected bank, or general evaluation launcher exists here.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from experiments.model_quality_gpu_v1 import native as mq_native
from experiments.model_quality_gpu_v1.integrity import canonical, digest, read_json
from experiments.reasoning_posttraining_v1.serialization import compile_row
from . import contracts as c

_LIVE = object()
FIXTURE = {"id": "SCP-NQ-MR-SYNTHETIC-001", "world_id": "SCP-NQ-MECHANICS-WORLD-001",
           "system": "Use only the stated facts. This is an unscored software qualification prompt.",
           "user": "A register starts at 2. Increase it by 3. Report the final register value."}


def public_fixture():
    from experiments.sequential_complementary_v1.data import PublicCase
    return PublicCase(**FIXTURE)


def render(tokenizer, messages, cap: int) -> list[int]:
    c.require(type(cap) is int and 1 <= cap <= 288, "unregistered generation allowance")
    c.require(type(messages) is list and len(messages) == 2
              and all(type(m) is dict and set(m) == {"role", "content"} for m in messages)
              and [m["role"] for m in messages] == ["system", "user"]
              and all(type(m["content"]) is str and m["content"].strip() and "<|" not in m["content"] for m in messages),
              "native inference accepts only explicit, control-token-free system/user input")
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    c.require(text.endswith("<think>\n\n</think>\n\n"), "native direct-mode prefix differs")
    ids = tokenizer.encode(text, add_special_tokens=False)
    native = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    c.require(type(ids) is list and ids == native and bool(ids)
              and all(type(t) is int and 0 <= t < 151936 for t in ids), "native tokenizer parity failed")
    c.require(len(ids) + cap <= c.HOST["max_sequence"], "OVERLENGTH: no truncation or silent skip")
    return ids


def bit_equal(a, b) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def verify_sparse_tensor(original, trained, candidate):
    """Reconstruct the exact inherited per-tensor support, never scalar k*delta."""
    from experiments.sequential_complementary_v1.delta import magnitude_mask
    c.require(original.dtype == trained.dtype == candidate.dtype == torch.float32, "FP32 reconstruction required")
    c.require(original.shape == trained.shape == candidate.shape, "reconstruction shape differs")
    expected = original + magnitude_mask(trained - original, 0.25)
    c.require(bit_equal(expected, candidate), "incumbent differs from source + M_0.25(trained-source)")


def state_identity(model) -> dict:
    buffers = {}
    for name, value in model.named_buffers():
        data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        import hashlib
        buffers[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                         "sha256": hashlib.sha256(data).hexdigest()}
    return {"parameters": mq_native.parameter_digest(model), "buffers": digest(buffers)}


def frozen(model):
    c.require(not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters()),
              "inference qualification requires eval, zero trainable tensors and no gradients")
    c.require(all(p.dtype == torch.float32 and not p.is_meta for p in model.parameters()), "actual computation must remain FP32")


def _load_subject(binding, guard):
    from safetensors import safe_open
    assets = binding["assets"]
    source, incumbent, trained = (Path(assets[k]) for k in ("source_path", "subject_path", "trained_path"))
    mq_native.runtime(require=True)
    mq_native.configure_runtime(17)
    guard(before_load=True)
    tokenizer = mq_native.load_tokenizer(source)
    # The inherited loader really loads Supa here. The distinct subject is applied
    # below, audited tensor-for-tensor, and NEVER labeled as that Supa source.
    model = mq_native.load_model(source, "cpu")
    model.requires_grad_(False)
    actual = dict(model.named_parameters())
    c.require({n: list(p.shape) for n, p in actual.items()} == c.shapes(), "loaded unique 310-tensor inventory differs")
    c.require(sum(p.numel() for p in actual.values()) == 596049920, "loaded parameter count differs")
    c.require(model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), "tied storage lost")
    base_identity = state_identity(model)
    allow = c.trained_shapes()
    for name, parameter in actual.items():
        guard()
        # Reopen per tensor so memory-mapped pages from all three archives do not
        # accumulate throughout the scan. Large unchanged embeddings are chunked.
        with safe_open(str(source / "model.safetensors"), framework="pt", device="cpu") as src, \
             safe_open(str(incumbent / "model.safetensors"), framework="pt", device="cpu") as dst:
            original, candidate = src.get_tensor(name), dst.get_tensor(name)
            c.require(candidate.dtype == torch.float32, "incumbent silently cast from FP32")
            left, right, loaded = original.reshape(-1), candidate.reshape(-1), parameter.detach().reshape(-1)
            for start in range(0, loaded.numel(), 262144):
                guard()
                base = left[start:start + 262144].float()
                target = right[start:start + 262144]
                c.require(bool(torch.isfinite(base).all()) and bool(torch.isfinite(target).all()), "nonfinite asset tensor")
                c.require(bit_equal(loaded[start:start + 262144], base), "source loader changed the registered source")
                if name not in allow:
                    c.require(bit_equal(base, target), "sparse subject changed an untrained tensor")
                del base, target
            if name in allow:
                with safe_open(str(trained), framework="pt", device="cpu") as partial:
                    update_source = partial.get_tensor(name)
                    verify_sparse_tensor(original.float(), update_source, candidate)
                    del update_source
            with torch.no_grad():
                parameter.copy_(candidate)
            c.require(bit_equal(parameter.detach(), candidate), "subject loading did not preserve exact FP32 bits")
            del original, candidate, left, right, loaded
    guard()
    # Hash again AFTER materialization; a path-shaped or stale hash is not a permit.
    c.require(c.audit_assets(source, trained, incumbent) == assets, "assets changed while loading")
    model = model.to("cuda:0").eval()
    frozen(model)
    c.require(model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), "device transfer lost tied storage")
    c.require(getattr(model.config, "_attn_implementation", None) == "eager", "unqualified attention backend")
    guard()
    return model, tokenizer, {"source_id": assets["source_id"], "subject_id": c.SUBJECT,
        "source_weight_sha256": c.SOURCE_SHA, "subject_weight_sha256": c.SUBJECT_SHA,
        "trained_partial_sha256": c.TRAINED_SHA, "source_state": base_identity,
        "subject_state": state_identity(model), "full_tensors_verified": 310,
        "trained_tensors_reconstructed": 220, "source_sidecar_and_explicit_fp32_loader": True,
        "storage_tying": True, "optimizer_steps": 0}


class GenerationBackend:
    """A live owned worker, not a callback or a portable hash-string credential."""
    evidence_class = "NATIVE_QUALIFICATION_ONLY"

    def __init__(self, model, tokenizer, reserve, guard, emit, run_dir=None, *, _key=None):
        self.model, self.tokenizer = model, tokenizer
        self.reserve, self.guard, self.emit = reserve, guard, emit
        self.run_dir, self._key = run_dir, _key
        self.closed = False
        self.forward_calls = 0
        self.generation_requests = 0
        self.generated_tokens = 0
        self._pending = None
        self._budget = None
        self._completed = 0
        self._arms_started = set()
        self._messages = []
        self._generated = []
        self.reserve_request = self._reserve_request  # stable callback identity

    def close(self):
        self.closed = True
        self._key = None
        self._budget = None

    def authorize(self, case, arm, budget, emit):
        c.require(self._key is _LIVE and not self.closed and self.run_dir is not None, "native backend is not live/qualified")
        claim = read_json(Path(self.run_dir) / "worker_claim.json")
        c.require(claim["pid"] == os.getpid() and not (Path(self.run_dir) / "supervisor.json").exists(), "owned worker is no longer live")
        c.require(case == public_fixture() and arm in ("MR", "REP"), "only the fixed unscored MR qualification case is permitted")
        c.require(budget.reserve is self.reserve_request and emit is self.emit
                  and budget.max_calls == 5 and budget.max_output_tokens == 288,
                  "foreign supervisor reservation/budget/output sink")
        c.require(self._pending is None, "failed prior request is not retryable")
        c.require(arm not in self._arms_started, "a qualification arm cannot be retried")
        from experiments.sequential_complementary_v1.reasoning import _prompt, ROLES, REPEAT
        self._messages = [_prompt(case, role) for role in (ROLES if arm == "MR" else REPEAT)]
        self._generated = []
        self._arms_started.add(arm)
        self._budget, self._completed = budget, 0
        frozen(self.model)

    def check_context(self, messages, cap):
        return render(self.tokenizer, messages, cap)

    def _reserve_request(self, charges):
        c.require(self._key is _LIVE and not self.closed and self._budget is not None, "no active inference qualification")
        c.require(type(charges) is dict and set(charges) == {"model_calls", "output_token_allowance"}
                  and type(charges["model_calls"]) is int and charges["model_calls"] == 1
                  and type(charges["output_token_allowance"]) is int and charges["output_token_allowance"] in (32, 64),
                  "unregistered generation request")
        c.require(self._pending is None and self._completed < 5, "reused/exhausted generation request")
        c.require(charges["output_token_allowance"] == (64 if self._completed < 4 else 32),
                  "role-specific output allowance differs")
        self.reserve({"generation_requests": 1, "generated_tokens_all_candidates": charges["output_token_allowance"]})
        self._pending = charges["output_token_allowance"]
        self.generation_requests += 1

    @torch.inference_mode()
    def forward(self, ids, *, past=None, cache=False, keep=1):
        c.require(not self.closed, "closed backend")
        frozen(self.model)
        self.guard()
        self.reserve({"evaluation_and_qualification_model_calls": 1})
        self.forward_calls += 1  # failed attempts remain charged
        value = self.model(input_ids=ids, past_key_values=past, use_cache=cache,
                           return_dict=True, logits_to_keep=keep)
        c.require(value.logits.dtype == torch.float32 and bool(torch.isfinite(value.logits).all()), "nonfinite/non-FP32 logits")
        if cache:
            c.require(value.past_key_values is not None, "generation backend returned no KV cache")
        self.guard()
        return value

    @torch.inference_mode()
    def generate(self, messages, cap):
        c.require(self._key is _LIVE and not self.closed and self._pending == cap and self._budget is not None,
                  "native generation requires a charged live request")
        c.require(self._budget.calls == self._completed + 1, "generation bypassed the frozen request budget")
        if self._completed < 4:
            expected = self._messages[self._completed]
        else:
            from experiments.sequential_complementary_v1.reasoning import SELECT_RULE
            case = public_fixture()
            expected = [{"role": "system", "content": case.system + "\n" + SELECT_RULE},
                        {"role": "user", "content": case.user + "\nCANDIDATE_DATA_JSON:\n" + canonical([
                            {"index": i, **record} for i, record in enumerate(self._generated)]).decode()}]
        c.require(messages == expected, "generation prompt differs from the fixed qualification policy")
        tokens = self.check_context(messages, cap)
        device = next(self.model.parameters()).device
        current = torch.tensor([tokens], dtype=torch.long, device=device)
        emitted, past, terminated = [], None, False  # NEVER shared between roles
        started = time.monotonic()
        try:
            for index in range(cap):
                value = self.forward(current, past=past, cache=True)
                past = value.past_key_values
                c.require(past.get_seq_length() == len(tokens) + index, "KV length/cache progression differs")
                token = int(value.logits[0, -1].argmax())
                emitted.append(token)
                self.generated_tokens += 1
                if token == self.tokenizer.eos_token_id:
                    terminated = True
                    break
                current = torch.tensor([[token]], dtype=torch.long, device=device)
            text = self.tokenizer.decode(emitted[:-1] if terminated else emitted, skip_special_tokens=False)
            self.emit({"event": "NATIVE_TOKEN_RECORD", "messages_sha256": digest(messages),
                       "generated_ids_including_terminal": emitted, "terminated": terminated,
                       "input_tokens": len(tokens), "output_tokens": len(emitted),
                       "wall_seconds": time.monotonic() - started})
            self._generated.append({"text": text, "terminated": terminated})
            self._pending = None
            self._completed += 1
            return {"text": text, "terminated": terminated, "input_tokens": len(tokens), "output_tokens": len(emitted)}
        except Exception as exc:
            self.emit({"event": "NATIVE_GENERATION_FAILED", "generated_ids_including_terminal": emitted,
                       "messages_sha256": digest(messages), "exception_type": type(exc).__name__, "message": str(exc)})
            raise


def require_native_backend(backend, case, arm, budget, emit):
    c.require(type(backend) is GenerationBackend, "SCP native backend/supervisor integration requires the exact live adapter")
    backend.authorize(case, arm, budget, emit)


def open_backend(binding, reserve, guard, emit, run_dir):
    # claim_worker() has already consumed and validated the one-use MQ context.
    claim = read_json(Path(run_dir) / "worker_claim.json")
    c.require(claim["pid"] == os.getpid() and not (Path(run_dir) / "supervisor.json").exists(), "MQ child claim missing/stale")
    model, tokenizer, loading = _load_subject(binding, guard)
    return GenerationBackend(model, tokenizer, reserve, guard, emit, run_dir, _key=_LIVE), loading


@torch.inference_mode()
def student_probes(backend):
    """Mechanical facts only. No backward, loss, optimizer, or answer scoring."""
    model, tokenizer = backend.model, backend.tokenizer
    frozen(model)
    before = state_identity(model)
    prompt = [{"role": "system", "content": "Answer with the requested number."},
              {"role": "user", "content": "Reply with 7."}]
    row = compile_row(prompt, "7", tokenizer, max_sequence=2048, completion_reserve=128)
    c.require(render(tokenizer, prompt, 128) == row["prompt_ids"], "RRP/native generation prefix differs")
    c.require(len(row["ids"]) <= 64, "short parity fixture exceeded its frozen bound")
    device = next(model.parameters()).device
    ids = torch.tensor([row["ids"]], dtype=torch.long, device=device)
    direct = backend.forward(ids, keep=0).logits
    backend.guard(); backend.reserve({"evaluation_and_qualification_model_calls": 1}); backend.forward_calls += 1
    hidden = mq_native.hidden_states(model, ids)
    projected = F.linear(hidden, model.lm_head.weight)
    c.require(projected.dtype == torch.float32 and bool(torch.isfinite(projected).all()), "hidden/head parity is nonfinite")
    torch.testing.assert_close(projected, direct, atol=0, rtol=0)
    parts, past = [], None
    for start in range(0, ids.shape[1], 3):
        value = backend.forward(ids[:, start:start + 3], past=past, cache=True, keep=0)
        parts.append(value.logits); past = value.past_key_values
        c.require(past.get_seq_length() == min(start + 3, ids.shape[1]), "short cached sequence length differs")
    cached = torch.cat(parts, dim=1)
    torch.testing.assert_close(cached, direct, atol=2e-4, rtol=2e-4)
    cache_error = float((cached - direct).abs().max())
    altered = ids.clone()
    ordinary = tokenizer.encode("8", add_special_tokens=False)
    c.require(bool(ordinary) and ordinary[0] != tokenizer.eos_token_id, "ordinary-token probe missing")
    altered[0, -3] = ordinary[0]
    perturbed = backend.forward(altered, keep=0).logits
    torch.testing.assert_close(perturbed[:, :-3], direct[:, :-3], atol=0, rtol=0)
    del direct, hidden, projected, parts, cached, past, value, perturbed
    # Real-source long-shape qualification without allocating [2048,vocab] logits.
    ordinary = tokenizer.encode("x y", add_special_tokens=False)
    c.require(bool(ordinary) and tokenizer.eos_token_id not in ordinary, "ordinary shape tokens missing")
    sequence = (ordinary * math.ceil(2048 / len(ordinary)))[:2048]
    long_ids = torch.tensor([sequence], dtype=torch.long, device=device)
    full = backend.forward(long_ids).logits
    prefill = backend.forward(long_ids[:, :-1], cache=True)
    c.require(prefill.past_key_values.get_seq_length() == 2047, "long prefill cache length differs")
    last = backend.forward(long_ids[:, -1:], past=prefill.past_key_values, cache=True)
    c.require(last.past_key_values.get_seq_length() == 2048, "long continuation cache length differs")
    torch.testing.assert_close(full, last.logits, atol=2e-4, rtol=2e-4)
    long_error = float((full - last.logits).abs().max())
    del full, prefill, last, long_ids
    frozen(model)
    backend.guard()
    after = state_identity(model)
    c.require(before == after, "zero-update qualification changed parameters/buffers")
    return {"state_before": before, "state_after": after, "zero_update": True, "backward_calls": 0,
            "optimizer_steps": 0, "currently_trainable_tensors": 0, "cached_max_abs": cache_error,
            "shape2048_cached_max_abs": long_error, "qualified_sequence_tokens": 2048,
            "checks": ["native_prefix_and_eos_masks", "hidden_logits", "short_cache", "causality",
                       "shape2048_last_logits_cache", "fp32", "no_gradients", "zero_update"],
            "training_shape_qualified": False, "answers_scored": 0}


def mr_probes(backend, max_seconds):
    from experiments.sequential_complementary_v1.reasoning import Budget, run_case, protocol_identity
    frozen(backend.model)
    before = state_identity(backend.model)
    results = []
    for arm in ("REP", "MR"):
        budget = Budget(5, 288, max_seconds, backend.reserve_request)
        value = run_case(public_fixture(), arm, backend, budget, backend.emit, native=True)
        c.require(value["classification"] == "NATIVE_QUALIFICATION_ONLY"
                  and value["calls"] == 5 and [r["role"] for r in value["generations"]] == [f"R{i}" for i in range(5)],
                  "MR request/role ledger differs")
        results.append({k: v for k, v in value.items() if k != "generations"})
    frozen(backend.model)
    after = state_identity(backend.model)
    c.require(before == after, "MR changed subject parameters/buffers")
    return {"protocol_sha256": protocol_identity(), "fixture_sha256": public_fixture().identity(),
            "state_before": before, "state_after": after, "results": results, "answers_scored": 0,
            "backward_calls": 0, "optimizer_steps": 0, "zero_update": True,
            "selector_validity_is_not_a_quality_gate": True,
            "note": "Malformed/truncated responses remain unscored mechanics observations, never filtered successes."}
