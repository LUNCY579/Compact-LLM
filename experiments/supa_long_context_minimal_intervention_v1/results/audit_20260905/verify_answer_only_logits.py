"""CPU tiny-model proof of answer-only LM-head loss/LoRA-gradient equivalence.

Does not load checkpoint weights or change live experiment source. Run at root:
../Chat-Bubble/.venv/Scripts/python.exe experiments/supa_long_context_minimal_intervention_v1/results/audit_20260905/verify_answer_only_logits.py
"""
from contextlib import nullcontext
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.loss.loss_utils import ForCausalLMLoss
from peft import LoraConfig, TaskType, get_peft_model

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "src"))
from slcmi.provenance import CORES, checkpoint_files, sha256


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20260905)
    before = {name: checkpoint_files(core["path"]) for name, core in CORES.items()}
    core = CORES["a11"]["path"]
    canonical_source = core / "modeling_supa_a11.py"
    spec = importlib.util.spec_from_file_location("canonical_forward_for_tiny_cpu_test", canonical_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = Qwen3Config(vocab_size=257, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=4, max_position_embeddings=4096, attention_dropout=0.0,
                         use_cache=False, tie_word_embeddings=True, eos_token_id=2,
                         pad_token_id=0, bos_token_id=1)
    config._attn_implementation = "sdpa"
    config.abstain_token_id = 5
    config.abstain_logit_bias = 0.0
    tiny = module.Qwen3AbstainForCausalLM(config).float().requires_grad_(False)
    model = get_peft_model(tiny, LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4,
                                         lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                         layers_to_transform=[3], layers_pattern="layers", bias="none"))
    # Test a nonzero adapter so both A and B receive nonzero gradients.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.01)
    base = model.get_base_model()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert all(".3." in name for name in params)
    input_lengths_seen = []
    head_shapes_seen = []
    input_hook = base.model.register_forward_pre_hook(lambda m, a, kw: input_lengths_seen.append(kw["input_ids"].shape[-1]), with_kwargs=True)
    head_hook = base.lm_head.register_forward_pre_hook(lambda m, a: head_shapes_seen.append(list(a[0].shape)))
    cases = []
    start = time.perf_counter()

    def measure(ids, labels, answer_tokens, reduced, amp_dtype):
        model.zero_grad(set_to_none=True)
        model.train()
        kw = {"input_ids": ids, "labels": labels, "use_cache": False}
        if reduced:
            keep = answer_tokens + 1
            kw.update(labels=labels[:, -keep:], logits_to_keep=keep)
        context = torch.autocast("cpu", dtype=amp_dtype) if amp_dtype else nullcontext()
        with context:
            out = model(**kw)
        out.loss.backward()
        return out.loss.detach().float(), {n: p.grad.detach().float().clone() for n, p in params.items()}, out.logits.detach().float()

    # Smallest valid prompt, EOS-only target, multi-token answers, observed
    # bounded maximum, and the proposed fixed limit. All include terminal EOS.
    for prompt_tokens, answer_tokens in ((1, 1), (1, 2), (73, 8), (3690, 8), (3832, 8)):
        sequence_length = prompt_tokens + answer_tokens
        ids = torch.randint(6, config.vocab_size, (1, sequence_length))
        ids[0, -1] = config.eos_token_id
        labels = ids.clone()
        labels[:, :prompt_tokens] = -100
        keep = answer_tokens + 1
        full_shift = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:]
        cropped_labels = labels[:, -keep:]
        cropped_shift = torch.nn.functional.pad(cropped_labels, (0, 1), value=-100)[:, 1:]
        assert torch.equal(full_shift[:, -keep:], cropped_shift)
        assert (full_shift[:, :-keep] == -100).all()
        assert int((cropped_shift != -100).sum()) == answer_tokens
        assert int(cropped_shift[0, -2]) == config.eos_token_id
        for precision, amp_dtype, atol, rtol in (("float32", None, 2e-7, 2e-5), ("cpu_autocast_bfloat16", torch.bfloat16, 2e-4, 0.02), ("cpu_autocast_float16", torch.float16, 2e-5, 0.002)):
            full_loss, full_grads, full_logits = measure(ids, labels, answer_tokens, False, amp_dtype)
            tail_loss, tail_grads, tail_logits = measure(ids, labels, answer_tokens, True, amp_dtype)
            loss_close = torch.allclose(full_loss, tail_loss, atol=atol, rtol=rtol)
            per_parameter = {}
            for name in params:
                first, second = full_grads[name], tail_grads[name]
                per_parameter[name] = {"max_abs_gradient_delta": float((first - second).abs().max()),
                                       "full_gradient_norm": float(first.norm()),
                                       "reduced_gradient_norm": float(second.norm()),
                                       "close": torch.allclose(first, second, atol=atol, rtol=rtol)}
            all_close = loss_close and all(v["close"] for v in per_parameter.values())
            cases.append({"prompt_tokens": prompt_tokens, "answer_tokens_including_eos": answer_tokens,
                          "sequence_length": sequence_length, "logits_to_keep": keep, "precision": precision,
                          "full_loss": float(full_loss), "reduced_loss": float(tail_loss),
                          "loss_abs_delta": float((full_loss - tail_loss).abs()), "loss_close": loss_close,
                          "full_logits_shape": list(full_logits.shape), "reduced_logits_shape": list(tail_logits.shape),
                          "tail_logits_max_abs_delta": float((full_logits[:, -keep:] - tail_logits).abs().max()),
                          "decoder_input_lengths": input_lengths_seen[-2:],
                          "lm_head_input_shapes": head_shapes_seen[-2:], "supervised_targets": answer_tokens,
                          "terminal_eos_supervised": True, "parameters": per_parameter, "all_close": all_close})
            assert all_close, cases[-1]
            assert input_lengths_seen[-2:] == [sequence_length, sequence_length]
            assert head_shapes_seen[-1][1] == keep
            print(f"PASS {precision} length={sequence_length} K={keep} loss_delta={cases[-1]['loss_abs_delta']:.3g}", flush=True)
        # Deliberate off-by-one negative control: K=answer_tokens alone drops
        # the first answer target after causal shifting (EOS-only becomes empty).
        wrong_shift = torch.nn.functional.pad(labels[:, -answer_tokens:], (0, 1), value=-100)[:, 1:]
        assert int((wrong_shift != -100).sum()) == answer_tokens - 1

    # Canonical wrapper with the real nonzero gate remains loss-equivalent under
    # no_grad validation. It must stay disabled for backward as in existing code.
    model.eval()
    model.config.abstain_logit_bias = 5.156249985098839
    with torch.no_grad():
        gate_full = model(input_ids=ids, labels=labels, use_cache=False)
        gate_tail = model(input_ids=ids, labels=labels[:, -keep:], logits_to_keep=keep, use_cache=False)
    gate_loss_close = torch.allclose(gate_full.loss, gate_tail.loss, atol=2e-7, rtol=2e-5)
    assert gate_loss_close
    input_hook.remove()
    head_hook.remove()
    after = {name: checkpoint_files(core["path"]) for name, core in CORES.items()}
    canonical_config = json.loads((core / "config.json").read_text(encoding="utf-8"))
    vocab = canonical_config["vocab_size"]
    receipt = {"python": sys.executable, "torch": torch.__version__, "canonical_model_loads": 0,
               "tiny_random_cpu_models": 1, "gpu_used": False, "gradient_checkpointing": True,
               "tiny_trainable_layers": [3], "canonical_layers_0_1_2_untouched": True,
               "cases": cases, "all_passed": all(r["all_close"] for r in cases),
               "gate_on_validation_loss_equivalence": bool(gate_loss_close),
               "off_by_one_negative_control_detected": True,
               "canonical_vocab_size": vocab,
               "theoretical_logits_storage_bytes_at_3698_tokens_and_K9": {
                   "full_fp16": 3698 * vocab * 2, "reduced_fp16": 9 * vocab * 2,
                   "full_loss_fp32_copy": 3698 * vocab * 4, "reduced_loss_fp32_copy": 9 * vocab * 4,
                   "note": "Tensor-size arithmetic only; not a measurement of peak VRAM or total saving."},
               "source_sha256": {str(p): sha256(p) for p in (Path(__file__), canonical_source, Path(inspect.getfile(Qwen3ForCausalLM)), Path(inspect.getfile(ForCausalLMLoss)))},
               "checkpoint_sha256_before": before, "checkpoint_sha256_after": after,
               "canonical_checkpoint_unchanged": before == after, "wall_seconds": time.perf_counter() - start}
    assert receipt["canonical_checkpoint_unchanged"]
    destination = Path(__file__).with_name("answer_only_logits_equivalence.json")
    destination.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(destination, flush=True)


if __name__ == "__main__":
    main()
