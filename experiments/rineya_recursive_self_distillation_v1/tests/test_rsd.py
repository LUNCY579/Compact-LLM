"""Mechanical evidence only: random CPU models and explicitly scripted fixtures."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from experiments.rineya_recursive_self_distillation_v1 import contracts as c
from experiments.rineya_recursive_self_distillation_v1.backend import TorchBackend, tensor_hash
from experiments.rineya_recursive_self_distillation_v1.data import (
    accept_rewrite, demo_catalog, from_prlc, load_catalog, normalized,
    require_training_roles, validate_catalog, verify, write_catalog,
)
from experiments.rineya_recursive_self_distillation_v1.engine import evaluate, scheduled, train, update
from experiments.rineya_recursive_self_distillation_v1.evidence import RunStore, read_json
from experiments.rineya_recursive_self_distillation_v1.metrics import paired_effect, retention_stop, summary
from experiments.rineya_recursive_self_distillation_v1.objectives import guidance_kl, token_ce
from experiments.rineya_recursive_self_distillation_v1.prompts import Views
from experiments.rineya_recursive_self_distillation_v1.__main__ import compare, main, read_run


torch.set_num_threads(1)


class Tokenizer:
    eos_token_id = 2
    all_special_ids = [0, 1, 2]
    def encode(self, text, add_special_tokens=False):
        if text == "FORBIDDEN_CONTROL":
            return [0]
        return [x + 3 for x in text.encode()]
    def decode(self, ids, skip_special_tokens=False):
        return bytes(max(0, min(255, x - 3)) for x in ids).decode(errors="replace")
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=True, add_generation_prompt=True, enable_thinking=False)
        return [1] + self.encode(messages[0]["content"]) + [1]


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(259, 8)
        with torch.no_grad():
            self.embed.weight.mul_(.1)
            self.embed.weight[:, 0] = 1
        self.embed.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(2, 8) * .1)
        self.lora_B = nn.Parameter(torch.zeros(8, 2))
    def forward(self, input_ids, use_cache=False):
        h = self.embed(input_ids).cumsum(1)
        h = h / torch.arange(1, h.shape[1] + 1, device=h.device)[None, :, None]
        h = h + F.linear(F.linear(h, self.lora_A), self.lora_B)
        return SimpleNamespace(last_hidden_state=h)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Body()
        self.lm_head = nn.Linear(8, 259, bias=False)
        self.lm_head.requires_grad_(False)
        with torch.no_grad():
            self.lm_head.weight.mul_(.1)
            self.lm_head.weight[2, 0] = 8


def make_backend(plan=None):
    torch.manual_seed(33)
    return TorchBackend(TinyModel(), Tokenizer(), plan or small_plan(), eos_ids=(2,))


def small_plan(**kw):
    return replace(c.Plan(), rounds=2, batch_tasks=2, eval_every=1,
                   max_new_tokens=64, chunk_tokens=2, bootstrap_samples=100, **kw)


def binding(plan, tasks, arm):
    return {"arm": arm, "plan": plan.fingerprint, "catalog": c.digest([asdict(t) for t in tasks]),
            "evidence": "random_CPU_fixture_NOT_native", "parent": "same-random-fixture"}


def rollout(task, text, version=0):
    tokenizer = Tokenizer()
    ids = tuple(tokenizer.encode(text)) + (2,)
    return c.Rollout(task.key, Views(tokenizer, (2,), 1024).student(task.public()), ids,
                     text, tuple(-1.0 for _ in ids), version, True)


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.p = torch.randn(5, 4, dtype=torch.double, requires_grad=True)
        self.q = torch.randn(5, 4, dtype=torch.double, requires_grad=True)
        self.w = torch.randn(11, 4, dtype=torch.double)
        self.y = torch.tensor([0, 2, 4, 6, 8])
    def test_kl_value_all_vocabulary(self):
        lp = F.linear(self.p, self.w).log_softmax(-1)
        lq = F.linear(self.q, self.w).log_softmax(-1)
        ref = (lq.exp() * (lq - lp)).sum(-1).mean()
        for chunk in (1, 2, 5, 8):
            self.assertTrue(torch.allclose(guidance_kl(self.p, self.q, self.w, chunk), ref, atol=1e-12))
    def test_kl_gradient_reference(self):
        lp = F.linear(self.p, self.w).log_softmax(-1)
        lq = F.linear(self.q.detach(), self.w).log_softmax(-1)
        ref = (lq.exp() * (lq - lp)).sum(-1).mean()
        a = torch.autograd.grad(ref, self.p)[0]
        b = torch.autograd.grad(guidance_kl(self.p, self.q, self.w, 2), self.p)[0]
        torch.testing.assert_close(a, b)
    def test_teacher_stop_gradient(self):
        guidance_kl(self.p, self.q, self.w).backward()
        self.assertIsNone(self.q.grad)
        self.assertIsNotNone(self.p.grad)
    def test_kl_gradcheck(self):
        self.assertTrue(torch.autograd.gradcheck(lambda x: guidance_kl(x, self.q, self.w, 2), (self.p,)))
    def test_ce_gradcheck(self):
        self.assertTrue(torch.autograd.gradcheck(lambda x: token_ce(x, self.w, self.y, 2), (self.p,)))
    def test_ce_value_gradient_reference(self):
        expected = F.cross_entropy(F.linear(self.p, self.w), self.y)
        actual = token_ce(self.p, self.w, self.y, 2)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, self.p)[0], torch.autograd.grad(expected, self.p)[0])
    def test_bfloat16_scaled_gradient(self):
        p, q, w = (x.detach().bfloat16() for x in (self.p, self.q, self.w))
        p.requires_grad_()
        actual = guidance_kl(p, q, w, 2) * .03
        lp, lq = F.linear(p, w).float().log_softmax(-1), F.linear(q, w).float().log_softmax(-1)
        ref = (lq.exp() * (lq - lp)).sum() / len(p) * .03
        self.assertTrue(torch.allclose(torch.autograd.grad(actual, p)[0].float(),
                                      torch.autograd.grad(ref, p)[0].float(), atol=.0003, rtol=.02))
    def test_identity_zero_loss(self):
        self.assertEqual(float(guidance_kl(self.p, self.p, self.w).detach()), 0)
    def test_direction_is_forward(self):
        self.assertFalse(torch.isclose(guidance_kl(self.p, self.q, self.w), guidance_kl(self.q, self.p, self.w)))
    def test_head_must_be_frozen(self):
        with self.assertRaises(ValueError):
            guidance_kl(self.p, self.q, self.w.requires_grad_())
    def test_invalid_shape_and_chunk(self):
        for args in ((self.p[:0], self.q[:0], self.w, 2), (self.p, self.q, self.w, 0)):
            with self.assertRaises(ValueError):
                guidance_kl(*args)
    def test_nonfinite_hidden_rejected(self):
        p = self.p.detach().clone()
        p[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            guidance_kl(p, self.q, self.w)
    def test_ce_out_of_range(self):
        with self.assertRaises(ValueError):
            token_ce(self.p, self.w, self.y + 20)


class ContractDataTests(unittest.TestCase):
    def setUp(self):
        self.tasks = demo_catalog()
    def test_valid_catalog_all_roles(self):
        self.assertEqual(validate_catalog(self.tasks)["rows"], 20)
        require_training_roles(self.tasks)
    def test_reference_wrong_rejected(self):
        with self.assertRaises(ValueError):
            validate_catalog([replace(self.tasks[0], answer="999")])
    def test_duplicate_key_rejected(self):
        with self.assertRaises(ValueError):
            validate_catalog(self.tasks + [self.tasks[0]])
    def test_parent_split_leak_rejected(self):
        with self.assertRaises(ValueError):
            validate_catalog([self.tasks[0], replace(self.tasks[0], key="other", split="calibration")])
    def test_normalized_prompt_leak_rejected(self):
        other = replace(self.tasks[0], key="other", parent="other", split="calibration",
                        prompt="   " + self.tasks[0].prompt.upper() + "  ")
        with self.assertRaises(ValueError):
            validate_catalog([self.tasks[0], other])
    def test_exclusion_parent(self):
        with self.assertRaises(ValueError):
            validate_catalog(self.tasks, {"parents": [self.tasks[0].parent], "normalized_prompt_sha256": []})
    def test_exclusion_prompt(self):
        with self.assertRaises(ValueError):
            validate_catalog(self.tasks, {"parents": [], "normalized_prompt_sha256": [c.digest(normalized(self.tasks[0].prompt))]})
    def test_unknown_split_rejected(self):
        for split in ("sealed", "test", "challenge", "FINAL", "unknown"):
            with self.assertRaises(ValueError):
                replace(self.tasks[0], split=split)
    def test_all_five_roles_required(self):
        with self.assertRaises(ValueError):
            require_training_roles(self.tasks[:4])
    def test_retention_source_coverage(self):
        tasks = [replace(t, source="new") if t.split == "retention_train" else t for t in self.tasks]
        with self.assertRaises(ValueError):
            require_training_roles(tasks)
    def test_verifier_requires_terminal_unique_answer(self):
        t = self.tasks[0]
        for text in (f"<final>{t.answer}</final> trailing", f"<final>{t.answer}</final><final>{t.answer}</final>",
                     f"<final>+{t.answer}</final>", "", "answer guessed"):
            self.assertFalse(verify(t, text)[0])
        self.assertTrue(verify(t, t.solution)[0])
        self.assertFalse(verify(t, t.solution, False)[0])
    def test_endpoint_not_reasoning_proof(self):
        t = self.tasks[0]
        self.assertEqual(verify(t, "Wrong intermediate reasoning. <final>" + t.answer + "</final>"), (True, "endpoint_only"))
    def test_shorter_correct_rewrite_accepted(self):
        t = self.tasks[0]
        self.assertTrue(accept_rewrite(t, rollout(t, t.solution), rollout(t, f"<final>{t.answer}</final>"))[0])
    def test_wrong_or_longer_rewrite_rejected(self):
        t = self.tasks[0]
        self.assertFalse(accept_rewrite(t, rollout(t, t.solution), rollout(t, "<final>999</final>"))[0])
        self.assertFalse(accept_rewrite(t, rollout(t, t.solution), rollout(t, t.solution))[0])
    def test_rewrite_stale_and_eval_rejected(self):
        t = self.tasks[0]
        with self.assertRaises(ValueError):
            accept_rewrite(t, rollout(t, t.solution), rollout(t, "x", 1))
        with self.assertRaises(ValueError):
            accept_rewrite(replace(t, split="calibration"), rollout(t, "a"), rollout(t, "b"))
    def test_prlc_conversion_preserves_identity(self):
        t = SimpleNamespace(key="s:i", source="s", revision="abc", group="s:p", split="train", prompt="2+3?",
                            answer="5", requirements=("text",), verifier="exact-v1")
        out = from_prlc(t, "5", "verified local origin")
        self.assertEqual((out.key, out.parent, out.split), ("s:i", "s:p", "train"))
        t.requirements = ("mimo:code",)
        with self.assertRaises(ValueError):
            from_prlc(t, "5", "source")
    def test_json_duplicate_keys_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            for text in ('{"seed":1,"seed":2}', '{"seed":NaN}', '{"seed":Infinity}'):
                p.write_text(text)
                with self.assertRaises(ValueError):
                    c.read_json(p)
    def test_unknown_plan_key_rejected(self):
        with self.assertRaises(TypeError):
            c.Plan(unknown=True)
    def test_guard_weakening_rejected(self):
        for kw in ({"host_free_gib": 7.3}, {"worker_free_gib": 5.9}, {"gpu_free_gib": 3.9},
                   {"wall_seconds": 1801}, {"max_context_tokens": 2048}, {"seed": True}, {"learning_rate": float("nan")}):
            with self.assertRaises(ValueError):
                c.Plan(**kw)
    def test_separate_rng_purposes(self):
        self.assertNotEqual(c.purpose_seed(1, 1, "k", "rollout"), c.purpose_seed(1, 1, "k", "rewrite"))
    def test_catalog_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rows.json"
            write_catalog(p, self.tasks)
            self.assertEqual(load_catalog(p)[0], self.tasks)
            with self.assertRaises(FileExistsError):
                write_catalog(p, self.tasks)


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.backend = make_backend()
        self.t = demo_catalog()[0]
    def test_public_problem_has_no_gold(self):
        self.assertEqual(set(asdict(self.t.public())), {"key", "text"})
        with self.assertRaises(ValueError):
            self.backend.views.student(self.t)
    def test_teacher_privilege_not_student(self):
        v = self.backend.views
        self.assertEqual(v.student(self.t.public()), v.privileged(self.t)[:len(v.student(self.t.public()))])
        altered = replace(self.t, solution="SECRET_SOURCE_REFERENCE <final>8</final>")
        self.assertEqual(v.student(self.t.public()), v.student(altered.public()))
        self.assertNotEqual(v.privileged(self.t), v.privileged(altered))
    def test_rewriter_gold_free_dto(self):
        v = self.backend.views
        text = v.tokenizer.decode(v.rewriter(self.t.public(), "original"))
        self.assertNotIn(self.t.solution, text)
        with self.assertRaises(ValueError):
            v.rewriter(self.t, "original")
    def test_evaluation_privilege_forbidden(self):
        for split in ("calibration", "retention_eval", "retention_train"):
            with self.assertRaises(ValueError):
                self.backend.views.privileged(replace(self.t, split=split))
    def test_selection_requires_explicit_probe(self):
        t = replace(self.t, split="train_selection")
        with self.assertRaises(ValueError):
            self.backend.views.privileged(t)
        self.backend.views.privileged(t, selection=True)
    def test_eos_and_overflow_rules(self):
        for ids, terminated in (((2, 2), True), ((4,), True), ((2, 4), False)):
            with self.assertRaises(ValueError):
                self.backend.views.response(ids, terminated)
        with self.assertRaises(ValueError):
            self.backend.views.fit((1,) * 1024, 1)
        self.assertEqual(self.backend.views.target("x")[-1], 2)
    def test_control_token_rejected(self):
        with self.assertRaises(ValueError):
            self.backend.views.content("FORBIDDEN_CONTROL")
    def test_native_generation_uses_public_view(self):
        with self.assertRaises(ValueError):
            self.backend.sample(self.t, seed=1)
    def test_causal_future_invariance(self):
        b = self.backend
        prefix = b.views.student(self.t.public())
        x = b.hidden(prefix, (5, 6, 2), grad=False)
        y = b.hidden(prefix, (5, 9, 2), grad=False)
        torch.testing.assert_close(x[:2], y[:2], rtol=0, atol=0)
        self.assertFalse(torch.equal(x[2], y[2]))
    def test_sample_logprob_alignment(self):
        r = self.backend.sample(self.t.public(), seed=101)
        self.backend.audit_rollout(self.t, r)
        self.assertLess(self.backend.sampled_logprob_error(r)[0], 1e-5)
    def test_stale_and_private_prompt_rejected(self):
        r = self.backend.sample(self.t.public(), seed=101)
        with self.assertRaises(ValueError):
            self.backend.audit_rollout(self.t, replace(r, policy_version=3))
        with self.assertRaises(ValueError):
            self.backend.audit_rollout(self.t, replace(r, prompt_ids=(4, 5)))
    def test_teacher_snapshot_restore_after_exception(self):
        b = self.backend
        with torch.no_grad():
            next(iter(b.params.values())).add_(.1)
        changed = tensor_hash(b.snapshot())
        with self.assertRaises(RuntimeError):
            with b.teacher_role(True):
                self.assertEqual(tensor_hash(b.snapshot()), b.initial_digest)
                raise RuntimeError("fixture fault")
        self.assertEqual(tensor_hash(b.snapshot()), changed)
    def test_frozen_evolving_initial_equality_then_difference(self):
        b = self.backend
        r = b.sample(self.t.public(), seed=101)
        a = b.guidance(self.t, r, frozen=True).detach()
        z = b.guidance(self.t, r, frozen=False).detach()
        torch.testing.assert_close(a, z, atol=0, rtol=0)
        with torch.no_grad():
            b.params['model.lora_B'].add_(.5)
        self.assertFalse(torch.equal(b.guidance(self.t, r, frozen=True), b.guidance(self.t, r, frozen=False)))
    def test_heldout_sft_blocked(self):
        with self.assertRaises(ValueError):
            self.backend.supervised(replace(self.t, split="calibration"), (4, 2))
    def test_update_changes_adapter_not_base(self):
        b = self.backend
        base = tensor_hash({n: p for n, p in b.model.named_parameters() if not p.requires_grad})
        initial = tensor_hash(b.snapshot())
        tasks = demo_catalog()
        opt = torch.optim.AdamW(b.params.values(), lr=1e-3)
        outcome = update(b, opt, tasks[:2], [next(t for t in tasks if t.split == "retention_train")], "evolving", b.plan)
        self.assertTrue(outcome["optimizer_step"])
        self.assertNotEqual(tensor_hash(b.snapshot()), initial)
        self.assertEqual(tensor_hash({n: p for n, p in b.model.named_parameters() if not p.requires_grad}), base)
    def test_duplicate_update_task_rejected(self):
        opt = torch.optim.AdamW(self.backend.params.values())
        with self.assertRaises(ValueError):
            update(self.backend, opt, [self.t, self.t], [replace(self.t, key="r", split="retention_train")], "sft", self.backend.plan)
    def test_srcl_empty_acceptance_no_replay_only_training(self):
        b = self.backend
        opt = torch.optim.AdamW(b.params.values())
        tasks = demo_catalog()
        out = update(b, opt, tasks[:2], [next(t for t in tasks if t.split == "retention_train")], "srcl_only", b.plan)
        self.assertEqual(out["status"], "NO_SIGNAL")
        self.assertEqual(b.version, 0)
    def test_scripted_accepted_rewrite_real_ce_update(self):
        # Scripted text is ONLY a mechanics fixture, not an on-policy efficacy result.
        b = self.backend
        tasks = demo_catalog()
        selected = tasks[:2]
        by_key = {t.key: t for t in selected}
        def scripted(problem, *, seed, prefix=None, **kwargs):
            t = by_key[problem.key]
            text = f"<final>{t.answer}</final>" if prefix is not None else t.solution
            r = rollout(t, text)
            return replace(r, prompt_ids=tuple(prefix)) if prefix is not None else r
        optimizer = torch.optim.AdamW(b.params.values(), lr=1e-3)
        with patch.object(b, "sample", side_effect=scripted):
            result = update(b, optimizer, selected, [next(t for t in tasks if t.split == "retention_train")],
                            "srcl_only", b.plan)
        self.assertEqual(result["accepted_rewrites"], 2)
        self.assertGreater(result["weighted_losses"]["concise"], 0)
        self.assertEqual(result["supervised_tokens"]["guidance"], 0)
        self.assertEqual(b.version, 1)
    def test_supervised_output_budget_enforced(self):
        with self.assertRaises(ValueError):
            self.backend.supervised(self.t, (4,) * 64 + (2,))

    def test_gold_free_evaluation_all_enrolled_rows(self):
        tasks = demo_catalog()
        rows = evaluate(self.backend, tasks, self.backend.plan)
        self.assertEqual(len(rows), 8)
        self.assertEqual({r["split"] for r in rows}, {"calibration", "retention_eval"})
    def test_parent_schedule_stable_and_unique(self):
        tasks = demo_catalog()
        for step in range(10):
            a = scheduled(tasks, "train", 2, 101, step)
            z = scheduled(list(reversed(tasks)), "train", 2, 101, step)
            self.assertEqual(a, z)
            self.assertEqual(len({t.parent for t in a}), 2)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tasks, self.plan = demo_catalog(), small_plan()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def run_arm(self, arm, name, updates=2, resume=False):
        b = make_backend(self.plan)
        result = train(b, self.tasks, self.plan, arm, self.root / name,
                       binding(self.plan, self.tasks, arm), max_updates=updates, resume=resume)
        return b, result
    def test_six_arm_controller_paths(self):
        for arm in c.ARMS:
            _, result = self.run_arm(arm, arm)
            self.assertEqual(result["status"], "NO_SIGNAL" if arm == "srcl_only" else "COMPLETE")
            self.assertFalse(result["native_efficacy_claim"])
    def test_clean_resume_bitwise(self):
        a, _ = self.run_arm("evolving", "whole")
        _, first = self.run_arm("evolving", "resumed", updates=1)
        self.assertEqual(first["status"], "PAUSED")
        b, last = self.run_arm("evolving", "resumed", updates=1, resume=True)
        self.assertEqual(tensor_hash(a.snapshot()), tensor_hash(b.snapshot()))
        x = read_run(self.root / "whole")[1]
        y = read_run(self.root / "resumed")[1]
        for name in ("history", "evaluations", "generated_tokens", "processed_tokens", "model_calls", "optimizer_steps"):
            self.assertEqual(x[name], y[name])
        self.assertEqual(last["steps"], 2)
    def test_completed_run_cannot_resume(self):
        self.run_arm("sft", "done")
        with self.assertRaises(ValueError):
            self.run_arm("sft", "done", resume=True)
    def test_checkpoint_hash_tamper(self):
        self.run_arm("sft", "bad", updates=1)
        path = self.root / "bad"
        directory = read_json(path / "LATEST.json")["directory"]
        with (path / directory / "STATE.json").open("a") as f:
            f.write(" ")
        with self.assertRaises(ValueError):
            self.run_arm("sft", "bad", resume=True)
        with self.assertRaises(ValueError):
            read_run(path)
    def test_inflight_quarantine(self):
        self.run_arm("sft", "bad", updates=1)
        (self.root / "bad" / "INFLIGHT.json").write_text("{}")
        with self.assertRaises(ValueError):
            self.run_arm("sft", "bad", resume=True)
    def test_lock_excludes_second_writer(self):
        store = RunStore(self.root / "lock", binding(self.plan, self.tasks, "sft"), self.plan)
        try:
            with self.assertRaises(FileExistsError):
                RunStore(self.root / "lock", binding(self.plan, self.tasks, "sft"), self.plan, resume=True)
        finally:
            store.close()
    def test_binding_drift_blocks_resume(self):
        self.run_arm("sft", "bound", updates=1)
        with self.assertRaises(ValueError):
            self.run_arm("frozen", "bound", resume=True)
    def test_catalog_digest_enforced(self):
        b = make_backend(self.plan)
        wrong = binding(self.plan, self.tasks, "sft")
        wrong["catalog"] = "bad"
        with self.assertRaises(ValueError):
            train(b, self.tasks, self.plan, "sft", self.root / "bad", wrong)
    def test_budget_fault_retains_inflight_and_cost(self):
        p = replace(self.plan, model_call_limit=1)
        b = make_backend(p)
        with self.assertRaises(ValueError):
            train(b, self.tasks, p, "sft", self.root / "budget", binding(p, self.tasks, "sft"))
        self.assertTrue((self.root / "budget" / "INFLIGHT.json").exists())
        failure = read_json(next((self.root / "budget").glob("FAILURE-*.json")))
        self.assertEqual(failure["last_state"]["model_calls"], 1)
    def test_compare_same_step_and_complete_coverage(self):
        self.run_arm("sft", "a")
        self.run_arm("evolving", "b")
        result = compare(self.root / "a", self.root / "b", 2, self.plan)
        self.assertEqual(result["calibration"]["parents"], 4)
        self.assertFalse(result["compute_matched"])
        with self.assertRaises(ValueError):
            compare(self.root / "a", self.root / "b", 99, self.plan)
    def test_offline_cli_end_to_end(self):
        plan = self.root / "plan.json"
        plan.write_text(json.dumps(asdict(self.plan)))
        self.assertEqual(main(["fixture", "--output", str(self.root / "catalog.json")]), 0)
        self.assertEqual(main(["plan", "--plan", str(plan), "--catalog", str(self.root / "catalog.json"),
                               "--output", str(self.root / "validated.json")]), 0)
        self.assertEqual(read_json(self.root / "validated.json")["optimizer_steps"], 0)
    def test_cli_planning_imports_no_torch(self):
        code = "import sys; import experiments.rineya_recursive_self_distillation_v1.__main__; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True)
    def test_native_without_permission_platform_blocks(self):
        from experiments.rineya_recursive_self_distillation_v1.native import launch
        with patch("experiments.rineya_recursive_self_distillation_v1.native.os.name", "nt"):
            with self.assertRaisesRegex(ValueError, "allow-model-load"):
                launch(SimpleNamespace(allow_model_load=False))
    def test_native_disposable_updates_need_training_permission(self):
        from experiments.rineya_recursive_self_distillation_v1.native import launch
        with patch("experiments.rineya_recursive_self_distillation_v1.native.os.name", "nt"):
            with self.assertRaisesRegex(ValueError, "allow-training"):
                launch(SimpleNamespace(allow_model_load=True, mode="qualify", allow_training=False))
    def test_native_unsupported_platform_fails_before_import(self):
        from experiments.rineya_recursive_self_distillation_v1.native import launch
        with patch("experiments.rineya_recursive_self_distillation_v1.native.os.name", "posix"):
            with self.assertRaisesRegex(ValueError, "Windows-only"):
                launch(SimpleNamespace())

    def test_parent_export_requires_review(self):
        from experiments.rineya_recursive_self_distillation_v1.native import export_parent
        b = make_backend(self.plan)
        export_parent(b, {"status": "COMPLETE", "steps": 2}, self.root,
                      {"evidence": "fixture_only"}, {"runtime": "fixture_only"})
        candidate = read_json(self.root / "PARENT_CANDIDATE.json")
        self.assertFalse(candidate["parent_evaluation_reviewed"])
        self.assertEqual(candidate["reviewed_by"], "")
        self.assertEqual(main(["review-parent", "--candidate", str(self.root / "PARENT_CANDIDATE.json"),
                          "--reviewed-by", "test fixture reviewer", "--output", str(self.root / "unapproved.json")]), 2)
        self.assertFalse((self.root / "unapproved.json").exists())
    def test_unfinished_parent_cannot_export(self):
        from experiments.rineya_recursive_self_distillation_v1.native import export_parent
        with self.assertRaises(ValueError):
            export_parent(make_backend(self.plan), {"status": "PAUSED", "steps": 1}, self.root, {}, {})
    def test_bind_hashes_exact_files(self):
        paths = {}
        for name in ("e0-config", "e0-result", "catalog", "exclusions"):
            paths[name] = self.root / (name + ".json")
            paths[name].write_text("{}")
        argv = ["bind", "--parent-stage", "--output", str(self.root / "binding.json")]
        for name, path in paths.items():
            argv += ["--" + name, str(path)]
        self.assertEqual(main(argv), 0)
        document = read_json(self.root / "binding.json")
        self.assertIsNone(document["parent"])
        self.assertEqual(document["catalog"]["sha256"], c.file_hash(paths["catalog"]))


class MetricsTests(unittest.TestCase):
    def rows(self):
        return [{"key": str(i), "parent": str(i), "source": "source", "split": "retention_eval",
                 "budget": 128, "correct": i % 2 == 0, "terminated": True, "tokens": 10} for i in range(4)]
    def test_paired_repair_damage_counts(self):
        a = self.rows()
        b = [dict(r, correct=not r["correct"]) for r in a]
        result = paired_effect(a, b, samples=100)
        self.assertEqual(result["parent_macro_delta"], 0)
        self.assertEqual(result["source_wrong_to_correct"], 2)
        self.assertEqual(result["source_correct_to_wrong"], 2)
    def test_parent_macro_not_row_weighted(self):
        rows = self.rows()[:2]
        rows += [dict(rows[0], key="duplicate-variant")]
        self.assertEqual(summary(rows)["parent_macro_accuracy"], .5)
        self.assertNotEqual(summary(rows)["row_accuracy"], .5)
    def test_missing_pair_rejected(self):
        with self.assertRaises(ValueError):
            paired_effect(self.rows(), self.rows()[:-1])
    def test_duplicate_pair_rejected(self):
        with self.assertRaises(ValueError):
            paired_effect(self.rows(), self.rows() + self.rows()[:1])
    def test_budget_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            paired_effect(self.rows(), [dict(r, budget=256) for r in self.rows()])
    def test_retention_drop_stops(self):
        stopped, _ = retention_stop(self.rows(), [dict(r, correct=False) for r in self.rows()], .02, .25)
        self.assertTrue(stopped)
    def test_truncation_stops(self):
        stopped, _ = retention_stop(self.rows(), [dict(r, terminated=False) for r in self.rows()], .02, .25)
        self.assertTrue(stopped)


if __name__ == "__main__":
    unittest.main()
