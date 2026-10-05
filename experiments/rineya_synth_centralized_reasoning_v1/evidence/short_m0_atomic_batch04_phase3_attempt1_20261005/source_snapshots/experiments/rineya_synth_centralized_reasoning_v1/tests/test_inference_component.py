"""Standard-library deterministic fake backend controls; never Torch/model/GPU."""
from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import inference_component as component

VOCAB = 151936


class Device:
    def __init__(self, name): self.name, self.type = name, name.split(":")[0]
    def __str__(self): return self.name


class Tensor:
    def __init__(self, *, shape, dtype, device="cpu", requires_grad=False):
        self.shape, self.dtype, self.device, self.requires_grad = tuple(shape), dtype, Device(device), requires_grad
    def data_ptr(self): return id(self)


class Array(Tensor):
    def __init__(self, values, *, dtype="long", device="cpu"):
        super().__init__(shape=(len(values), len(values[0])), dtype=dtype, device=device)
        self.values = deepcopy(values)
    def tolist(self): return deepcopy(self.values)


class Parameter(Tensor):
    def __init__(self, device="cpu", requires_grad=False):
        super().__init__(shape=(2, 2), dtype="bfloat16", device=device, requires_grad=requires_grad)
        self._version, self.grad, self.values = 0, None, [1, 2, 3, 4]


class Scalar:
    def __init__(self, value): self.value = value
    def item(self): return self.value


class SparseRow(Tensor):
    """Exact sparse finite logits, with all 151936 vocabulary entries represented."""
    def __init__(self, token, *, selected=10.0, baseline=0.0, device="cpu", dtype="float32", poison=None):
        super().__init__(shape=(VOCAB,), dtype=dtype, device=device)
        self.token, self.selected, self.baseline, self.poison = token, selected, baseline, poison
    def argmax(self): return Scalar(self.token)
    def __getitem__(self, index):
        if index == self.poison: return Scalar(float("nan"))
        return Scalar(self.selected if index == self.token else self.baseline)
    def finite(self): return self.poison is None and math.isfinite(self.selected) and math.isfinite(self.baseline)


class Logits(Tensor):
    def __init__(self, rows, device="cpu"):
        super().__init__(shape=(1, len(rows), VOCAB), dtype="float32", device=device)
        self.rows = rows
    def __getitem__(self, key):
        batch, position = key
        assert batch == 0
        return self.rows[position]
    def finite(self): return all(row.finite() for row in self.rows)


class FakeTorch:
    _synth_fake_component = True
    long, float32, bfloat16 = "long", "float32", "bfloat16"
    def __init__(self): self.no_grad_depth, self.tensor_fault, self.mask_fault, self.normalization_fault = 0, None, None, None
    def is_tensor(self, value): return isinstance(value, Tensor)
    def tensor(self, values, *, dtype, device):
        result = Array(values, dtype=dtype, device=device)
        if self.tensor_fault: self.tensor_fault(result)
        return result
    def ones_like(self, tensor):
        result = Array([[1] * tensor.shape[1]], dtype=tensor.dtype, device=str(tensor.device))
        if self.mask_fault: self.mask_fault(result)
        return result
    def no_grad(self):
        torch = self
        class NoGrad:
            def __enter__(self): torch.no_grad_depth += 1
            def __exit__(self, *args): torch.no_grad_depth -= 1
        return NoGrad()
    def isfinite(self, tensor): return SimpleNamespace(all=lambda: tensor.finite())
    def log_softmax(self, row, *, dim):
        assert dim == -1
        normalizer = row.selected + math.log(1 + (VOCAB - 1) * math.exp(row.baseline - row.selected))
        result = SparseRow(row.token, selected=row.selected - normalizer, baseline=row.baseline - normalizer,
                           device=str(row.device), dtype=row.dtype, poison=row.poison)
        if self.normalization_fault: self.normalization_fault(result)
        return result


class Model:
    training = False
    def __init__(self, torch, tokens=(9, 10, 151645), prompt_length=2, device="cpu"):
        self.torch, self.tokens, self.prompt_length, self.device = torch, list(tokens), prompt_length, device
        self.parameters = {"base.weight": Parameter(device), "adapter.weight": Parameter(device, requires_grad=True)}
        self.parameters["adapter.weight"].dtype = "float32"
        self.parameters["adapter.weight"].grad = Parameter(device)
        self.calls, self.fault = [], None
    def named_parameters(self, *, remove_duplicate=False): return list(self.parameters.items())
    def __call__(self, **kwargs):
        assert self.torch.no_grad_depth > 0, "actual forward must run inside no_grad"
        assert set(kwargs) == {"input_ids", "attention_mask", "use_cache", "return_dict"}
        assert kwargs["use_cache"] is False and kwargs["return_dict"] is True
        ids, mask = kwargs["input_ids"].tolist()[0], kwargs["attention_mask"].tolist()[0]
        assert mask == [1] * len(ids)
        self.calls.append(list(ids))
        rows = [SparseRow(self.tokens[min(max(0, index + 1 - self.prompt_length), len(self.tokens) - 1)],
                          device=self.device) for index in range(len(ids))]
        logits = Logits(rows, self.device)
        if self.fault: self.fault(self, logits)
        return SimpleNamespace(logits=logits)


class Journal:
    def __init__(self): self.events, self.saved, self.fail_save, self.wrapper_fault = [], [], None, None
    def call(self, operation, function):
        self.events.append(("attempt", operation))
        try:
            if self.wrapper_fault: return self.wrapper_fault(function)
            output = function()
        except BaseException:
            self.events.append(("failed", operation))
            raise
        self.events.append(("returned", operation))
        return output
    def preserve(self, name, value):
        if self.fail_save: self.fail_save(name, value)
        self.saved.append((name, deepcopy(value)))


class Decoder:
    def __init__(self): self.calls, self.fault = [], None
    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        self.calls.append(list(ids))
        if self.fault: return self.fault(ids)
        return "".join({9: '{"answer":', 10: '[0]}', 151644: 'x'}.get(i, "") for i in ids)


def row(index=0, ids=None):
    ids = [5, 6] if ids is None else list(ids)
    return {"row_id": component.digest(["source", index]), "prompt_sha256": component.digest(["prompt", index]),
            "prompt_ids": ids, "prompt_tokens": len(ids), "attention_mask": [1] * len(ids)}


class ComponentTests(unittest.TestCase):
    def fixture(self, *, tokens=(9, 10, 151645), prompt_length=2, device="cpu", **limits):
        torch, journal, decoder = FakeTorch(), Journal(), Decoder()
        model = Model(torch, tokens, prompt_length, device)
        guards = []
        engine = component.InferenceComponent(torch, model, decoder, journal.call, guards.append, journal.preserve,
                    max_attempted_forwards=limits.pop("max_attempted_forwards", 200),
                    max_input_positions=limits.pop("max_input_positions", 102400), device=device, **limits)
        return engine, torch, model, journal, decoder, guards

    def collect(self, engine, rows=None):
        return engine.collect([row()] if rows is None else rows, source_packet_sha256=component.digest("packet"))

    def test_framework_free_import_and_no_scorer_or_launch_imports(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import inference_component; "
                "assert not any(x in sys.modules for x in ['torch','transformers','cpu_tokenizer','rs_verifier','native_worker']); "
                "print('FRAMEWORK_FREE_COMPONENT')")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(PACKAGE)],
                                capture_output=True, text=True, timeout=30, check=True)
        self.assertEqual("FRAMEWORK_FREE_COMPONENT", result.stdout.strip())

    def test_exact_prefix_generated_only_decode_raw_likelihood_and_hashes(self):
        engine, torch, model, journal, decoder, guards = self.fixture()
        source = [row()]
        before = deepcopy(source)
        result = self.collect(engine, source)
        self.assertEqual(before, source)
        self.assertEqual([[5, 6], [5, 6, 9], [5, 6, 9, 10]], model.calls)
        self.assertEqual([[9, 10, 151645]], decoder.calls)
        output = result["rows"][0]
        self.assertEqual('{"answer":[0]}', output["decoded_response"])
        self.assertEqual([9, 10, 151645], output["raw_generated_ids"])
        self.assertEqual("eos", output["finish_reason"])
        self.assertFalse(output["truncated"])
        self.assertEqual([2, 3, 4], [step["prefix_length"] for step in output["trace"]])
        self.assertTrue(all(math.isfinite(x) and x < 0 for x in output["selected_logprobs"]))
        self.assertEqual(component.digest([5, 6]), output["prompt_ids_sha256"])
        self.assertEqual(component.digest(source[0]), output["input_row_sha256"])
        self.assertEqual("CPU_FAKE_COMPONENT", output["scope"])
        self.assertFalse(output["native_launch_authorized"])
        self.assertFalse(output["native_qualification_established"])
        self.assertEqual(3, engine.costs()["attempted_forwards"])
        self.assertEqual({"attempted": 9, "returned": 9, "failed": 0, "pending": 0}, engine.costs()["input_positions"])
        self.assertEqual(0, torch.no_grad_depth)
        self.assertEqual(6, len(guards))
        self.assertEqual("inference_row_0001", journal.saved[0][0])

    def test_both_exact_eos_ids_stop_and_other_special_does_not(self):
        for eos in (151645, 151643):
            engine, _, model, _, _, _ = self.fixture(tokens=(151644, eos, 10))
            output = self.collect(engine)["rows"][0]
            self.assertEqual([151644, eos], output["raw_generated_ids"])
            self.assertEqual(eos, output["eos_token_id"])
            self.assertEqual(2, len(model.calls))

    def test_default_32_cap_marks_truncation_and_exact_context_reservation(self):
        engine, _, model, _, _, _ = self.fixture(tokens=(9,), prompt_length=480)
        output = self.collect(engine, [row(ids=[5] * 480)])["rows"][0]
        self.assertEqual(32, len(output["raw_generated_ids"]))
        self.assertEqual(511, len(model.calls[-1]))
        self.assertEqual("generation_cap", output["finish_reason"])
        self.assertTrue(output["truncated"])
        self.assertFalse(output["prompt_truncated"])

    def test_overlong_last_row_blocks_whole_packet_without_forward_or_dropping(self):
        engine, _, model, journal, _, _ = self.fixture()
        source = [row(), row(1, [5] * 481)]
        with self.assertRaisesRegex(ValueError, "context window"): self.collect(engine, source)
        self.assertFalse(model.calls)
        self.assertFalse(journal.saved)
        self.assertEqual(481, len(source[-1]["prompt_ids"]))

    def test_leaky_fields_masks_and_bad_ids_rejected_before_any_row(self):
        changes = (lambda x: x.update(target="answer"), lambda x: x.update(labels=[-100]),
                   lambda x: x.update(oracle_state={}), lambda x: x.update(stratum="solved"),
                   lambda x: x.update(prompt_ids=[True, 6]), lambda x: x.update(prompt_ids=[5, -1]),
                   lambda x: x.update(prompt_ids=[5, VOCAB]), lambda x: x.update(prompt_ids=[5, 151643]),
                   lambda x: x.update(attention_mask=[1, 0]), lambda x: x.update(attention_mask=[True, 1]),
                   lambda x: x.update(prompt_tokens=True), lambda x: x.update(row_id="unhashed"))
        for change in changes:
            engine, _, model, _, _, _ = self.fixture()
            altered = row(1)
            change(altered)
            with self.subTest(change=change), self.assertRaises(ValueError): self.collect(engine, [row(), altered])
            self.assertFalse(model.calls)

    def test_duplicate_rows_bad_source_hash_and_untyped_packet_rejected(self):
        for source in ([row(), row()], (), [], None):
            engine, _, model, _, _, _ = self.fixture()
            with self.subTest(source=source), self.assertRaises(ValueError):
                engine.collect(source, source_packet_sha256=component.digest("packet"))
            self.assertFalse(model.calls)
        engine, *_ = self.fixture()
        with self.assertRaises(ValueError): engine.collect([row()], source_packet_sha256="bad")

    def test_exact_tensor_and_mask_validated_before_supplied_call(self):
        for kind in ("input", "mask"):
            engine, torch, model, journal, _, _ = self.fixture()
            if kind == "input": torch.tensor_fault = lambda x: x.values[0].reverse()
            else: torch.mask_fault = lambda x: x.values[0].__setitem__(0, 0)
            with self.assertRaisesRegex(ValueError, "input tensor|attention mask"): self.collect(engine)
            self.assertFalse(model.calls)
            self.assertFalse(journal.events)
            self.assertEqual("COLLECTION_FAILED", journal.saved[-1][1]["status"])

    def test_full_logits_reject_poison_away_from_selected_last_row_dtype_shape_device_and_grad(self):
        faults = (lambda m, x: setattr(x.rows[0], "poison", 1234),
                  lambda m, x: setattr(x, "dtype", "bfloat16"),
                  lambda m, x: setattr(x, "shape", (1, 1, VOCAB)),
                  lambda m, x: setattr(x, "device", Device("cuda:0")),
                  lambda m, x: setattr(x, "requires_grad", True))
        for fault in faults:
            engine, _, model, journal, _, _ = self.fixture()
            model.fault = fault
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, "full batch-one FP32 finite"): self.collect(engine)
            self.assertEqual(1, engine.costs()["operations"]["inference_forward"]["returned"])
            self.assertEqual([], journal.saved[-1][1]["raw_generated_ids"])

    def test_generated_argmax_and_log_softmax_validation(self):
        for token in (True, -1, VOCAB):
            engine, _, model, _, _, _ = self.fixture(tokens=(token,))
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "argmax token"): self.collect(engine)
        for value in (float("nan"), float("inf"), .1):
            engine, torch, _, _, _, _ = self.fixture()
            torch.normalization_fault = lambda x: setattr(x, "selected", value)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "log-softmax|log probability"): self.collect(engine)

    def test_failure_accounting_via_supplied_callable_preserves_partial_generation(self):
        engine, _, model, journal, _, _ = self.fixture()
        def fault(model, logits):
            if len(model.calls) == 2: raise RuntimeError("simulated forward failure")
        model.fault = fault
        with self.assertRaisesRegex(RuntimeError, "forward failure"): self.collect(engine, [row(), row(1)])
        self.assertEqual([("attempt", "inference_forward"), ("returned", "inference_forward"),
                          ("attempt", "inference_forward"), ("failed", "inference_forward")], journal.events)
        self.assertEqual({"attempted": 2, "returned": 1, "failed": 1, "pending": 0},
                         engine.costs()["operations"]["inference_forward"])
        self.assertEqual({"attempted": 5, "returned": 2, "failed": 3, "pending": 0}, engine.costs()["input_positions"])
        self.assertEqual([9], journal.saved[-1][1]["raw_generated_ids"])
        self.assertEqual("error", journal.saved[-1][1]["finish_reason"])
        self.assertEqual(2, len(model.calls))

    def test_wrapper_failure_after_forward_does_not_erase_actual_return(self):
        engine, _, model, journal, _, _ = self.fixture()
        def fail_after(function):
            function()
            raise RuntimeError("journal post-forward failure")
        journal.wrapper_fault = fail_after
        with self.assertRaisesRegex(RuntimeError, "post-forward"): self.collect(engine)
        self.assertEqual(1, len(model.calls))
        self.assertEqual({"attempted": 1, "returned": 1, "failed": 0, "pending": 0},
                         engine.costs()["operations"]["inference_forward"])

    def test_budget_counts_failed_attempts_and_refuses_next_forward(self):
        for limits, positions in (({"max_attempted_forwards": 1}, 2), ({"max_input_positions": 4}, 2)):
            engine, _, model, journal, _, _ = self.fixture(**limits)
            with self.subTest(limits=limits), self.assertRaisesRegex(ValueError, "budget exhausted"): self.collect(engine)
            self.assertEqual(1, len(model.calls))
            self.assertEqual(positions, engine.costs()["input_positions"]["attempted"])
            self.assertEqual([9], journal.saved[-1][1]["raw_generated_ids"])

    def test_preservation_happens_before_next_row_and_sink_failure_stops_it(self):
        engine, _, model, journal, _, _ = self.fixture()
        observed = []
        journal.fail_save = lambda name, value: observed.append(len(model.calls))
        result = self.collect(engine, [row(), row(1)])
        self.assertEqual([3, 6], observed)
        self.assertEqual(2, len(result["rows"]))
        engine, _, model, journal, _, _ = self.fixture()
        def fail(name, value): raise OSError("durable sink unavailable")
        journal.fail_save = fail
        with self.assertRaisesRegex(OSError, "sink unavailable"): self.collect(engine, [row(), row(1)])
        self.assertEqual(3, len(model.calls))
        self.assertEqual([9, 10, 151645], engine.last_receipt["raw_generated_ids"])
        journal.fail_save = None
        with self.assertRaisesRegex(ValueError, "halted"): self.collect(engine, [row(1)])
        self.assertEqual(3, len(model.calls))

    def test_sink_mutation_halts_component_before_next_row(self):
        engine, _, model, journal, _, _ = self.fixture()
        journal.fail_save = lambda name, value: setattr(model.parameters["base.weight"], "_version", 1)
        with self.assertRaisesRegex(ValueError, "changed"): self.collect(engine, [row(), row(1)])
        self.assertEqual(3, len(model.calls))
        self.assertTrue(engine.halted)
        self.assertEqual(1, len(journal.saved))

    def test_model_and_existing_adapter_gradients_remain_unchanged_without_optimizer(self):
        engine, _, model, _, _, _ = self.fixture()
        original = {name: (p, p.data_ptr(), p.dtype, p.requires_grad, p._version, p.grad, deepcopy(p.values),
                          None if p.grad is None else deepcopy(p.grad.values)) for name, p in model.parameters.items()}
        self.collect(engine)
        engine.qualify(row(1), source_packet_sha256=component.digest("packet"))
        for name, p in model.parameters.items():
            before = original[name]
            self.assertIs(before[0], p)
            self.assertEqual(before[1:5], (p.data_ptr(), p.dtype, p.requires_grad, p._version))
            self.assertIs(before[5], p.grad)
            self.assertEqual(before[6], p.values)
            if p.grad is not None: self.assertEqual(before[7], p.grad.values)
        self.assertFalse(model.training)

    def test_parameter_or_gradient_mutation_and_registration_drift_stop_next_row(self):
        mutations = (lambda m: setattr(m.parameters["base.weight"], "_version", 1),
                     lambda m: setattr(m.parameters["base.weight"], "requires_grad", True),
                     lambda m: setattr(m.parameters["base.weight"], "dtype", "float32"),
                     lambda m: setattr(m.parameters["adapter.weight"].grad, "_version", 1),
                     lambda m: setattr(m.parameters["base.weight"], "grad", Parameter()),
                     lambda m: m.parameters.update(extra=Parameter()),
                     lambda m: m.parameters.update({"base.weight": Parameter()}))
        for mutation in mutations:
            engine, _, model, journal, _, _ = self.fixture()
            model.fault = lambda m, x: mutation(m)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "changed"): self.collect(engine, [row(), row(1)])
            self.assertEqual(1, len(model.calls))
            self.assertEqual("COLLECTION_FAILED", journal.saved[-1][1]["status"])

    def test_invalid_decoder_response_is_preserved_as_failure_without_next_row(self):
        engine, _, model, journal, decoder, _ = self.fixture()
        decoder.fault = lambda ids: {"answer": [0]}
        with self.assertRaisesRegex(ValueError, "decoded response"): self.collect(engine, [row(), row(1)])
        self.assertEqual(3, len(model.calls))
        self.assertEqual([9, 10, 151645], journal.saved[-1][1]["raw_generated_ids"])

    def test_numerical_qualification_two_exact_repeats_and_generated_only_bulk(self):
        engine, _, model, journal, decoder, _ = self.fixture()
        result = engine.qualify(row(), source_packet_sha256=component.digest("packet"))
        expected = [[5, 6], [5, 6, 9], [5, 6, 9, 10]]
        self.assertEqual(expected + expected + [[5, 6, 9, 10]], model.calls)
        self.assertFalse(decoder.calls)
        self.assertTrue(result["repeat_exact"])
        self.assertEqual([0, 0, 0], result["selected_logprob_errors"])
        self.assertEqual(.03125, result["selected_logprob_max_error_limit"])
        self.assertEqual(.01, result["selected_logprob_mean_error_limit"])
        self.assertEqual(7, result["costs"]["attempted_forwards"])
        self.assertFalse(result["collection_scope_qualified"])
        self.assertFalse(result["native_qualification_established"])
        self.assertEqual("COMPONENT_NUMERICAL_CONTROL_PASSED", journal.saved[-1][1]["status"])
        self.assertEqual(7, len(journal.saved))
        self.assertEqual("inference_q0001_r0_t00", journal.saved[0][0])
        self.assertEqual(9, journal.saved[0][1]["token"])

    def test_repeat_failure_preserves_both_raw_traces(self):
        engine, _, model, journal, _, _ = self.fixture()
        def fault(model, logits):
            if len(model.calls) == 4: logits.rows[-1].selected = 11.0
        model.fault = fault
        with self.assertRaisesRegex(ValueError, "repeat greedy"): engine.qualify(row(), source_packet_sha256=component.digest("packet"))
        failure = journal.saved[-1][1]
        self.assertEqual(2, len(failure["traces"]))
        self.assertEqual(3, len(failure["traces"][1]))
        self.assertFalse(failure["repeat_exact"])

    def test_bulk_argmax_and_selected_probability_gate_failures_preserve_errors(self):
        for kind in ("argmax", "probability"):
            engine, _, model, journal, _, _ = self.fixture()
            def fault(model, logits):
                if len(model.calls) == 7:
                    if kind == "argmax": logits.rows[1].token = 8
                    else: logits.rows[1].selected = 9.0
            model.fault = fault
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "prefix/bulk"):
                engine.qualify(row(), source_packet_sha256=component.digest("packet"))
            failure = journal.saved[-1][1]
            self.assertGreater(failure["selected_logprob_max_error"], .03125)
            self.assertEqual([0] if kind == "argmax" else [], failure["argmax_differences"])
            self.assertEqual("COMPONENT_NUMERICAL_CONTROL_FAILED", failure["status"])

    def test_cuda_injection_remains_unqualified_and_does_not_touch_cuda_api(self):
        engine, _, _, _, _, _ = self.fixture(device="cuda:0")
        result = self.collect(engine)
        self.assertEqual("UNQUALIFIED_NATIVE_IMPLEMENTATION", result["scope"])
        self.assertFalse(result["native_launch_authorized"])
        self.assertFalse(result["native_qualification_established"])

    def test_invalid_or_expanded_limits_rejected(self):
        for limits in ({"context_cap": 513}, {"generation_cap": 33}, {"generation_cap": True},
                       {"max_attempted_forwards": 0}, {"max_input_positions": True}, {"vocabulary_size": 100}):
            with self.subTest(limits=limits), self.assertRaises(ValueError): self.fixture(**limits)


if __name__ == "__main__": unittest.main()
