import math
import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3 import fresh_native as f


def packet():
    raw = [{"case_id": str(i), "raw_prompt": "original " + str(i),
            "raw_prompt_sha256": f.text_sha("original " + str(i))} for i in range(2)]
    tokens = [{**row, "rendered_prompt": "USER:" + row["raw_prompt"],
               "rendered_prompt_sha256": f.text_sha("USER:" + row["raw_prompt"]),
               "input_token_ids": [1, 2], "input_tokens": 2} for row in raw]
    return raw, {"status": "FROZEN_FRESH_PROMPT_ONLY_TOKEN_BINDING", "canonical_source": "source",
                 "model_loads": 0, "rows": tokens}


def test_prompt_only_projection_strict_schema_hash_counts_and_shapes():
    raw, tokens = packet(); rows = f.project_inputs(raw, tokens, 2, "source", "SCIENTIFIC")
    assert len(rows) == 2 and "gold" not in rows[0] and "category" not in rows[0]
    raw[0]["gold"] = "answer"
    with pytest.raises(PermissionError): f.project_inputs(raw, tokens, 2, "source", "SCIENTIFIC")
    del raw[0]["gold"]; tokens["rows"][0]["input_token_ids"] = [151936]
    tokens["rows"][0]["input_tokens"] = 1
    with pytest.raises(RuntimeError): f.project_inputs(raw, tokens, 2, "source", "SCIENTIFIC")
    raw, tokens = packet(); tokens["rows"][0]["rendered_prompt_sha256"] = "0" * 64
    with pytest.raises(RuntimeError): f.project_inputs(raw, tokens, 2, "source", "SCIENTIFIC")
    raw, tokens = packet(); raw[1]["case_id"] = raw[0]["case_id"]
    with pytest.raises(RuntimeError): f.project_inputs(raw, tokens, 2, "source", "SCIENTIFIC")


def test_exact_complete_caps_and_forecast_include_controls_nonforward_reserve():
    counts = f.count_contract()
    assert counts["task_physical_rows"] == 9 * 456 == 4104
    assert counts["model_API_cap"] == 74 * 456 + 67 + 97 * 8 + 9 == 34596
    good = f.full_forecast([.1, .12], [.03, .04], .001, 100.)
    assert good["admitted"] and good["remaining_forward_seconds"] == pytest.approx(33744 * .12 * 1.5)
    assert good["remaining_nonforward_seconds"] == pytest.approx(4104 * (.04 + .001) * 1.5)
    assert not f.full_forecast([1.], [0.], 0., 100.)["admitted"]
    for times, extras, parser, elapsed in (([], [0.], 0., 0.), ([.1], [], 0., 0.), ([math.nan], [0.], 0., 0.), ([.1], [.1], math.inf, 0.)):
        with pytest.raises(RuntimeError): f.full_forecast(times, extras, parser, elapsed)


def test_global_ledger_unique_backend_indexes_counts_cache_partial_and_alias():
    rows = []; ledger = f.GlobalLedger(SimpleNamespace(append=rows.append), f.time.perf_counter(), cap=3)
    ledger.accept({"event": "FORWARD_STARTED", "API_attempt": 1}, "session")
    ledger.accept({"event": "FORWARD_COMPLETED", "API_attempt": 1, "prefix_tokens": 2}, "session")
    mapped = ledger.accept({"event": "PILOT_COMMITTED", "API_attempt": 1}, "session")
    assert mapped["API_attempt"] == 1
    ledger.accept({"event": "FORWARD_STARTED", "full_model_call": 1}, "reference")
    ledger.accept({"event": "FORWARD_FAILED", "full_model_call": 1, "partial_compute": "UNKNOWN"}, "reference")
    ledger.accept({"event": "CACHE_REJECTION_STARTED", "API_attempt": 1}, "cache")
    ledger.accept({"event": "CACHE_REJECTED", "API_attempt": 1, "model_compute_started": False}, "cache")
    assert ledger.counts() == {"model_API_attempts": 3, "completed_forwards": 1, "failed_forwards": 1,
                               "full_forward_attempts": 2, "cache_rejections": 1, "completed_prefix_positions": 2}
    with pytest.raises(RuntimeError, match="duplicate"):
        ledger.accept({"event": "FORWARD_COMPLETED", "API_attempt": 1}, "session")
    with pytest.raises(RuntimeError, match="cap"):
        ledger.accept({"event": "FORWARD_STARTED", "API_attempt": 2}, "session")


class FakeDriver:
    def __init__(self, unequal=False, repeat_error=False):
        self.pilots = 0; self.current = None; self.unequal = unequal; self.repeat_error = repeat_error
        self.generated, self.rows_seen = [], []
    def begin_path(self, row, index, path): self.current = path
    def make_pilot(self, row, started):
        self.pilots += 1
        return SimpleNamespace(probability=.75, logits_sha256="changed" if self.unequal and self.pilots == 2 else "same")
    def generate(self, row, k, pilot, started):
        assert "gold" not in row and "scorer" not in row
        self.generated.append((self.current, k, pilot is not None)); self.rows_seen.append(row["raw_prompt"])
        tokens = [3 if k == 0 else 4, 9]
        if self.repeat_error and self.current == "PRIMARY_REPEAT": tokens = [7, 9]
        return {"token_ids": tokens, "response": str(tokens[0]), "termination": "eos", "K": k,
                "pilot_probability": None if pilot is None else pilot.probability,
                "pilot_logits_sha256": None if pilot is None else pilot.logits_sha256,
                "selected_first_logits_sha256": "k" + str(k), "safety_stopped": False,
                "full_model_calls": 2 + (k if pilot is not None else 0)}


def input_row():
    return {"case_id": "opaque", "cohort": "SCIENTIFIC", "raw_prompt": "original input",
            "prefix_ids_sha256": "0" * 64, "input_token_ids": [1, 2]}


def test_nine_real_paths_independent_pilots_no_probability_for_guard_and_repeat():
    driver, observed = FakeDriver(), []
    result = f.execute_item(input_row(), 0, driver, 0, 1, observed.append)
    assert len(observed) == 9 and set(result) == set(f.PATHS) and driver.pilots == 2
    assert result[f.CONFIDENT]["K"] == 1 and result[f.UNCERTAIN]["K"] == 0
    assert result["RANDOM17"]["K"] == 0 and result["RANDOM29"]["K"] == 1
    assert result[f.GUARD]["certificate"] == "UNRECOGNIZED" and result[f.GUARD]["K"] == 1
    assert f._repeat_signature(result[f.GUARD]) == f._repeat_signature(result["PRIMARY_REPEAT"])
    assert all(prompt == "original input" for prompt in driver.rows_seen)


@pytest.mark.parametrize("kind", ["unequal", "repeat_error"])
def test_independent_pilot_or_actual_repeat_divergence_fails_complete_item(kind):
    driver = FakeDriver(**{kind: True})
    with pytest.raises(RuntimeError, match="pilots|repeat"):
        f.execute_item(input_row(), 0, driver, 0, 1, lambda row: None)


def test_parent_clock_rejects_missing_future_nonfinite_and_expired(monkeypatch):
    monkeypatch.setattr(f.time, "perf_counter", lambda: 200.)
    monkeypatch.delenv(f.PARENT_CLOCK, raising=False)
    with pytest.raises(RuntimeError): f.parent_clock()
    for value in ("nan", "inf", "201", "-20000", "bad"):
        monkeypatch.setenv(f.PARENT_CLOCK, value)
        with pytest.raises(RuntimeError): f.parent_clock()
    monkeypatch.setenv(f.PARENT_CLOCK, "100")
    assert f.parent_clock() == 100.


def test_tokenizer_identity_requires_actual_qualified_class_assets_runtime_and_after_hashes():
    assets = {"checkpoint": "hash"}
    binding = {"checkpoint_files": assets, "tokenizer_class": "Qwen2Tokenizer", "tokenizer_length": 151669}
    runtime = {"python": "3.11.15", "torch": "2.12.1+cu126", "transformers": "5.14.1",
               "tokenizers": "qualified", "qwen_implementation_sha256": "actualQwen"}
    identity = {"qualified_manifest_sha256": "b0b08bb354ce7bc48c760f59a87a7e7d8b8ba432aedddee3e05b55a0f0221131",
                "checkpoint_files": assets, "runtime": {k: runtime[k] for k in ("python", "torch", "transformers", "tokenizers")},
                "qwen_implementation_sha256": "actualQwen", "canonical_source": "source", "interpreter": "python",
                "class": "Qwen2Tokenizer", "vocabulary": 151669}
    manifest = {"tokenizer_identity": identity, "checkpoint_files_after": assets,
                "tokenizer_identity_after": {k: v for k, v in identity.items() if k not in ("class", "vocabulary")}}
    assert f.validate_tokenizer_identity(manifest, binding, runtime, "source", "python") == identity
    identity["class"] = "unqualified"
    with pytest.raises(RuntimeError): f.validate_tokenizer_identity(manifest, binding, runtime, "source", "python")
    identity["class"] = "Qwen2Tokenizer"; manifest["checkpoint_files_after"] = {}
    with pytest.raises(RuntimeError): f.validate_tokenizer_identity(manifest, binding, runtime, "source", "python")
    manifest["checkpoint_files_after"] = assets
    manifest["tokenizer_identity_after"] = {**manifest["tokenizer_identity_after"], "canonical_source": "other"}
    with pytest.raises(RuntimeError): f.validate_tokenizer_identity(manifest, binding, runtime, "source", "python")


def test_actual_max_driver_sequence_nine_cpu_fixture_attempts_eight_complete_and_restore():
    from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.tests.test_opposed_gate_generation import Host, Tokenizer
    from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.opposed_gate_generation import GenerationSession, _hook_snapshot
    rows = []; ledger = f.GlobalLedger(SimpleNamespace(append=rows.append), f.time.perf_counter())
    model = Host(sequences={0: (3,), 1: (4,)})
    session = GenerationSession(model, Tokenizer(), run_id="fixture", source_sha256="a" * 64,
        resident_state_sha256="b" * 64, tokenizer_sha256="c" * 64, api_cap=20,
        guard=lambda: {}, synchronize=lambda: None, enforce_device=False,
        event_sink=lambda e: ledger.accept(e, "session", {"phase": "QUALIFICATION_MAX_DRIVER"}))
    before = _hook_snapshot(model)
    ids = torch.tensor([[1, 2] + [1] * 355])
    proof = f.qualify_max_driver(session, ids, {})
    assert ledger.attempts == 9 and len(ledger.completed) == 8 and len(ledger.failed) == 1
    assert ledger.counts()["completed_prefix_positions"] == 8 * 357
    assert ledger.failed[0]["partial_compute"] == "UNKNOWN" and proof["forced_exception_restoration"]
    assert _hook_snapshot(model) == before and session.trajectory_sink is None


def test_all_worker_text_reads_are_explicit_utf8_and_prompt_export_handles_unicode(tmp_path, monkeypatch):
    tree = ast.parse(inspect.getsource(f))
    reads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "read_text"]
    assert reads and all(any(k.arg == "encoding" and isinstance(k.value, ast.Constant) and k.value.value == "utf-8" for k in n.keywords) for n in reads)
    monkeypatch.setattr(f, "INPUTS", tmp_path)
    admission, hashes = {}, {}
    for key, filename, count in (("scientific_bank", "SCIENTIFIC_PROMPTS.json", 432),
                                 ("preservation_bank", "PRESERVATION_PROMPTS.json", 24)):
        raw = [{"case_id": key + str(i), "raw_prompt": "原文🙂 " + str(i),
                "raw_prompt_sha256": f.text_sha("原文🙂 " + str(i))} for i in range(count)]
        tokens = [{**x, "rendered_prompt": x["raw_prompt"], "rendered_prompt_sha256": x["raw_prompt_sha256"],
                   "input_token_ids": [1, 2], "input_tokens": 2} for x in raw]
        token_key = key.replace("bank", "token_binding"); token_name = filename.replace("PROMPTS", "TOKEN_BINDING")
        for field, name, value in ((key, filename, raw), (token_key, token_name,
            {"status": "FROZEN_FRESH_PROMPT_ONLY_TOKEN_BINDING", "canonical_source": "source", "model_loads": 0, "rows": tokens})):
            path = tmp_path / name; path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            hashes[name] = f.sha(path); admission[field + "_path"] = str(path); admission[field + "_sha256"] = hashes[name]
    manifest = {"status": "FROZEN_FRESH_OPEN_MODEL_EXPORT", "scientific_cases": 432, "preservation_cases": 24,
                "maximum_input_tokens": 350, "maximum_forward_prefix_tokens": 357, "labels_excluded": True,
                "tokenizer_source": "source", "model_loads": 0, "files": hashes}
    path = tmp_path / "MODEL_EXPORT_MANIFEST.json"; path.write_text(json.dumps(manifest), encoding="utf-8")
    admission.update(prompt_export_manifest_path=str(path), prompt_export_manifest_sha256=f.sha(path))
    rows = f.read_inputs(admission, "source")
    assert len(rows) == 456 and rows[0]["raw_prompt"] == "原文🙂 0"


def test_real_anchor_context_guards_compiled_qualification_call_and_reference_scope():
    from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.tests.test_opposed_gate_generation import Host
    tree = ast.parse(inspect.getsource(f)); qualifier = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "qualifier")
    reference = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "reference")
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "one_shot_anchors" for n in ast.walk(reference))
    model = Host()
    def invoke(host, ids, *args):
        with torch.inference_mode():
            return (host(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1).logits, 0.)
    namespace = {"model": model, "one_shot_anchors": f.one_shot_anchors, "invoke_prefix": invoke,
                 "prefix": None, "guard": None, "synchronize": None, "FixedBlock": None}
    module = ast.fix_missing_locations(ast.Module(body=[qualifier], type_ignores=[])); exec(compile(module, "qualification fixture", "exec"), namespace)
    namespace["qualifier"](torch.tensor([[1, 2]]), "INTACT", {})
    class DoubleAnchor(Host):
        def forward(self, input_ids, **kwargs):
            self.model.layers[0](input_ids.float().unsqueeze(-1))
            return super().forward(input_ids, **kwargs)
    namespace["model"] = DoubleAnchor()
    with pytest.raises(RuntimeError, match="one-shot"):
        namespace["qualifier"](torch.tensor([[1, 2]]), "INTACT", {})
    assert not namespace["model"]._forward_hooks and not namespace["model"]._forward_pre_hooks
