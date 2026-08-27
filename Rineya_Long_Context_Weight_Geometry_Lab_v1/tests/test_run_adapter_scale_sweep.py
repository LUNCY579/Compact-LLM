from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_adapter_scale_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_adapter_scale_sweep", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_scale_validation_is_finite_unique_and_ordered() -> None:
    assert MODULE.validate_scales(["1.25", "-0.25", "0"]) == (
        1.25,
        -0.25,
        0.0,
    )

    with pytest.raises(ValueError, match="Duplicate adapter scale"):
        MODULE.validate_scales(["0.50", ".5"])
    with pytest.raises(ValueError, match="finite"):
        MODULE.validate_scales(["nan"])
    with pytest.raises(ValueError, match="At least one"):
        MODULE.validate_scales([])


def test_candidate_paths_are_stable_and_existing_files_are_rejected(
    tmp_path: Path,
) -> None:
    plans = MODULE.plan_candidates(
        tmp_path,
        "nimbus_step200",
        MODULE.validate_scales(["0.8125", "-0.25"]),
    )

    assert [plan.candidate_id for plan in plans] == [
        "nimbus_step200_scale0p8125",
        "nimbus_step200_scaleneg0p25",
    ]
    assert plans[0].manifest_path.name.endswith(".jsonl.manifest.json")

    plans[0].manifest_path.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        MODULE.refuse_existing_outputs(plans)


def test_exclusive_jsonl_writer_preserves_exact_iteration_order(tmp_path: Path) -> None:
    output = tmp_path / "candidate.jsonl"
    rows = [
        {"id": "third", "prediction": "3"},
        {"id": "first", "prediction": "1"},
        {"id": "second", "prediction": "2"},
    ]

    assert MODULE.write_jsonl_exclusive(output, iter(rows)) == 3
    written = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in written] == ["third", "first", "second"]
    assert not MODULE.temporary_path(output).exists()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        MODULE.write_jsonl_exclusive(output, rows)


def test_every_active_adapter_on_every_lora_layer_is_scaled() -> None:
    class FakeLoraLayer:
        def __init__(self, active: list[str]) -> None:
            self.active_adapters = active
            self.calls: list[tuple[str, float]] = []

        def set_scale(self, adapter_name: str, scale: float) -> None:
            self.calls.append((adapter_name, scale))

    class FakeModel:
        def __init__(self) -> None:
            self.first = FakeLoraLayer(["default"])
            self.second = FakeLoraLayer(["default", "aux"])

        def named_modules(self):
            yield "not_lora", object()
            yield "layers.0.q_proj", self.first
            yield "layers.1.down_proj", self.second

    model = FakeModel()
    bindings = MODULE.discover_lora_layers(model, FakeLoraLayer)

    assert len(bindings) == 2
    assert MODULE.set_all_lora_scales(bindings, -0.25) == 3
    assert model.first.calls == [("default", -0.25)]
    assert model.second.calls == [("default", -0.25), ("aux", -0.25)]


def test_preparation_preserves_order_disables_thinking_and_forbids_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thinking_values: list[bool] = []

    def render_chat(tokenizer, row, enable_thinking=None):
        del tokenizer
        thinking_values.append(enable_thinking)
        return str(row["tokens"])

    class FakeTensor:
        def __init__(self, tokens: int) -> None:
            self.shape = (1, tokens)

    class FakeTokenizer:
        def __call__(self, rendered: str, **kwargs):
            assert kwargs == {"return_tensors": "pt", "add_special_tokens": False}
            return {"input_ids": FakeTensor(int(rendered))}

    monkeypatch.setattr(MODULE.CANONICAL, "render_chat", render_chat)
    rows = [{"id": "z", "tokens": 7}, {"id": "a", "tokens": 9}]
    prepared = MODULE.prepare_rows(rows, FakeTokenizer())

    assert [row.row_id for row in prepared] == ["z", "a"]
    assert thinking_values == [False, False]

    overflow = MODULE.MAX_CONTEXT - MODULE.MAX_NEW_TOKENS + 1
    with pytest.raises(RuntimeError, match="truncation is forbidden"):
        MODULE.prepare_rows([{"id": "too-long", "tokens": overflow}], FakeTokenizer())


def test_manifest_records_frozen_protocol_and_order_digest(tmp_path: Path) -> None:
    input_path = tmp_path / "prompts.jsonl"
    input_path.write_text('{"id":"b"}\n{"id":"a"}\n', encoding="utf-8")
    plan = MODULE.plan_candidates(tmp_path, "candidate", [0.75])[0]
    prepared = (
        MODULE.PreparedRow("b", 2, {}),
        MODULE.PreparedRow("a", 3, {}),
    )

    manifest = MODULE.candidate_manifest(
        plan=plan,
        all_scales=[0.75, 1.25],
        candidate_index=1,
        model={"kind": "test"},
        adapter={"kind": "test"},
        input_path=input_path,
        prepared_rows=prepared,
        output_sha256="f" * 64,
        lora_layer_count=196,
        scale_set_calls=196,
    )

    assert manifest["protocol"] == {
        "id": MODULE.PROTOCOL_ID,
        "torch_dtype": "bfloat16",
        "seed": 17,
        "decoding": "greedy",
        "do_sample": False,
        "thinking": False,
        "max_new_tokens": 24,
        "max_context": 4096,
        "truncation": "forbidden",
    }
    assert manifest["input"]["ordered_ids_sha256"] == MODULE.ordered_ids_sha256(
        ["b", "a"]
    )
    assert manifest["single_model_and_adapter_load"] is True
