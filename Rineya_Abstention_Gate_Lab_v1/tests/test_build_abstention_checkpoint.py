from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

LAB = Path(__file__).resolve().parents[1]
SCRIPT = LAB / "scripts" / "build_abstention_checkpoint.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_abstention_checkpoint", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_custom_model_source_compiles_and_contains_no_generation_bias() -> None:
    builder = load_builder()
    compile(builder.MODELING_ABSTAIN, "modeling_abstain.py", "exec")
    assert "Qwen3AbstainForCausalLM" in builder.MODELING_ABSTAIN
    assert "sequence_bias" not in builder.MODELING_ABSTAIN


def test_configure_abstention_registers_auto_map_and_knobs() -> None:
    builder = load_builder()
    config = SimpleNamespace()
    builder.configure_abstention(config, 51378, 5.156249985098839)
    assert config.architectures == ["Qwen3AbstainForCausalLM"]
    assert config.auto_map == {
        "AutoModelForCausalLM": "modeling_abstain.Qwen3AbstainForCausalLM"
    }
    assert config.abstain_token_id == 51378
    assert config.abstain_logit_bias == 5.156249985098839
