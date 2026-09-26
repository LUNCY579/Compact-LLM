import copy
import json
from pathlib import Path

import pytest

from armr.preregistration import budget_only_config, function_hash

ROOT = Path(__file__).resolve().parents[1]


def configurations():
    base = json.loads((ROOT / "configs/reapplication_screen.json").read_text())
    return base, {**base, "max_prompt_seconds": 900}


def test_only_time_budget_changes_without_mutating_original():
    base, effective = configurations()
    before = copy.deepcopy(base)
    assert budget_only_config(base, effective) == effective
    assert base == before


@pytest.mark.parametrize("key,value", [("threads",8),("use_cache",True),("max_new_tokens",48),
                                      ("seed",18),("candidate_layers",[12]),("dtype","float16")])
def test_other_experimental_changes_rejected(key,value):
    base, effective = configurations()
    effective[key] = value
    with pytest.raises(ValueError,match="only max_prompt_seconds"):
        budget_only_config(base,effective)


@pytest.mark.parametrize("seconds",[90,300,1800])
def test_unregistered_deadline_rejected(seconds):
    base, effective = configurations()
    effective["max_prompt_seconds"] = seconds
    with pytest.raises(ValueError):
        budget_only_config(base,effective)


def test_comparison_function_hash_ignores_other_functions(tmp_path):
    p = tmp_path / "runner.py"
    p.write_text("def comparison():\n    return 1\n",encoding="utf-8")
    first = function_hash(p,"comparison")
    p.write_text("def unrelated():\n    return 2\n\ndef comparison():\n    return 1\n",encoding="utf-8")
    assert function_hash(p,"comparison") == first
