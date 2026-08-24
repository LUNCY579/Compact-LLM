from __future__ import annotations

import importlib.util
from pathlib import Path


def test_answer_normalization_handles_expected_short_forms() -> None:
    lab = Path(__file__).resolve().parents[1]
    script = lab / "scripts" / "run_halo_capability.py"
    spec = importlib.util.spec_from_file_location("halo_capability", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._normalize("Answer: The 17.") == "17"
    assert module._normalize("-1, 3") == "-1, 3"
    assert module._task_correct("R1", "7 + 9 = 16.", ["16"])
    assert module._task_correct("R3", "Room 23.", ["23"])
    assert module._task_correct("R2", "N2 is at (-1, 2).", ["-1,2"])
