from __future__ import annotations

from rineya_fwa.arms import ARM_SPECS, arm_spec


def test_arm_table():
    assert set(ARM_SPECS) == {"A", "B", "C", "D"}
    assert arm_spec("c").recurrent is True
    assert arm_spec("B").recurrent is False
    assert arm_spec("D").significance_gate is True
    assert arm_spec("A").install is False
