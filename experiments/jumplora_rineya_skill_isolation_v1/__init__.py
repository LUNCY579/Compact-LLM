"""Qualification-first JumpLoRA mechanics for the Rineya skill study.

The package deliberately keeps the qualification path independent of
Transformers/PEFT so the numerical contracts can be tested on CPU before a
model or dataset is loaded.
"""

__version__ = "1.0.0"
