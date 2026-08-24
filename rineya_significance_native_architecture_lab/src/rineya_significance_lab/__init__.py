"""Cornerstone-1 significance-native architecture experiments for Rineya.

The package keeps the research mechanism separate from the existing ChatBubble
model and evaluation code.  Importing it never loads Torch, Transformers, a
checkpoint, or CUDA.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
