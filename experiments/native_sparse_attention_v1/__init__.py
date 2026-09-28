"""NSA-QUAL-001: isolated, opt-in compact-Rineya attention research."""
from .core import NSAConfig, BlockCompressor, sparse_attention
from .qwen3 import Qwen3NSAAttention, convert_qwen3, restore_qwen3

__all__ = ["NSAConfig", "BlockCompressor", "sparse_attention", "Qwen3NSAAttention",
           "convert_qwen3", "restore_qwen3"]
