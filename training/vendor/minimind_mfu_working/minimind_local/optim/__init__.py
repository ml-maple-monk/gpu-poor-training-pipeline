"""Optimizer implementations and candidates."""

from .hybrid import HybridOptimizer
from .muon8bit import Muon8Bit, dequantize_blockwise_int8, quantize_blockwise_int8, zeropower_via_newtonschulz5

__all__ = [
    "HybridOptimizer",
    "Muon8Bit",
    "dequantize_blockwise_int8",
    "quantize_blockwise_int8",
    "zeropower_via_newtonschulz5",
]
