"""
TensorRT Edge-LLM - A Python package for quantizing and exporting LLMs for edge deployment.

This package provides utilities for quantizing large language models using NVIDIA ModelOpt
and preparing them for ONNX export and edge deployment. It supports various quantization
schemes including FP8, INT4 AWQ, and NVFP4 for efficient inference on edge devices.

Key Features:
- LLM quantization with calibration support
- Multiple quantization schemes (FP8, INT4 AWQ, NVFP4)
- Automatic model type detection
- HuggingFace model compatibility
- Quantization configuration management

Example Usage:
    from tensorrt_edgellm import quantize_and_save_model
    
    # Quantize and save a model
    quantize_and_save_model(
        torch_dir="path/to/model",
        output_dir="path/to/output",
        quantization="fp8"
    )
"""

from .quantization.llm_quantization import quantize_and_save_model
from .quantization.quantization_utils import quantize_model

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

__author__ = "NVIDIA"
__email__ = "TBD@nvidia.com"

__all__ = [
    "quantize_and_save_model",
    "quantize_model",
]
