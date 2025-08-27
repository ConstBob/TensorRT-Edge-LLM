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
- ONNX export for LLM and visual models

Example Usage:
    from tensorrt_edgellm import quantize_and_save_model, llm_export, visual_export
    
    # Quantize and save a model
    quantize_and_save_model(
        model_dir="path/to/model",
        output_dir="path/to/output",
        quantization="fp8"
    )
    
    # Export LLM to ONNX (standard model)
    llm_export(
        model_dir="path/to/model",
        output_dir="path/to/output"
    )
    
    # Export EAGLE base model to ONNX (EAGLE3)
    llm_export(
        model_dir="path/to/model",
        output_dir="path/to/output",
        eagle_base=True
    )
    
    # Export EAGLE base model to ONNX (EAGLE2)
    llm_export(
        model_dir="path/to/model",
        output_dir="path/to/output",
        eagle_base=True,
        eagle2=True
    )
    
    # Export EAGLE draft model to ONNX (EAGLE3)
    llm_export(
        model_dir="path/to/model",
        output_dir="path/to/output",
        eagle_draft=True
    )
    
    # Export EAGLE draft model to ONNX (EAGLE2)
    llm_export(
        model_dir="path/to/model",
        output_dir="path/to/output",
        eagle_draft=True,
        eagle2=True
    )
    
    # Export visual model to ONNX
    visual_export(
        model_dir="path/to/model",
        output_dir="path/to/output",
        dtype="fp16"
    )
"""

from .onnx_export.llm_export import llm_export
from .onnx_export.visual_export import visual_export
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
    "llm_export",
    "visual_export",
]
