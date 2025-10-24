"""
Valid precisions for TensorRT Edge-LLM tests

This module contains valid precisions that are used across multiple test modules.
"""

# Valid precision types for different model components
VALID_LLM_PRECISIONS = [
    'fp16', 'fp8', 'int8_sq', 'int4_awq', 'nvfp4', 'int4_gptq'
]
VALID_LM_HEAD_PRECISIONS = ['fp16', 'fp8', 'int4_awq', 'nvfp4']
VALID_VISUAL_PRECISIONS = ['fp16', 'fp8']
