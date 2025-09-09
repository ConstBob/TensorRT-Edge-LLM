"""
Test configuration classes for TensorRT Edge-LLM.

Provides configuration classes for LLM and VLM pipeline tests with parameter parsing
and path generation for ONNX models and TensorRT engines.
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from valid_precisions import VALID_LLM_PRECISIONS, VALID_LM_HEAD_PRECISIONS


@dataclass
class BaseRuntimeTestConfig:
    """Base configuration class for LLM and VLM tests."""
    param_str: str
    model_name: str
    precision: str
    onnx_dir: str
    engine_dir: str
    lm_head_precision: str = "fp16"
    max_batch_size: int = 1
    max_input_len: int = 2048
    max_seq_len: int = 4096
    output_seq_len: int = 128
    test_case_file: str = "llm_basic"

    def get_model_id(self) -> str:
        """Generate a unique model identifier combining model name, precision, and sequence length."""
        return f"{self.model_name}-{self.precision}-{self.lm_head_precision}-{self.max_seq_len}"

    def get_llm_engine_dir(self) -> str:
        """Get the directory path for LLM TensorRT engines."""
        return os.path.join(self.engine_dir, self.get_model_id(), "llm")

    def get_llm_onnx_dir(self) -> str:
        """Get the directory path for LLM ONNX model files."""
        return os.path.join(self.onnx_dir, self.get_model_id(), "llm")

    def get_test_case_file(self) -> str:
        """Get the test case file name."""
        return "tests/test_cases/" + self.test_case_file + ".json"

    def get_output_json_file(self) -> str:
        """Get the output JSON file name."""
        return os.path.join(self.engine_dir, self.get_model_id(),
                            self.param_str + ".json")

    @classmethod
    def parse_name_and_precision(cls,
                                 param_str: str) -> Tuple[str, str, List[str]]:
        """
        Extract model name, precision, and remaining parameters from a test parameter string.
        
        Example:
            >>> BaseRuntimeTestConfig.parse_name_and_precision("Qwen2.5-0.5B-fp16-mxbs1-mxil2048")
            ('Qwen2.5-0.5B', 'fp16', ['mxbs1', 'mxil2048'])
        """
        parts = param_str.split('-')

        precision: Optional[str] = None
        lm_head_precision: Optional[str] = "fp16"
        model_parts: List[str] = []
        remaining: List[str] = []

        # Find the precision part and split the string accordingly
        for i, part in enumerate(parts):
            if part in VALID_LLM_PRECISIONS:
                precision = part
                model_parts = parts[:i]
                if i + 1 < len(parts) and parts[i + 1].startswith('lm'):
                    lm_head_precision = parts[i + 1][2:]
                    assert lm_head_precision in VALID_LM_HEAD_PRECISIONS, f"Invalid LM head precision: {lm_head_precision}. Valid precisions: {VALID_LM_HEAD_PRECISIONS}"
                    remaining = parts[i + 2:]
                else:
                    remaining = parts[i + 1:]
                break

        if not precision:
            raise ValueError(
                f"No valid precision found in parameter string: {param_str}. "
                f"Valid precisions: {VALID_LLM_PRECISIONS}")

        model_name = '-'.join(model_parts)
        return model_name, precision, lm_head_precision, remaining

    def parse_common_params(self, remaining_parts: List[str]) -> None:
        """Parse and apply common parameters from the remaining parts of the test string."""
        for part in remaining_parts:
            if part.startswith('mxbs'):
                self.max_batch_size = int(part[4:])
            elif part.startswith('mxil'):
                self.max_input_len = int(part[4:])
            elif part.startswith('mxsl'):
                self.max_seq_len = int(part[4:])
            elif part.startswith('osl'):
                self.output_seq_len = int(part[3:])
            else:
                self.test_case_file = part


@dataclass
class LLMRuntimeTestConfig(BaseRuntimeTestConfig):
    """Configuration class for Large Language Model (LLM) tests."""
    type: str = "llm"

    @classmethod
    def from_test_string(cls, param_str: str, onnx_dir: str,
                         engine_dir: str) -> 'LLMRuntimeTestConfig':
        """
        Create an LLMRuntimeTestConfig instance from a test parameter string.
        
        Example:
            >>> config = LLMRuntimeTestConfig.from_test_string(
            ...     "Qwen2.5-0.5B-fp16-mxbs1-mxil2048", "/onnx", "/engines"
            ... )
            >>> config.model_name, config.precision, config.max_input_len
            ('Qwen2.5-0.5B', 'fp16', 2048)
        """
        model_name, precision, lm_head_precision, remaining = cls.parse_name_and_precision(
            param_str)
        config = cls(param_str=param_str,
                     model_name=model_name,
                     precision=precision,
                     lm_head_precision=lm_head_precision,
                     onnx_dir=onnx_dir,
                     engine_dir=engine_dir)
        config.parse_common_params(remaining)
        return config


@dataclass
class VLMRuntimeTestConfig(BaseRuntimeTestConfig):
    """Configuration class for Vision-Language Model (VLM) tests."""
    model_type: str = ""
    visual_precision: str = "fp16"
    min_image_tokens: int = 128
    max_image_tokens: int = 2048
    text_token_length: int = 1024
    image_token_length: int = 1024
    type: str = "vlm"

    @classmethod
    def from_test_string(cls, param_str: str, onnx_dir: str,
                         engine_dir: str) -> 'VLMRuntimeTestConfig':
        """
        Create a VLMRuntimeTestConfig instance from a test parameter string.
        
        Example:
            >>> config = VLMRuntimeTestConfig.from_test_string(
            ...     "Qwen2.5-VL-3B-fp16-mxbs1-mxil2048", "/onnx", "/engines"
            ... )
            >>> config.model_type, config.visual_precision
            ('qwen2_5_vl', 'fp16')
        """
        model_name, precision, lm_head_precision, remaining = cls.parse_name_and_precision(
            param_str)
        config = cls(param_str=param_str,
                     model_name=model_name,
                     precision=precision,
                     lm_head_precision=lm_head_precision,
                     onnx_dir=onnx_dir,
                     engine_dir=engine_dir)

        # Auto-detect VLM model type based on model name
        if 'qwen2-vl' in model_name.lower():
            config.model_type = 'qwen2_vl'
        elif 'qwen2.5-vl' in model_name.lower():
            config.model_type = 'qwen2_5_vl'
        elif 'internvl3' in model_name.lower():
            config.model_type = 'internvl3'
        else:
            raise ValueError(
                f"Unsupported VLM model name: {model_name}. "
                f"Please add support for this model type in runtime_test_config.py"
            )

        # Parse common parameters first
        config.parse_common_params(remaining)

        # Parse VLM-specific parameters
        for part in remaining:
            if part.startswith('mnit'):
                config.min_image_tokens = int(part[4:])
            elif part.startswith('mxit'):
                config.max_image_tokens = int(part[4:])
            elif part.startswith('ttl'):
                config.text_token_length = int(part[3:])
            elif part.startswith('itl'):
                config.image_token_length = int(part[3:])
            elif part == 'vitfp8':
                config.visual_precision = "fp8"

        return config

    def get_visual_onnx_dir(self) -> str:
        """Get the directory path for visual ONNX model files."""
        return os.path.join(self.onnx_dir, self.get_model_id(),
                            f"visual-{self.visual_precision}")

    def get_visual_engine_dir(self) -> str:
        """Get the directory path for visual TensorRT engines."""
        return os.path.join(self.engine_dir, self.get_model_id(),
                            f"visual-{self.visual_precision}")
