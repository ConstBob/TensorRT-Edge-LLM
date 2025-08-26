"""
Test configuration classes for tensorrt-edge-llm
"""
import os
from dataclasses import dataclass


@dataclass
class BaseTestConfig:
    """Base config with common fields and methods"""
    model_name: str
    precision: str
    batch_size: int = 1
    max_input_len: int = 2048
    max_seq_len: int = 4096
    output_seq_len: int = 128

    def get_model_id(self) -> str:
        """Universal model ID generation"""
        return f"{self.model_name}-{self.precision}-{self.max_seq_len}"

    def get_onnx_llm_path(self, onnx_dir: str) -> str:
        """Get ONNX model file path - universal for all model types"""
        return os.path.join(onnx_dir, self.get_model_id(), "model.onnx")

    def get_engine_llm_path(self, engine_dir: str) -> str:
        """Get LLM engine file path - universal"""
        return os.path.join(engine_dir, "llm_engines", self.get_model_id(),
                            "llm.engine")

    def get_engine_llm_dir(self, engine_dir: str) -> str:
        """Get LLM engine directory - universal"""
        return os.path.join(engine_dir, "llm_engines", self.get_model_id())

    def get_onnx_model_dir(self, onnx_dir: str) -> str:
        """Get ONNX model directory - universal"""
        return os.path.join(onnx_dir, self.get_model_id())

    @classmethod
    def _parse_common_params(cls, param_str: str):
        """Extract common parameters from test string"""
        parts = param_str.split('-')

        precision = None
        model_parts = []
        remaining = []

        for i, part in enumerate(parts):
            if part in ['fp16', 'fp8', 'int4', 'nvfp4']:
                precision = part
                model_parts = parts[:i]
                remaining = parts[i + 1:]
                break

        if not precision:
            raise ValueError(f"No precision found in: {param_str}")

        model_name = '-'.join(model_parts)
        return model_name, precision, remaining

    def _apply_common_params(self, remaining_parts: list):
        """Apply common parameter parsing"""
        for part in remaining_parts:
            if part.startswith('bs'):
                self.batch_size = int(part[2:])
            elif part.startswith('mxil'):
                self.max_input_len = int(part[4:])
            elif part.startswith('mxsl'):
                self.max_seq_len = int(part[4:])
            elif part.startswith('osl'):
                self.output_seq_len = int(part[3:])


@dataclass
class LLMTestConfig(BaseTestConfig):
    """LLM-specific config"""
    dynamic_shape: bool = False

    @classmethod
    def from_test_param(cls, param_str: str):
        """Parse test parameter string to create LLMTestConfig"""
        model_name, precision, remaining = cls._parse_common_params(param_str)
        config = cls(model_name=model_name, precision=precision)
        config._apply_common_params(remaining)
        return config


@dataclass
class VLMTestConfig(BaseTestConfig):
    """VLM-specific config - inherits common LLM paths, adds VLM-specific visual paths"""
    visual_precision: str = "fp16"
    model_type: str = "qwen2_vl"
    max_batch_size: int = 2
    min_image_tokens: int = 0
    max_image_tokens: int = 0
    image_tokens: int = 512
    is_dynamic: bool = False
    text_token_length: int = 1024
    image_token_length: int = 1024
    vitfp8: bool = False

    @classmethod
    def from_test_param(cls, param_str: str):
        """Parse test parameter string to create VLMTestConfig"""
        model_name, precision, remaining = cls._parse_common_params(param_str)
        config = cls(model_name=model_name, precision=precision)

        if 'qwen2-vl' in model_name.lower():
            config.model_type = 'qwen2_vl'
        elif 'qwen2.5-vl' in model_name.lower():
            config.model_type = 'qwen2_5_vl'
        elif 'internvl3' in model_name.lower():
            config.model_type = 'internvl3'

        config._apply_common_params(remaining)

        for part in remaining:
            if part.startswith('mxbs'):
                config.max_batch_size = int(part[4:])
            elif part.startswith('it'):
                config.image_tokens = int(part[2:])
            elif part.startswith('mnit'):
                config.min_image_tokens = int(part[4:])
            elif part.startswith('mxit'):
                config.max_image_tokens = int(part[4:])
            elif part.startswith('ttl'):
                config.text_token_length = int(part[3:])
            elif part.startswith('itl'):
                config.image_token_length = int(part[3:])
            elif part == 'dynamic':
                config.is_dynamic = True
            elif part == 'vitfp8':
                config.vitfp8 = True
                config.visual_precision = "fp8"

        return config

    def get_onnx_visual_path(self, onnx_dir: str) -> str:
        """Get visual ONNX file path - VLM-specific"""
        visual_precision = self.visual_precision
        return os.path.join(onnx_dir, self.get_model_id(),
                            f"visual_enc_onnx_{visual_precision}",
                            "model.onnx")

    def get_engine_visual_path(self, engine_dir: str) -> str:
        """Get visual engine file path - VLM-specific"""
        return os.path.join(engine_dir, "visual_engines", self.get_model_id(),
                            "visual.engine")

    def get_onnx_visual_dir(self, onnx_dir: str) -> str:
        """Get visual ONNX directory - VLM-specific"""
        return os.path.join(onnx_dir, self.get_model_id(),
                            f"visual_enc_onnx_{self.visual_precision}")

    def get_engine_visual_dir(self, engine_dir: str) -> str:
        return os.path.join(engine_dir, "visual_engines", self.get_model_id())
