import enum
import os
from dataclasses import dataclass
from typing import List

import pytest
from valid_precisions import (VALID_LLM_PRECISIONS, VALID_LM_HEAD_PRECISIONS,
                              VALID_VISUAL_PRECISIONS)


class ExportType(enum.Enum):
    LLM = "LLM"
    VLM = "VLM"


@dataclass
class ExportConfig:
    """Config for model export tests"""
    model_name: str
    llm_precision: str
    visual_precision: str
    lm_head_precision: str
    export_type: ExportType = ExportType.LLM
    torch_dir: str = None
    onnx_dir: str = None
    max_seq_len: int = 4096

    @classmethod
    def from_test_string(cls, test_param: str, export_type: ExportType,
                         global_config) -> 'ExportConfig':
        parts = test_param.split('-')

        model_parts = []
        llm_precision = None
        lm_head_precision = "fp16"
        visual_precision = "fp16"
        max_seq_len = None

        for i, part in enumerate(parts):
            if part in VALID_LLM_PRECISIONS:
                llm_precision = part
                model_parts = parts[:i]
            elif part.startswith('lm'):
                lm_head_precision = part[2:]
            elif part.startswith('vit'):
                visual_precision = part[3:]
            elif part.isdigit():
                max_seq_len = int(part)

        if not llm_precision:
            raise ValueError(
                f"Invalid export test parameter format: {test_param}. Can't find LLM precision"
            )
        if llm_precision not in VALID_LLM_PRECISIONS:
            raise ValueError(
                f"Invalid export test parameter format: {test_param}. Invalid LLM precision: {llm_precision}"
            )
        if lm_head_precision not in VALID_LM_HEAD_PRECISIONS:
            raise ValueError(
                f"Invalid export test parameter format: {test_param}. Invalid LM head precision: {lm_head_precision}"
            )
        if visual_precision not in VALID_VISUAL_PRECISIONS:
            raise ValueError(
                f"Invalid export test parameter format: {test_param}. Invalid visual precision: {visual_precision}"
            )

        model_name = '-'.join(model_parts)

        return cls(model_name=model_name,
                   llm_precision=llm_precision,
                   lm_head_precision=lm_head_precision,
                   visual_precision=visual_precision,
                   export_type=export_type,
                   max_seq_len=max_seq_len,
                   torch_dir=global_config['torch_dir'],
                   onnx_dir=global_config['onnx_dir'])

    def get_torch_model_dir(self) -> str:
        """Get torch model directory path"""
        return os.path.join(self.torch_dir, self.model_name)

    def get_cnn_dailymail_dataset_dir(self) -> str:
        """Get dataset directory path"""
        return os.path.join(self.torch_dir, "datasets/cnn_dailymail")

    def get_mmmu_dataset_dir(self) -> str:
        """Get dataset directory path"""
        return os.path.join(self.torch_dir, "datasets/MMMU")

    def get_onnx_model_dir(self) -> str:
        """Get output directory for ONNX model"""
        output_name = f"{self.model_name}-{self.llm_precision}-{self.lm_head_precision}-{self.max_seq_len}"
        return os.path.join(self.onnx_dir, output_name)

    def get_quantized_model_dir(self) -> str:
        """Get quantized model directory path (if quantization is needed)"""
        if self.llm_precision == "fp16":
            return self.get_torch_model_dir()
        quantized_name = f"{self.model_name}-quantized-{self.llm_precision}-{self.lm_head_precision}-{self.max_seq_len}"
        return os.path.join(self.get_onnx_model_dir(), quantized_name)

    def get_llm_onnx_model_dir(self) -> str:
        """Get LLM ONNX model directory path"""
        return os.path.join(self.get_onnx_model_dir(), f"llm")

    def get_visual_onnx_model_dir(self, visual_precision: str) -> str:
        """Get visual ONNX model directory path"""
        return os.path.join(self.get_onnx_model_dir(),
                            f"visual-{visual_precision}")

    def __str__(self) -> str:
        result = f"{self.export_type.value}-{self.model_name}-{self.llm_precision}-{self.lm_head_precision}-{self.max_seq_len}"
        if self.max_seq_len:
            result += f"-{self.max_seq_len}"
        return result


def build_export_commands(config: ExportConfig) -> List[List[str]]:
    """Build export commands - data-driven approach using latest API"""
    torch_model_dir = config.get_torch_model_dir()
    config.get_onnx_model_dir()
    commands = []

    # If precision is not fp16, we need to quantize first
    if config.llm_precision != "fp16":
        quantized_model_dir = config.get_quantized_model_dir()

        # Quantize command
        quantize_cmd = [
            "tensorrt-edgellm-quantize-llm", f"--model_dir={torch_model_dir}",
            f"--output_dir={quantized_model_dir}",
            f"--quantization={config.llm_precision}",
            f"--dataset_dir={config.get_cnn_dailymail_dataset_dir()}"
        ]
        if config.lm_head_precision != "fp16":
            quantize_cmd.append(
                f"--lm_head_quantization={config.lm_head_precision}")
        commands.append(quantize_cmd)

        # Export command (using quantized model)
        llm_cmd = [
            "tensorrt-edgellm-export-llm",
            f"--model_dir={quantized_model_dir}",
            f"--output_dir={config.get_llm_onnx_model_dir()}"
        ]
    else:
        # Direct export for fp16
        llm_cmd = [
            "tensorrt-edgellm-export-llm", f"--model_dir={torch_model_dir}",
            f"--output_dir={config.get_llm_onnx_model_dir()}"
        ]

    if config.max_seq_len:
        llm_cmd.append(f"--max_position_embeddings={config.max_seq_len}")
    commands.append(llm_cmd)

    if config.export_type == ExportType.VLM:
        # Always export fp16 visual model regardless of the precision
        visual_fp16_dir = config.get_visual_onnx_model_dir(
            visual_precision="fp16")
        visual_fp16_cmd = [
            "tensorrt-edgellm-export-visual",
            f"--model_dir={torch_model_dir}",
            f"--output_dir={visual_fp16_dir}",
            f"--dtype=fp16",
        ]
        commands.append(visual_fp16_cmd)

        if config.visual_precision == "fp8":
            # Export fp8 visual model if specifies fp8 precision is fp8 to allow both fp8 and fp16 visual model tests
            visual_fp8_dir = config.get_visual_onnx_model_dir(
                visual_precision="fp8")
            visual_fp8_cmd = [
                "tensorrt-edgellm-export-visual",
                f"--model_dir={torch_model_dir}",
                f"--output_dir={visual_fp8_dir}",
                f"--dtype=fp16",
                f"--quantization=fp8",
                f"--dataset_dir={config.get_mmmu_dataset_dir()}",
            ]
            commands.append(visual_fp8_cmd)

    return commands


def validate_export_result(config: ExportConfig) -> None:
    """Simple file validation - fail fast"""
    output_dir = config.get_llm_onnx_model_dir()

    llm_onnx = os.path.join(output_dir, "model.onnx")
    if not os.path.exists(llm_onnx):
        raise FileNotFoundError(f"LLM ONNX model not found: {llm_onnx}")

    if config.export_type == ExportType.VLM:
        # Visual model should be in separate visual subdirectory
        fp16_visual_onnx = config.get_visual_onnx_model_dir(
            visual_precision="fp16")
        if not os.path.exists(fp16_visual_onnx):
            raise FileNotFoundError(
                f"FP16 Visual ONNX model not found: {fp16_visual_onnx}")
        if config.visual_precision == "fp8":
            fp8_visual_onnx = config.get_visual_onnx_model_dir(
                visual_precision="fp8")
            if not os.path.exists(fp8_visual_onnx):
                raise FileNotFoundError(
                    f"FP8 Visual ONNX model not found: {fp8_visual_onnx}")


class TestModelExport:
    """Unified test suite for model export"""

    def test_model_export(self, test_param: str, test_logger,
                          export_type: ExportType, global_config):
        """Universal export test - handles both LLM and VLM"""
        from pytest_helpers import run_command, timer_context

        config = ExportConfig.from_test_string(test_param, export_type,
                                               global_config)

        # Simple validation
        torch_dir = config.get_torch_model_dir()
        if not os.path.exists(torch_dir):
            raise FileNotFoundError(f"Torch model not found: {torch_dir}")

        llm_onnx_dir = config.get_llm_onnx_model_dir()
        os.makedirs(llm_onnx_dir, exist_ok=True)

        # Create quantized model directory if needed
        if config.llm_precision != "fp16":
            quantized_model_dir = config.get_quantized_model_dir()
            os.makedirs(quantized_model_dir, exist_ok=True)

        test_logger.info(
            f"Starting {config.export_type.value} export test for {config}")

        commands = build_export_commands(config)

        with timer_context(
                f"Exporting {config.export_type.value} {config.model_name} to {config.llm_precision}"
        ):
            for i, cmd in enumerate(commands):
                task_name = f"Export step {i+1}/{len(commands)}"
                test_logger.info(f"$ {' '.join(cmd)}")

                result = run_command(cmd, timeout=3600)  # 1 hour timeout

                if not result['success']:
                    pytest.fail(
                        f"{task_name} failed: {result.get('error', 'Unknown error')}"
                    )

                if test_logger and result.get('output'):
                    for line in result['output'].splitlines():
                        if line.strip():
                            test_logger.info(line)

        validate_export_result(config)
        test_logger.info(
            f"{config.export_type.value} export completed successfully")

    def test_llm_model_export(self, test_param: str, test_logger,
                              global_config):
        """LLM export test entry point"""
        self.test_model_export(test_param, test_logger, ExportType.LLM,
                               global_config)

    def test_vlm_model_export(self, test_param: str, test_logger,
                              global_config):
        """VLM export test entry point"""
        self.test_model_export(test_param, test_logger, ExportType.VLM,
                               global_config)
