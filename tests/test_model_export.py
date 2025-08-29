import enum
import os
from dataclasses import dataclass
from typing import List

import pytest


class ExportType(enum.Enum):
    LLM = "llm"
    VLM = "vlm"


class ExportLLMPrecision(enum.Enum):
    FP16 = "fp16"
    INT4 = "int4"
    FP8 = "fp8"
    NVFP4 = "nvfp4"


class ExportVisualPrecision(enum.Enum):
    FP16 = "fp16"
    FP8 = "fp8"


@dataclass
class ExportConfig:
    """Config for model export tests"""
    model_name: str
    llm_precision: str
    visual_precision: str
    export_type: ExportType = ExportType.LLM
    torch_model_dir: str = None
    output_base_dir: str = None
    max_seq_len: int = 4096

    def __post_init__(self):
        if self.torch_model_dir is None:
            self.torch_model_dir = os.environ.get(
                'TORCH_MODEL_DIR', '/scratch.trt_llm_data/llm-models')
        if self.output_base_dir is None:
            self.output_base_dir = os.environ.get('ONNX_MODEL_DIR', 'models')

    @classmethod
    def from_test_param(cls, test_param: str) -> 'ExportConfig':
        parts = test_param.split('-')

        model_parts = []
        llm_precision = None
        visual_precision = "fp16"
        max_seq_len = None

        valid_llm_precisions = [p.value for p in ExportLLMPrecision]
        for i, part in enumerate(parts):
            if part in valid_llm_precisions:
                llm_precision = part
                model_parts = parts[:i]
                remaining_parts = parts[i + 1:]
                break

        if not llm_precision:
            raise ValueError(
                f"Invalid export test parameter format: {test_param}")

        valid_visual_precisions = [p.value for p in ExportVisualPrecision]
        for i, part in enumerate(remaining_parts):
            if part.startswith('vit') and len(part) > 3:
                vit_precision = part[3:]  # Remove 'vit' prefix
                if vit_precision in valid_visual_precisions:
                    visual_precision = vit_precision
            elif part.isdigit():
                max_seq_len = int(part)

        model_name = '-'.join(model_parts)
        export_type = ExportType.VLM if 'VL' in model_name.upper(
        ) else ExportType.LLM

        return cls(model_name=model_name,
                   llm_precision=llm_precision,
                   visual_precision=visual_precision,
                   export_type=export_type,
                   max_seq_len=max_seq_len)

    def get_torch_model_path(self) -> str:
        """Get torch model directory path"""
        return os.path.join(self.torch_model_dir, self.model_name)

    def get_output_dir(self) -> str:
        """Get output directory for ONNX model"""
        output_name = f"{self.model_name}-{self.llm_precision}"
        if self.max_seq_len:
            output_name += f"-{self.max_seq_len}"
        return os.path.join(self.output_base_dir, output_name)

    def __str__(self) -> str:
        result = f"{self.export_type.value}-{self.model_name}-{self.llm_precision}"
        if self.max_seq_len:
            result += f"-{self.max_seq_len}"
        return result


def build_export_commands(config: ExportConfig) -> List[List[str]]:
    """Build export commands - data-driven approach"""
    torch_model_path = config.get_torch_model_path()
    output_dir = config.get_output_dir()

    commands = []

    llm_cmd = [
        "python3", "export/llm_export.py", f"--torch_dir={torch_model_path}",
        f"--output_dir={output_dir}", f"--dtype={config.llm_precision}"
    ]
    if config.max_seq_len:
        llm_cmd.append(f"--max_seq_len={config.max_seq_len}")
    if config.export_type == ExportType.VLM:
        llm_cmd.append("--use_prompt_tuning=True")
    commands.append(llm_cmd)

    if config.export_type == ExportType.VLM:
        multimodal_cmd = [
            "python3", "export/multimodal_export.py",
            f"--torch_dir={torch_model_path}", f"--output_dir={output_dir}",
            f"--visualType={config.visual_precision}"
        ]
        commands.append(multimodal_cmd)

    return commands


def validate_export_result(config: ExportConfig) -> None:
    """Simple file validation - fail fast"""
    output_dir = config.get_output_dir()

    llm_onnx = os.path.join(output_dir, "model.onnx")
    if not os.path.exists(llm_onnx):
        raise FileNotFoundError(f"LLM ONNX model not found: {llm_onnx}")

    if config.export_type == ExportType.VLM:
        visual_onnx = os.path.join(
            output_dir, f"visual_enc_onnx_{config.visual_precision}",
            "model.onnx")
        if not os.path.exists(visual_onnx):
            raise FileNotFoundError(
                f"Visual ONNX model not found: {visual_onnx}")


class TestModelExport:
    """Unified test suite for model export"""

    def test_model_export(self, test_param: str, test_config, test_logger):
        """Universal export test - handles both LLM and VLM"""
        from pytest_helpers import run_command, timer_context

        config = ExportConfig.from_test_param(test_param)

        # Simple validation
        torch_path = config.get_torch_model_path()
        if not os.path.exists(torch_path):
            raise FileNotFoundError(f"Torch model not found: {torch_path}")
        valid_llm_precisions = [p.value for p in ExportLLMPrecision]
        if config.llm_precision not in valid_llm_precisions:
            raise ValueError(
                f"Invalid LLM precision: {config.llm_precision}. Valid options: {valid_llm_precisions}"
            )

        valid_visual_precisions = [p.value for p in ExportVisualPrecision]
        if config.visual_precision not in valid_visual_precisions:
            raise ValueError(
                f"Invalid visual precision: {config.visual_precision}. Valid options: {valid_visual_precisions}"
            )

        output_dir = config.get_output_dir()
        os.makedirs(output_dir, exist_ok=True)

        test_logger.info(
            f"Starting {config.export_type.value.upper()} export test for {config}"
        )

        commands = build_export_commands(config)
        export_type_name = "VLM" if config.export_type == ExportType.VLM else "LLM"

        with timer_context(
                f"Exporting {export_type_name} {config.model_name} to {config.llm_precision}"
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
        test_logger.info(f"{export_type_name} export completed successfully")

    def test_llm_model_export(self, test_param: str, test_config, test_logger):
        """LLM export test entry point"""
        config = ExportConfig.from_test_param(test_param)
        if config.export_type != ExportType.LLM:
            pytest.skip(f"Not an LLM export test: {config.export_type}")
        self.test_model_export(test_param, test_config, test_logger)

    def test_vlm_model_export(self, test_param: str, test_config, test_logger):
        """VLM export test entry point"""
        config = ExportConfig.from_test_param(test_param)
        if config.export_type != ExportType.VLM:
            pytest.skip(f"Not a VLM export test: {config.export_type}")
        self.test_model_export(test_param, test_config, test_logger)
