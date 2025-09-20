import os

import pytest
from conftest import EnvironmentConfig
from pytest_helpers import timer_context

from .config import ModelType, TaskType, TestConfig
from .utils.command_generation import generate_export_commands


def validate_export_result(config: TestConfig) -> None:
    """Simple file validation - fail fast"""
    output_dir = config.get_llm_onnx_dir()

    llm_onnx = os.path.join(output_dir, "model.onnx")
    if not os.path.exists(llm_onnx):
        raise FileNotFoundError(f"LLM ONNX model not found: {llm_onnx}")

    if config.lora:
        lora_onnx = os.path.join(config.get_llm_onnx_dir(), "lora_model.onnx")
        if not os.path.exists(lora_onnx):
            raise FileNotFoundError(f"LoRA ONNX model not found: {lora_onnx}")

    if config.model_type == ModelType.VLM:
        # Visual model should be in separate visual subdirectory
        fp16_visual_onnx_dir = config.get_visual_onnx_dir("fp16")
        if not os.path.exists(fp16_visual_onnx_dir):
            raise FileNotFoundError(
                f"Visual ONNX model not found: {fp16_visual_onnx_dir}")
        if config.visual_precision == "fp8":
            fp8_visual_onnx_dir = config.get_visual_onnx_dir("fp8")
            if not os.path.exists(fp8_visual_onnx_dir):
                raise FileNotFoundError(
                    f"Visual ONNX model not found: {fp8_visual_onnx_dir}")


class TestModelExport:
    """Unified test suite for model export"""

    def test_model_export(self, test_param: str, test_logger,
                          model_type: ModelType,
                          env_config: EnvironmentConfig):
        """Universal export test - handles both LLM and VLM"""

        config = TestConfig.from_param_string(test_param, model_type,
                                              TaskType.EXPORT, env_config)

        # Simple validation
        torch_dir = config.get_torch_model_dir()
        if not os.path.exists(torch_dir):
            raise FileNotFoundError(f"Torch model not found: {torch_dir}")

        llm_onnx_dir = config.get_llm_onnx_dir()
        os.makedirs(llm_onnx_dir, exist_ok=True)

        # Create quantized model directory if needed
        if config.llm_precision != "fp16":
            quantized_model_dir = config.get_quantized_model_dir()
            os.makedirs(quantized_model_dir, exist_ok=True)

        commands = generate_export_commands(config)

        with timer_context(
                f"Exporting {config.model_type.value} {config.model_name} to {config.llm_precision}",
                test_logger):
            for i, (cmd, timeout) in enumerate(commands):
                task_name = f"Export step {i+1}/{len(commands)}"

                from pytest_helpers import run_command
                result = run_command(cmd,
                                     timeout=timeout,
                                     remote_config=None,
                                     logger=test_logger)
                if not result['success']:
                    pytest.fail(
                        f"{task_name} failed: {result.get('error', 'Unknown error')}"
                    )

        validate_export_result(config)

    def test_llm_model_export(self, test_param: str, test_logger,
                              env_config: EnvironmentConfig):
        """LLM export test entry point"""
        self.test_model_export(test_param, test_logger, ModelType.LLM,
                               env_config)

    def test_vlm_model_export(self, test_param: str, test_logger,
                              env_config: EnvironmentConfig):
        """VLM export test entry point"""
        self.test_model_export(test_param, test_logger, ModelType.VLM,
                               env_config)
