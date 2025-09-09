"""
Test suite for VLM pipeline functionality.

Tests engine building, chat inference, benchmarking, and batch inference
using parameterized configurations. VLM models handle both text and visual inputs.
"""
from typing import Any, Dict

from runtime_test_config import VLMRuntimeTestConfig
from utils.remote_utils import RemoteConfig

from tests.pytest_helpers import PipelineTestType, execute_pipeline_test


class TestVLMPipeline:
    """Test suite for VLM pipeline functionality."""

    def test_engine_build(self, test_param: str, executable_files: Dict[str,
                                                                        str],
                          execution_mode: str, remote_config: RemoteConfig,
                          test_logger: Any, global_config: Dict[str,
                                                                Any]) -> None:
        """Test TensorRT engine building for VLM models."""
        execute_pipeline_test(test_param, executable_files, execution_mode,
                              remote_config, test_logger,
                              PipelineTestType.BUILD, VLMRuntimeTestConfig,
                              global_config)

    def test_chat(self, test_param: str, executable_files: Dict[str, str],
                  execution_mode: str, remote_config: RemoteConfig,
                  test_logger: Any, global_config: Dict[str, Any]) -> None:
        """Test interactive chat inference for VLM models."""
        execute_pipeline_test(test_param, executable_files, execution_mode,
                              remote_config, test_logger,
                              PipelineTestType.CHAT, VLMRuntimeTestConfig,
                              global_config)

    def test_benchmark(self, test_param: str, executable_files: Dict[str, str],
                       execution_mode: str, remote_config: RemoteConfig,
                       test_logger: Any, global_config: Dict[str,
                                                             Any]) -> None:
        """Test performance benchmarking for VLM models."""
        execute_pipeline_test(test_param, executable_files, execution_mode,
                              remote_config, test_logger,
                              PipelineTestType.BENCHMARK, VLMRuntimeTestConfig,
                              global_config)

    def test_inference(self, test_param: str, executable_files: Dict[str, str],
                       execution_mode: str, remote_config: RemoteConfig,
                       test_logger: Any, global_config: Dict[str,
                                                             Any]) -> None:
        """Test batch inference for VLM models."""
        execute_pipeline_test(test_param, executable_files, execution_mode,
                              remote_config, test_logger,
                              PipelineTestType.INFERENCE, VLMRuntimeTestConfig,
                              global_config)
