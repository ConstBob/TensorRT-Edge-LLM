"""
Test suite for LLM pipeline functionality.

Tests engine building, chat inference, benchmarking, and batch inference
using parameterized configurations.
"""
import logging
from typing import Dict, Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import timer_context

from .config import ModelType, TaskType, TestConfig
from .utils.command_execution import (execute_benchmark_test,
                                      execute_build_test,
                                      execute_inference_test)


class TestLLMPipeline:
    """Test suite for LLM pipeline functionality."""

    def test_engine_build(self, test_param: str, executable_files: Dict[str,
                                                                        str],
                          remote_config: Optional[RemoteConfig],
                          test_logger: logging.Logger,
                          env_config: EnvironmentConfig) -> None:
        """Test TensorRT engine building for LLM models."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.BUILD, env_config)

        with timer_context(f"LLM build for {config.model_name}", test_logger):
            result = execute_build_test(config, executable_files,
                                        remote_config, test_logger, env_config)
            if not result['success']:
                pytest.fail(f"Build failed: {result['error']}")

    def test_benchmark(self, test_param: str, executable_files: Dict[str, str],
                       remote_config: Optional[RemoteConfig],
                       test_logger: logging.Logger,
                       env_config: EnvironmentConfig) -> None:
        """Test performance benchmarking for LLM models."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.BENCHMARK, env_config)

        with timer_context(f"LLM benchmark for {config.model_name}",
                           test_logger):
            result = execute_benchmark_test(config, executable_files,
                                            remote_config, test_logger,
                                            env_config)
            if not result['success']:
                pytest.fail(f"Benchmark failed: {result['error']}")

    def test_inference(self, test_param: str, executable_files: Dict[str, str],
                       remote_config: Optional[RemoteConfig],
                       test_logger: logging.Logger,
                       env_config: EnvironmentConfig) -> None:
        """Test batch inference for LLM models."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)

        with timer_context(f"LLM inference for {config.model_name}",
                           test_logger):
            result = execute_inference_test(config, executable_files,
                                            remote_config, test_logger,
                                            env_config)
            if not result['success']:
                pytest.fail(f"Inference failed: {result['error']}")
