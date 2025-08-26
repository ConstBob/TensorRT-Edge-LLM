from utils.test_config import LLMTestConfig

from tests.pytest_helpers import PipelineTestType, execute_pipeline_test


class TestLLMPipeline:
    """Test suite for LLM pipeline functionality"""

    def test_engine_build(self, test_param: str, test_config, executable_files,
                          execution_mode, remote_config, test_logger):
        """Test engine build with parameterized configs"""
        execute_pipeline_test(test_param, test_config, executable_files,
                              execution_mode, remote_config, test_logger,
                              PipelineTestType.BUILD, LLMTestConfig, "LLM")

    def test_inference_chat(self, test_param: str, test_config,
                            executable_files, execution_mode, remote_config,
                            test_logger):
        """Test chat inference with parameterized configs"""
        execute_pipeline_test(test_param, test_config, executable_files,
                              execution_mode, remote_config, test_logger,
                              PipelineTestType.CHAT, LLMTestConfig, "LLM")

    def test_inference_benchmark(self, test_param: str, test_config,
                                 executable_files, execution_mode,
                                 remote_config, test_logger):
        """Test benchmark inference with parameterized configs"""
        execute_pipeline_test(test_param, test_config, executable_files,
                              execution_mode, remote_config, test_logger,
                              PipelineTestType.BENCHMARK, LLMTestConfig, "LLM")
