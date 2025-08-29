from pytest_helpers import PipelineTestType, execute_pipeline_test
from utils.test_config import VLMTestConfig


class TestVLMPipeline:
    """Test suite for VLM pipeline functionality"""

    def test_engine_build(self, test_param: str, test_config, executable_files,
                          execution_mode, remote_config, test_logger):
        """Test engine build with parameterized configs"""
        execute_pipeline_test(test_param, test_config, executable_files,
                              execution_mode, remote_config, test_logger,
                              PipelineTestType.BUILD, VLMTestConfig, "VLM")

    def test_inference_chat(self, test_param: str, test_config,
                            executable_files, execution_mode, remote_config,
                            test_logger):
        """Test chat inference with parameterized configs"""
        execute_pipeline_test(test_param, test_config, executable_files,
                              execution_mode, remote_config, test_logger,
                              PipelineTestType.CHAT, VLMTestConfig, "VLM")

    def test_inference_benchmark(self, test_param: str, test_config,
                                 executable_files, execution_mode,
                                 remote_config, test_logger):
        """Test benchmark inference with parameterized configs"""
        execute_pipeline_test(test_param, test_config, executable_files,
                              execution_mode, remote_config, test_logger,
                              PipelineTestType.BENCHMARK, VLMTestConfig, "VLM")

    def test_inference(self, test_param: str, test_config, executable_files,
                       execution_mode, remote_config, test_logger):
        """Test inference with parameterized configs"""
        execute_pipeline_test(test_param, test_config, executable_files,
                              execution_mode, remote_config, test_logger,
                              PipelineTestType.INFERENCE, VLMTestConfig, "VLM")
