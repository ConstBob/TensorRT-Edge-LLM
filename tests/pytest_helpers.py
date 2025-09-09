import enum
import os
import shlex
import subprocess
import time
from contextlib import contextmanager
from typing import Any, Dict, List

from runtime_test_config import BaseRuntimeTestConfig
from utils.accuracy_utils import check_rouge_score
from utils.command_config import build_command, get_command_timeout
from utils.remote_utils import RemoteConfig


class ExecutionMode:
    LOCAL = "local"
    REMOTE = "remote"


class PipelineTestType(enum.Enum):
    BUILD = "build"
    CHAT = "chat"
    BENCHMARK = "benchmark"
    INFERENCE = "inference"


def run_command(cmd: List[str],
                remote_host: str = None,
                remote_workspace: str = None,
                timeout: int = 300,
                remote_password: str = None) -> Dict[str, Any]:
    """Simple unified command execution"""
    if remote_host:
        ssh_cmd = [
            'sshpass', '-p', remote_password, 'ssh', '-o',
            'StrictHostKeyChecking=no', remote_host
        ]

        if remote_workspace:
            cmd_str = f"cd {shlex.quote(remote_workspace)} && {' '.join(shlex.quote(arg) for arg in cmd)}"
        else:
            cmd_str = ' '.join(shlex.quote(arg) for arg in cmd)
        ssh_cmd.append(cmd_str)
        final_cmd = ssh_cmd
    else:
        final_cmd = cmd

    result = subprocess.run(final_cmd,
                            capture_output=True,
                            text=True,
                            timeout=timeout)
    return {
        'success': result.returncode == 0,
        'returncode': result.returncode,
        'output': result.stdout,
        'error': result.stderr if result.returncode != 0 else None
    }


def check_file_exists(filepath: str,
                      remote_host: str = None,
                      remote_password: str = None) -> bool:
    """Simple file existence check"""
    if remote_host:
        result = run_command(['test', '-f', filepath], remote_host, None, 300,
                             remote_password)
        return result['success']
    else:
        return os.path.exists(filepath)


def get_file_content(filepath: str,
                     remote_host: str = None,
                     remote_password: str = None) -> str:
    """Get file content with proper error handling"""
    if remote_host:
        # First check if file exists
        if not check_file_exists(filepath, remote_host, remote_password):
            raise FileNotFoundError(f"Remote file not found: {filepath}")

        result = run_command(['cat', filepath], remote_host, None, 300,
                             remote_password)
        if not result['success']:
            raise RuntimeError(
                f"Failed to read remote file {filepath}: {result.get('error', 'Unknown error')}"
            )

        content = result['output']
        if not content or not content.strip():
            raise ValueError(
                f"Remote file {filepath} is empty or contains only whitespace")

        return content
    else:
        # Local file handling
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Local file not found: {filepath}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content or not content.strip():
                raise ValueError(
                    f"Local file {filepath} is empty or contains only whitespace"
                )

            return content
        except Exception as e:
            raise RuntimeError(
                f"Failed to read local file {filepath}: {str(e)}")


@contextmanager
def timer_context(description: str):
    """Simple timer context manager"""
    print(f"\nStarting: {description}")
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        print(f"Completed: {description} ({elapsed:.2f}s)")


class UnifiedTaskExecutor:
    """Universal task executor for both LLM and VLM"""

    def __init__(self,
                 config: BaseRuntimeTestConfig,
                 executable_files: Dict[str, str],
                 execution_mode: str = ExecutionMode.LOCAL,
                 remote_config: RemoteConfig = None,
                 logger=None):
        self.config = config
        self.executable_files = executable_files
        self.execution_mode = execution_mode
        self.remote_config = remote_config
        self.logger = logger

        # Simple command executor setup
        if execution_mode == ExecutionMode.REMOTE:
            self.remote_host = f"{remote_config.user}@{remote_config.host}"
            self.remote_workspace = remote_config.remote_workspace
            self.remote_password = remote_config.password
            self.run_cmd = lambda cmd, timeout=300: run_command(
                cmd, self.remote_host, self.remote_workspace, timeout, self.
                remote_password)
        else:
            self.remote_host = None
            self.remote_workspace = None
            self.remote_password = None
            self.run_cmd = lambda cmd, timeout=300: run_command(
                cmd, None, None, timeout)

    def execute_test(self, test_type: PipelineTestType) -> Dict[str, Any]:
        """Execute any test type for any model type"""
        if test_type == PipelineTestType.BUILD:
            return self.execute_build_test()
        elif test_type == PipelineTestType.CHAT:
            return self.execute_chat_test()
        elif test_type == PipelineTestType.BENCHMARK:
            return self.execute_benchmark_test()
        elif test_type == PipelineTestType.INFERENCE:
            return self.execute_inference_test()
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def execute_build_test(self) -> Dict[str, Any]:
        """Execute build test - adapts to config type"""
        self.create_engine_dirs()

        if self.config.type == "vlm":  # VLM
            llm_cmd = build_command('vlm_llm_build', self.config,
                                    self.executable_files)
            llm_result = self.run_cmd(llm_cmd,
                                      get_command_timeout('vlm_llm_build'))

            if not llm_result['success']:
                return {
                    'success': False,
                    'error':
                    f"LLM engine build failed: {llm_result.get('error', 'Unknown error')}",
                    'test_type': PipelineTestType.BUILD.value
                }

            visual_cmd = build_command('vlm_visual_build', self.config,
                                       self.executable_files)
            visual_result = self.run_cmd(
                visual_cmd, get_command_timeout('vlm_visual_build'))

            return {
                'success':
                visual_result['success'],
                'error':
                f"Visual engine build failed: {visual_result.get('error', 'Unknown error')}"
                if not visual_result['success'] else None,
                'test_type':
                PipelineTestType.BUILD.value
            }
        elif self.config.type == "llm":  # LLM
            cmd = build_command('llm_build', self.config,
                                self.executable_files)
            result = self.run_cmd(cmd, get_command_timeout('llm_build'))
            result['test_type'] = PipelineTestType.BUILD.value
            return result

    def execute_chat_test(self) -> Dict[str, Any]:
        """Execute chat test - adapts to config type"""
        cmd_key = 'vlm_chat' if self.config.type == "vlm" else 'llm_chat'
        cmd = build_command(cmd_key, self.config, self.executable_files)
        result = self.run_cmd(cmd, get_command_timeout(cmd_key))
        result['test_type'] = PipelineTestType.CHAT.value
        return result

    def execute_benchmark_test(self) -> Dict[str, Any]:
        """Execute benchmark test - adapts to config type"""
        cmd_key = 'vlm_benchmark' if self.config.type == "vlm" else 'llm_benchmark'
        cmd = build_command(cmd_key, self.config, self.executable_files)
        result = self.run_cmd(cmd, get_command_timeout(cmd_key))
        result['test_type'] = PipelineTestType.BENCHMARK.value
        return result

    def execute_inference_test(self) -> Dict[str, Any]:
        """Execute inference test - adapts to config type"""
        cmd_key = 'vlm_llm_inference' if self.config.type == "vlm" else 'llm_inference'
        cmd = build_command(cmd_key, self.config, self.executable_files)
        result = self.run_cmd(cmd, get_command_timeout(cmd_key))
        result['test_type'] = PipelineTestType.INFERENCE.value
        rouge_score = check_rouge_score(
            get_file_content(self.config.get_output_json_file(),
                             self.remote_host, self.remote_password),
            # The test case file is not hosted on the remote server, so we pass None for the remote host and password
            get_file_content(self.config.get_test_case_file(), None, None))
        result['rouge_score'] = rouge_score
        return result

    def create_engine_dirs(self):
        """Create engine directories as needed"""

        llm_engine_dir = self.config.get_llm_engine_dir()
        self.run_cmd(['mkdir', '-p', llm_engine_dir], 30)

        if self.config.type == "vlm":
            visual_engine_dir = self.config.get_visual_engine_dir()
            self.run_cmd(['mkdir', '-p', visual_engine_dir], 30)


def execute_pipeline_test(test_param: str, executable_files: Dict[str, str],
                          execution_mode: str, remote_config: RemoteConfig,
                          test_logger: Any, test_type: PipelineTestType,
                          config_class: type,
                          global_config: Dict[str, Any]) -> None:
    """
    Execute a pipeline test with the specified configuration.
    
    This function provides a unified interface for executing pipeline tests
    across different model types (LLM/VLM) and test types (build, chat, benchmark, inference).
    It creates the appropriate configuration object and executes the test using
    the UnifiedTaskExecutor.
    
    Args:
        test_param: Test parameter string defining model configuration
        executable_files: Dictionary mapping task names to executable paths
        execution_mode: Execution mode (local or remote)
        remote_config: Remote execution configuration for SSH-based testing
        test_logger: Logger instance for test output and debugging
        test_type: Type of pipeline test to execute
        config_class: Configuration class to use (LLMRuntimeTestConfig or VLMRuntimeTestConfig)
        global_config: Global test configuration including paths and settings
        
    Raises:
        pytest.fail: If the test execution fails
        
    Example:
        >>> execute_pipeline_test(
        ...     "Qwen2.5-0.5B-fp16-bs1-mxil2048", 
        ...     executables, "local", None, logger,
        ...     PipelineTestType.BUILD, LLMRuntimeTestConfig, global_config
        ... )
    """
    # Create configuration object from test parameter string
    config = config_class.from_test_string(test_param,
                                           global_config['onnx_dir'],
                                           global_config['engine_dir'])

    # Create task executor
    executor = UnifiedTaskExecutor(config, executable_files, execution_mode,
                                   remote_config, test_logger)

    if test_logger:
        test_logger.info(
            f"Starting {config.type} {test_type.value} test for {config.model_name}-{config.precision}"
        )

    # Execute the test with timing context
    with timer_context(
            f"{config.type} {test_type.value} for {config.model_name}"):
        result = executor.execute_test(test_type)

        # Log output if logger is available
        if test_logger:
            all_output = []
            if result.get('output'):
                all_output.extend(result['output'].strip().split('\n'))
            if result.get('error'):
                all_output.extend(result['error'].strip().split('\n'))

            if all_output:
                for line in all_output:
                    if line.strip():
                        test_logger.info(f"  {line}")

        # Fail the test if execution was unsuccessful
        if not result['success']:
            import pytest
            pytest.fail(
                f"{config.type} {test_type.value} failed: {result.get('error', 'Unknown error')}"
            )
