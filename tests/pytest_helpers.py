import enum
import os
import shlex
import subprocess
import time
from contextlib import contextmanager
from typing import Any, Dict, List

from utils.command_config import build_command, get_command_timeout
from utils.remote_utils import enhance_config_with_remote_paths


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


def validate_model_files(config,
                         test_config: Dict[str, str],
                         remote_host: str = None) -> None:
    """Simple validation - check required ONNX files"""
    files = [config.get_onnx_llm_path(test_config['onnx_model_dir'])]

    if hasattr(config, 'get_onnx_visual_path'):  # VLM
        files.append(config.get_onnx_visual_path(
            test_config['onnx_model_dir']))

    for filepath in files:
        if not check_file_exists(filepath, remote_host):
            raise FileNotFoundError(f"Required file not found: {filepath}")


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
                 config,
                 executable_files: Dict[str, str],
                 execution_mode: str = ExecutionMode.LOCAL,
                 remote_config=None,
                 logger=None):
        self.config = config
        self.executable_files = executable_files
        self.execution_mode = execution_mode
        self.remote_config = remote_config
        self.logger = logger

        # Simple command executor setup
        if execution_mode == ExecutionMode.REMOTE:
            remote_host = f"{remote_config.user}@{remote_config.host}"
            remote_workspace = remote_config.remote_workspace
            remote_password = remote_config.password
            self.run_cmd = lambda cmd, timeout=300: run_command(
                cmd, remote_host, remote_workspace, timeout, remote_password)
        else:
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
        self._ensure_engine_directories_exist()

        if hasattr(self.config, 'get_engine_visual_path'):  # VLM
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
        else:  # LLM
            cmd = build_command('llm_build', self.config,
                                self.executable_files)
            result = self.run_cmd(cmd, get_command_timeout('llm_build'))
            result['test_type'] = PipelineTestType.BUILD.value
            return result

    def execute_chat_test(self) -> Dict[str, Any]:
        """Execute chat test - adapts to config type"""
        cmd_key = 'vlm_chat' if hasattr(
            self.config, 'get_engine_visual_path') else 'llm_chat'

        cmd = build_command(cmd_key, self.config, self.executable_files)
        result = self.run_cmd(cmd, get_command_timeout(cmd_key))
        result['test_type'] = PipelineTestType.CHAT.value
        return result

    def execute_benchmark_test(self) -> Dict[str, Any]:
        """Execute benchmark test - adapts to config type"""
        cmd_key = 'vlm_benchmark' if hasattr(
            self.config, 'get_engine_visual_path') else 'llm_benchmark'

        cmd = build_command(cmd_key, self.config, self.executable_files)
        result = self.run_cmd(cmd, get_command_timeout(cmd_key))
        result['test_type'] = PipelineTestType.BENCHMARK.value
        return result

    def execute_inference_test(self) -> Dict[str, Any]:
        """Execute inference test - adapts to config type"""
        cmd_key = 'vlm_llm_inference' if hasattr(
            self.config, 'get_engine_visual_path') else 'llm_inference'
        cmd = build_command(cmd_key, self.config, self.executable_files)
        result = self.run_cmd(cmd, get_command_timeout(cmd_key))
        result['test_type'] = PipelineTestType.INFERENCE.value
        return result

    def execute_command(self, cmd, task_name="", timeout=300):
        """Simple command execution interface for compatibility"""
        return self.run_cmd(cmd, timeout)

    def _ensure_engine_directories_exist(self):
        """Create engine directories as needed"""
        base_dirs = getattr(self.config, '_base_dirs', {})
        engine_dir = base_dirs.get('engine_dir', 'engines')

        llm_engine_path = self.config.get_engine_llm_path(engine_dir)
        llm_engine_dir = os.path.dirname(llm_engine_path)
        self.run_cmd(['mkdir', '-p', llm_engine_dir], 30)

        if hasattr(self.config, 'get_engine_visual_path'):
            visual_engine_path = self.config.get_engine_visual_path(engine_dir)
            visual_engine_dir = os.path.dirname(visual_engine_path)
            self.run_cmd(['mkdir', '-p', visual_engine_dir], 30)


def execute_pipeline_test(test_param: str, test_config: Dict[str, str],
                          executable_files: Dict[str, str],
                          execution_mode: str, remote_config, test_logger,
                          test_type: PipelineTestType, config_class,
                          pipeline_name: str):
    """Simple test execution"""

    config = config_class.from_test_param(test_param)
    enhanced_config = enhance_config_with_remote_paths(config, test_config,
                                                       execution_mode,
                                                       remote_config)
    executor = UnifiedTaskExecutor(enhanced_config, executable_files,
                                   execution_mode, remote_config, test_logger)

    if test_logger:
        test_logger.info(
            f"Starting {pipeline_name} {test_type.value} test for {config.model_name}-{config.precision}"
        )

    with timer_context(
            f"{pipeline_name} {test_type.value} for {config.model_name}"):
        result = executor.execute_test(test_type)

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

        if not result['success']:
            import pytest
            pytest.fail(
                f"{pipeline_name} {test_type.value} failed: {result.get('error', 'Unknown error')}"
            )
