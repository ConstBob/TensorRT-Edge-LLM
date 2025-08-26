import os
from typing import List

import pytest
from pytest_helpers import ExecutionMode, run_command, timer_context
from utils.device_utils import DeviceConfig, EnvironmentConfig


class ProjectBuilder:
    """Handles project compilation and build tasks"""

    def __init__(self,
                 logger=None,
                 remote_host=None,
                 remote_workspace=None,
                 remote_password=None):
        self.logger = logger
        self.remote_host = remote_host
        self.remote_workspace = remote_workspace
        self.remote_password = remote_password

    def build(self, device_config: DeviceConfig) -> bool:
        if self.logger:
            cap_str = device_config.compute_capability if device_config.compute_capability else "unknown"
            self.logger.info(
                f"Building for {device_config.target} (compute cap {cap_str})")
            if not device_config.cuda_version:
                self.logger.warning("CUDA version not detected")
            if not device_config.trt_package_dir:
                self.logger.warning("TensorRT package not found")
                raise RuntimeError("TensorRT package not found")

        cmake_cmd = ['cmake', '..', '-DBUILD_UNIT_TESTS=ON']

        if device_config.trt_package_dir:
            cmake_cmd.append(
                f'-DTRT_PACKAGE_DIR={device_config.trt_package_dir}')

        if device_config.cuda_version:
            cmake_cmd.append(f'-DCUDA_VERSION={device_config.cuda_version}')

        if device_config.target in ['orin', 'thor']:
            cmake_cmd.append(f'-DAUTO_TARGET={device_config.target}')

        build_cmd = ' && '.join(
            ['mkdir -p build', 'cd build', ' '.join(cmake_cmd), 'make -j16'])

        if self.logger:
            self.logger.info("Executing: PROJECT BUILD")

        result = run_command(cmd=['bash', '-c', build_cmd],
                             remote_host=self.remote_host,
                             remote_workspace=self.remote_workspace,
                             timeout=1800,
                             remote_password=self.remote_password)

        if self.logger:
            if result['success']:
                self.logger.info("Command succeeded: PROJECT BUILD")
            else:
                self.logger.error(
                    f"Command failed: PROJECT BUILD - {result.get('error', 'Unknown error')}"
                )
        return result['success']


def build_project(arch: str, cuda_version: str, trt_package_dir: str,
                  target: str, logger) -> bool:
    """Standalone project build function"""
    builder = ProjectBuilder(logger=logger)
    device_config = DeviceConfig(arch, cuda_version, trt_package_dir, target,
                                 None)
    return builder.build(device_config)


class TestProjectCommon:
    """Test suite for common project tasks (build and unit tests)"""

    def test_remote_setup(self, test_config, execution_mode, remote_config,
                          test_logger):
        """Test remote workspace setup - only runs in remote mode"""
        if execution_mode != ExecutionMode.REMOTE:
            pytest.skip("Remote setup test only runs in remote execution mode")

        if not remote_config:
            pytest.fail("Remote config required for remote execution mode")

        test_logger.info(
            f"Setting up remote workspace on {remote_config.host}")

    def test_build_project(self, request, test_config, execution_mode,
                           remote_config, test_logger):
        """Test project build - builds all components"""
        target = getattr(request, 'param', 'auto')
        test_logger.info(
            f"Building project (target: {target}, mode: {execution_mode})")

        config = EnvironmentConfig(execution_mode, remote_config)
        device_config = self._detect_device_config(config, test_logger)

        if device_config.cuda_version:
            test_logger.info(f"CUDA version: {device_config.cuda_version}")
        else:
            test_logger.warning("CUDA version not detected")

        if device_config.trt_package_dir:
            test_logger.info(
                f"TensorRT package: {device_config.trt_package_dir}")
        else:
            test_logger.warning("TensorRT package not found")

        if execution_mode == ExecutionMode.REMOTE:
            builder = ProjectBuilder(
                logger=test_logger,
                remote_host=remote_config.host,
                remote_workspace=remote_config.remote_workspace,
                remote_password=remote_config.password)
        else:
            builder = ProjectBuilder(logger=test_logger)

        with timer_context(f"Building ({execution_mode})"):
            success = builder.build(device_config)

        if not success:
            pytest.fail("Build failed")

        expected_files = [
            'unitTest', 'examples/llm/llm_build', 'examples/llm/llm_chat',
            'examples/llm/llm_benchmark', 'examples/multimodal/visual_build',
            'examples/multimodal/vlm_chat', 'examples/multimodal/vlm_benchmark'
        ]
        remote_host = remote_config.host if execution_mode == ExecutionMode.REMOTE else None
        remote_workspace = remote_config.remote_workspace if execution_mode == ExecutionMode.REMOTE else None
        remote_password = remote_config.password if execution_mode == ExecutionMode.REMOTE else None
        self._verify_build_artifacts(expected_files, 'build', remote_host,
                                     remote_workspace, remote_password,
                                     test_logger)

        test_logger.info(f"Build completed for {device_config.target}")

    def test_unit_tests(self, test_config, executable_files, execution_mode,
                        remote_config, test_logger):
        """Test unit tests execution - model independent"""
        test_logger.info(
            f"Starting unit tests execution in {execution_mode} mode")

        build_dir = test_config.get('build_dir', 'build')
        if execution_mode == ExecutionMode.REMOTE:
            test_logger.info("Executing: UNIT TESTS")
            remote_host = f"{remote_config.user}@{remote_config.host}"
            test_logger.info(f"Remote host: {remote_host}")
            test_logger.info(
                f"Remote workspace: {remote_config.remote_workspace}")

            unit_test_cmd = ["./unitTest"]
            test_logger.info(f"Unit test command: {unit_test_cmd}")
            result = run_command(
                cmd=unit_test_cmd,
                remote_host=remote_host,
                remote_workspace=
                f"{remote_config.remote_workspace}/{build_dir}",
                timeout=1800,
                remote_password=remote_config.password)

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

            if test_logger:
                if result['success']:
                    test_logger.info("Command succeeded: UNIT TESTS")
                else:
                    test_logger.error(
                        f"Command failed: UNIT TESTS - {result.get('error', 'Unknown error')}"
                    )
        else:
            unit_test_path = executable_files['unit_test']
            if not os.path.exists(unit_test_path):
                pytest.fail(
                    f"Unit test executable not found: {unit_test_path}")

            test_logger.info(f"Unit test executable found: {unit_test_path}")

            test_logger.info("Executing: UNIT TESTS")
            if os.path.exists(build_dir):
                unit_test_cmd = f"cd {build_dir} && ./unitTest"
                result = run_command(cmd=['bash', '-c', unit_test_cmd],
                                     timeout=1800)
            else:
                test_logger.warning(
                    f"Build directory {build_dir} not found, running from current directory"
                )
                result = run_command(cmd=[unit_test_path], timeout=1800)

            if test_logger:
                if result['success']:
                    test_logger.info("Command succeeded: UNIT TESTS")
                else:
                    test_logger.error(
                        f"Command failed: UNIT TESTS - {result.get('error', 'Unknown error')}"
                    )

        if not result['success']:
            pytest.fail(
                f"Unit tests failed: {result.get('error', 'Unknown error')}")

    def _detect_device_config(self, test_config: EnvironmentConfig,
                              logger) -> DeviceConfig:
        """Detect device configuration for the current environment"""

        if test_config.execution_mode == ExecutionMode.REMOTE:
            remote_host = f"{test_config.remote_config.user}@{test_config.remote_config.host}"
            remote_workspace = test_config.remote_config.remote_workspace
            remote_password = test_config.remote_config.password
            run_cmd_func = lambda cmd, timeout=300: run_command(
                cmd, remote_host, remote_workspace, timeout, remote_password)
        else:
            run_cmd_func = lambda cmd, timeout=300: run_command(
                cmd, None, None, timeout)

        workspace = test_config.get_workspace()
        device_config = DeviceConfig.auto_detect(run_cmd_func, workspace)

        if logger:
            cap_str = device_config.compute_capability if device_config.compute_capability else "unknown"
            logger.info(
                f"Detected: {device_config.arch}, CUDA {device_config.cuda_version}, "
                f"target {device_config.target} (compute cap {cap_str})")

        return device_config

    def _verify_build_artifacts(self, expected_artifacts: List[str],
                                build_dir: str, remote_host: str,
                                remote_workspace: str, remote_password: str,
                                logger):
        """Verify build artifacts exist using unified command execution"""
        for artifact in expected_artifacts:
            artifact_path = os.path.join(build_dir, artifact)

            result = run_command(cmd=['test', '-f', artifact_path],
                                 remote_host=remote_host,
                                 remote_workspace=remote_workspace,
                                 timeout=10,
                                 remote_password=remote_password)
            if result['success']:
                logger.info(f"Build artifact verified: {artifact_path}")
            else:
                logger.warning(
                    f"Expected build artifact not found: {artifact_path}")
