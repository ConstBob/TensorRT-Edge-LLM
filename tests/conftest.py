import datetime
import logging
import os
import sys
from pathlib import Path

# Add tests directory to path for imports
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

import pytest
import yaml
from pytest_helpers import run_command
from utils.device_utils import DeviceDetector


@pytest.fixture(scope="session")
def global_config():
    """Load test config from environment"""
    return {
        'llm_sdk_dir':
        os.environ.get('LLM_SDK_DIR', os.getcwd()),
        'torch_dir':
        os.environ.get('TORCH_DIR', '/scratch.trt_llm_data/llm-models'),
        'onnx_dir':
        os.environ.get('ONNX_DIR', 'models'),
        'engine_dir':
        os.environ.get('ENGINE_DIR', 'engines'),
        'trt_lib_path':
        os.environ.get('TRT_LIB_PATH', 'TensorRT-linux/lib'),
        'build_dir':
        'build',
        'test_log_dir':
        os.environ.get('TEST_LOG_DIR', 'logs'),
    }


@pytest.fixture(scope="session", autouse=True)
def setup_environment(global_config):
    """Setup environment and library paths"""
    llm_sdk_dir = global_config['llm_sdk_dir']
    trt_lib_path = os.path.join(llm_sdk_dir, global_config['trt_lib_path'])

    def _contains_trt_lib(dir_path: str) -> bool:
        try:
            return os.path.isdir(dir_path) and any(
                name.startswith('libnvinfer.so')
                for name in os.listdir(dir_path))
        except Exception:
            return False

    if not _contains_trt_lib(trt_lib_path):
        detector = DeviceDetector(run_command)
        detected_lib = detector.get_tensorrt_lib_dir(llm_sdk_dir)
        if detected_lib:
            trt_lib_path = detected_lib
            os.environ['TRT_LIB_PATH'] = trt_lib_path

    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{current_ld_path}:{trt_lib_path}"
    else:
        os.environ['LD_LIBRARY_PATH'] = trt_lib_path

    os.makedirs(global_config['onnx_dir'], exist_ok=True)
    os.makedirs(global_config['engine_dir'], exist_ok=True)

    log_dir_path = global_config['test_log_dir']
    os.makedirs(log_dir_path, exist_ok=True)


@pytest.fixture
def executable_files(global_config):
    """Paths to build executables"""
    build_dir = global_config['build_dir']
    return {
        'llm_build': f"{build_dir}/examples/llm/llm_build",
        'llm_chat': f"{build_dir}/examples/llm/llm_chat",
        'llm_benchmark': f"{build_dir}/examples/llm/llm_benchmark",
        'llm_accuracy': f"{build_dir}/examples/llm/llm_accuracy",
        'llm_inference': f"{build_dir}/examples/llm/llm_inference",
        'visual_build': f"{build_dir}/examples/multimodal/visual_build",
        'vlm_chat': f"{build_dir}/examples/multimodal/vlm_chat",
        'vlm_benchmark': f"{build_dir}/examples/multimodal/vlm_benchmark",
        'vlm_accuracy': f"{build_dir}/examples/multimodal/vlm_accuracy",
        'unit_test': f"{build_dir}/unitTest"
    }


@pytest.fixture
def execution_mode(request):
    """Get execution mode from command line"""
    mode_str = request.config.getoption("--execution-mode")
    # Import from pytest_helpers
    from pytest_helpers import ExecutionMode
    return ExecutionMode.LOCAL if mode_str == "local" else ExecutionMode.REMOTE


@pytest.fixture
def remote_config(request):
    """Get remote configuration from command line"""
    execution_mode_str = request.config.getoption("--execution-mode")

    if execution_mode_str == "remote":
        from utils.remote_utils import RemoteConfig

        password = request.config.getoption("--remote-password")
        if not password:
            password = os.environ.get('BOARD_PASSWORD_NVKS')

        if not password:
            pytest.fail(
                "Remote password required for remote execution. "
                "Use --remote-password or set BOARD_PASSWORD_NVKS environment variable."
            )

        return RemoteConfig(
            host=request.config.getoption("--remote-host"),
            user=request.config.getoption("--remote-user"),
            password=password,
            remote_workspace=request.config.getoption("--remote-workspace"))

    return None


@pytest.fixture(autouse=True)
def test_logger(request, global_config):
    """Create individual logger for each test"""
    test_name = request.node.name
    test_function = request.function.__name__

    log_dir_path = global_config['test_log_dir']
    log_dir = Path(log_dir_path)
    log_dir.mkdir(exist_ok=True, parents=True)

    if hasattr(request, 'param') or '[' in test_name:
        if '[' in test_name and ']' in test_name:
            param_part = test_name.split('[')[1].split(']')[0]
            param_clean = param_part.replace('/', '_').replace(':',
                                                               '_').replace(
                                                                   '-', '_')
            log_filename = f"{test_function}_{param_clean}.log"
        else:
            log_filename = f"{test_name}.log"
    else:
        log_filename = f"{test_function}.log"

    log_file = log_dir / log_filename

    logger = logging.getLogger(f"test_{test_name}")
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 80)
    logger.info(f"Starting test: {test_name}")
    logger.info(f"Test function: {test_function}")
    logger.info(f"Test file: {request.fspath}")
    logger.info(f"Timestamp: {datetime.datetime.now().isoformat()}")

    logger.info("Environment Information:")
    logger.info(f"  LLM_SDK_DIR: {os.environ.get('LLM_SDK_DIR', 'Not set')}")
    logger.info(f"  ONNX_DIR: {os.environ.get('ONNX_DIR', 'Not set')}")
    logger.info(f"  ENGINE_DIR: {os.environ.get('ENGINE_DIR', 'Not set')}")
    logger.info(
        f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    logger.info("=" * 80)

    request.node.test_logger = logger

    yield logger

    logger.info("=" * 80)
    logger.info(f"Completed test: {test_name}")
    logger.info(f"Timestamp: {datetime.datetime.now().isoformat()}")
    logger.info("=" * 80)

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption("--priority",
                     action="store",
                     default="l0",
                     help="Test priority level (l0, l1, etc.)")
    parser.addoption("--execution-mode",
                     action="store",
                     default="local",
                     choices=["local", "remote"],
                     help="Execution mode: local or remote")
    parser.addoption("--remote-host",
                     action="store",
                     default="192.168.55.1",
                     help="Remote host for remote execution")
    parser.addoption("--remote-user",
                     action="store",
                     default="nvidia",
                     help="Remote user for remote execution")
    parser.addoption("--remote-password",
                     action="store",
                     help="Remote password for remote execution")
    parser.addoption("--remote-workspace",
                     action="store",
                     default="/home/nvidia/tensorrt-edge-llm",
                     help="Remote workspace directory")


def pytest_runtest_setup(item):
    """Setup before each test"""
    if hasattr(item, 'test_logger'):
        item.test_logger.info(f"Setting up test: {item.name}")


def pytest_runtest_teardown(item):
    """Teardown after each test"""
    if hasattr(item, 'test_logger'):
        item.test_logger.info(f"Tearing down test: {item.name}")


_test_config_cache = {}


def _get_test_config(priority):
    """Get test configuration with caching"""
    if priority not in _test_config_cache:
        config_file = f"tests/configs/{priority}.yml"
        try:
            with open(config_file, 'r') as f:
                _test_config_cache[priority] = yaml.safe_load(f)
        except FileNotFoundError:
            _test_config_cache[priority] = None
    return _test_config_cache[priority]


def pytest_generate_tests(metafunc):
    """Generate parameterized tests based on YAML configuration"""
    if "test_param" in metafunc.fixturenames:
        priority = metafunc.config.getoption("--priority", "l0")
        config = _get_test_config(priority)

        if not config:
            return

        test_cases = config.get('tests', [])
        current_test_name = metafunc.function.__name__

        relevant_tests = []
        current_test_file = metafunc.module.__name__

        for test_case in test_cases:
            if isinstance(test_case, str):
                if '::' in test_case:
                    test_file_path, test_function_part = test_case.split(
                        '::', 1)
                    test_module = test_file_path.replace('/', '.').replace(
                        '.py', '')

                    if (test_module == current_test_file
                            and current_test_name in test_function_part):
                        if '[' in test_case and ']' in test_case:
                            param = test_case.split('[')[1].split(']')[0]
                            relevant_tests.append(param)

        if relevant_tests:
            metafunc.parametrize("test_param", relevant_tests)


def pytest_collection_modifyitems(config, items):
    """Keep only test functions that are explicitly configured in the YAML file"""
    priority = config.getoption("--priority", "l0")
    test_config = _get_test_config(priority)

    if not test_config:
        return

    configured_tests = set()
    for test_case in test_config.get('tests', []):
        if isinstance(test_case, str):
            test_name = test_case.split('::')[-1]
            configured_tests.add(test_name)

    items[:] = [item for item in items if item.name in configured_tests]
