"""
Remote device utilities for test execution
"""
import os
from dataclasses import dataclass
from typing import Dict


@dataclass
class RemoteConfig:
    """Configuration for remote test execution"""
    host: str
    user: str = "nvidia"
    password: str = None
    remote_workspace: str = "/home/nvidia/tensorrt-edge-llm"

    def __post_init__(self):
        if self.password is None:
            self.password = os.environ.get('REMOTE_PASSWORD', '')


class PathConverter:
    """Convert local paths to remote paths when needed"""

    def __init__(self,
                 test_config: Dict[str, str],
                 execution_mode: str = 'local',
                 remote_config: RemoteConfig = None):
        self.test_config = test_config
        self.execution_mode = execution_mode
        self.remote_config = remote_config

    def convert_to_remote_path(self, local_path: str) -> str:
        """Convert local path to remote path if needed"""
        if self.execution_mode == 'remote' and self.remote_config:
            llm_sdk_dir = self.test_config['llm_sdk_dir']
            if local_path.startswith(llm_sdk_dir):
                relative_path = os.path.relpath(local_path, llm_sdk_dir)
                remote_absolute_path = os.path.join(
                    self.remote_config.remote_workspace, relative_path)
                return remote_absolute_path
        return local_path


def enhance_config_with_remote_paths(config,
                                     test_config: Dict[str, str],
                                     execution_mode: str = 'local',
                                     remote_config: RemoteConfig = None):
    """Enhance config object with remote path conversion capability"""
    converter = PathConverter(test_config, execution_mode, remote_config)

    original_methods = {}
    path_methods = [
        'get_onnx_model_dir', 'get_engine_llm_dir', 'get_engine_llm_path',
        'get_onnx_llm_path'
    ]

    if hasattr(config, 'get_onnx_visual_dir'):
        path_methods.extend([
            'get_onnx_visual_dir', 'get_engine_visual_dir',
            'get_onnx_visual_path', 'get_engine_visual_path'
        ])

    for method_name in path_methods:
        if hasattr(config, method_name):
            original_methods[method_name] = getattr(config, method_name)

    def create_enhanced_method(original_method, method_name):

        def enhanced_method(self, *args, **kwargs):
            local_path = original_method(*args, **kwargs)
            return converter.convert_to_remote_path(local_path)

        return enhanced_method

    import types
    for method_name, original_method in original_methods.items():
        enhanced_method = create_enhanced_method(original_method, method_name)
        setattr(config, method_name, types.MethodType(enhanced_method, config))

    config._base_dirs = {
        'onnx_dir': test_config['onnx_model_dir'],
        'engine_dir': test_config['engine_dir']
    }

    return config
