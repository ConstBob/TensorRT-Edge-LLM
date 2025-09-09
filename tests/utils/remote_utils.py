"""
Remote device utilities for test execution
"""
import os
from dataclasses import dataclass


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
