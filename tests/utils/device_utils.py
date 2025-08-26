"""Device detection and config for local/remote execution"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pytest_helpers import ExecutionMode


@dataclass
class DeviceConfig:
    arch: str
    cuda_version: Optional[str]
    trt_package_dir: Optional[str]
    target: str  # orin, thor, n1, or auto
    compute_capability: Optional[int]

    @classmethod
    def auto_detect(cls, run_command_func, workspace: str):
        """Auto-detect device configuration using run_command function"""
        detector = DeviceDetector(run_command_func)

        arch = detector._get_arch()
        cuda_version = detector._get_cuda_version()
        trt_package_dir = detector._get_tensorrt_package(workspace)

        compute_cap = detector._get_compute_capability()
        target = cls._map_compute_cap_to_target(compute_cap)

        return cls(arch, cuda_version, trt_package_dir, target, compute_cap)

    @staticmethod
    def _map_compute_cap_to_target(compute_cap: Optional[int]) -> str:
        if compute_cap == 87:
            return 'orin'
        elif compute_cap == 101 or compute_cap == 110:
            return 'thor'
        elif compute_cap == 121:
            return 'n1'
        else:
            return 'auto'  # let cmake decide


class DeviceDetector:

    def __init__(self, run_command_func):
        self.run_command = run_command_func

    def _get_arch(self) -> str:
        result = self.run_command(['uname', '-m'], timeout=10)
        return result['output'].strip() if result['success'] else 'unknown'

    def _get_cuda_version(self) -> str:
        # find libcudart.so files
        find_result = self.run_command(
            ['find', '/usr', '/opt', '-name', 'libcudart.so*', '-type', 'f'],
            timeout=30)

        if find_result['success'] and find_result['output'].strip():
            cuda_libs = find_result['output'].strip().split('\n')

            for lib_path in cuda_libs:
                if not lib_path.strip():
                    continue

                # use ctypes to get real version from CUDA runtime
                python_cmd = f'''python3 -c "
import ctypes
try:
    cuda_rt = ctypes.CDLL({repr(lib_path.strip())})
    version = ctypes.c_int()
    result = cuda_rt.cudaRuntimeGetVersion(ctypes.byref(version))
    if result == 0:
        major = int(version.value // 1000)
        minor = int(int(version.value % 1000) // 10)
        print(f'{{major}}.{{minor}}')
    else:
        exit(1)
except:
    exit(1)
"'''

                version_result = self.run_command(['bash', '-c', python_cmd],
                                                  timeout=10)
                if version_result['success'] and version_result[
                        'output'].strip():
                    version = version_result['output'].strip()
                    return version

        return None

    def _get_tensorrt_package(self, workspace: str) -> str:
        find_result = self.run_command(
            ['find', workspace, '-name', 'libnvinfer.so*', '-type', 'f'],
            timeout=30)

        if find_result['success'] and find_result['output'].strip():
            lib_path = find_result['output'].strip().split('\n')[0]

            possible_roots = []

            if '/targets/' in lib_path:
                possible_roots.append(lib_path.split('/targets/')[0])

            for trt_root in possible_roots:
                if self._verify_tensorrt(trt_root):
                    return trt_root

        return None

    def _get_compute_capability(self) -> Optional[int]:
        # use CUDA runtime API via ctypes
        find_result = self.run_command(
            ['find', '/usr', '/opt', '-name', 'libcudart.so*', '-type', 'f'],
            timeout=30)

        if find_result['success'] and find_result['output'].strip():
            cuda_libs = find_result['output'].strip().split('\n')

            for lib_path in cuda_libs:
                if not lib_path.strip():
                    continue

                python_cmd = f'''python3 -c "
import ctypes

class cudaDeviceProp(ctypes.Structure):
    _fields_ = [
        ('name', ctypes.c_char * 256),
        ('uuid', ctypes.c_ubyte * 16),
        ('luid', ctypes.c_char * 8),
        ('luidDeviceNodeMask', ctypes.c_uint),
        ('totalGlobalMem', ctypes.c_size_t),
        ('sharedMemPerBlock', ctypes.c_size_t),
        ('regsPerBlock', ctypes.c_int),
        ('warpSize', ctypes.c_int),
        ('memPitch', ctypes.c_size_t),
        ('maxThreadsPerBlock', ctypes.c_int),
        ('maxThreadsDim', ctypes.c_int * 3),
        ('maxGridSize', ctypes.c_int * 3),
        ('clockRate', ctypes.c_int),
        ('totalConstMem', ctypes.c_size_t),
        ('major', ctypes.c_int),
        ('minor', ctypes.c_int)
    ]

try:
    cuda = ctypes.CDLL('{repr(lib_path.strip())}')
    cuda.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cuda.cudaGetDeviceCount.restype = ctypes.c_int
    cuda.cudaGetDeviceProperties.argtypes = [ctypes.POINTER(cudaDeviceProp), ctypes.c_int]
    cuda.cudaGetDeviceProperties.restype = ctypes.c_int
    
    device_count = ctypes.c_int()
    result = cuda.cudaGetDeviceCount(ctypes.byref(device_count))
    
    if result == 0 and device_count.value > 0:
        prop = cudaDeviceProp()
        result = cuda.cudaGetDeviceProperties(ctypes.byref(prop), 0)
        if result == 0:
            print(f'{{prop.major}}.{{prop.minor}}')
        else:
            exit(1)
    else:
        exit(1)
except:
    exit(1)
"'''

                cap_result = self.run_command(['bash', '-c', python_cmd],
                                              timeout=15)
                if cap_result['success'] and cap_result['output'].strip():
                    try:
                        cap_str = cap_result['output'].strip()
                        major, minor = cap_str.split('.')
                        compute_cap = int(major) * 10 + int(minor)
                        return compute_cap
                    except:
                        continue

        return None

    def _verify_tensorrt(self, trt_dir: str) -> bool:
        result = self.run_command(
            ['bash', '-c', f'test -f {trt_dir}/include/NvInfer.h'], timeout=5)
        return result['success']

    def get_tensorrt_lib_dir(self, workspace: str) -> Optional[str]:
        """Return the TensorRT lib directory that contains libnvinfer.so* or None if not found."""
        trt_root = self._get_tensorrt_package(workspace)
        if not trt_root:
            return None

        arch = self._get_arch()
        candidates = [
            f"{trt_root}/lib",
            f"{trt_root}/lib64",
        ]
        if arch:
            candidates.append(f"{trt_root}/targets/{arch}-linux-gnu/lib")

        for cand in candidates:
            check = self.run_command(
                ['bash', '-c', f'ls "{cand}"/libnvinfer.so* >/dev/null 2>&1'],
                timeout=5)
            if check.get('success'):
                return cand

        return None


class EnvironmentConfig:

    def __init__(self,
                 execution_mode: str = ExecutionMode.LOCAL,
                 remote_config=None):
        self.execution_mode = execution_mode
        self.remote_config = remote_config

        self.llm_sdk_dir = os.environ.get('LLM_SDK_DIR', os.getcwd())
        self.onnx_model_dir = os.environ.get('ONNX_MODEL_DIR', 'models')
        self.engine_dir = os.environ.get('ENGINE_DIR', 'engines')
        self.build_dir = 'build'

    def get_workspace(self) -> str:
        if self.execution_mode == ExecutionMode.REMOTE and self.remote_config:
            return self.remote_config.remote_workspace
        return self.llm_sdk_dir
