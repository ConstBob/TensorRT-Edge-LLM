# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GPU inventory selection helpers for EdgeLLM orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import OrchestrationError


@dataclass(frozen=True)
class GPUDescriptor:
    """User-directed GPU selection criteria for one execution endpoint.

    Attributes:
        index: Explicit GPU index. It is mutually exclusive with discovery criteria.
        min_memory_gib: Minimum total GPU memory for discovery.
        model_name: Case-insensitive GPU-name substring for discovery.
    """

    index: int | None = None
    min_memory_gib: int | None = None
    model_name: str | None = None

    def __post_init__(self) -> None:
        """Validate the GPU selection criteria.

        Returns:
            None.

        Raises:
            ValueError: If no criterion is provided or the criteria conflict.
        """
        if self.index is not None and self.index < 0:
            raise ValueError("GPU index must be non-negative")
        if self.min_memory_gib is not None and self.min_memory_gib <= 0:
            raise ValueError("minimum GPU memory must be positive")
        if self.index is not None and (self.min_memory_gib is not None
                                       or self.model_name is not None):
            raise ValueError(
                "GPU index cannot be combined with discovery criteria")
        if self.index is None and self.min_memory_gib is None and not self.model_name:
            raise ValueError(
                "GPU descriptor requires an index, minimum memory, or model name"
            )


NVIDIA_SMI_QUERY_FIELDS = (
    "index",
    "name",
    "uuid",
    "memory.free",
    "memory.total",
    "utilization.gpu",
    "compute_cap",
)


def nvidia_smi_discovery_argv() -> list[str]:
    """Return the argv used to query GPU inventory.

    Returns:
        Command arguments producing the CSV schema consumed by ``select_gpu``.
    """
    return [
        "nvidia-smi",
        f"--query-gpu={','.join(NVIDIA_SMI_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]


def cuda_runtime_discovery_command() -> str:
    """Return a Python command that discovers one or more CUDA devices.
    This is needed because on some devices (such as Thor), nvidia-smi is 
    not present.

    Returns:
        A shell command that prints rows in the same CSV schema as
        ``nvidia_smi_discovery_argv``. The command is useful on Tegra devices
        where ``nvidia-smi`` is not present.
    """
    return r"""python3 - <<'PY'
import ctypes

class cudaDeviceProp(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("uuid", ctypes.c_ubyte * 16),
        ("luid", ctypes.c_char * 8),
        ("luidDeviceNodeMask", ctypes.c_uint),
        ("totalGlobalMem", ctypes.c_size_t),
        ("sharedMemPerBlock", ctypes.c_size_t),
        ("regsPerBlock", ctypes.c_int),
        ("warpSize", ctypes.c_int),
        ("memPitch", ctypes.c_size_t),
        ("maxThreadsPerBlock", ctypes.c_int),
        ("maxThreadsDim", ctypes.c_int * 3),
        ("maxGridSize", ctypes.c_int * 3),
        ("clockRate", ctypes.c_int),
        ("totalConstMem", ctypes.c_size_t),
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
    ]

cuda = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
cuda.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
cuda.cudaGetDeviceCount.restype = ctypes.c_int
cuda.cudaGetDeviceProperties.argtypes = [ctypes.POINTER(cudaDeviceProp), ctypes.c_int]
cuda.cudaGetDeviceProperties.restype = ctypes.c_int

device_count = ctypes.c_int()
if cuda.cudaGetDeviceCount(ctypes.byref(device_count)) != 0:
    raise SystemExit(1)

for index in range(device_count.value):
    prop = cudaDeviceProp()
    if cuda.cudaGetDeviceProperties(ctypes.byref(prop), index) != 0:
        raise SystemExit(1)
    name = prop.name.decode(errors="replace").strip("\x00") or "CUDA GPU"
    memory_mib = int(prop.totalGlobalMem // (1024 * 1024))
    print(f"{index}, {name}, {index}, {memory_mib}, {memory_mib}, 0, {prop.major}.{prop.minor}")
PY"""


@dataclass(frozen=True)
class GPUSelection:
    """One GPU inventory entry selected from ``nvidia-smi`` output.

    Attributes:
        index: CUDA GPU index reported by ``nvidia-smi``.
        name: GPU model name reported by ``nvidia-smi``.
        uuid: Stable GPU UUID used in ``CUDA_VISIBLE_DEVICES``.
        free_memory_mib: Currently free GPU memory in MiB.
        total_memory_mib: Total GPU memory in MiB.
        utilization_percent: Current GPU utilization percentage.
        compute_capability: CUDA compute capability reported as major.minor.
    """

    index: int
    name: str
    uuid: str
    free_memory_mib: int
    total_memory_mib: int
    utilization_percent: int
    compute_capability: str

    @property
    def sm_arch(self) -> str:
        """Return the CUDA SM tag for this GPU.

        Returns:
            The normalized CUDA SM tag, such as ``sm_86``.

        Raises:
            ValueError: If the GPU capability is not major.minor.
        """
        major, separator, minor = self.compute_capability.partition(".")
        if not separator or not major.isdigit() or not minor.isdigit():
            raise ValueError(
                f"invalid GPU compute capability: {self.compute_capability!r}")
        return f"sm_{int(major)}{int(minor)}"


def select_gpu(
    output: str,
    descriptor: GPUDescriptor,
) -> GPUSelection:
    """Select a GPU from ``nvidia-smi`` CSV output.

    Args:
        output: CSV output for index, name, UUID, free memory, total memory,
            utilization, and CUDA compute capability.
        descriptor: Explicit-index or discovery criteria supplied by the caller.

    Returns:
        The matching explicit GPU, or the least-used eligible GPU with the most
        free memory.

    Raises:
        OrchestrationError: If output has no GPUs or no GPU matches criteria.
    """
    candidates: list[GPUSelection] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            candidates.append(
                GPUSelection(
                    index=int(parts[0]),
                    name=parts[1],
                    uuid=parts[2],
                    free_memory_mib=int(parts[3]),
                    total_memory_mib=int(parts[4]),
                    utilization_percent=int(parts[5]),
                    compute_capability=parts[6],
                ))
        except ValueError:
            continue
    if not candidates:
        raise OrchestrationError(
            f"No GPUs found. nvidia-smi output: {output!r}")

    eligible = candidates
    if descriptor.index is not None:
        eligible = [
            candidate for candidate in candidates
            if candidate.index == descriptor.index
        ]
    if descriptor.min_memory_gib is not None:
        minimum_mib = descriptor.min_memory_gib * 1024
        eligible = [
            candidate for candidate in eligible
            if candidate.total_memory_mib >= minimum_mib
        ]
    if descriptor.model_name:
        requested_name = descriptor.model_name.casefold()
        eligible = [
            candidate for candidate in eligible
            if requested_name in candidate.name.casefold()
        ]
    if not eligible:
        inventory = ", ".join(
            f"{candidate.index}:{candidate.name}:{candidate.total_memory_mib}MiB"
            for candidate in candidates)
        raise OrchestrationError(
            f"No GPU matches {descriptor!r}. Available: {inventory}")

    if descriptor.index is not None:
        return eligible[0]
    low_utilization = [
        candidate for candidate in eligible
        if candidate.utilization_percent <= 10
    ]
    pool = low_utilization or eligible
    return min(
        pool,
        key=lambda candidate: (
            -candidate.free_memory_mib,
            candidate.utilization_percent,
            candidate.index,
        ),
    )
