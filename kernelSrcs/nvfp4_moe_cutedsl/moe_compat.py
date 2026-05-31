# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Small MoE compatibility shim for standalone CuTeDSL exports."""

from enum import IntEnum


class ActivationType(IntEnum):
    """Activation enum values used by the grouped MoE kernels."""

    InvalidType = 0
    Identity = 1
    Gelu = 2
    Relu = 3
    Silu = 4
    Swiglu = 5
    Geglu = 6
    SwigluBias = 7
    Relu2 = 8


def is_gated_activation(activation_type: ActivationType) -> bool:
    return activation_type in (
        ActivationType.Swiglu,
        ActivationType.SwigluBias,
        ActivationType.Geglu,
    )
