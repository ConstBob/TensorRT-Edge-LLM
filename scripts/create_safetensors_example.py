# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
from safetensors.torch import load_file, save_file

# Create tensor0 with values [-2.0, -1.25, 0, 1.1, 2.0] in FP16
tensor0 = torch.tensor([-2.0, -1.25, 0.0, 1.1, 2.0],
                       dtype=torch.float16).reshape(1, 5)

# Create tensor1 with values [-3.8, 0.0, 2.7] in BF16
tensor1 = torch.tensor([-3.8, 0.0, 2.7], dtype=torch.bfloat16).reshape(1, 3)

# Create a dictionary of tensors
tensors = {"tensor0": tensor0, "tensor1": tensor1}

# Save to safetensor file
save_file(tensors, "test_safetensors.safetensors")

print("Created safetensors file with tensors:")
print("\ntensor0:")
print(f"Shape: {tensor0.shape}")
print(f"Dtype: {tensor0.dtype}")
print(f"Values: {tensor0}")

print("\ntensor1:")
print(f"Shape: {tensor1.shape}")
print(f"Dtype: {tensor1.dtype}")
print(f"Values: {tensor1}")

# Verify the ranges
tensor0_min = torch.min(tensor0).item()
tensor0_max = torch.max(tensor0).item()
tensor1_min = torch.min(tensor1).item()
tensor1_max = torch.max(tensor1).item()

print("\nVerification:")
print(f"tensor0 range: [{tensor0_min}, {tensor0_max}]")
print(f"tensor1 range: [{tensor1_min}, {tensor1_max}]")

# Load and verify the saved file
print("\nLoading saved file to verify:")
loaded_tensors = load_file("test_safetensors.safetensors")

print("\ntensor0 from file:")
print(f"Shape: {loaded_tensors['tensor0'].shape}")
print(f"Dtype: {loaded_tensors['tensor0'].dtype}")
print(f"Values: {loaded_tensors['tensor0']}")

print("\ntensor1 from file:")
print(f"Shape: {loaded_tensors['tensor1'].shape}")
print(f"Dtype: {loaded_tensors['tensor1'].dtype}")
print(f"Values: {loaded_tensors['tensor1']}")

# Verify loaded ranges
loaded_tensor0_min = torch.min(loaded_tensors['tensor0']).item()
loaded_tensor0_max = torch.max(loaded_tensors['tensor0']).item()
loaded_tensor1_min = torch.min(loaded_tensors['tensor1']).item()
loaded_tensor1_max = torch.max(loaded_tensors['tensor1']).item()

print("\nVerification of loaded tensors:")
print(f"tensor0 range: [{loaded_tensor0_min}, {loaded_tensor0_max}]")
print(f"tensor1 range: [{loaded_tensor1_min}, {loaded_tensor1_max}]")
