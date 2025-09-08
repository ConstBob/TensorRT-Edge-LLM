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
"""
Quantization utilities for TensorRT Edge-LLM.

This module provides core quantization functionality using NVIDIA ModelOpt.
"""

from typing import Any, Dict

import modelopt.torch.quantization as mtq
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def quantize_model(
    model: torch.nn.Module,
    quant_config: Dict[str, Any],
    calib_dataloader: DataLoader,
) -> torch.nn.Module:
    """
    Quantize a PyTorch model using the specified configuration and calibration data.
    
    Args:
        model: PyTorch model to quantize
        quant_config: Quantization configuration dictionary
        calib_dataloader: DataLoader for calibration data
        
    Returns:
        Quantized PyTorch model
    """

    # Define calibration loop
    def calibrate_loop(model: torch.nn.Module) -> None:
        """
        Calibration loop that adjusts weights and scaling factors.
        
        Args:
            model: Model to calibrate
        """
        # Create progress bar for calibration
        pbar = tqdm(calib_dataloader, desc="Calibrating", unit="num_samples")
        print(f"Calibrating model on {len(calib_dataloader)} samples...")
        for data in pbar:
            if isinstance(data, dict):
                data = {k: v.to(model.device) for k, v in data.items()}
                model(**data)
            else:
                data = data.to(model.device)
                model(data)

    # Get quantization config and perform quantization
    mtq.quantize(model, quant_config, forward_loop=calibrate_loop)
    mtq.print_quant_summary(model)
    return model
