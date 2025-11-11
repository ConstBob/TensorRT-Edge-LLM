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

import modelopt.torch.quantization as mtq
import torch
from datasets import (concatenate_datasets, get_dataset_config_names,
                      load_dataset)
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import \
    Qwen2_5_VisionTransformerPretrainedModel
from transformers.models.qwen2_vl.modeling_qwen2_vl import \
    Qwen2VisionTransformerPretrainedModel
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from ..visual_models.internvl3_model import InternVLVisionModel
from .quantization_utils import quantize_model


def get_visual_calib_dataloader(
    model,
    processor,
    dataset_dir="lmms-lab/MMMU",
):
    # There are 2 possible dataset: lmms-lab/MMMU and MMMU/MMMU. The first one does not have any configs.
    # https://huggingface.co/datasets/lmms-lab/MMMU
    # https://huggingface.co/datasets/MMMU/MMMU

    if "lmms-lab/MMMU" in dataset_dir:
        # Default use MMMU_DEV. It's recommended to use your own dataset for calibration.
        dataset = load_dataset(dataset_dir, split="dev")
    elif "MMMU" in dataset_dir:
        dataset_configs = get_dataset_config_names(dataset_dir)
        dataset = concatenate_datasets([
            load_dataset(dataset_dir, config, split="dev")
            for config in dataset_configs
        ])
    else:
        raise NotImplementedError(
            f"Unsupported dataset name or local repo directory: {dataset_dir}."
        )

    def _preprocess(data, processor):
        image_inputs = []
        for (key, value) in data.items():
            if "image" in key and isinstance(value, Image.Image):
                image_inputs.append(value.convert("RGB"))
        inputs = processor(
            text="",
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        return {
            "hidden_states": inputs["pixel_values"],
            "grid_thw": inputs["image_grid_thw"],
        }

    def _preprocess_internvl(data, processor):
        image_inputs = []
        for (key, value) in data.items():
            if "image" in key and isinstance(value, Image.Image):
                image_inputs.append(value.convert("RGB"))
        inputs = processor(images=image_inputs, )
        return {"pixel_values": inputs["pixel_values"]}

    preprocess_fn = _preprocess_internvl if isinstance(
        model, InternVLVisionModel) else _preprocess

    dataset = dataset.map(preprocess_fn,
                          batched=False,
                          fn_kwargs={"processor": processor},
                          remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=dataset.column_names)

    if isinstance(model, (
            Qwen3VLVisionModel,
            Qwen2_5_VisionTransformerPretrainedModel,
            Qwen2VisionTransformerPretrainedModel,
    )):
        # Initialize additional inputs for model
        class QwenViTDataset(Dataset):

            def __init__(self, data, model):
                self.data = data
                self.model = model

            def __len__(self):
                return len(self.data)

            def get_attention_mask(self, cu_seqlens, seq_length):
                attention_mask = torch.full([1, seq_length, seq_length],
                                            torch.finfo(self.model.dtype).min,
                                            dtype=self.model.dtype)
                for i in range(1, len(cu_seqlens)):
                    attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i],
                                   cu_seqlens[i - 1]:cu_seqlens[i]] = 0
                return attention_mask

            def __getitem__(self, idx):
                raw_data = self.data[idx]
                hidden_states = raw_data["hidden_states"].to(self.model.dtype)
                grid_thw = raw_data["grid_thw"]
                rotary_pos_emb = self.model.rot_pos_emb(grid_thw)
                cu_seqlens = torch.repeat_interleave(
                    grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                        dim=0,
                        dtype=torch.int32,
                    )
                cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
                seq_length = hidden_states.shape[0]
                attention_mask = self.get_attention_mask(
                    cu_seqlens, seq_length)
                inputs = {
                    "hidden_states": hidden_states,
                    "rotary_pos_emb": rotary_pos_emb,
                    "attention_mask": attention_mask,
                }

                if isinstance(self.model,
                              Qwen2_5_VisionTransformerPretrainedModel):
                    window_index, cu_window_seqlens = self.model.get_window_index(
                        grid_thw)
                    cu_window_seqlens = torch.tensor(
                        cu_window_seqlens,
                        dtype=torch.int32,
                    )
                    cu_window_seqlens = torch.unique_consecutive(
                        cu_window_seqlens)
                    window_attention_mask = self.get_attention_mask(
                        cu_window_seqlens, seq_length)
                    reverse_window_index = torch.argsort(window_index)
                    inputs["window_attention_mask"] = window_attention_mask
                    inputs["window_index"] = window_index
                    inputs["reverse_window_index"] = reverse_window_index
                elif isinstance(self.model, Qwen3VLVisionModel):
                    fast_pos_embed_idx, fast_pos_embed_weight = self.model.fast_pos_embed_interpolate_optimized(
                        grid_thw)
                    inputs["fast_pos_embed_idx"] = fast_pos_embed_idx
                    inputs["fast_pos_embed_weight"] = fast_pos_embed_weight

                return inputs

        dataset = QwenViTDataset(dataset, model)
    elif isinstance(model, InternVLVisionModel):

        class InternVLDataset(Dataset):

            def __init__(self, data, model):
                self.data = data
                self.model = model

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                raw_data = self.data[idx]
                pixel_values = raw_data["pixel_values"].to(self.model.dtype)
                return {"pixel_values": pixel_values}

        dataset = InternVLDataset(dataset, model)

    return dataset


def quantize_visual(model, precision, processor, dataset_dir="lmms-lab/MMMU"):
    assert isinstance(
        model, (Qwen3VLVisionModel, Qwen2_5_VisionTransformerPretrainedModel,
                Qwen2VisionTransformerPretrainedModel,
                InternVLVisionModel)), f"Invalid model type {type(model)}"
    assert precision in [
        "fp8"
    ], f"Only fp8(W8A8) is supported for visual model. You passed an unsupported precision: {precision}."
    assert "MMMU" in dataset_dir, f"Unsupported dataset name or local repo directory: {dataset_dir}."

    quant_config = mtq.FP8_DEFAULT_CFG.copy()

    # Enable FP8 MHA and FP8 GEMM
    quant_config["quant_cfg"]["*[qkv]_bmm_quantizer"] = {
        "num_bits": (4, 3),
        "axis": None
    }
    quant_config["quant_cfg"]["*softmax_quantizer"] = {
        "num_bits": (4, 3),
        "axis": None
    }

    # Disable Conv to avoid accuracy degradation
    quant_config["quant_cfg"]["nn.Conv3d"] = {"*": {"enable": False}}
    quant_config["quant_cfg"]["nn.Conv2d"] = {"*": {"enable": False}}
    data_loader = get_visual_calib_dataloader(model, processor, dataset_dir)
    quantized_model = quantize_model(model, quant_config, data_loader)
    mtq.print_quant_summary(quantized_model)
    return quantized_model
