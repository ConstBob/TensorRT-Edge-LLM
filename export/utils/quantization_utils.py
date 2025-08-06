# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import time

import modelopt.torch.quantization as mtq
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor


def get_calib_dataloader(dataset_name_or_dir="cnn_dailymail",
                         tokenizer=None,
                         batch_size=1,
                         calib_size=512,
                         block_size=512):
    print(f"Loading calibration dataset from {dataset_name_or_dir}")
    if "cnn_dailymail" in dataset_name_or_dir:
        dataset = load_dataset(dataset_name_or_dir,
                               name="3.0.0",
                               split="train")
        dataset = dataset["article"][:calib_size]
    elif os.path.isdir(dataset_name_or_dir):
        print(
            f"Recognized local dataset repo {dataset_name_or_dir} for calibration; "
            "assuming the calibration data are in the train split and text column."
        )
        dataset = load_dataset(dataset_name_or_dir, split="train")
        dataset = dataset["text"][:calib_size]
    else:
        raise NotImplementedError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}."
        )

    batch_encoded = tokenizer.batch_encode_plus(dataset,
                                                return_tensors="pt",
                                                padding=True,
                                                truncation=True,
                                                max_length=block_size)

    calib_dataloader = DataLoader(batch_encoded["input_ids"],
                                  batch_size=batch_size,
                                  shuffle=False)

    return calib_dataloader


def get_quant_config(precision, lm_head_precision="fp16"):

    if precision == "fp8":
        quant_cfg = mtq.FP8_DEFAULT_CFG
    # Include both int4 plugin and ootb solution
    elif "int4" in precision:
        quant_cfg = mtq.INT4_AWQ_CFG

    elif precision == "nvfp4":
        if hasattr(mtq, "NVFP4_AWQ_FULL_CFG"):
            quant_cfg = mtq.NVFP4_AWQ_FULL_CFG

    if lm_head_precision == "fp8":
        quant_cfg["quant_cfg"]["*lm_head.input_quantizer"] = {
            "num_bits": (4, 3),
            "axis": None
        }
        quant_cfg["quant_cfg"]["*lm_head.weight_quantizer"] = {
            "num_bits": (4, 3),
            "axis": None
        }
    elif "int4" in lm_head_precision:
        quant_cfg["quant_cfg"]["*lm_head.weight_quantizer"] = {
            "num_bits": 4,
            "block_sizes": {
                -1: 128
            },
            "enable": True
        }
    elif lm_head_precision == "nvfp4":
        quant_cfg["quant_cfg"]["*lm_head.input_quantizer"] = {
            "num_bits": (2, 1),
            "block_sizes": {
                -1: 16,
                "type": "dynamic",
                "scale_bits": (4, 3)
            },
            "axis": None,
            "enable": True,
        }
        quant_cfg["quant_cfg"]["*lm_head.weight_quantizer"] = {
            "num_bits": (2, 1),
            "block_sizes": {
                -1: 16,
                "type": "dynamic",
                "scale_bits": (4, 3)
            },
            "axis": None,
            "enable": True,
        }
    return quant_cfg


def _quantize_model(model, quant_config, calib_dataloader=None):
    """
    The calibration loop for the model can be setup using the modelopt API.

    Example usage:
    from modelopt.torch.utils.dataset_utils import create_forward_loop
    model = ...  # Initialize the model
    tokenizer = ...  # Initialize the tokenizer
    quant_cfg = ...  # Setup quantization configuration
    forward_loop = create_forward_loop(model=model, dataset_name="cnn_dailymail", tokenizer=tokenizer)
    mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    """

    def calibrate_loop(model):
        """Adjusts weights and scaling factors based on selected algorithms."""
        for idx, data in enumerate(calib_dataloader):
            if idx % 10 == 0:
                print(f"Calibrating batch {idx}...")
            if isinstance(data, dict):
                data = {k: v.to(model.device) for k, v in data.items()}
                model(**data)
            else:
                data = data.to(model.device)
                model(data)

    print("Starting quantization...")
    start_time = time.time()
    mtq.quantize(model, quant_config, forward_loop=calibrate_loop)
    end_time = time.time()
    print(f"Quantization finishes in {end_time - start_time}s.")

    return model


def quantize(model,
             tokenizer,
             precision,
             lm_head_precision="fp16",
             dataset_dir=None):
    """
    Quantize the PyTorch model to fp8 or int4_awq
    """
    assert precision in [
        "fp8", "int4", "nvfp4", "int4_ootb"
    ], f"Only fp8(W8A8), int4(W4A16) and nvfp4(W4A4) is supported. You passed an unsupported precision: {precision}."

    assert lm_head_precision in [
        "fp16"
    ], f"Only fp16(unquantized) is supported for lm_head. You passed an unsupported precision: {lm_head_precision}."

    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not dataset_dir:
        dataset_dir = "cnn_dailymail"

    if "int4" in precision:
        batch_size = 16
    else:
        batch_size = 1
    data_loader = get_calib_dataloader(dataset_name_or_dir=dataset_dir,
                                       tokenizer=tokenizer,
                                       batch_size=batch_size)
    quant_config = get_quant_config(precision, lm_head_precision)
    quantized_model = _quantize_model(model, quant_config, data_loader)
    mtq.print_quant_summary(quantized_model)
    return quantized_model


def get_vit_calib_dataloader(
    model,
    model_type,
    dataset_name_or_dir="MMMU",
    torch_dir=None,
):
    # Default use MMMU_DEV. It's recommended to use your own dataset for calibration.
    if dataset_name_or_dir == "MMMU":
        dataset = load_dataset("lmms-lab/MMMU", split="dev")

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

        preprocess_fn = _preprocess_internvl if model_type == "internvl" else _preprocess
        if model_type == "internvl":
            processor = AutoProcessor.from_pretrained(torch_dir)
            processor = processor.image_processor
        else:
            # Limit pixels to reasonable size. Too large images will cause OOM.
            processor = AutoProcessor.from_pretrained(torch_dir,
                                                      min_pixels=128 * 28 * 28,
                                                      max_pixels=2048 * 28 *
                                                      28)

        dataset = dataset.map(preprocess_fn,
                              batched=False,
                              fn_kwargs={"processor": processor},
                              remove_columns=dataset.column_names)
        dataset.set_format(type="torch", columns=dataset.column_names)
    else:
        raise NotImplementedError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}."
        )

    if model_type in ["qwen2_vl", "qwen2_5_vl"]:
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

                if model_type == "qwen2_5_vl":
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

                return inputs

        dataset = QwenViTDataset(dataset, model)
    elif model_type == "internvl":

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
    else:
        raise NotImplementedError(f"Invalid model type {model_type}")

    return dataset


def quantize_visual(model, precision, model_type, torch_dir):
    assert precision in [
        "fp8"
    ], f"Only fp8(W8A8) is recommended for vit. You passed an unsupported precision: {precision}."

    # Set quantization config, this will only enable FP8 GEMMs that not belong to multihead attention modules.
    # Also disable Conv3d to avoid accuracy degradation.
    quant_config = mtq.FP8_DEFAULT_CFG
    quant_config["quant_cfg"]["nn.Conv3d"] = {"*": {"enable": False}}
    quant_config["quant_cfg"]["nn.Conv2d"] = {"*": {"enable": False}}

    # Disable `attn.proj` layers to avoid performance degradation.
    quant_config["quant_cfg"]["*attn.proj*"] = {"enable": False}
    quant_config["quant_cfg"]["*attention.proj*"] = {"enable": False}

    data_loader = get_vit_calib_dataloader(model,
                                           model_type,
                                           torch_dir=torch_dir)
    quantized_model = _quantize_model(model, quant_config, data_loader)
    mtq.print_quant_summary(quantized_model)
    return quantized_model
