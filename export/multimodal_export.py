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

import onnx
import onnx_graphsurgeon as gs
import torch
import torch.nn as nn
from llm_export import get_config_path, llm_arguments
from utils.export_utils import ModelLoader, QwenVisionAttention, torch_to_onnx
from utils.quantization_utils import quantize_visual


def multimodal_arguments():
    parser = llm_arguments()
    parser.add_argument('--visualType',
                        type=str,
                        default="fp16",
                        choices=["fp16", "fp8"],
                        help="The precision of visual encoder onnx export")
    return parser


def export_qwen2_vl_visual(hf_model, output_dir, dtype, torch_dir):
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VisionTransformerPretrainedModel, Qwen2VLVisionBlock)

    class Qwen2VLVisionBlockOpt(Qwen2VLVisionBlock):

        def __init__(self, config, attn_implementation: str = "eager") -> None:
            super().__init__(config)
            self.attn = QwenVisionAttention(config.embed_dim,
                                            num_heads=config.num_heads)

        def forward(self, hidden_states, attention_mask,
                    position_embeddings) -> torch.Tensor:
            hidden_states = hidden_states + self.attn(
                self.norm1(hidden_states),
                attention_mask=attention_mask,
                position_embeddings=position_embeddings)
            hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
            return hidden_states

    class Qwen2VisionTransformerPretrainedModelOpt(
            Qwen2VisionTransformerPretrainedModel):

        def __init__(self, config) -> None:
            super().__init__(config)
            self.blocks = nn.ModuleList([
                Qwen2VLVisionBlockOpt(config, config._attn_implementation)
                for _ in range(config.depth)
            ])

        def forward(self, hidden_states: torch.Tensor,
                    rotary_pos_emb: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
            hidden_states = self.patch_embed(hidden_states)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())
            for blk in self.blocks:
                hidden_states = blk(hidden_states,
                                    attention_mask=attention_mask,
                                    position_embeddings=position_embeddings)
            res = self.merger(hidden_states)
            return res

    model = Qwen2VisionTransformerPretrainedModelOpt._from_config(
        hf_model.config,
        torch_dtype=torch.float16,
    )
    model.load_state_dict(hf_model.state_dict())
    model.eval().cuda()

    # Quantize
    if dtype == "fp8":
        model = quantize_visual(model, dtype, hf_model.config.model_type,
                                torch_dir)

    hw = 16
    in_chans = model.config.in_chans
    temporal_patch_size = model.config.temporal_patch_size
    patch_size = model.config.patch_size
    rotary_pos_emb_dim = model.config.embed_dim // model.config.num_heads // 2

    input = torch.randn(
        (hw, in_chans * temporal_patch_size * patch_size * patch_size),
        dtype=torch.float16,
        device=model.device)
    rotary_pos_emb = torch.randn((hw, rotary_pos_emb_dim),
                                 dtype=torch.float32,
                                 device=model.device)
    attention_mask = torch.randn((1, hw, hw),
                                 dtype=torch.float16,
                                 device=model.device)

    dynamic_axes = {
        'input': {
            0: 'hw'
        },
        'rotary_pos_emb': {
            0: 'hw'
        },
        'attention_mask': {
            1: 'hw',
            2: 'hw'
        },
    }

    start_time = time.time()
    torch_to_onnx(
        model,
        (input, rotary_pos_emb, attention_mask),
        output_dir,
        "model.onnx",
        input_names=["input", "rotary_pos_emb", "attention_mask"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    end_time = time.time()
    print(
        f"Qwen2-VL visual encoder ONNX Export from torch completed in {end_time - start_time}s. ONNX file is saved to {output_dir}."
    )


def export_qwen2_5_vl_visual(hf_model, output_dir, dtype, torch_dir):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLMLP,
        Qwen2_5_VLPatchMerger, Qwen2_5_VLVisionBlock)

    class Qwen2_5_VLMLPWar(Qwen2_5_VLMLP):
        """Cast Down Proj to FP32 to avoid FP16 overflow"""

        def __init__(self, config, bias: bool = False):
            super().__init__(config, bias)

        def forward(self, hidden_state):
            hidden_state = self.act_fn(
                self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
            hidden_state = hidden_state.to(torch.float32)
            self.down_proj.weight.data = self.down_proj.weight.data.to(
                torch.float32)
            self.down_proj.bias.data = self.down_proj.bias.data.to(
                torch.float32)
            return self.down_proj(hidden_state)

    class VisionBlockWar(Qwen2_5_VLVisionBlock):
        "WAR for Qwen2.5-VL 3B FP16 overflow"

        def __init__(self, config, attn_implementation: str = "eager") -> None:
            super().__init__(config)
            self.attn = QwenVisionAttention(config.hidden_size,
                                            num_heads=config.num_heads)
            self.mlp = Qwen2_5_VLMLPWar(config, bias=True)

        def forward(self, hidden_states, attention_mask,
                    position_embeddings) -> torch.Tensor:
            hidden_states = hidden_states + self.attn(
                self.norm1(hidden_states),
                attention_mask=attention_mask,
                position_embeddings=position_embeddings)
            hidden_states = hidden_states.to(torch.float32) + self.mlp(
                self.norm2(hidden_states))
            return hidden_states

    class Qwen2_5_VLPatchMergerWar(Qwen2_5_VLPatchMerger):
        "WAR for Qwen2.5-VL 3B FP16 overflow"

        def __init__(self,
                     dim: int,
                     context_dim: int,
                     spatial_merge_size: int = 2) -> None:
            super().__init__(dim, context_dim, spatial_merge_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.mlp(
                self.ln_q(x).to(torch.float16).view(-1, self.hidden_size))
            return x

    class VisionBlockOpt(Qwen2_5_VLVisionBlock):

        def __init__(self, config, attn_implementation: str = "eager") -> None:
            super().__init__(config)
            self.attn = QwenVisionAttention(config.hidden_size,
                                            num_heads=config.num_heads)

        def forward(self, hidden_states, attention_mask,
                    position_embeddings) -> torch.Tensor:
            hidden_states = hidden_states + self.attn(
                self.norm1(hidden_states),
                attention_mask=attention_mask,
                position_embeddings=position_embeddings)
            hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
            return hidden_states

    class VisionTransformerPretrainedModelOpt(
            Qwen2_5_VisionTransformerPretrainedModel):

        def __init__(self, config) -> None:
            super().__init__(config)
            self.blocks = nn.ModuleList([
                VisionBlockOpt(config, config._attn_implementation)
                for _ in range(config.depth)
            ])
            # Qwen2.5-VL 3B VIT has overflow issue with FP16 and only happens in /blocks.31/mlp/down_proj
            # Apply WAR to cast /blocks.31/mlp/down_proj to FP32 to avoid this issue.
            if config.out_hidden_size == 2048:
                self.blocks[-1] = VisionBlockWar(config,
                                                 config._attn_implementation)
                self.merger = Qwen2_5_VLPatchMergerWar(
                    dim=config.out_hidden_size,
                    context_dim=config.hidden_size,
                    spatial_merge_size=config.spatial_merge_size,
                )

        def forward(self, hidden_states: torch.Tensor,
                    rotary_pos_emb: torch.Tensor, attention_mask: torch.Tensor,
                    window_attention_mask: torch.Tensor,
                    window_index: torch.Tensor,
                    reverse_window_index: torch.Tensor) -> torch.Tensor:
            hidden_states = self.patch_embed(hidden_states)

            seq_len, _ = hidden_states.size()
            hidden_states = hidden_states.reshape(
                seq_len // self.spatial_merge_unit, self.spatial_merge_unit,
                -1)
            hidden_states = hidden_states[window_index, :, :]
            hidden_states = hidden_states.reshape(seq_len, -1)
            rotary_pos_emb = rotary_pos_emb.reshape(
                seq_len // self.spatial_merge_unit, self.spatial_merge_unit,
                -1)
            rotary_pos_emb = rotary_pos_emb[window_index, :, :]
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())

            for layer_num, blk in enumerate(self.blocks):
                if layer_num in self.fullatt_block_indexes:
                    attention_mask_now = attention_mask
                else:
                    attention_mask_now = window_attention_mask
                hidden_states = blk(hidden_states,
                                    attention_mask=attention_mask_now,
                                    position_embeddings=position_embeddings)

            hidden_states = self.merger(hidden_states)
            hidden_states = hidden_states[reverse_window_index, :]

            return hidden_states

    model = VisionTransformerPretrainedModelOpt._from_config(
        hf_model.config,
        torch_dtype=torch.float16,
    )
    model.load_state_dict(hf_model.state_dict())
    model.eval().cuda()

    # Quantize
    if dtype == "fp8":
        model = quantize_visual(model, dtype, hf_model.config.model_type,
                                torch_dir)

    # Dummy input sizes will be replaced by dynamic axes
    grid_t = 1
    grid_h = 8
    grid_w = 16
    hw = grid_t * grid_h * grid_w
    in_chans = model.config.in_chans
    temporal_patch_size = model.config.temporal_patch_size
    patch_size = model.config.patch_size
    rotary_pos_emb_dim = model.config.hidden_size // model.config.num_heads // 2

    input = torch.randn(
        (hw, in_chans * temporal_patch_size * patch_size * patch_size),
        dtype=torch.float16,
        device=model.device)
    rotary_pos_emb = torch.randn((hw, rotary_pos_emb_dim),
                                 dtype=torch.float32,
                                 device=model.device)
    attention_mask = torch.randn((1, hw, hw),
                                 dtype=torch.float16,
                                 device=model.device)
    window_attention_mask = torch.randn((1, hw, hw),
                                        dtype=torch.float16,
                                        device=model.device)
    window_index = torch.arange(hw // 4,
                                dtype=torch.int64,
                                device=model.device)
    window_index = window_index.reshape(grid_t, grid_h // 8, 4, grid_w // 8, 4)
    window_index = window_index.permute(0, 1, 3, 2, 4).reshape(-1)
    # TensorRT TopK max K = 3840. Compute reverse index outside to support longer image tokens.
    reverse_window_index = torch.argsort(window_index)

    dynamic_axes = {
        'input': {
            0: 'hw'
        },
        'rotary_pos_emb': {
            0: 'hw'
        },
        'attention_mask': {
            1: 'hw',
            2: 'hw'
        },
        'window_attention_mask': {
            1: 'hw',
            2: 'hw'
        },
        'window_index': {
            0: 'hw//4'
        },
        'reverse_window_index': {
            0: 'hw//4'
        }
    }

    start_time = time.time()
    torch_to_onnx(
        model,
        (input, rotary_pos_emb, attention_mask, window_attention_mask,
         window_index, reverse_window_index),
        output_dir,
        "model.onnx",
        input_names=[
            "input", "rotary_pos_emb", "attention_mask",
            "window_attention_mask", "window_index", "reverse_window_index"
        ],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    end_time = time.time()
    print(
        f"Qwen2.5-VL visual encoder ONNX Export from torch completed in {end_time - start_time}s. ONNX file is saved to {output_dir}."
    )


def export_internvl3_visual(hf_model, output_dir, dtype, torch_dir):

    class InternVLVisionModel(torch.nn.Module):

        def __init__(self, hf_model):
            super().__init__()
            self.channels = hf_model.config.vision_config.num_channels
            self.image_size = hf_model.config.vision_config.image_size
            self.vision_tower = hf_model.model.vision_tower
            self.multi_modal_projector = hf_model.model.multi_modal_projector
            self.downsample_ratio = hf_model.config.downsample_ratio
            self.pixel_shuffle = hf_model.model.pixel_shuffle
            self.device = hf_model.device
            self.dtype = hf_model.dtype

        def forward(self, pixel_values):
            pixel_values = pixel_values.reshape(-1, self.channels,
                                                self.image_size[0],
                                                self.image_size[1])
            vision_features = self.vision_tower(pixel_values).last_hidden_state
            vision_features = vision_features[:, 1:, :]
            channels = vision_features.shape[1]
            feature_size = int(channels**0.5)
            batch_size = vision_features.shape[0]

            # Reshape tensor to spatial dimensions
            vision_features = vision_features.reshape(batch_size, feature_size,
                                                      feature_size, -1)

            # Apply downsampling using pixel shuffle
            vision_features = self.pixel_shuffle(
                vision_features, scale_factor=self.downsample_ratio)

            # Reshape tensor to prepare for projection
            vision_features = vision_features.reshape(
                batch_size, -1, vision_features.shape[-1])

            # Project features through multi-modal projector
            vision_features = self.multi_modal_projector(vision_features)
            return vision_features.reshape(-1, vision_features.shape[-1])

    model = InternVLVisionModel(hf_model)

    # Quantize
    if dtype == "fp8":
        model = quantize_visual(model, dtype, hf_model.config.model_type,
                                torch_dir)

    # dummy input
    hw = 32 * 32
    in_chans = hf_model.config.vision_config.num_channels
    patch_size = hf_model.config.vision_config.patch_size
    input = torch.randn((hw, in_chans * patch_size[0] * patch_size[1]),
                        dtype=torch.float16,
                        device=model.device)
    dynamic_axes = {
        'input': {
            0: 'hw'
        },
    }

    start_time = time.time()
    torch_to_onnx(
        model,
        (input),
        output_dir,
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    end_time = time.time()
    print(
        f"InternVL3 visual encoder ONNX Export from torch completed in {end_time - start_time}s. ONNX file is saved to {output_dir}."
    )


def export_visual(hf_model, args):
    # 1. Export raw onnx
    onnx_dir = os.path.join(args.output_dir,
                            f"visual_enc_onnx_{args.visualType}")

    if args.model_type == 'qwen2_vl':
        export_qwen2_vl_visual(hf_model.visual, onnx_dir, args.visualType,
                               args.torch_dir)
    elif args.model_type == 'qwen2_5_vl':
        export_qwen2_5_vl_visual(hf_model.visual, onnx_dir, args.visualType,
                                 args.torch_dir)
    elif args.model_type == 'internvl':
        export_internvl3_visual(hf_model, onnx_dir, args.visualType,
                                args.torch_dir)
    else:
        raise ValueError(f"Invalid model type {args.model_type}")

    # 2. Surgeon onnx for FP8
    if args.visualType == "fp8":
        from utils.surgeon_utils import fold_fp8_qdq_to_dq

        onnx_path = os.path.join(onnx_dir, "model.onnx")
        graph = gs.import_onnx(onnx.load(onnx_path))
        graph = fold_fp8_qdq_to_dq(graph)
        graph.fold_constants().cleanup().toposort()
        onnx_model = gs.export_onnx(graph)

        print(
            f"Saving ONNX files in {onnx_dir}. All existing ONNX in the folder will be overwritten."
        )
        for filename in os.listdir(onnx_dir):
            file_path = os.path.join(onnx_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    if ".json" not in file_path:
                        os.unlink(file_path)

            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        onnx.save_model(onnx_model,
                        onnx_path,
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        location=f"onnx_model.data",
                        convert_attribute=True)


def main(args):
    model_loader = ModelLoader(args.torch_dir, args.config_path)
    args.model_type = model_loader.get_model_type()
    if args.model_type not in ['qwen2_vl', 'qwen2_5_vl', 'internvl']:
        raise ValueError(f"Invalid model type {args.model_type}")
    hf_model = model_loader.load_model()
    export_visual(hf_model, args)


if __name__ == '__main__':
    parser = multimodal_arguments()
    args = parser.parse_args()
    args.config_path = get_config_path(args)
    main(args)
