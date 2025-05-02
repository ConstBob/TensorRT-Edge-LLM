# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import math
import os
import time

import numpy as np
import onnx_graphsurgeon as gs
import torch
import torch.nn as nn
from llm_export import (check_dtype_support, export_raw_llm, get_config_path,
                        llm_arguments, surgeon_llm)
from transformers.cache_utils import DynamicCache
from utils.export_utils import (WrapperModelForCausalLM, load_model_with_lora,
                                torch_to_onnx)
from utils.surgeon_utils import RopeType


class Qwen2VLWrapper(WrapperModelForCausalLM):
    """
    Model wrapper for the original Qwen2-VL model.
    1. Handles the combination of text tokens with virtual tokens.
    2. Handles onnx export for DynamicCache.
    """

    def __init__(self, model):
        super().__init__(model)

    def forward(self, input_ids, past_key_values, image_embeds):
        # Handles combination of text tokens with visual tokens
        image_mask = input_ids > (self.config.vocab_size - 1)

        # clip tokens in the [0, vocab_size) range
        normal_tokens = torch.where(image_mask, self.config.vocab_size - 1,
                                    input_ids)
        normal_embeddings = self.model.embed_tokens(normal_tokens)

        # put virtual tokens in the [0, max_visual_vocab_size) range
        visual_tokens = torch.where(image_mask,
                                    input_ids - self.config.vocab_size, 0)
        image_embeds = nn.functional.embedding(visual_tokens, image_embeds)

        # image_mask: [batch_size, seq_len] -> [batch_size, seq_len, 1]
        # combine the correct sources of embedding: normal/prompt
        inputs_embeds = torch.where(image_mask.unsqueeze(-1), image_embeds,
                                    normal_embeddings)

        # Convert kv cache to DynamicCache to satisfy Qwen2VLModel requirement
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        outputs = self.model(
            input_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # Convert kv cache back to list for onnx export
        past_key_values = outputs.past_key_values.to_legacy_cache()
        return logits, past_key_values


def export_qwen2_vl_visual(hf_model, output_dir):
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VisionTransformerPretrainedModel, Qwen2VLVisionBlock,
        VisionAttention, apply_rotary_pos_emb_vision)

    class VisionAttentionOpt(VisionAttention):

        def __init__(self, dim: int, num_heads: int = 16):
            super().__init__(dim, num_heads)

        def forward(self,
                    hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor,
                    position_embeddings: torch.Tensor = None) -> torch.Tensor:
            seq_length = hidden_states.shape[0]
            q, k, v = self.qkv(hidden_states).reshape(seq_length, 3,
                                                      self.num_heads,
                                                      -1).permute(1, 0, 2,
                                                                  3).unbind(0)
            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                    "removed and `position_embeddings` will be mandatory.")
                emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
                cos = emb.cos().float()
                sin = emb.sin().float()
            else:
                cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(
                self.head_dim)
            attn_weights = attn_weights + attention_mask

            attn_weights = nn.functional.softmax(attn_weights,
                                                 dim=-1,
                                                 dtype=torch.float32).to(
                                                     v.dtype)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(0, 1)
            attn_output = attn_output.reshape(seq_length, -1)
            attn_output = self.proj(attn_output)
            return attn_output

    class Qwen2VLVisionBlockOpt(Qwen2VLVisionBlock):

        def __init__(self, config, attn_implementation: str = "eager") -> None:
            super().__init__(config)
            self.attn = VisionAttentionOpt(config.embed_dim,
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
        }
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


def export_qwen2_5_vl_visual(hf_model, output_dir):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLVisionAttention,
        Qwen2_5_VLVisionBlock, apply_rotary_pos_emb_vision)

    class VisionAttentionOpt(Qwen2_5_VLVisionAttention):

        def __init__(self, dim: int, num_heads: int = 16):
            super().__init__(dim, num_heads)

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            position_embeddings: torch.Tensor,
        ) -> torch.Tensor:
            seq_length = hidden_states.shape[0]
            q, k, v = self.qkv(hidden_states).reshape(seq_length, 3,
                                                      self.num_heads,
                                                      -1).permute(1, 0, 2,
                                                                  3).unbind(0)
            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                    "removed and `position_embeddings` will be mandatory.")
                emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
                cos = emb.cos().float()
                sin = emb.sin().float()
            else:
                cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(
                self.head_dim)
            attn_weights = attn_weights + attention_mask
            attn_weights = nn.functional.softmax(attn_weights,
                                                 dim=-1,
                                                 dtype=torch.float32).to(
                                                     q.dtype)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(0, 1)
            attn_output = attn_output.reshape(seq_length, -1)
            attn_output = self.proj(attn_output)
            return attn_output

    class VisionBlockOpt(Qwen2_5_VLVisionBlock):

        def __init__(self, config, attn_implementation: str = "eager") -> None:
            super().__init__(config)
            self.attn = VisionAttentionOpt(config.hidden_size,
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
            0: 'hw//4',
        },
        'reverse_window_index': {
            0: 'hw//4',
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


def export_multimodal(args):
    if not check_dtype_support(args):
        return

    if args.model_type == 'qwen2_vl':
        from transformers import Qwen2VLForConditionalGeneration
        hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.torch_dir,
            torch_dtype=torch.float16,
        ).cuda()

        # 1. export visual encoder
        export_qwen2_vl_visual(
            hf_model.visual, os.path.join(args.output_dir, "visual_enc_onnx"))
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration
        hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.torch_dir,
            torch_dtype=torch.float16,
        ).cuda()

        # 1. export visual encoder
        # Qwen2.5-VL 3B VIT has FP16 overflow issue on certain inputs.
        # Apply ../scripts/upcast_fp32_gemm_war.py after exporting ONNX as work-around.
        export_qwen2_5_vl_visual(
            hf_model.visual, os.path.join(args.output_dir, "visual_enc_onnx"))

    # 2. export raw llm
    llm_output_dir = os.path.join(args.output_dir, "llm_onnx")
    if args.save_original:
        raw_onnx_dir = llm_output_dir + "_raw"
    else:
        raw_onnx_dir = llm_output_dir

    dummy_len = 10
    image_embeds = torch.randn((dummy_len, hf_model.config.hidden_size),
                               dtype=torch.float16).cuda()
    # Handle LoRA weights if provided
    if args.lora_mode == "static":
        hf_model, lora_config, lora_weights = load_model_with_lora(
            hf_model, args.lora_dir, args.lora_mode)
    elif args.lora_mode == "merged":
        hf_model = load_model_with_lora(hf_model, args.lora_dir,
                                        args.lora_mode)
    state_dict = export_raw_llm(hf_model,
                                raw_onnx_dir,
                                args.dtype,
                                args.config_path,
                                args.torch_dir,
                                lm_head_precision=args.lm_head,
                                wrapper_cls=Qwen2VLWrapper,
                                extra_inputs={"image_embeds": image_embeds},
                                extra_dyn_axes={
                                    "image_embeds": {
                                        0: "image_token_length"
                                    },
                                })

    # 3. surgeon llm
    mrope_rotary_cos_sin = gs.Variable(
        "mrope_rotary_cos_sin", np.float32,
        ['batch_size', hf_model.config.max_position_embeddings * 128
         ])  # head_size = 128
    mrope_position_deltas = gs.Variable("mrope_position_deltas", np.int64,
                                        ['batch_size', 1])
    surgeon_llm(
        f"{raw_onnx_dir}/model.onnx",
        llm_output_dir,
        args.dtype,
        args.mode,
        args.config_path,
        state_dict,
        args.max_seq_length,
        rope_type=RopeType.kMROPE,
        extra_plugin_inputs=[mrope_rotary_cos_sin, mrope_position_deltas],
        lm_head_precision=args.lm_head,
        lora_config=lora_config if args.lora_mode == "static" else None,
        lora_weights=lora_weights if args.lora_mode == "static" else None)


if __name__ == '__main__':
    parser = llm_arguments()
    args = parser.parse_args()
    args.config_path = get_config_path(args)

    with open(args.config_path) as f:
        args.model_type = json.load(f).get("model_type")

    if args.model_type == 'qwen2_vl' or args.model_type == 'qwen2_5_vl':
        export_multimodal(args)
    else:
        raise RuntimeError(f"Invalid model type {args.model_type}")
