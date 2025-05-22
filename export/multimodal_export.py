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
import onnx
import onnx_graphsurgeon as gs
import torch
import torch.nn as nn
from llm_export import (check_dtype_support, export_raw_llm, get_config_path,
                        llm_arguments, surgeon_llm)
from transformers.cache_utils import DynamicCache
from utils.export_utils import (WrapperModelForCausalLM, load_model_with_lora,
                                torch_to_onnx, QwenVisionAttention)
from utils.surgeon_utils import RopeType
from utils.quantization_utils import quantize_visual


def multimodal_arguments():
    parser = llm_arguments()
    parser.add_argument('--visualOnly',
                        action='store_true',
                        default=False,
                        help="Export visual encoder only")
    parser.add_argument('--llmOnly',
                        action='store_true',
                        default=False,
                        help="Export llm only")
    parser.add_argument('--visualType',
                        type=str,
                        default="fp16",
                        choices=["fp16", "fp8"],
                        help="The precision of visual encoder onnx export")
    return parser


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
        model = quantize_visual(model, dtype, hf_model.config.model_type, torch_dir)
    
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
        'input': {0: 'hw'},
        'rotary_pos_emb': {0: 'hw'},
        'attention_mask': {1: 'hw', 2: 'hw'},
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
        Qwen2_5_VLVisionBlock, Qwen2_5_VLPatchMerger)
    
    class Qwen2_5_VLMLPWar(Qwen2_5_VLMLP):
        """Cast Down Proj to FP32 to avoid FP16 overflow"""
        def __init__(self, config, bias: bool = False):
            super().__init__(config, bias)
        
        def forward(self, hidden_state):
            hidden_state = self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
            hidden_state = hidden_state.to(torch.float32)
            self.down_proj.weight.data = self.down_proj.weight.data.to(torch.float32)
            self.down_proj.bias.data = self.down_proj.bias.data.to(torch.float32)
            return self.down_proj(hidden_state)
    
    class VisionBlockWar(Qwen2_5_VLVisionBlock):
        "WAR for Qwen2.5-VL 3B FP16 overflow"
        def __init__(self, config, attn_implementation: str = "eager") -> None:
            super().__init__(config)
            self.attn = QwenVisionAttention(config.hidden_size, num_heads=config.num_heads)
            self.mlp = Qwen2_5_VLMLPWar(config, bias=True)

        def forward(self, hidden_states, attention_mask,
                    position_embeddings) -> torch.Tensor:
            hidden_states = hidden_states + self.attn(
                self.norm1(hidden_states),
                attention_mask=attention_mask,
                position_embeddings=position_embeddings)
            hidden_states = hidden_states.to(torch.float32) + self.mlp(self.norm2(hidden_states))
            return hidden_states
    
    class Qwen2_5_VLPatchMergerWar(Qwen2_5_VLPatchMerger):
        "WAR for Qwen2.5-VL 3B FP16 overflow"
        def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
            super().__init__(dim, context_dim, spatial_merge_size)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.mlp(self.ln_q(x).to(torch.float16).view(-1, self.hidden_size))
            return x

    class VisionBlockOpt(Qwen2_5_VLVisionBlock):
        def __init__(self, config, attn_implementation: str = "eager") -> None:
            super().__init__(config)
            self.attn = QwenVisionAttention(config.hidden_size, num_heads=config.num_heads)

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
                self.blocks[-1] = VisionBlockWar(config, config._attn_implementation)
                self.merger = Qwen2_5_VLPatchMergerWar(
                    dim=config.out_hidden_size,
                    context_dim=config.hidden_size,
                    spatial_merge_size=config.spatial_merge_size,
                )

        def forward(
            self,
            hidden_states: torch.Tensor,
            rotary_pos_emb: torch.Tensor,
            attention_mask: torch.Tensor,
            window_attention_mask: torch.Tensor,
            window_index: torch.Tensor,
            reverse_window_index: torch.Tensor
        ) -> torch.Tensor:
            hidden_states = self.patch_embed(hidden_states)

            seq_len, _ = hidden_states.size()
            hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
            hidden_states = hidden_states[window_index, :, :]
            hidden_states = hidden_states.reshape(seq_len, -1)
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
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
        model = quantize_visual(model, dtype, hf_model.config.model_type, torch_dir)
    
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
        'input': {0: 'hw'},
        'rotary_pos_emb': {0: 'hw'},
        'attention_mask': {1: 'hw', 2: 'hw'},
        'window_attention_mask': {1: 'hw', 2: 'hw'},
        'window_index': {0: 'hw//4'},
        'reverse_window_index': {0: 'hw//4'}
    }

    start_time = time.time()
    torch_to_onnx(
        model,
        (input, rotary_pos_emb, attention_mask, window_attention_mask, window_index, reverse_window_index),
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


def export_llm(hf_model, args):
    if not check_dtype_support(args):
        return
    
    # 1. export raw llm
    llm_output_dir = os.path.join(args.output_dir, f"llm_onnx_{args.dtype}")
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

    # 2. surgeon llm
    mrope_rotary_cos_sin = gs.Variable(
        "mrope_rotary_cos_sin", np.float32,
        ['batch_size', hf_model.config.max_position_embeddings * 128]
    )  # head_size = 128
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


def export_visual(hf_model, args):
    # 1. Export raw onnx
    onnx_dir = os.path.join(args.output_dir, f"visual_enc_onnx_{args.visualType}")
    
    if args.model_type == 'qwen2_vl':
        export_qwen2_vl_visual(hf_model, onnx_dir, args.visualType, args.torch_dir)
    elif args.model_type == 'qwen2_5_vl':
        export_qwen2_5_vl_visual(hf_model, onnx_dir, args.visualType, args.torch_dir)
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
                
        onnx.save_model(
            onnx_model,
            onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"onnx_model.data",
            convert_attribute=True
        )
        

def load_hf_model(args):
    if args.model_type == 'qwen2_vl':
        from transformers import Qwen2VLForConditionalGeneration
        hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.torch_dir,
            torch_dtype=torch.float16,
        )
    elif args.model_type == 'qwen2_5_vl':
        from transformers import Qwen2_5_VLForConditionalGeneration
        hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.torch_dir,
            torch_dtype=torch.float16,
        )
    else:
        raise ValueError(f"Invalid model type {args.model_type}")

    return hf_model.eval().cuda()


def main(args):
    hf_model = None
    exportVisual = True
    exportLLM = True
    if args.visualOnly:
        exportLLM = False
    if args.llmOnly:
        exportVisual = False
        
    if exportVisual:
        if hf_model is None:
            hf_model = load_hf_model(args)
        export_visual(hf_model.visual, args)
        
    if exportLLM:
        if hf_model is None:
            hf_model = load_hf_model(args)
        export_llm(hf_model, args)


if __name__ == '__main__':
    parser = multimodal_arguments()
    args = parser.parse_args()
    args.config_path = get_config_path(args)

    with open(args.config_path) as f:
        args.model_type = json.load(f).get("model_type")

    if args.model_type in ['qwen2_vl', 'qwen2_5_vl']:
        main(args)
    else:
        raise ValueError(f"Invalid model type {args.model_type}")
