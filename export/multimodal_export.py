import math
import os
import time

import numpy as np
import onnx_graphsurgeon as gs
import torch
import torch.nn as nn
from llm_export import (export_raw_llm, get_config_path, llm_arguments,
                        surgeon_llm)
from transformers.cache_utils import DynamicCache
from utils.export_utils import WrapperModelForCausalLM, torch_to_onnx
from utils.surgeon_utils import RopeType


def multimodal_arguments():
    parser = llm_arguments()
    parser.add_argument('--model_type',
                        type=str,
                        default='qwen2_vl',
                        choices=['qwen2_vl'],
                        help="Model type")
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
        # Handles combination of text tokens with virtual tokens
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
            self.head_dim = dim / num_heads

        def forward(self,
                    hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor,
                    rotary_pos_emb: torch.Tensor = None) -> torch.Tensor:
            seq_length = hidden_states.shape[0]
            q, k, v = self.qkv(hidden_states).reshape(seq_length, 3,
                                                      self.num_heads,
                                                      -1).permute(1, 0, 2,
                                                                  3).unbind(0)
            q = apply_rotary_pos_emb_vision(q.unsqueeze(0),
                                            rotary_pos_emb).squeeze(0)
            k = apply_rotary_pos_emb_vision(k.unsqueeze(0),
                                            rotary_pos_emb).squeeze(0)

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

    class Qwen2VLVisionBlockOpt(Qwen2VLVisionBlock):

        def __init__(self, config, attn_implementation: str = "eager") -> None:
            super().__init__(config)
            self.attn = VisionAttentionOpt(config.embed_dim,
                                           num_heads=config.num_heads)

        def forward(self, hidden_states, attention_mask,
                    rotary_pos_emb) -> torch.Tensor:
            hidden_states = hidden_states + self.attn(
                self.norm1(hidden_states),
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb)
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
            for blk in self.blocks:
                hidden_states = blk(hidden_states,
                                    attention_mask=attention_mask,
                                    rotary_pos_emb=rotary_pos_emb)
            res = self.merger(hidden_states)
            return res

    model = Qwen2VisionTransformerPretrainedModelOpt._from_config(
        hf_model.config,
        torch_dtype=torch.float16,
    )
    model.load_state_dict(hf_model.state_dict())
    model.eval()

    hw = 16
    in_chans = model.config.in_chans
    temporal_patch_size = model.config.temporal_patch_size
    patch_size = model.config.patch_size
    rotary_pos_emb_dim = model.config.embed_dim // model.config.num_heads // 2

    input = torch.randn(
        (hw, in_chans * temporal_patch_size * patch_size * patch_size),
        dtype=torch.float16)
    rotary_pos_emb = torch.randn((hw, rotary_pos_emb_dim), dtype=torch.float32)
    attention_mask = torch.randn((1, hw, hw), dtype=torch.float16)

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


def export_qwen2_vl(args):
    from transformers import Qwen2VLForConditionalGeneration

    hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.torch_dir,
        torch_dtype=torch.float16,
    ).cuda()

    # 1. export visual encoder
    export_qwen2_vl_visual(hf_model.visual,
                           os.path.join(args.output_dir, "visual_enc_onnx"))

    # 2. export raw llm
    llm_output_dir = os.path.join(args.output_dir, "llm_onnx")
    if args.save_original:
        raw_onnx_dir = llm_output_dir + "_raw"
    else:
        raw_onnx_dir = llm_output_dir

    dummy_len = 10
    image_embeds = torch.randn((dummy_len, hf_model.config.hidden_size),
                               dtype=torch.float16).cuda()
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
    mrope_rotary_cos_sin = gs.Variable("mrope_rotary_cos_sin", np.float32,
                                       ['batch_size', 4194304])
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
        lm_head_precision=args.lm_head)


if __name__ == '__main__':
    parser = multimodal_arguments()
    args = parser.parse_args()
    args.config_path = get_config_path(args)

    if args.model_type == 'qwen2_vl':
        export_qwen2_vl(args)
    else:
        raise RuntimeError(f"Invalid model type {args.model_type}")
