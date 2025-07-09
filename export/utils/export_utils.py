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
from typing import Optional

import numpy as np
import onnx_graphsurgeon as gs
import torch
from peft import PeftConfig, PeftModel, load_peft_weights
from torch import Tensor
from torch.nn import Embedding
from transformers import DynamicCache
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    VisionAttention, apply_rotary_pos_emb_vision)
from utils.surgeon_utils import RopeType


class PromptTuningEmbedding(Embedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weight: Tensor,
        padding_idx: Optional[int] = None,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)

        self.vocab_size = num_embeddings
        self.weight = weight

    def forward(self, input_ids, image_embeds):
        # Handles combination of text tokens with visual tokens
        image_mask = input_ids > (self.vocab_size - 1)

        # clip tokens in the [0, vocab_size) range
        normal_tokens = torch.where(image_mask, self.vocab_size - 1, input_ids)
        normal_embeddings = torch.nn.functional.embedding(
            normal_tokens, self.weight.data)

        # put virtual tokens in the [0, max_visual_vocab_size) range
        visual_tokens = torch.where(image_mask, input_ids - self.vocab_size, 0)
        image_embeds = torch.nn.functional.embedding(visual_tokens,
                                                     image_embeds)

        # image_mask: [batch_size, seq_len] -> [batch_size, seq_len, 1]
        # combine the correct sources of embedding: normal/prompt
        inputs_embeds = torch.where(image_mask.unsqueeze(-1), image_embeds,
                                    normal_embeddings)
        return inputs_embeds


class WrapperModelForCausalLM(torch.nn.Module):
    """
    Wrapper Model to ensure all models have the same I/O
    """

    def __init__(self, model, eagle3=False, use_prompt_tuning=False):
        super().__init__()
        try:
            self.model = model.model
        except:
            self.model = model
        self.lm_head = model.lm_head
        self.config = model.config
        self.use_prompt_tuning = use_prompt_tuning
        self.eagle3 = eagle3
        if self.use_prompt_tuning:
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Embedding):
                    padding_idx = getattr(module, 'padding_idx', None)
                    self.prompt_tuning_embedding = PromptTuningEmbedding(
                        module.num_embeddings, module.embedding_dim,
                        module.weight, padding_idx)
                    break

    def forward(self,
                input_ids,
                past_key_values,
                image_embeds: Optional[Tensor] = None):
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        if self.use_prompt_tuning:
            inputs_embeds = self.prompt_tuning_embedding(
                input_ids, image_embeds)
            input_ids = None
        else:
            inputs_embeds = None
        outputs = self.model(input_ids=input_ids,
                             past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds,
                             use_cache=True)
        hidden_states = outputs[0]
        past_key_values = outputs.past_key_values.to_legacy_cache()
        logits = self.lm_head(hidden_states)
        return logits, past_key_values


class WrapperEagleBaseModelForCausalLM(WrapperModelForCausalLM):
    """
    Wrapper Model to ensure all models have the same I/O
    """

    def __init__(self, model, eagle3=False, use_prompt_tuning=False):
        # Call parent constructor to set up prompt_tuning_embedding
        super().__init__(model,
                         eagle3=eagle3,
                         use_prompt_tuning=use_prompt_tuning)

    def forward(self,
                input_ids,
                past_key_values,
                image_embeds: Optional[Tensor] = None):
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        # Handle prompt tuning if enabled
        if self.use_prompt_tuning:
            inputs_embeds = self.prompt_tuning_embedding(
                input_ids, image_embeds)
            input_ids = None
        else:
            inputs_embeds = None

        outputs = self.model(input_ids=input_ids,
                             past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds,
                             output_hidden_states=True)

        last_hidden_states = outputs[0]
        all_hidden_states = outputs['hidden_states']
        if self.eagle3:
            idx = [
                2, ((len(all_hidden_states) - 1) // 2),
                len(all_hidden_states) - 4
            ]
            hidden_states_0 = all_hidden_states[idx[0]]
            hidden_states_1 = all_hidden_states[idx[1]]
            hidden_states_2 = all_hidden_states[idx[2]]
            hidden_states = torch.cat(
                [hidden_states_0, hidden_states_1, hidden_states_2], dim=-1)

        past_key_values = outputs.past_key_values.to_legacy_cache()
        hidden_states_reshape = last_hidden_states.reshape(
            -1, last_hidden_states.size(2))
        logits = self.lm_head(hidden_states_reshape)
        if self.eagle3:
            return logits, past_key_values, hidden_states
        else:
            return logits, past_key_values, last_hidden_states


class WrapperEagleDraftModelForCausalLM(WrapperModelForCausalLM):
    """
    Wrapper Model to ensure all models have the same I/O
    """

    def __init__(self, model, eagle3=False, use_prompt_tuning=False):
        # Call parent constructor to set up prompt_tuning_embedding
        super().__init__(model,
                         eagle3=eagle3,
                         use_prompt_tuning=use_prompt_tuning)
        # Override lm_head and logsoftmax for draft model
        self.model.lm_head = torch.nn.Identity()
        self.logsoftmax = model.logsoftmax
        self.model.logsoftmax = torch.nn.Identity()

    def forward(
        self,
        input_ids,
        past_key_values,
        hidden_states_input,
        hidden_states_from_draft,
        image_embeds: Optional[Tensor] = None,
    ):
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        # Handle prompt tuning if enabled (for input_ids if provided)
        if self.use_prompt_tuning and input_ids is not None:
            inputs_embeds = self.prompt_tuning_embedding(
                input_ids, image_embeds)
            input_ids = None
        else:
            inputs_embeds = None

        if self.eagle3:
            #hidden_states_input: go through the fc layer
            outputs = self.model(
                hidden_states=hidden_states_input,
                input_ids=input_ids,
                past_key_values=past_key_values,
                hidden_states_from_draft=hidden_states_from_draft,
                inputs_embeds=inputs_embeds,
                use_cache=True)
            hidden_states = outputs[0]
            hidden_states_reshape = hidden_states.reshape(
                -1, hidden_states.size(2))
            hidden_states_reshape = self.model.norm(hidden_states_reshape)
        else:
            outputs = self.model(
                hidden_states=hidden_states_input,
                hidden_states_from_draft=hidden_states_from_draft,
                input_ids=input_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True)
            hidden_states = outputs[0]
            hidden_states_reshape = hidden_states.reshape(
                -1, hidden_states.size(2))

        past_key_values = outputs[1]

        logits = self.lm_head(hidden_states_reshape)
        #hidden_states will added as output in insert_gather_last_token_eagle
        return logits, past_key_values


def torch_to_onnx(model, inputs, onnx_dir, onnx_name, input_names,
                  output_names, dynamic_axes):
    os.makedirs(onnx_dir, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            inputs,
            f'{onnx_dir}/{onnx_name}',
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=19,
            do_constant_folding=True,
        )


class ModelLoader:
    """
    A class to handle HuggingFace model loading and configuration
    """

    def __init__(self,
                 torch_dir,
                 config_path,
                 eagle_torch_dir=None,
                 eagle_base=False,
                 eagle_draft=False,
                 eagle3=False):
        self.config_path = config_path
        self.torch_dir = torch_dir
        self.model_type = self.get_model_type()
        self.hf_model = None
        self.eagle_torch_dir = eagle_torch_dir
        self.eagle_base = eagle_base
        self.eagle_draft = eagle_draft
        self.eagle3 = eagle3
        self.rope_type = RopeType.kROPE_ROTATE_NEOX

    def get_model_type(self):
        """Get model type from config file"""
        with open(self.config_path) as f:
            return json.load(f).get("model_type")

    def _get_eagle_draft_model(self):
        assert self.eagle_torch_dir, "Need to provide --eagle_torch_dir when you want to export eagle draft model"
        from eagle.ea_model import EagleModel
        model = EagleModel.from_pretrained(
            base_model_path=self.torch_dir,
            ea_model_path=self.eagle_torch_dir,
            use_eagle3=self.eagle3,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu",
            local_files_only=True,
        )
        self.hidden_size = model.config.hidden_size
        draft_model = model.ea_layer
        draft_model.to(torch.float16).to('cuda')
        return draft_model

    def _prepare_draft_model_extra_inputs(self):
        dummy_len_input_hidden = 10
        dummy_bs = 1
        targetModelOutputHiddenDim = self.hidden_size * 3 if self.eagle3 else self.hidden_size
        hidden_states_input = torch.randn(
            (dummy_bs, dummy_len_input_hidden, targetModelOutputHiddenDim),
            dtype=torch.float16).cuda()
        hidden_states_from_draft = torch.randn(
            (dummy_bs, dummy_len_input_hidden, self.hidden_size),
            dtype=torch.float16).cuda()

        extra_inputs = {
            "hidden_states_input": hidden_states_input,
            "hidden_states_from_draft": hidden_states_from_draft
        }
        extra_dyn_axes = {
            "hidden_states_input": {
                0: "batch_size",
                1: "seq_len",
            },
            "hidden_states_from_draft": {
                0: "batch_size",
                1: "seq_len",
            }
        }
        return extra_inputs, extra_dyn_axes

    def prepare_extra_inputs(self):
        extra_inputs = {}
        extra_dyn_axes = {}
        if self.eagle_draft:
            extra_inputs_draft, extra_dyn_axes_draft = self._prepare_draft_model_extra_inputs(
            )
            extra_inputs.update(extra_inputs_draft)
            extra_dyn_axes.update(extra_dyn_axes_draft)
        if self.model_type in ['qwen2_vl', 'qwen2_5_vl', 'internvl_chat']:
            dummy_len = 10
            image_embeds = torch.randn(
                (dummy_len, self.hf_model.config.hidden_size),
                dtype=torch.float16).cuda()
            extra_inputs["image_embeds"] = image_embeds
            extra_dyn_axes["image_embeds"] = {0: "image_token_length"}

        return extra_inputs, extra_dyn_axes

    def get_wrapper_cls(self):
        if self.eagle_draft:
            return WrapperEagleDraftModelForCausalLM
        elif self.eagle_base:
            return WrapperEagleBaseModelForCausalLM
        else:
            return WrapperModelForCausalLM

    def load_model(self):
        """Load HuggingFace model based on model type"""
        print(
            f"Loading HF model from {self.torch_dir} with model type {self.model_type}"
        )
        if self.model_type in ['qwen2_vl', 'qwen2_5_vl']:
            self.rope_type = RopeType.kMROPE
        if self.eagle_draft:
            self.hf_model = self._get_eagle_draft_model()
        elif self.model_type == 'qwen2_vl':
            from transformers import Qwen2VLForConditionalGeneration
            self.hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.torch_dir,
                torch_dtype=torch.float16,
            )
        elif self.model_type == 'qwen2_5_vl':
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.torch_dir,
                torch_dtype=torch.float16,
            )
        elif self.model_type == 'internvl_chat':
            from transformers import AutoModel
            self.hf_model = AutoModel.from_pretrained(
                self.torch_dir,
                torch_dtype=torch.float16,
                trust_remote_code=True)
        else:
            from transformers import AutoModelForCausalLM
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                self.torch_dir,
                torch_dtype=torch.float16,
                trust_remote_code=True)

        return self.hf_model.eval().cuda()

    def get_rope_type(self):
        return self.rope_type

    def add_extra_plugin_inputs(self):
        """Add extra inputs based on model type and Eagle configuration"""
        extra_plugin_inputs = []
        extra_plugin_attributes = {}

        # Ensure model is loaded
        if self.hf_model is None:
            self.load_model()

        # Add inputs for vision models
        if self.model_type == 'qwen2_vl' or self.model_type == 'qwen2_5_vl':

            mrope_rotary_cos_sin = gs.Variable(
                "mrope_rotary_cos_sin", np.float32, [
                    'batch_size',
                    self.hf_model.config.max_position_embeddings * 128
                ])  # head_size = 128
            mrope_position_deltas = gs.Variable("mrope_position_deltas",
                                                np.int64, ['batch_size', 1])
            extra_plugin_inputs.append(mrope_rotary_cos_sin)
            extra_plugin_inputs.append(mrope_position_deltas)

        # Add inputs for Eagle models
        if self.eagle_base or self.eagle_draft:
            attention_mask = gs.Variable(
                "attention_mask", np.int32,
                ['batch_size', 'q_len', 'q_len_aligned'])
            attention_pos_id = gs.Variable("attention_pos_id", np.int32,
                                           ['batch_size', 'q_len'])
            extra_plugin_inputs.append(attention_mask)
            extra_plugin_inputs.append(attention_pos_id)
            extra_plugin_attributes["enable_tree_attention"] = 1

        return extra_plugin_inputs, extra_plugin_attributes

    def load_model_with_lora(self, base_model, lora_dir, lora_mode):
        """
        Load and handle LoRA weights for a model.

        Args:
            base_model: The base model loaded from HuggingFace
            lora_dir: Directory containing LoRA weights
            lora_mode: LoRA mode to use ("merged" or "static")

        Returns:
            The model with LoRA weights applied (merged for merged mode)
        """
        if not lora_dir:
            return base_model

        if lora_mode == "merged":
            print(f"Loading LoRA weights from {lora_dir} in merged mode...")
            model = PeftModel.from_pretrained(base_model, lora_dir)
            print("Merging LoRA weights into base model...")
            model = model.merge_and_unload()
            return model
        elif lora_mode == "static":
            print(f"Loading LoRA config from {lora_dir} in static mode...")

            # Load LoRA config
            config = PeftConfig.from_pretrained(lora_dir)

            # Load LoRA weights
            weights = load_peft_weights(lora_dir)

            return base_model, config, weights

        return base_model

    def save_d2t_for_eagle3_draft(self, onnx_dir):
        load_model_path = os.path.join(self.eagle_torch_dir,
                                       "pytorch_model.bin")
        ea_layer_state_dict = torch.load(load_model_path, weights_only=True)
        # When the  draft vocab size is not equal to the base vocab size, we need to map the token id from draft to base using d2t.bin
        d2t_tensor = ea_layer_state_dict['d2t']
        d2t_path = os.path.join(onnx_dir, "d2t.bin")
        with open(d2t_path, 'wb') as f:
            f.write(d2t_tensor.numpy().astype(np.int64).tobytes())


def llm_to_onnx(model, output_dir, extra_inputs={}, extra_dyn_axes={}):
    """
    Export the WrapperModelForCausalLM to ONNX with fixed I/O names and shape definitions and save to `output_dir`

    Parameters:
        model: torch.Module
        output_dir: str, the output_dir of the original ONNX.
        extra_inputs: dict, append additional inputs after kv_cache. Usually for VL models
        extra_dyn_axes: dict. Usually for VL models
    """
    start_time = time.time()
    config = model.config
    num_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    hidden_size = config.hidden_size
    hidden_size_per_layer = hidden_size // num_attention_heads

    dummy_bs = 1
    dummy_len = 10
    dummy_input_ids = torch.randint(100, (dummy_bs, dummy_len),
                                    dtype=torch.int64).cuda()
    input_names = ["input_ids"]
    output_names = ["logits"]
    dynamic_axes = {"input_ids": {0: "batch_size", 1: "seq_len"}}
    dummy_kv_cache = ()
    for i in range(num_layers):
        dummy_k = torch.rand(
            (dummy_bs, num_key_value_heads, dummy_len, hidden_size_per_layer),
            dtype=torch.float16).cuda()
        dummy_v = torch.rand(
            (dummy_bs, num_key_value_heads, dummy_len, hidden_size_per_layer),
            dtype=torch.float16).cuda()
        dummy_kv_cache = dummy_kv_cache + ((dummy_k, dummy_v), )
        input_names.extend(
            [f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
        output_names.extend(
            [f"present_key_values.{i}.key", f"present_key_values.{i}.value"])
        input_dynamic_axes = {0: "batch_size", 2: "past_len"}
        dynamic_axes[f"past_key_values.{i}.key"] = input_dynamic_axes
        dynamic_axes[f"past_key_values.{i}.value"] = input_dynamic_axes

    if isinstance(model, WrapperEagleBaseModelForCausalLM):
        output_names.extend(['hidden_states'])

    torch_to_onnx(
        model,
        (dummy_input_ids, {
            "past_key_values": dummy_kv_cache,
            **extra_inputs
        }),
        output_dir,
        "model.onnx",
        input_names=input_names + list(extra_inputs.keys()),
        output_names=output_names,
        dynamic_axes=dynamic_axes | extra_dyn_axes,
    )

    end_time = time.time()
    print(
        f"Native ONNX Export from torch completed in {end_time - start_time}s. ONNX file is saved to {output_dir}."
    )


class QwenVisionAttention(VisionAttention):

    def __init__(self, dim: int, num_heads: int = 16):
        super().__init__(dim, num_heads)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_embeddings: torch.Tensor) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3,
                                                  self.num_heads,
                                                  -1).permute(1, 0, 2,
                                                              3).unbind(0)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(
            self.head_dim)
        attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights,
                                                   dim=-1,
                                                   dtype=torch.float32).to(
                                                       v.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output
