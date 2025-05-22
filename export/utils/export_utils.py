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

import torch
from peft import PeftConfig, PeftModel, load_peft_weights
from transformers import DynamicCache


def load_model_with_lora(base_model, lora_dir, lora_mode):
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


class WrapperModelForCausalLM(torch.nn.Module):
    """
    Wrapper Model to ensure all models have the same I/O
    """

    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.lm_head = model.lm_head
        self.config = model.config

    def forward(self, input_ids, past_key_values):
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        outputs = self.model(input_ids=input_ids,
                             past_key_values=past_key_values)
        hidden_states = outputs[0]
        past_key_values = outputs.past_key_values.to_legacy_cache()
        logits = self.lm_head(hidden_states)
        return logits, past_key_values

class WrapperEagleBaseModelForCausalLM(torch.nn.Module):
    """
    Wrapper Model to ensure all models have the same I/O
    """

    def __init__(self, model, eagle3=False):
        super().__init__()
        self.model = model.model
        self.lm_head = model.lm_head
        self.config = model.config
        self.eagle3 = eagle3

    def forward(self, input_ids, past_key_values):
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        outputs = self.model(input_ids=input_ids,
                             past_key_values=past_key_values,
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

        past_key_values = outputs.past_key_values
        hidden_states_reshape = last_hidden_states.reshape(
            -1, last_hidden_states.size(2))
        logits = self.lm_head(hidden_states_reshape)
        if self.eagle3:
            return logits, past_key_values, hidden_states
        else:
            return logits, past_key_values, last_hidden_states


class WrapperEagleDraftModelForCausalLM(torch.nn.Module):
    """
    Wrapper Model to ensure all models have the same I/O
    """

    def __init__(self, model, eagle3=False):
        super().__init__()
        self.model = model
        self.lm_head = model.lm_head
        self.model.lm_head = torch.nn.Identity()
        self.config = model.config
        self.logsoftmax = model.logsoftmax
        self.model.logsoftmax = torch.nn.Identity()
        self.eagle3 = eagle3

    def forward(self,
                input_ids,
                past_key_values,
                hidden_states_input,
                hidden_states_from_draft=None):
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        if self.eagle3:
            #hidden_states_input: go through the fc layer
            outputs = self.model(
                hidden_states=hidden_states_input,
                input_ids=input_ids,
                past_key_values=past_key_values,
                hidden_states_from_draft=hidden_states_from_draft,
                use_cache=True)
            hidden_states = outputs[0]
            hidden_states_reshape = hidden_states.reshape(
                -1, hidden_states.size(2))
            hidden_states_reshape = self.model.norm(hidden_states_reshape)
        else:
            outputs = self.model(hidden_states=hidden_states_input,
                                 input_ids=input_ids,
                                 past_key_values=past_key_values,
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
