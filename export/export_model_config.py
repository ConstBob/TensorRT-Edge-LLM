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

import argparse
import json
import os

from transformers import AutoConfig


def export_llm_config(config_dict):
    """Export LLM configuration with required fields."""
    required_fields = [
        "vocab_size", "max_position_embeddings", "hidden_size",
        "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "rope_theta", "rope_scaling"
    ]

    llm_config = {}
    for field in required_fields:
        if field not in config_dict:
            raise KeyError(f"Required field '{field}' not found in config")
        llm_config[field] = config_dict[field]

    # Handle head_dim (optional)
    if "head_dim" in config_dict:
        llm_config["head_dim"] = config_dict["head_dim"]
    else:
        print(
            "Warning: head_dim not found in config, calculating as hidden_size // num_attention_heads"
        )
        llm_config["head_dim"] = config_dict["hidden_size"] // config_dict[
            "num_attention_heads"]

    llm_config["model_type"] = "llm"
    return llm_config


def export_eagle_base_config(config_dict, eagle_version="eagle3"):
    """Export EAGLE base configuration with required fields."""
    required_fields = [
        "vocab_size", "max_position_embeddings", "hidden_size",
        "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "rope_theta", "rope_scaling"
    ]

    eagle_config = {}
    for field in required_fields:
        if field not in config_dict:
            raise KeyError(f"Required field '{field}' not found in config")
        eagle_config[field] = config_dict[field]

    # Handle head_dim (optional)
    if "head_dim" in config_dict:
        eagle_config["head_dim"] = config_dict["head_dim"]
    else:
        print(
            "Warning: head_dim not found in config, calculating as hidden_size // num_attention_heads"
        )
        eagle_config["head_dim"] = config_dict["hidden_size"] // config_dict[
            "num_attention_heads"]

    eagle_config["model_type"] = f"{eagle_version}_base"
    return eagle_config


def export_eagle_draft_config(config_dict, eagle_version="eagle3"):
    """Export EAGLE draft configuration with required fields."""
    required_fields = [
        "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads"
    ]

    draft_config = {}
    for field in required_fields:
        if field not in config_dict:
            raise KeyError(f"Required field '{field}' not found in config")
        draft_config[field] = config_dict[field]

    # Handle head_dim (optional)
    if "head_dim" in config_dict:
        draft_config["head_dim"] = config_dict["head_dim"]
    else:
        print(
            "Warning: head_dim not found in config, calculating as hidden_size // num_attention_heads"
        )
        draft_config["head_dim"] = config_dict["hidden_size"] // config_dict[
            "num_attention_heads"]

    # Handle draft_vocab_size based on EAGLE version
    if eagle_version == "eagle2":
        if "vocab_size" not in config_dict:
            raise KeyError("Required field 'vocab_size' not found in config")
        draft_config["draft_vocab_size"] = config_dict["vocab_size"]
    elif eagle_version == "eagle3":
        if "draft_vocab_size" not in config_dict:
            raise KeyError(
                "Required field 'draft_vocab_size' not found in config")
        draft_config["draft_vocab_size"] = config_dict["draft_vocab_size"]
    else:
        raise ValueError(f"Unsupported EAGLE version: {eagle_version}")

    # Set model_type for draft
    draft_config["model_type"] = f"{eagle_version}_draft"

    return draft_config


def export_vision_config(config_dict):
    """Export vision configuration without modification."""
    if "vision_config" not in config_dict:
        raise KeyError("Required field 'vision_config' not found in config")

    # Return the original config_dict as-is without any modification
    # Since MRoPE needs LLM config, ViTRunner will use the LLM config.
    return config_dict


def export_config(config, model_type, eagle2=False):
    """Export configuration based on model type and EAGLE version."""
    config_dict = config.to_dict()

    # Extract model name from config class
    config_class_name = config.__class__.__name__
    model_name = config_class_name.lower().replace('config', '')

    # Determine EAGLE version
    eagle_version = "eagle2" if eagle2 else "eagle3"

    # Export based on model type
    if model_type == 'vision':
        # For vision, use the full config_dict
        output_config = export_vision_config(config_dict)
    else:
        # For other model types, use text_config if available
        if "text_config" in config_dict:
            print("Detected multimodal model, using text_config")
            config_dict = config_dict["text_config"]

        if model_type == 'llm':
            output_config = export_llm_config(config_dict)
        elif model_type == 'eagle_base':
            output_config = export_eagle_base_config(config_dict,
                                                     eagle_version)
        elif model_type == 'eagle_draft':
            output_config = export_eagle_draft_config(config_dict,
                                                      eagle_version)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    # Add model name to output
    output_config["model"] = model_name

    return output_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_dir',
                        required=True,
                        help='Path to the model folder')
    parser.add_argument('--output', required=True, help='Output JSON filename')
    parser.add_argument('--model_type',
                        choices=['llm', 'eagle_base', 'eagle_draft', 'vision'],
                        required=True,
                        help='Type of configuration to export')
    parser.add_argument('--eagle2',
                        action='store_true',
                        help='Use EAGLE2 instead of EAGLE3')
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.torch_dir, trust_remote_code=True)

    # Export configuration using the unified function
    output_config = export_config(config, args.model_type, args.eagle2)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(output_config, f, indent=2)

    print(f"Config saved to {args.output}")


if __name__ == "__main__":
    main()
