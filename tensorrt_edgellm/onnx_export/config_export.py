# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Any, Dict


def _export_native_llm_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
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

    # Handle head_dim
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


def _export_eagle_base_config(config_dict: Dict[str, Any],
                              eagle_version: str = "eagle3") -> Dict[str, Any]:
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

    # Handle head_dim
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


def _export_eagle_draft_config(
        config_dict: Dict[str, Any],
        eagle_version: str = "eagle3") -> Dict[str, Any]:
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

    # Handle head_dim
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


def export_vision_config(config: Any) -> Dict[str, Any]:
    """Export vision configuration without modification."""
    config_dict = config.to_dict()

    if "vision_config" not in config_dict:
        raise KeyError("Required field 'vision_config' not found in config")

    # Return the original config_dict as-is without any modification
    # Since MRoPE needs LLM config, ViTRunner will use the LLM config.
    return config_dict


def export_llm_config(config: Any,
                      model_type: str,
                      eagle2: bool = False) -> Dict[str, Any]:
    """Export configuration based on model type and EAGLE version."""
    config_dict = config.to_dict()

    # Extract model name from config class
    config_class_name = config.__class__.__name__
    model_name = config_class_name.lower().replace('config', '')

    # Determine EAGLE version
    eagle_version = "eagle2" if eagle2 else "eagle3"
    # For other model types, use text_config if available
    if "text_config" in config_dict:
        print("Detected multimodal model, using text_config")
        config_dict = config_dict["text_config"]

    if model_type == 'llm':
        output_config = _export_native_llm_config(config_dict)
    elif model_type == 'eagle_base':
        output_config = _export_eagle_base_config(config_dict, eagle_version)
    elif model_type == 'eagle_draft':
        output_config = _export_eagle_draft_config(config_dict, eagle_version)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Add model name to output
    output_config["model"] = model_name

    return output_config
