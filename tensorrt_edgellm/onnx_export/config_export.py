# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Dict

from ..version import __version__


def _export_native_llm_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Export LLM configuration with required fields.

    Args:
        config_dict: Raw model configuration dictionary.

    Returns:
        Dict[str, Any]: Sanitized LLM configuration for Edge-LLM export.
    """
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

    # Handle LongRoPE (rope_scaling already validated in required_fields)
    rope_scaling = config_dict["rope_scaling"]
    if rope_scaling and rope_scaling.get("type", None) == "longrope":
        if "original_max_position_embeddings" not in config_dict:
            raise KeyError(
                f"Required field 'original_max_position_embeddings' not found in config"
            )
        llm_config["original_max_position_embeddings"] = config_dict[
            "original_max_position_embeddings"]

    # Handle head_dim
    if "head_dim" in config_dict:
        llm_config["head_dim"] = config_dict["head_dim"]
    else:
        print(
            "Warning: head_dim not found in config, calculating as hidden_size // num_attention_heads"
        )
        llm_config["head_dim"] = config_dict["hidden_size"] // config_dict[
            "num_attention_heads"]

    if "partial_rotary_factor" in config_dict:
        llm_config["partial_rotary_factor"] = config_dict[
            "partial_rotary_factor"]
    else:
        llm_config["partial_rotary_factor"] = 1.0

    # Gemma3n LAuReL config
    if "laurel_rank" in config_dict:
        llm_config["laurel_rank"] = config_dict["laurel_rank"]

    llm_config["model_type"] = "llm"
    return llm_config


def _export_hybrid_mamba_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Export hybrid Mamba model configuration with Mamba-specific fields."""
    required_fields = [
        "vocab_size",
        "max_position_embeddings",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
    ]
    optional_fields_with_defaults = {
        "rope_theta": 10000.0,
        "rope_scaling": None,
    }

    llm_config = {}
    for field in required_fields:
        if field not in config_dict:
            raise KeyError(f"Required field '{field}' not found in config")
        llm_config[field] = config_dict[field]

    for field, default in optional_fields_with_defaults.items():
        llm_config[field] = config_dict.get(field, default)

    if "head_dim" in config_dict:
        llm_config["head_dim"] = config_dict["head_dim"]
    else:
        llm_config["head_dim"] = config_dict["hidden_size"] // config_dict[
            "num_attention_heads"]

    if "partial_rotary_factor" in config_dict:
        llm_config["partial_rotary_factor"] = config_dict[
            "partial_rotary_factor"]
    else:
        llm_config["partial_rotary_factor"] = 1.0

    layers_block_type = config_dict.get("layers_block_type", [])
    if not layers_block_type:
        pattern = config_dict.get("hybrid_override_pattern", "")
        num_mamba = pattern.count("M")
        num_attention = pattern.count("*")
    else:
        num_mamba = sum(1 for t in layers_block_type if t == "mamba")
        num_attention = sum(1 for t in layers_block_type if t == "attention")

    llm_config["num_mamba_layers"] = num_mamba
    llm_config["num_attention_layers"] = num_attention
    llm_config["mamba_num_heads"] = config_dict.get("mamba_num_heads", 0)
    llm_config["mamba_head_dim"] = config_dict.get("mamba_head_dim", 0)
    llm_config["ssm_state_size"] = config_dict.get("ssm_state_size", 0)

    mamba_num_heads = llm_config["mamba_num_heads"]
    mamba_head_dim = llm_config["mamba_head_dim"]
    ssm_state_size = llm_config["ssm_state_size"]
    n_groups = config_dict.get("n_groups",
                               config_dict.get("mamba_n_groups", 1))
    llm_config[
        "conv_dim"] = mamba_num_heads * mamba_head_dim + 2 * n_groups * ssm_state_size
    llm_config["conv_kernel"] = config_dict.get(
        "conv_kernel", config_dict.get("mamba_d_conv", 4))

    llm_config["use_rope"] = "rope_theta" in config_dict

    llm_config["model_type"] = "hybrid_mamba"
    return llm_config


def _export_eagle_base_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
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
    if "partial_rotary_factor" in config_dict:
        eagle_config["partial_rotary_factor"] = config_dict[
            "partial_rotary_factor"]
    else:
        eagle_config["partial_rotary_factor"] = 1.0

    eagle_config["model_type"] = f"eagle3_base"
    return eagle_config


def _export_eagle_draft_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Export EAGLE draft configuration with required fields."""
    required_fields = [
        "hidden_size", "max_position_embeddings", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "rope_theta", "rope_scaling"
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
    if "draft_vocab_size" not in config_dict:
        raise KeyError("Required field 'draft_vocab_size' not found in config")
    draft_config["draft_vocab_size"] = config_dict["draft_vocab_size"]

    # Add base model configuration fields
    # The target_hidden_size from the model config represents the base model's hidden dimension
    if "target_hidden_size" in config_dict:
        # Use target_hidden_size * 3 as the base model hidden dimension (as per llm_export.py logic)
        draft_config[
            "base_model_hidden_size"] = config_dict["target_hidden_size"] * 3
    else:
        # Fallback: assume base model hidden size is 3x draft model (Eagle3 default)
        draft_config["base_model_hidden_size"] = config_dict["hidden_size"] * 3
        print(
            f"Warning: target_hidden_size not found, using default 3x draft hidden size: {draft_config['base_model_hidden_size']}"
        )

    # Set model_type for draft
    draft_config["model_type"] = f"eagle3_draft"

    return draft_config


def export_vision_config(config: Any) -> Dict[str, Any]:
    """Export vision encoder configuration with proper model_type."""
    config_dict = config.to_dict()

    has_vision = "vision_config" in config_dict
    has_phi4_vision = "image_embd_layer" in config_dict.get("embd_layer", {})
    if not (has_vision or has_phi4_vision):
        raise KeyError(
            "Required field 'vision_config' or 'image_embd_layer' in 'embd_layer' not found in config"
        )
    # Add TensorRT Edge-LLM version
    config_dict['edgellm_version'] = __version__

    # Set top-level model_type for C++ builder/runtime
    # Check if this is Qwen3-Omni vision (has vision_config.model_type = qwen3_omni_vision_encoder)
    if 'vision_config' in config_dict and config_dict['vision_config'].get(
            'model_type') == 'qwen3_omni_vision_encoder':
        config_dict['model_type'] = 'qwen3_omni_vision_encoder'

    # Return the config_dict. Since MRoPE needs LLM config, ViTRunner will use the LLM config.
    return config_dict


def export_llm_config(config: Any,
                      model_type: str,
                      trt_native_ops: bool = False) -> Dict[str, Any]:
    """Export configuration based on model type and EAGLE version."""
    config_dict = config.to_dict()

    # Extract model name from config class
    config_class_name = config.__class__.__name__
    model_name = config_class_name.lower().replace('config', '')

    # For multimodal models, preserve token IDs before switching to text_config
    multimodal_token_ids = {}
    if "text_config" in config_dict:
        print("Detected multimodal model, using text_config")
        # Automatically preserve any field ending with '_token_id' or '_token_ids' at the top level
        for key, value in config_dict.items():
            if key.endswith('_token_id') or key.endswith('_token_ids'):
                multimodal_token_ids[key] = value
        if multimodal_token_ids:
            print(
                f"Preserved multimodal token IDs: {list(multimodal_token_ids.keys())}"
            )
        config_dict = config_dict["text_config"]

    if model_type == 'llm':
        output_config = _export_native_llm_config(config_dict)
    elif model_type == 'hybrid_mamba':
        output_config = _export_hybrid_mamba_config(config_dict)
    elif model_type == 'eagle3_base':
        output_config = _export_eagle_base_config(config_dict)
    elif model_type == 'eagle_draft':
        output_config = _export_eagle_draft_config(config_dict)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Add model name to output
    output_config["model"] = model_name

    # Add TensorRT Edge-LLM version
    output_config['edgellm_version'] = __version__

    # Add trt_native_ops to output_config
    output_config["trt_native_ops"] = trt_native_ops

    # Restore multimodal token IDs if any were saved
    if multimodal_token_ids:
        output_config.update(multimodal_token_ids)
        print(
            f"Restored multimodal token IDs to output config: {list(multimodal_token_ids.keys())}"
        )

    return output_config


def export_audio_config(config: Any) -> Dict[str, Any]:
    """Export audio encoder configuration with proper model_type."""
    config_dict = config.to_dict()

    has_audio = "audio_config" in config_dict
    if not (has_audio):
        raise KeyError("Required field 'audio_config' not found in config")
    # Add TensorRT Edge-LLM version
    config_dict['edgellm_version'] = __version__

    # Set top-level model_type for C++ builder (reads top-level, not audio_config.model_type)
    config_dict['model_type'] = 'qwen3_omni_audio_encoder'

    return config_dict


def export_code2wav_config(config: Any) -> Dict[str, Any]:
    """Export code2wav configuration with proper model_type."""
    config_dict = config.to_dict()

    has_code2wav = "code2wav_config" in config_dict
    if not (has_code2wav):
        raise KeyError("Required field 'code2wav_config' not found in config")

    # Add TensorRT Edge-LLM version
    config_dict['edgellm_version'] = __version__

    # Set model_type to code2wav for proper type identification
    # Override any existing model_type in code2wav_config
    if 'code2wav_config' in config_dict:
        config_dict['code2wav_config']['model_type'] = 'qwen3_omni_code2wav'

    # Also set top-level model_type for easier parsing
    config_dict['model_type'] = 'qwen3_omni_code2wav'

    return config_dict


def export_tts_talker_config(full_tts_config: Any) -> Dict[str, Any]:
    """Export Qwen3-TTS Talker config for C++ runtime."""
    config_dict = full_tts_config.to_dict()
    talker_raw = config_dict["talker_config"]

    talker_config = export_llm_config(full_tts_config.talker_config, 'llm',
                                      False)
    talker_config["model_type"] = "qwen3_tts_talker"
    talker_config["use_embeddings_input"] = True

    # TTS text embedding dimensions
    talker_config["text_hidden_size"] = talker_raw["text_hidden_size"]
    if "text_vocab_size" in talker_raw:
        talker_config["text_vocab_size"] = talker_raw["text_vocab_size"]
    talker_config["num_code_groups"] = talker_raw["num_code_groups"]

    # Codec control tokens
    for key in [
            "codec_eos_token_id", "codec_think_id", "codec_nothink_id",
            "codec_think_bos_id", "codec_think_eos_id", "codec_pad_id",
            "codec_bos_id"
    ]:
        if key in talker_raw:
            talker_config[key] = talker_raw[key]

    # TTS special tokens (top-level config)
    for key in [
            "tts_pad_token_id", "tts_bos_token_id", "tts_eos_token_id",
            "im_start_token_id", "im_end_token_id"
    ]:
        if key in config_dict:
            talker_config[key] = config_dict[key]

    # Language / Speaker (TTS field name is spk_id, not speaker_id)
    if "codec_language_id" in talker_raw:
        talker_config["codec_language_id"] = talker_raw["codec_language_id"]
    if "spk_id" in talker_raw:
        talker_config["speaker_id"] = talker_raw["spk_id"]
        talker_config["available_speakers"] = list(talker_raw["spk_id"].keys())
    if "spk_is_dialect" in talker_raw:
        talker_config["spk_is_dialect"] = talker_raw["spk_is_dialect"]

    talker_config = {k: v for k, v in talker_config.items() if v is not None}
    print(f"Exported TTS Talker config with {len(talker_config)} fields")
    return talker_config


def export_talker_config(full_qwen3_omni_config: Any) -> Dict[str, Any]:
    """Export Talker config: preserve original model structure and fields."""
    config_dict = full_qwen3_omni_config.to_dict()

    if "talker_config" not in config_dict:
        raise KeyError("Required field 'talker_config' not found in config")

    # Top-level LLM fields (vocab_size, hidden_size, etc.)
    result = export_llm_config(
        full_qwen3_omni_config.talker_config.text_config,
        model_type='llm',
        trt_native_ops=False)

    talker_config_raw = config_dict["talker_config"]

    # Override model_type to preserve original Talker model type
    # Original: "qwen3_omni_moe_talker" or "qwen3_omni_talker"
    if "model_type" in talker_config_raw:
        result["model_type"] = talker_config_raw["model_type"]

    # Core talker fields
    result["thinker_hidden_size"] = talker_config_raw["thinker_hidden_size"]
    result["accept_hidden_layer"] = talker_config_raw["accept_hidden_layer"]

    # Architecture metadata
    if "num_code_groups" in talker_config_raw:
        result["num_code_groups"] = talker_config_raw["num_code_groups"]

    # Mark as embedding-input model (no tokenizer needed for build)
    result["use_embeddings_input"] = True

    # Multimodal token IDs
    result["audio_token_id"] = talker_config_raw["audio_token_id"]
    if "audio_start_token_id" in talker_config_raw:
        result["audio_start_token_id"] = talker_config_raw[
            "audio_start_token_id"]
    if "audio_end_token_id" in talker_config_raw:
        result["audio_end_token_id"] = talker_config_raw["audio_end_token_id"]

    result["image_token_id"] = talker_config_raw["image_token_id"]
    if "video_token_id" in talker_config_raw:
        result["video_token_id"] = talker_config_raw["video_token_id"]

    # Role tokens from config (used for chat template)
    # Note: eos_token_id is NOT exported here - it should be obtained from tokenizer at runtime
    # This aligns with standard LLM behavior where stopping logic is in the application layer
    token_id_fields = [
        "user_token_id", "assistant_token_id", "system_token_id",
        "im_start_token_id", "im_end_token_id"
    ]

    for key in token_id_fields:
        if key in config_dict and config_dict[key] is not None:
            result[key] = config_dict[key]

    # Codec control tokens (preserve original field names for consistency)
    result["codec_nothink_id"] = talker_config_raw["codec_nothink_id"]
    result["codec_think_bos_id"] = talker_config_raw["codec_think_bos_id"]
    result["codec_think_eos_id"] = talker_config_raw["codec_think_eos_id"]
    result["codec_pad_id"] = talker_config_raw["codec_pad_id"]
    result["codec_bos_id"] = talker_config_raw["codec_bos_id"]
    result["codec_eos_token_id"] = talker_config_raw[
        "codec_eos_token_id"]  # Keep original field name

    # TTS special tokens (from top-level config)
    result["tts_pad_token_id"] = config_dict.get("tts_pad_token_id", 151671)
    result["tts_bos_token_id"] = config_dict.get("tts_bos_token_id", 151672)
    result["tts_eos_token_id"] = config_dict.get("tts_eos_token_id", 151673)

    # Speaker ID mapping for multi-speaker support
    if "speaker_id" in talker_config_raw and talker_config_raw["speaker_id"]:
        result["speaker_id"] = talker_config_raw["speaker_id"]
        # Set default speaker to first speaker in mapping (typically f245: 2301)
        speaker_ids = list(talker_config_raw["speaker_id"].values())
        result["default_speaker_id"] = speaker_ids[0]
        result["available_speakers"] = list(
            talker_config_raw["speaker_id"].keys())
        print(
            f"Exported {len(result['available_speakers'])} speaker IDs, default: {result['default_speaker_id']}"
        )
    else:
        # Fallback: if no speaker_id mapping in config, use default
        print(
            "Warning: No speaker_id mapping found in config, using default f245"
        )
        result["default_speaker_id"] = 2301  # f245 as fallback

    # Optional metadata fields (preserve if present for full compatibility)
    optional_fields = [
        "output_router_logits", "position_id_per_seconds", "seconds_per_chunk",
        "spatial_merge_size"
    ]
    for field in optional_fields:
        if field in talker_config_raw:
            result[field] = talker_config_raw[field]

    # Validate all required fields are present
    required_fields = ["user_token_id", "assistant_token_id"]
    missing_fields = [
        f for f in required_fields if f not in result or result[f] is None
    ]
    if missing_fields:
        raise ValueError(
            f"Required token ID fields missing from config: {missing_fields}")

    # Filter out remaining None values (optional fields only)
    result = {k: v for k, v in result.items() if v is not None}

    print(f"Exported Talker config with {len(result)} fields")

    return result
