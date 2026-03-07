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
import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (LlamaAttention, LlamaMLP,
                                                      LlamaRMSNorm,
                                                      apply_rotary_pos_emb,
                                                      repeat_kv)
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2MLP
from transformers.models.qwen3_moe.modeling_qwen3_moe import \
    Qwen3MoeSparseMoeBlock

from .attention_plugin import attention_plugin
from .attention_trt import EdgeLLMAttentionTRTNative
from .layer_utils import EdgeLLMQKNorm, EdgeLLMQKVProj

# FP8 (E4M3) quantization constants
# Max finite value representable by NVIDIA FP8 E4M3 format; used to derive per-tensor KV cache scale.
FP8_E4M3_MAX: float = 448.0


class PromptTuningEmbedding(torch.nn.Module):
    """
    Prompt Tuning Embedding for multimodal models.
    
    This module combines text embeddings with visual embeddings for models that support
    both text and image inputs. It handles the mapping between text tokens and visual
    tokens in the vocabulary space by using token IDs beyond the normal vocabulary size
    to represent visual tokens.
    
    Attributes:
        embedding: Base token embedding layer for text tokens
        vocab_size: Size of the vocabulary for text tokens
    """

    def __init__(
        self,
        embedding: torch.nn.Embedding,
    ) -> None:
        """
        Initialize the PromptTuningEmbedding module.
        
        Args:
            embedding: Base token embedding layer for text tokens
        """
        super().__init__()
        self.embedding = embedding
        self.vocab_size = embedding.num_embeddings

    def forward(self, input_ids: torch.Tensor,
                image_embeds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining text and visual embeddings.
        
        Args:
            input_ids: Token IDs with visual tokens having IDs > vocab_size
            image_embeds: Visual embeddings for image tokens
            
        Returns:
            Combined embeddings of text and visual tokens
        """
        # Identify visual tokens (IDs > vocab_size)
        image_mask = input_ids > (self.vocab_size - 1)

        # Clip normal tokens to valid vocabulary range
        normal_tokens = torch.where(image_mask, self.vocab_size - 1, input_ids)
        normal_embeddings = self.embedding(normal_tokens)

        # Map visual tokens to embedding indices
        visual_tokens = torch.where(image_mask, input_ids - self.vocab_size, 0)
        image_embeds = torch.nn.functional.embedding(visual_tokens,
                                                     image_embeds)

        # Combine normal and visual embeddings based on mask
        inputs_embeds = torch.where(image_mask.unsqueeze(-1), image_embeds,
                                    normal_embeddings)
        return inputs_embeds


class EdgeLLMAttention(nn.Module):
    """
    Multi-headed attention using the custom attention plugin for optimized inference.
    
    This module replaces the standard attention mechanism with a custom TensorRT plugin
    that fuses RoPE application, KV cache management, and attention computation.
    It supports both standard models and EAGLE draft variants.
    
    For EAGLE3 draft models, the input dimension is doubled (2x hidden_size) to handle
    concatenated input embeddings and hidden states.
    
    Attributes:
        q_proj: Query projection layer
        k_proj: Key projection layer
        v_proj: Value projection layer
        o_proj: Output projection layer
        q_norm: Query normalization layer (optional, for Qwen3 models)
        k_norm: Key normalization layer (optional, for Qwen3 models)
        qk_norm: QK normalization layer (optional, for Llama4 models)
        hidden_size: Hidden dimension size
        num_key_value_heads: Number of key-value heads
        num_attention_heads: Number of attention heads
        head_dim: Dimension of each attention head
        max_position_embeddings: Maximum sequence length for positional embeddings
        eagle3_draft: Whether this is an EAGLE3 draft model (affects input dimension)
    """

    def __init__(self,
                 attention_module: nn.Module,
                 eagle3_draft: bool = False) -> None:
        """
        Initialize the EdgeLLMAttention module.
        
        Args:
            attention_module: Original attention module to extract components from
            eagle3_draft: Whether this is an EAGLE3 draft model
        """
        super().__init__()

        # Copy configuration attributes from the original attention module
        self.hidden_size: int = attention_module.config.hidden_size
        self.num_key_value_heads: int = attention_module.config.num_key_value_heads
        self.num_attention_heads: int = attention_module.config.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
        self.num_key_value_groups: int = self.num_attention_heads // self.num_key_value_heads

        # Set head dimension
        if hasattr(attention_module.config, 'head_dim'):
            self.head_dim: int = attention_module.config.head_dim
        else:
            self.head_dim: int = attention_module.config.hidden_size // self.num_attention_heads

        self.qkv_proj = EdgeLLMQKVProj(attention_module, eagle3_draft)
        self.o_proj = attention_module.o_proj

        self.qk_norm = EdgeLLMQKNorm(attention_module)

        # Maximum sequence length for positional embeddings
        self.max_position_embeddings: int = attention_module.config.max_position_embeddings

        # EAGLE3 draft uses 2x hidden_size input dimension
        self.eagle3_draft: bool = eagle3_draft

        if hasattr(attention_module, 'k_bmm_quantizer') and hasattr(
                attention_module, 'v_bmm_quantizer'):
            k_amax = getattr(attention_module.k_bmm_quantizer, 'amax', None)
            v_amax = getattr(attention_module.v_bmm_quantizer, 'amax', None)

            # Derive scale so that max(K,V) maps into FP8 E4M3 dynamic range.
            def _scale_quant_orig(amax):
                return (amax.cpu().float() / FP8_E4M3_MAX
                        ).view(1) if amax is not None else torch.tensor(
                            [1.0], dtype=torch.float32)

            k_scale_quant_orig = _scale_quant_orig(k_amax)
            v_scale_quant_orig = _scale_quant_orig(v_amax)
            # Pack dequant scales into a single tensor input for AttentionPlugin:
            # [k_scale_quant_orig, v_scale_quant_orig]
            k_v_scale_quant_orig = torch.cat(
                [k_scale_quant_orig.view(1),
                 v_scale_quant_orig.view(1)],
                dim=0).float()
            self.register_buffer("k_v_scale_quant_orig", k_v_scale_quant_orig)
        else:
            # Always define these buffers so downstream code can safely access them.
            self.register_buffer("k_v_scale_quant_orig", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for attention computation using the attention plugin for ONNX export.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_size)
            past_key_value: Past key-value cache of shape (batch_size, 2, num_kv_heads, max_position_embeddings, head_dim)
            rope_rotary_cos_sin: RoPE rotary embeddings of shape (batch_size, seq_len, rotary_dim)
            context_lengths: Context length tensor of shape (batch_size,)
            kvcache_start_index: Start index of KV cache of shape (kv_cache_start_batch_size,), required
            attention_mask: Attention mask of shape (batch_size, seq_len, seq_len + past_len), optional
            position_ids: Position IDs of shape (batch_size, seq_len), optional
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Attention output and updated key-value cache
        """
        bsz, q_len, _ = hidden_states.size()

        # Apply Q, K, V projections
        query_states, key_states, value_states = self.qkv_proj(hidden_states)

        norm_shape = [bsz, q_len, -1, self.head_dim]
        query_states, key_states = self.qk_norm(query_states, key_states,
                                                norm_shape)

        dtype = query_states.dtype

        # Convert to FP16 for plugin compatibility
        # For int8 quantization, we always need to explicitly convert to FP16
        query_states = query_states.to(torch.float16)
        key_states = key_states.to(torch.float16)
        value_states = value_states.to(torch.float16)

        fp8_kv_cache = past_key_value.dtype == torch.float8_e4m3fn
        if fp8_kv_cache:
            assert self.k_v_scale_quant_orig is not None, \
                "k_v_scale_quant_orig must be set when past_key_value is float8_e4m3fn"
        else:
            assert past_key_value.dtype == torch.float16, "past_key_value must be FP16 or FP8 E4M3"
        # Ensure rope embeddings are FP32
        assert rope_rotary_cos_sin.dtype == torch.float32, "rope_rotary_cos_sin must be FP32"

        # Enable tree attention if position info is available
        enable_tree_attention = attention_mask is not None and position_ids is not None

        # Call fused attention plugin
        attn_output, present_key_value = attention_plugin(
            query_states,
            key_states,
            value_states,
            past_key_value,
            context_lengths,
            rope_rotary_cos_sin,
            kvcache_start_index,
            self.num_attention_heads,
            self.num_key_value_heads,
            enable_tree_attention,
            self.head_dim,
            fp8_kv_cache,
            attention_mask,
            position_ids,
            k_v_scale_quant_orig=self.k_v_scale_quant_orig,
        )

        # Reshape output and apply final projection
        attn_output = attn_output.reshape(bsz, q_len, -1).to(dtype)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value

    def quant_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for attention computation for quantization.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_size)
            position_embeddings: Tuple of tensors, containing cos and sin
            
        Returns:
            Attention output of shape (batch_size, seq_len, hidden_size)
        """
        bsz, q_len, _ = hidden_states.size()

        # Apply Q, K, V projections
        query_states, key_states, value_states = self.qkv_proj(hidden_states)

        norm_shape = [bsz, q_len, -1, self.head_dim]
        query_states, key_states = self.qk_norm(query_states, key_states,
                                                norm_shape)

        query_states = query_states.view(bsz, q_len, self.num_attention_heads,
                                         self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                         self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(
            2, 3)) / math.sqrt(self.head_dim)

        # Causal Mask Addition
        if q_len > 1:
            # Create a causal mask to prevent attending to future tokens
            # The mask shape is (1, 1, q_len, q_len) to be broadcastable
            mask = torch.triu(torch.ones((1, 1, q_len, q_len),
                                         device=hidden_states.device,
                                         dtype=torch.bool),
                              diagonal=1)
            # Fill the masked positions with a large negative value
            attn_weights.masked_fill_(mask,
                                      torch.finfo(attn_weights.dtype).min)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1,
                                             dtype=torch.float32).to(
                                                 query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output


class EdgeLLMDecoderLayer(nn.Module):
    """
    Decoder layer with custom attention and support for EAGLE draft models.
    
    This module implements a transformer decoder layer with custom attention
    and support for different model architectures including Llama, Qwen, and
    EAGLE draft variants. It handles the differences in layer normalization
    and model structure between these architectures.
    
    For EAGLE3 draft models, this layer processes both input embeddings and
    hidden states separately, applying normalization to each before concatenation.
    For standard, it processes only hidden states.
    
    Attributes:
        hidden_size: Hidden dimension size
        mlp: Multi-layer perceptron component
        input_layernorm: Input layer normalization (optional, for EAGLE3 draft)
        post_attention_layernorm: Post-attention layer normalization
        hidden_norm: Hidden normalization for EAGLE3 draft (optional)
        self_attn: Custom attention module with fused operations
        eagle3_draft: Whether this is an EAGLE3 draft model
    """

    def __init__(self,
                 config_or_module: Union[nn.Module, Any],
                 index: int = 0,
                 torch_dtype: torch.dtype = torch.float16,
                 eagle3_draft: bool = False) -> None:
        """
        Initialize the EdgeLLMDecoderLayer module.
        
        Args:
            config_or_module: Either a decoder layer module or configuration object
            index: Layer index (used for determining layer normalization setup)
            eagle3_draft: Whether this is an EAGLE3 draft model
        """
        super().__init__()

        self.eagle3_draft = eagle3_draft
        self.torch_dtype = torch_dtype
        self.laurel = None
        self.pre_feedforward_layernorm = None
        self.post_feedforward_layernorm = None

        # Handle both config and module inputs
        if isinstance(config_or_module, nn.Module):
            # Use existing components from base model
            decoder_layer = config_or_module
            if hasattr(decoder_layer, 'hidden_size'):
                self.hidden_size: int = decoder_layer.hidden_size
            else:
                self.hidden_size: int = decoder_layer.self_attn.hidden_size
            self.mlp = decoder_layer.mlp
            self.input_layernorm = decoder_layer.input_layernorm.to(
                torch_dtype)
            self.post_attention_layernorm = decoder_layer.post_attention_layernorm.to(
                torch_dtype)

            laurel_module = getattr(decoder_layer, 'laurel', None)
            if laurel_module is not None:
                if self.eagle3_draft:
                    raise ValueError(
                        "LAuReL is not supported for EAGLE3 draft models. "
                        "EAGLE3 concatenates inputs to 2x hidden size before attention, "
                        "which is incompatible with Gemma3n LAuReL modules.")

                self.pre_feedforward_layernorm = getattr(
                    decoder_layer, 'pre_feedforward_layernorm', None)
                self.post_feedforward_layernorm = getattr(
                    decoder_layer, 'post_feedforward_layernorm', None)

                missing = []
                if self.pre_feedforward_layernorm is None:
                    missing.append("pre_feedforward_layernorm")
                if self.post_feedforward_layernorm is None:
                    missing.append("post_feedforward_layernorm")
                if missing:
                    missing_str = ", ".join(missing)
                    raise ValueError(
                        "LAuReL requires pre/post feedforward layernorms. "
                        f"Missing: {missing_str}.")

                self.laurel = laurel_module.to(torch_dtype)
                self.pre_feedforward_layernorm = self.pre_feedforward_layernorm.to(
                    torch_dtype)
                self.post_feedforward_layernorm = self.post_feedforward_layernorm.to(
                    torch_dtype)

            # Replace attention with custom implementation
            self.self_attn = EdgeLLMAttention(decoder_layer.self_attn,
                                              eagle3_draft=eagle3_draft)
        else:
            # Construct new components from config (for draft models)
            config = config_or_module
            self.hidden_size: int = config.hidden_size
            self.post_attention_layernorm = LlamaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps).to(torch_dtype)

            # Handle input layernorm based on model type and layer index
            if eagle3_draft:
                # EAGLE3 draft: all layers have input_layernorm and hidden_norm
                self.hidden_norm = LlamaRMSNorm(
                    config.hidden_size,
                    eps=config.rms_norm_eps).to(torch_dtype)
                self.input_layernorm = LlamaRMSNorm(
                    config.hidden_size,
                    eps=config.rms_norm_eps).to(torch_dtype)

            # Create attention module from config based on model type
            if "qwen" in config.model_type:
                attention_module = Qwen2Attention(config, index)
                self.mlp = Qwen2MLP(config)
            else:
                attention_module = LlamaAttention(config, index)
                self.mlp = LlamaMLP(config)

            self.self_attn = EdgeLLMAttention(attention_module,
                                              eagle3_draft=eagle3_draft)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the decoder layer for ONNX export.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, embed_dim)
            past_key_value: Cached past key-value states of shape (batch_size, 2, num_kv_heads, max_position_embeddings, head_dim)
            rope_rotary_cos_sin: RoPE rotary embeddings of shape (batch_size, seq_len, rotary_dim)
            context_lengths: Context length tensor of shape (batch_size,)
            kvcache_start_index: Start index of KV cache of shape (kv_cache_start_batch_size,), required
            inputs_embeds: Input embeddings for EAGLE3 draft of shape (batch_size, seq_len, embed_dim), optional
            attention_mask: Attention mask of shape (batch_size, seq_len, seq_len + past_len), optional
            position_ids: Position IDs of shape (batch_size, seq_len), optional
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output hidden states and updated key-value cache
        """
        residual = hidden_states

        if self.eagle3_draft:
            if inputs_embeds is None:
                raise ValueError("inputs_embeds is required for EAGLE3 draft")
            # EAGLE3 draft: apply layernorm to both inputs and concatenate
            hidden_states = self.hidden_norm(hidden_states)
            inputs_embeds = self.input_layernorm(inputs_embeds)
            hidden_states = torch.cat((inputs_embeds, hidden_states), dim=-1)
        else:
            # Standard processing: apply input layernorm if available
            if self.input_layernorm is not None:
                hidden_states = self.input_layernorm(hidden_states)

        laurel_output: Optional[torch.Tensor] = None
        if self.laurel is not None:
            laurel_output = self.laurel(hidden_states)

        # Self attention with residual connection
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            kvcache_start_index=kvcache_start_index,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            rope_rotary_cos_sin=rope_rotary_cos_sin,
            context_lengths=context_lengths,
        )
        if self.laurel is not None:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + hidden_states
            hidden_states = (hidden_states + laurel_output) * (2.0**-0.5)

            residual = hidden_states
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.post_feedforward_layernorm(hidden_states)
            hidden_states = residual + hidden_states
        else:
            hidden_states = residual + hidden_states

            # MLP with residual connection
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            if isinstance(self.mlp, Qwen3MoeSparseMoeBlock):
                hidden_states = hidden_states[0]
            hidden_states = residual + hidden_states

        return hidden_states, present_key_value

    def quant_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Forward pass through the decoder layer for quantization.

        Args:
            hidden_states: Hidden states of shape (batch_size, seq_len, embed_dim)
            position_embeddings: Tuple of tensors, containing cos and sin
            inputs_embeds: Input embeddings of shape (batch_size, seq_len, embed_dim)
        """

        residual = hidden_states

        if self.eagle3_draft:
            if inputs_embeds is None:
                raise ValueError("inputs_embeds is required for EAGLE3 draft")
            # EAGLE3 draft: apply layernorm to both inputs and concatenate
            hidden_states = self.hidden_norm(hidden_states)
            inputs_embeds = self.input_layernorm(inputs_embeds)
            hidden_states = torch.cat((inputs_embeds, hidden_states), dim=-1)
        else:
            # Standard processing: apply input layernorm if available
            if self.input_layernorm is not None:
                hidden_states = self.input_layernorm(hidden_states)

        laurel_output: Optional[torch.Tensor] = None
        if self.laurel is not None:
            laurel_output = self.laurel(hidden_states)

        # Self Attention
        hidden_states = self.self_attn.quant_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings)
        if self.laurel is not None:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + hidden_states
            hidden_states = (hidden_states + laurel_output) * (2.0**-0.5)

            residual = hidden_states
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.post_feedforward_layernorm(hidden_states)
            hidden_states = residual + hidden_states
        else:
            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            if isinstance(self.mlp, Qwen3MoeSparseMoeBlock):
                hidden_states = hidden_states[0]
            hidden_states = residual + hidden_states

        return hidden_states


class EdgeLLMDecoderLayerTRTNative(nn.Module):
    """
    Decoder layer with TensorRT native attention operations.
    
    This module implements a transformer decoder layer with TensorRT native attention operation.
    Attributes:
        hidden_size: Hidden dimension size
        mlp: Multi-layer perceptron component
        post_attention_layernorm: Post-attention layer normalization
        self_attn: TensorRT native attention module with fused operations
    """

    def __init__(self,
                 config_or_module: Union[nn.Module, Any],
                 torch_dtype: torch.dtype = torch.float16,
                 layer_index: int = 0,
                 eagle3_draft: bool = False) -> None:
        """
        Initialize the EdgeLLMDecoderLayerTRTNative module.
        
        Args:
            config_or_module: Decoder layer module
            torch_dtype: Data type of the module
        """
        super().__init__()

        self.torch_dtype = torch_dtype
        self.eagle3_draft = eagle3_draft
        self.layer_index = layer_index
        if hasattr(config_or_module, 'hidden_size'):
            self.hidden_size: int = config_or_module.hidden_size
        else:
            self.hidden_size: int = config_or_module.self_attn.hidden_size
        self.self_attn = None
        self.mlp = None
        self.input_layernorm = None
        self.post_attention_layernorm = None
        self.hidden_norm = None
        self.laurel = None
        self.pre_feedforward_layernorm = None
        self.post_feedforward_layernorm = None

        if isinstance(config_or_module, nn.Module):
            self._init_with_module(config_or_module)
        else:
            self._init_with_config(config_or_module)

        assert self.self_attn is not None
        assert self.mlp is not None

    def _init_with_module(self, decoder_layer: nn.Module):

        self.mlp = decoder_layer.mlp
        self.input_layernorm = decoder_layer.input_layernorm.to(
            self.torch_dtype)
        self.post_attention_layernorm = decoder_layer.post_attention_layernorm.to(
            self.torch_dtype)

        laurel_module = getattr(decoder_layer, 'laurel', None)
        if laurel_module is not None:
            if self.eagle3_draft:
                raise ValueError(
                    "LAuReL is not supported for EAGLE3 draft models. "
                    "EAGLE3 concatenates inputs to 2x hidden size before attention, "
                    "which is incompatible with Gemma3n LAuReL modules.")

            self.pre_feedforward_layernorm = getattr(
                decoder_layer, 'pre_feedforward_layernorm', None)
            self.post_feedforward_layernorm = getattr(
                decoder_layer, 'post_feedforward_layernorm', None)

            missing = []
            if self.pre_feedforward_layernorm is None:
                missing.append("pre_feedforward_layernorm")
            if self.post_feedforward_layernorm is None:
                missing.append("post_feedforward_layernorm")
            if missing:
                missing_str = ", ".join(missing)
                raise ValueError(
                    "LAuReL requires pre/post feedforward layernorms. "
                    f"Missing: {missing_str}.")

            self.laurel = laurel_module.to(self.torch_dtype)
            self.pre_feedforward_layernorm = self.pre_feedforward_layernorm.to(
                self.torch_dtype)
            self.post_feedforward_layernorm = self.post_feedforward_layernorm.to(
                self.torch_dtype)
        self.self_attn = EdgeLLMAttentionTRTNative(
            decoder_layer.self_attn, eagle3_draft=self.eagle3_draft)

    def _init_with_config(self, config: Any):
        # Construct new components from config (for draft models)
        self.post_attention_layernorm = LlamaRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps).to(self.torch_dtype)

        # Handle input layernorm based on model type and layer index
        if self.eagle3_draft:
            # EAGLE3 draft: all layers have input_layernorm and hidden_norm
            self.hidden_norm = LlamaRMSNorm(self.hidden_size,
                                            eps=config.rms_norm_eps).to(
                                                self.torch_dtype)
            self.input_layernorm = LlamaRMSNorm(self.hidden_size,
                                                eps=config.rms_norm_eps).to(
                                                    self.torch_dtype)

        # Create attention module from config based on model type
        if "qwen" in config.model_type:
            attention_module = Qwen2Attention(config, self.layer_index)
            self.mlp = Qwen2MLP(config)
        else:
            attention_module = LlamaAttention(config, self.layer_index)
            self.mlp = LlamaMLP(config)

        self.self_attn = EdgeLLMAttentionTRTNative(
            attention_module, eagle3_draft=self.eagle3_draft)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the decoder layer for ONNX export.
        
        Args:
            hidden_states: Input hidden states of shape (batch, seq_len, embed_dim)
            rope_rotary_cos_sin: RoPE rotary embeddings of shape (batch, seq_len, head_dim)
            context_lengths: Context length tensor of shape (batch,)
            kvcache_start_index: Start index of KV cache of shape (kv_cache_start_batch_size,), required
            k_cache: Key cache (batch, num_heads, capacity, head_dim)
            v_cache: Value cache (batch, num_heads, capacity, head_dim)
            attention_mask: Attention mask of shape (batch, seq_len, seq_len + past_len), optional
            position_ids: Position IDs of shape (batch, seq_len), optional
            inputs_embeds: Input embeddings for EAGLE3 draft (batch, seq_len, hidden_size), optional
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (hidden_states, present_k_cache, present_v_cache)
        """
        residual = hidden_states

        if self.eagle3_draft:
            if inputs_embeds is None:
                raise ValueError("inputs_embeds is required for EAGLE3 draft")
            # EAGLE3 draft: apply layernorm to both inputs and concatenate
            hidden_states = self.hidden_norm(hidden_states)
            inputs_embeds = self.input_layernorm(inputs_embeds)
            hidden_states = torch.cat((inputs_embeds, hidden_states), dim=-1)
        else:
            # Standard processing: apply input layernorm if available
            if self.input_layernorm is not None:
                hidden_states = self.input_layernorm(hidden_states)

        laurel_output: Optional[torch.Tensor] = None
        if self.laurel is not None:
            laurel_output = self.laurel(hidden_states)

        hidden_states, present_k_cache, present_v_cache = self.self_attn(
            hidden_states=hidden_states,
            k_cache=k_cache,
            v_cache=v_cache,
            rope_rotary_cos_sin=rope_rotary_cos_sin,
            context_lengths=context_lengths,
            kvcache_start_index=kvcache_start_index,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        if self.laurel is not None:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + hidden_states
            hidden_states = (hidden_states + laurel_output) * (2.0**-0.5)

            residual = hidden_states
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.post_feedforward_layernorm(hidden_states)
            hidden_states = residual + hidden_states
        else:
            hidden_states = residual + hidden_states

            # MLP with residual connection
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states, present_k_cache, present_v_cache
