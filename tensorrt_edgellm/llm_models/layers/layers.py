# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (LlamaAttention, LlamaMLP,
                                                      LlamaRMSNorm)
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2MLP

from .attention_plugin import attention_plugin


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

        # Copy projection layers from original attention module
        self.q_proj = attention_module.q_proj
        self.k_proj = attention_module.k_proj
        self.v_proj = attention_module.v_proj
        self.o_proj = attention_module.o_proj

        # Qwen3 models have QK normalization layers
        if hasattr(attention_module, 'q_norm'):
            self.q_norm = attention_module.q_norm
        else:
            self.q_norm = None

        if hasattr(attention_module, 'k_norm'):
            self.k_norm = attention_module.k_norm
        else:
            self.k_norm = None

        # Llama4 models have QK normalization layers
        if hasattr(attention_module, 'qk_norm'):
            self.qk_norm = attention_module.qk_norm
        else:
            self.qk_norm = None

        # Copy configuration attributes from the original attention module
        self.hidden_size: int = attention_module.config.hidden_size
        self.num_key_value_heads: int = attention_module.config.num_key_value_heads
        self.num_attention_heads: int = attention_module.config.num_attention_heads

        # Set head dimension
        if hasattr(attention_module.config, 'head_dim'):
            self.head_dim: int = attention_module.config.head_dim
        else:
            self.head_dim: int = attention_module.config.hidden_size // self.num_attention_heads

        # Maximum sequence length for positional embeddings
        self.max_position_embeddings: int = attention_module.config.max_position_embeddings

        # EAGLE3 draft uses 2x hidden_size input dimension
        self.eagle3_draft: bool = eagle3_draft

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for attention computation using the attention plugin.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_size)
            past_key_value: Past key-value cache of shape (batch_size, 2, num_kv_heads, max_position_embeddings, head_dim)
            rope_rotary_cos_sin: RoPE rotary embeddings of shape (batch_size, seq_len, head_dim)
            context_lengths: Context length tensor of shape (batch_size,)
            attention_mask: Attention mask of shape (batch_size, seq_len, seq_len + past_len), optional
            position_ids: Position IDs of shape (batch_size, seq_len), optional
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Attention output and updated key-value cache
        """
        bsz, q_len, _ = hidden_states.size()

        # Apply Q, K, V projections
        query_states = self.q_proj(hidden_states)
        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
        key_states = self.k_proj(hidden_states)
        if self.k_norm is not None:
            key_states = self.k_norm(key_states)

        if self.qk_norm is not None:
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        value_states = self.v_proj(hidden_states)

        # Concatenate QKV for the plugin
        qkv = torch.concat([query_states, key_states, value_states], dim=-1)

        dtype = qkv.dtype

        # Convert to FP16 for plugin compatibility
        if qkv.dtype != torch.float16:
            qkv = qkv.to(torch.float16)
        if past_key_value.dtype != torch.float16:
            past_key_value = past_key_value.to(torch.float16)

        # Ensure rope embeddings are FP32
        assert rope_rotary_cos_sin.dtype == torch.float32, "rope_rotary_cos_sin must be FP32"

        # Enable tree attention if position info is available
        enable_tree_attention = attention_mask is not None and position_ids is not None

        # Call fused attention plugin
        attn_output, present_key_value = attention_plugin(
            qkv,
            past_key_value,
            context_lengths,
            rope_rotary_cos_sin,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.max_position_embeddings,
            enable_tree_attention,
            self.head_dim,
            attention_mask,
            position_ids,
        )

        # Reshape output and apply final projection
        attn_output = attn_output.reshape(bsz, q_len, -1).to(dtype)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value


class EdgeLLMDecoderLayer(nn.Module):
    """
    Decoder layer with custom attention and support for EAGLE draft models.
    
    This module implements a transformer decoder layer with custom attention
    and support for different model architectures including Llama, Qwen, and
    EAGLE draft variants. It handles the differences in layer normalization
    and model structure between these architectures.
    
    For EAGLE3 draft models, this layer processes both input embeddings and
    hidden states separately, applying normalization to each before concatenation.
    For standard and EAGLE2 draft models, it processes only hidden states.
    
    Attributes:
        hidden_size: Hidden dimension size
        mlp: Multi-layer perceptron component
        input_layernorm: Input layer normalization (optional, for EAGLE2 draft layer 0+ and EAGLE3 draft)
        post_attention_layernorm: Post-attention layer normalization
        hidden_norm: Hidden normalization for EAGLE3 draft (optional)
        self_attn: Custom attention module with fused operations
        eagle3_draft: Whether this is an EAGLE3 draft model
    """

    def __init__(self,
                 config_or_module: Union[nn.Module, Any],
                 index: int = 0,
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

        # Handle both config and module inputs
        if isinstance(config_or_module, nn.Module):
            # Use existing components from base model
            decoder_layer = config_or_module
            self.hidden_size: int = decoder_layer.hidden_size
            self.mlp = decoder_layer.mlp
            self.input_layernorm = decoder_layer.input_layernorm
            self.post_attention_layernorm = decoder_layer.post_attention_layernorm

            # Replace attention with custom implementation
            self.self_attn = EdgeLLMAttention(decoder_layer.self_attn,
                                              eagle3_draft=eagle3_draft)
        else:
            # Construct new components from config (for draft models)
            config = config_or_module
            self.hidden_size: int = config.hidden_size
            self.post_attention_layernorm = LlamaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps)

            # Handle input layernorm based on model type and layer index
            if eagle3_draft:
                # EAGLE3 draft: all layers have input_layernorm and hidden_norm
                self.hidden_norm = LlamaRMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
                self.input_layernorm = LlamaRMSNorm(config.hidden_size,
                                                    eps=config.rms_norm_eps)
            else:
                # EAGLE2 draft: layer 0 doesn't have input_layernorm
                if not config.input_layernorm:
                    self.input_layernorm = None
                else:
                    self.input_layernorm = LlamaRMSNorm(
                        config.hidden_size, eps=config.rms_norm_eps)

            # Create attention module from config based on model type
            if "qwen" in config.model_type:
                attention_module = Qwen2Attention(config, index)
                self.mlp = Qwen2MLP(config)
            else:
                attention_module = LlamaAttention(config, index)
                self.mlp = LlamaMLP(config)

            self.self_attn = EdgeLLMAttention(attention_module,
                                              eagle3_draft=eagle3_draft)
            if eagle3_draft:
                # Double the input dimension for the attention module
                self.self_attn.q_proj = nn.Linear(
                    attention_module.q_proj.in_features * 2,
                    attention_module.q_proj.out_features,
                    bias=attention_module.q_proj.bias)
                self.self_attn.k_proj = nn.Linear(
                    attention_module.k_proj.in_features * 2,
                    attention_module.k_proj.out_features,
                    bias=attention_module.k_proj.bias)
                self.self_attn.v_proj = nn.Linear(
                    attention_module.v_proj.in_features * 2,
                    attention_module.v_proj.out_features,
                    bias=attention_module.v_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the decoder layer.
        
        Args:
            hidden_states: Input hidden states of shape (batch, seq_len, embed_dim)
            past_key_value: Cached past key-value states of shape (batch_size, 2, num_kv_heads, max_position_embeddings, head_dim)
            rope_rotary_cos_sin: RoPE rotary embeddings of shape (batch, seq_len, head_dim)
            context_lengths: Context length tensor of shape (batch,)
            inputs_embeds: Input embeddings for EAGLE3 draft of shape (batch, seq_len, embed_dim), optional
            attention_mask: Attention mask of shape (batch, seq_len, seq_len + past_len), optional
            position_ids: Position IDs of shape (batch, seq_len), optional
            
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

        # Self attention with residual connection
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            rope_rotary_cos_sin=rope_rotary_cos_sin,
            context_lengths=context_lengths,
        )
        hidden_states = residual + hidden_states

        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value
