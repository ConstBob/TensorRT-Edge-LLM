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
"""
Qwen3-Omni Audio Models for ONNX Export.

This module contains ONNX export adapters for Qwen3-Omni's audio processing components:
- AudioEncoder: Processes raw audio into features
- Code2Wav: Converts RVQ codes to audio waveform (vocoder)

Note: LLM-based audio generation models (Talker, CodePredictor) are in llm_models/qwen3_omni_talker.py
"""

from typing import Any

import torch
import torch.nn as nn
from transformers.models.qwen3_omni.modeling_qwen3_omni import (
    Qwen3OmniAudioAttention, Qwen3OmniAudioEncoder, Qwen3OmniAudioEncoderLayer,
    Qwen3OmniCode2Wav, Qwen3OmniCode2WavTransformerModel)

from ..onnx_export.onnx_utils import export_onnx, export_onnx_dynamo
from .audio_utils import eager_attention_forward


class Qwen3OmniAudioAttentionPatch(Qwen3OmniAudioAttention):
    """
    Patched version of Qwen3-Omni audio attention for ONNX export.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with custom attention implementation.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Optional attention mask with shape [num_attention_elems, num_attention_elems]
            
        Returns:
            Attention output
        """
        seq_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).reshape(
            seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(
            seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(
            seq_length, self.num_heads, -1)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = eager_attention_forward

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            scaling=self.scaling,
            attention_mask=attention_mask,
        )

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output


class Qwen3OmniAudioEncoderLayerPatch(Qwen3OmniAudioEncoderLayer):
    """
    Patched version of Qwen3OmniAudioEncoderLayer with custom attention mechanism.
    
    This class replaces the original attention mechanism with a custom implementation
    that is compatible with ONNX export.
    """

    def __init__(self,
                 config: Any,
                 attn_implementation: str = "eager") -> None:
        super().__init__(config)
        self.self_attn = Qwen3OmniAudioAttentionPatch(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer
            attention_mask (`torch.FloatTensor`, optional): attention mask with shape [num_attention_elems, num_attention_elems]

        Returns:
            hidden_states (`torch.FloatTensor`): output of the layer
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states,
                                       attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states,
                                        min=-clamp_value,
                                        max=clamp_value)

        return hidden_states


class Qwen3OmniAudioEncoderPatch(Qwen3OmniAudioEncoder):
    """
    Patched version of Qwen3OmniAudioEncoder for ONNX export.
    
    This class provides a wrapper around the original Qwen3-Omni audio encoder
    with custom blocks that are compatible with ONNX export.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize the patched audio encoder.
        
        Args:
            config: Model configuration object
        """
        super().__init__(config)
        # Replace all blocks with patched versions
        self.layers = nn.ModuleList([
            Qwen3OmniAudioEncoderLayerPatch(config,
                                            config._attn_implementation)
            for _ in range(config.encoder_layers)
        ])

    def forward(self, padded_feature: torch.Tensor,
                padded_mask_after_cnn_indices: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the audio encoder.
        Three changes have been made to the original forward pass:
        1. padded_feature preprocessing is not TRT friendly and is moved to runtime.
        2. boolean indexing using padded_mask_after_cnn results in NonZero node in ONNX and is not TRT friendly.
        Instead, use padded_mask_after_cnn_indices as model input: padded_mask_after_cnn_indices = torch.nonzero(padded_mask_after_cnn).
        3. cu_seqlens is only used in Flash Attention, not in eager Attention, so after_cnn_lens input is dropped. In its place, attention_mask is used.
        
        Args:
            padded_feature: Padded feature tensor [num_chunks, num_mel_bins, n_window]
            padded_mask_after_cnn_indices: Indices for the boolean padded mask after CNN layers [num_attention_elems, 2]
            attention_mask: Optional attention mask with shape [num_attention_elems, num_attention_elems]

        Returns:
            `torch.Tensor`: hidden_states.
        """
        padded_feature = padded_feature.unsqueeze(1)

        padded_embed = torch.nn.functional.gelu(self.conv2d1(padded_feature))
        padded_embed = torch.nn.functional.gelu(self.conv2d2(padded_embed))
        padded_embed = torch.nn.functional.gelu(self.conv2d3(padded_embed))

        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(
            padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        positional_embedding = (
            self.positional_embedding.
            positional_embedding[:padded_embed.shape[1], :].unsqueeze(0).to(
                padded_embed.dtype))
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn_indices[:, 0],
                                     padded_mask_after_cnn_indices[:, 1]]

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(hidden_states,
                                          attention_mask=attention_mask)
            hidden_states = layer_outputs

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states


class Qwen3OmniCode2WavTransformerModelPatch(Qwen3OmniCode2WavTransformerModel
                                             ):
    """
    Patched version of Qwen3OmniCode2WavTransformerModel for ONNX export.
    This class replaces the original sliding window attention mask generation logic with a custom implementation
    that is compatible with ONNX export.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)

    def forward(
        self,
        inputs_embeds=None,
        **kwargs,
    ) -> torch.Tensor:

        past_seen_tokens = 0
        cache_position = torch.arange(past_seen_tokens,
                                      past_seen_tokens +
                                      inputs_embeds.shape[1],
                                      device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        # Create sliding window attention mask with a window size of 72.
        # A token at query position q can attend to key position k if: (q - 72) < k <= q
        seq_len = inputs_embeds.shape[1]
        q_pos = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(1)
        k_pos = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
        valid_mask = (k_pos > q_pos - 72) & (k_pos <= q_pos)
        attention_mask = torch.where(valid_mask, 0,
                                     torch.finfo(inputs_embeds.dtype).min).to(
                                         inputs_embeds.dtype)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3OmniCode2WavModelPatch(Qwen3OmniCode2Wav):
    """
    Patched version of Qwen3OmniCode2WavModel for ONNX export.
    The pre-transformer is replaced with a patched version whose sliding window attention mask generation logic is compatible with ONNX export.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        # Replace pre_transformer with patched version
        self.pre_transformer = Qwen3OmniCode2WavTransformerModelPatch(config)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the code2wav model.
        """
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(
                f"Expected {self.config.num_quantizers} layer of codes, got {codes.shape[1]}"
            )
        hidden = self.code_embedding(codes + self.code_offset).mean(1)
        hidden = self.pre_transformer(inputs_embeds=hidden)
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)


def export_qwen3_omni_audio(
    model: Qwen3OmniAudioEncoderPatch,
    output_dir: str,
    torch_dtype: torch.dtype,
) -> None:
    """
    Export Qwen3-Omni audio encoder model to ONNX format.
    
    This function takes a patched Qwen3-Omni audio encoder model, prepares dummy inputs 
    for ONNX export, and saves the model in ONNX format.
    
    Args:
        model: Patched Qwen3-Omni audio encoder model
        output_dir: Directory to save the exported ONNX model
        torch_dtype: PyTorch data type for the model
    """

    # Prepare dummy inputs for ONNX export
    num_mel_bins = model.config.num_mel_bins
    n_window = model.config.n_window
    num_chunks = 3
    padded_mask_chunk_length = 13
    num_attention_elems = 38

    # Create input tensors with appropriate shapes and dtypes
    padded_feature = torch.randn(num_chunks,
                                 num_mel_bins,
                                 n_window * 2,
                                 dtype=torch_dtype,
                                 device=model.device)
    padded_mask_after_cnn = torch.tensor(
        [True] * num_attention_elems + [False] *
        (num_chunks * padded_mask_chunk_length - num_attention_elems),
        device=model.device).reshape(num_chunks, padded_mask_chunk_length)
    padded_mask_after_cnn_indices = torch.nonzero(padded_mask_after_cnn)

    # Block-diagonal attention mask matching _prepare_attention_mask + cu_seqlens logic.
    # Tokens within the same window attend to each other; cross-window attention is masked.
    attention_mask = torch.full(
        [num_attention_elems, num_attention_elems],
        torch.finfo(torch_dtype).min,
        device=model.device,
        dtype=torch_dtype,
    )
    cu_seqlens = [0, 26, 38]
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i],
                       cu_seqlens[i - 1]:cu_seqlens[i]] = 0

    inputs = (padded_feature, padded_mask_after_cnn_indices, attention_mask)
    input_names = [
        "padded_feature", "padded_mask_after_cnn_indices", "attention_mask"
    ]
    output_names = ["last_hidden_state"]

    # Define dynamic axes for variable input sizes
    dynamic_axes = {
        # Model inputs
        'padded_feature': {
            0: 'num_chunks'
        },
        'padded_mask_after_cnn_indices': {
            0: 'num_attention_elems'
        },
        'attention_mask': {
            0: 'num_attention_elems',
            1: 'num_attention_elems'
        },
        # Model outputs
        'last_hidden_state': {
            0: 'num_attention_elems'
        },
    }

    export_onnx(model, inputs, output_dir, input_names, output_names,
                dynamic_axes)


def export_qwen3_omni_code2wav(
    model: Qwen3OmniCode2WavModelPatch,
    output_dir: str,
) -> None:
    """
    Export Qwen3-Omni code2wav model to ONNX format.
    The ONNX dynamo exporter is used here as torchscript cannot resolve some dynamic shapes
    in the Qwen3OmniCausalConvNet module correctly.
    """
    # Model configuration
    num_quantizers = model.config.num_quantizers  # 16 for Qwen3-Omni

    # Dummy input dimensions
    batch_size = 2
    # Chunked decode uses code length of 300
    opt_code_len = 300

    # Prepare dummy input with optimal length
    codes = torch.randint(0,
                          model.config.codebook_size,
                          (batch_size, num_quantizers, opt_code_len),
                          dtype=torch.int64,
                          device=model.device)
    # Pack inputs
    inputs = (codes, )

    input_names = ['codes']
    output_names = ['waveform']

    # Define dynamic axes for variable-length codes
    input_dynamic_shapes = (
        # Input 0: 'codes'
        {
            0: 'batch',
            2: 'code_len'
        },  # dim 1 is fixed (num_quantizers=16)
    )
    output_dynamic_shapes = (
        # Output 0: 'waveform'
        {
            0: 'batch',
            2: 'waveform_len'
        },  # dim 1 is fixed (1 channel)
    )

    # Use dynamo exporter with OPSET 22 to avoid RMSNormalization
    # OPSET 23+ introduces RMSNormalization as a native op, which TensorRT 10.13 doesn't support
    # OPSET 22 forces RMSNorm to be decomposed into basic ops (Pow, ReduceMean, Sqrt, Mul)
    export_onnx_dynamo(model,
                       inputs,
                       output_dir,
                       input_names=input_names,
                       output_names=output_names,
                       input_dynamic_shapes=input_dynamic_shapes,
                       output_dynamic_shapes=output_dynamic_shapes,
                       opset_version=22)
