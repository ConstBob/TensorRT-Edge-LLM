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
Qwen3-TTS Audio Models for ONNX Export.

- Tokenizer-12Hz Decoder: CausalConv vocoder (codes → waveform)
- Speaker Encoder: ECAPA-TDNN (mel → speaker embedding)

Note: LLM-based audio generation models (Talker, CodePredictor) are in
llm_models/models/qwen3_omni_talker.py (shared with Qwen3-Omni).
"""

import json
import os

import torch
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder, Qwen3TTSTokenizerV2DecoderTransformerModel)

from ..onnx_export.onnx_utils import export_onnx

# ============================================================================
# Tokenizer-12Hz Decoder — ONNX Export Patches
# ============================================================================
#
# The pre_transformer uses create_causal_mask / create_sliding_window_causal_mask
# which are not compatible with torch.export (dynamo). We replace it with a
# patched version that builds the attention mask explicitly, same approach as
# Qwen3OmniCode2WavTransformerModelPatch in qwen3_omni_model.py.


class Qwen3TTSTokenizer12HzTransformerPatch(
        Qwen3TTSTokenizerV2DecoderTransformerModel):
    """Patched pre_transformer with explicit attention mask for ONNX export."""

    def __init__(self, config):
        super().__init__(config)

    def forward(self, inputs_embeds=None):
        inputs_embeds = self.input_proj(inputs_embeds)

        past_seen_tokens = 0
        cache_position = torch.arange(past_seen_tokens,
                                      past_seen_tokens +
                                      inputs_embeds.shape[1],
                                      device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        hidden_states = inputs_embeds

        seq_len = inputs_embeds.shape[1]
        q_pos = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(1)
        k_pos = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)

        # Full causal mask: k <= q
        full_causal = (k_pos <= q_pos)
        full_mask = torch.where(full_causal, 0,
                                torch.finfo(inputs_embeds.dtype).min).to(
                                    inputs_embeds.dtype)

        # Sliding window causal mask: (q - window) < k <= q
        if self.has_sliding_layers:
            sliding_causal = (k_pos > q_pos - self.window_size) & (k_pos
                                                                   <= q_pos)
            sliding_mask = torch.where(
                sliding_causal, 0,
                torch.finfo(inputs_embeds.dtype).min).to(inputs_embeds.dtype)

        mask_mapping = {"full_attention": full_mask}
        if self.has_sliding_layers:
            mask_mapping["sliding_attention"] = sliding_mask

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.output_proj(hidden_states)

        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class Qwen3TTSTokenizer12HzDecoderPatch(Qwen3TTSTokenizerV2Decoder):
    """Patched decoder with ONNX-compatible pre_transformer."""

    def __init__(self, config):
        super().__init__(config)
        self.pre_transformer = Qwen3TTSTokenizer12HzTransformerPatch(config)


def export_qwen3_tts_tokenizer_decoder(model, output_dir, torch_dtype):
    """Export Tokenizer-12Hz decoder to ONNX.

    Uses patched pre_transformer (explicit attention mask) and torchscript exporter
    (dynamo fails on aten.unbind in SplitResidualVectorQuantizer.decode).
    """
    os.makedirs(output_dir, exist_ok=True)

    patched = Qwen3TTSTokenizer12HzDecoderPatch(model.config)
    patched.load_state_dict(model.state_dict())
    patched.eval().to(model.device)

    num_quantizers = patched.config.num_quantizers
    codes = torch.randint(0,
                          patched.config.codebook_size,
                          (2, num_quantizers, 300),
                          dtype=torch.int64,
                          device=patched.device)

    export_onnx(patched, (codes, ),
                output_dir,
                input_names=['codes'],
                output_names=['waveform'],
                dynamic_axes={
                    'codes': {
                        0: 'batch',
                        2: 'code_len'
                    },
                    'waveform': {
                        0: 'batch',
                        2: 'waveform_len'
                    },
                })

    config_dict = model.config.to_dict()
    config_dict['model_type'] = 'qwen3_tts_code2wav'
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Exported tokenizer_decoder to {output_dir}")


# ============================================================================
# Speaker Encoder (ECAPA-TDNN)
# ============================================================================


def export_qwen3_tts_speaker_encoder(model, config, output_dir, torch_dtype):
    """Export ECAPA-TDNN speaker encoder to ONNX.

    Args:
        model: Qwen3TTSSpeakerEncoder (nn.Module, not PreTrainedModel)
        config: Qwen3TTSSpeakerEncoderConfig from the parent model
        output_dir: Directory to save the exported ONNX model
        torch_dtype: PyTorch data type for the model
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    device = next(model.parameters()).device
    mel = torch.randn(1, 200, config.mel_dim, dtype=torch_dtype, device=device)

    export_onnx(model, (mel, ),
                output_dir,
                input_names=['mel_spectrogram'],
                output_names=['speaker_embedding'],
                dynamic_axes={
                    'mel_spectrogram': {
                        0: 'batch',
                        1: 'mel_len'
                    },
                    'speaker_embedding': {
                        0: 'batch'
                    },
                })

    config_dict = {
        k: v
        for k, v in config.__dict__.items()
        if isinstance(v, (int, float, str, bool, list, dict, type(None)))
    }
    config_dict['model_type'] = 'qwen3_tts_speaker_encoder'
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Exported speaker_encoder to {output_dir}")
