# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
""" PyTorch LLaMA model."""
from typing import List, Optional


import torch.utils.checkpoint
from torch import nn

from .layers import LlamaDecoderLayeremb,load_weight_from_safetensors,LlamaRMSNorm

class Eagle3(nn.Module):

    def __init__(self,
                 config,
                 load_emb=False,
                 path=None,
                 bias=False,
                 ):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         self.padding_idx)
        self.draft_vocab_size = getattr(config, "draft_vocab_size",
                                        config.vocab_size)
        self.lm_head = nn.Linear(config.hidden_size,
                                 self.draft_vocab_size,
                                 bias=False)
        if load_emb and not hasattr(config, "target_hidden_size"):
            self.embed_tokens.weight.data = load_weight_from_safetensors(path,"model.embed_tokens.weight")
        self.hidden_size = config.hidden_size
        self.layers = nn.ModuleList([
            LlamaDecoderLayeremb(config, index, eagle3=True)
            for index in range(config.num_hidden_layers)
        ])
        if hasattr(config, "target_hidden_size"):
            self.fc = nn.Linear(config.target_hidden_size * 3,
                                self.hidden_size,
                                bias=False)
        else:
            self.fc = nn.Linear(config.hidden_size * 3,
                                self.hidden_size,
                                bias=False)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        d2t = torch.zeros((config.draft_vocab_size), dtype=torch.long)
        t2d = torch.zeros((config.vocab_size), dtype=torch.bool)
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)


    def forward(self,
                hidden_states,
                input_ids,
                hidden_states_from_draft,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                ):

        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        hidden_states = self.fc(hidden_states)
        hidden_states = hidden_states_from_draft + hidden_states
        
        next_decoder_cache = () if use_cache else None

        past_key_value = past_key_values[
            0] if past_key_values is not None else None

        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                input_emb=inputs_embeds,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=True,
            )
        if use_cache:
            next_decoder_cache += (
                layer_outputs[2 if output_attentions else 1], )
        hidden_states = layer_outputs[0]

        if use_cache:
            return hidden_states, next_decoder_cache

        return hidden_states
