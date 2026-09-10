# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import onnx

from tensorrt_edgellm.config import (LAYER_ATTN, LAYER_MAMBA, MambaConfig,
                                     ModelConfig)
from tensorrt_edgellm.models.nemotron_h.modeling_nemotron_h import \
    NemotronHCausalLM
from tensorrt_edgellm.models.nemotron_h.modeling_nemotron_h_mtp import \
    NemotronHMtpDraftModel
from tensorrt_edgellm.onnx.export import _export_model


def _config() -> ModelConfig:
    return ModelConfig(model_type="nemotron_h",
                       hidden_size=16,
                       num_hidden_layers=2,
                       num_attention_heads=2,
                       num_key_value_heads=1,
                       intermediate_size=32,
                       head_dim=8,
                       rms_norm_eps=1e-6,
                       vocab_size=32,
                       rope_theta=10_000.0,
                       max_position_embeddings=4096,
                       default_attention_scale=1.0 / math.sqrt(8.0),
                       layer_types=[LAYER_MAMBA, LAYER_ATTN],
                       mamba_cfg=MambaConfig(num_heads=2,
                                             head_dim=8,
                                             ssm_state_size=8,
                                             conv_dim=32,
                                             conv_kernel=4,
                                             n_groups=1))


def test_nemotron_h_export_uses_token_major_resident_state_contract():
    spec = NemotronHCausalLM(_config()).onnx_export_spec()
    args = dict(zip(spec.input_names, spec.args))
    shapes = dict(zip(spec.input_names, spec.dynamic_shapes))

    assert args["inputs_embeds"].ndim == 2
    assert args["conv_state_0"].shape[0] > args["query_lengths"].shape[0]
    assert args["recurrent_state_0"].shape[0] > args["query_lengths"].shape[0]
    assert "context_lengths" not in args
    assert "state_indices" in args
    assert args["execution_phase_marker"].shape == (2, )
    assert args["context_sequence_count_carrier"].shape == args[
        "query_lengths"].shape
    assert shapes["context_sequence_count_carrier"][0] != shapes[
        "query_lengths"][0]
    assert "execution_phase_marker" in args
    assert "context_sequence_count_carrier" in args

    outputs = spec.wrapped(*spec.args)
    assert outputs[0].shape == (2, 32)
    assert outputs[1].shape == args["past_key_values_0"].shape
    assert outputs[2].shape == args["conv_state_0"].shape
    assert outputs[3].shape == args["recurrent_state_0"].shape


def test_nemotron_h_export_routes_ragged_metadata_to_plugins(tmp_path):
    output = tmp_path / "nemotron-h-ragged.onnx"
    _export_model(NemotronHCausalLM(_config()), str(output), optimize=False)
    graph = onnx.load(str(output), load_external_data=False).graph
    nodes = {
        node.op_type: node
        for node in graph.node if node.op_type in
        {"AttentionPlugin", "causal_conv1d", "update_ssm_state"}
    }

    assert set(nodes) == {
        "AttentionPlugin", "causal_conv1d", "update_ssm_state"
    }
    assert set(nodes["AttentionPlugin"].input) >= {
        "query_start_offsets", "query_lengths", "past_lengths",
        "attention_sequence_lengths", "execution_phase_marker",
        "context_sequence_count_carrier"
    }
    for op_type in ("causal_conv1d", "update_ssm_state"):
        assert set(nodes[op_type].input) >= {
            "query_start_offsets", "query_lengths", "state_indices",
            "execution_phase_marker", "context_sequence_count_carrier"
        }


def test_nemotron_h_spec_base_logits_selection_has_independent_dynamic_axis(
        tmp_path):
    config = _config()
    config.mtp_base = True
    output = tmp_path / "nemotron-h-spec-base-ragged.onnx"

    _export_model(NemotronHCausalLM(config), str(output), optimize=False)

    graph = onnx.load(str(output), load_external_data=False).graph
    inputs = {tensor.name: tensor for tensor in graph.input}
    assert inputs["inputs_embeds"].type.tensor_type.shape.dim[
        0].dim_param == "physical_tokens"
    assert inputs["logits_indices"].type.tensor_type.shape.dim[
        0].dim_param == "logits_rows"


def test_nemotron_h_mtp_draft_exports_token_major_attention(tmp_path):
    config = _config()
    config.num_hidden_layers = 1
    config.layer_types = [LAYER_ATTN]
    output = tmp_path / "nemotron-h-mtp-ragged.onnx"
    model = NemotronHMtpDraftModel(config)
    spec = model.onnx_export_spec()
    args = dict(zip(spec.input_names, spec.args))
    shapes = dict(zip(spec.input_names, spec.dynamic_shapes))

    assert args["inputs_embeds"].ndim == 2
    assert "context_lengths" not in args
    assert "state_indices" in args
    assert args["execution_phase_marker"].shape == (2, )
    assert args["context_sequence_count_carrier"].shape == args[
        "query_lengths"].shape
    assert shapes["context_sequence_count_carrier"][0] != shapes[
        "query_lengths"][0]
    outputs = spec.wrapped(*spec.args)
    assert outputs[0].shape == (2, 32)
    assert outputs[1].shape == (2, 16)

    _export_model(model, str(output), optimize=False)
    graph = onnx.load(str(output), load_external_data=False).graph
    attention = next(node for node in graph.node
                     if node.op_type == "AttentionPlugin")
    assert set(attention.input) >= {
        "query_start_offsets", "query_lengths", "past_lengths",
        "attention_sequence_lengths", "execution_phase_marker",
        "context_sequence_count_carrier"
    }
