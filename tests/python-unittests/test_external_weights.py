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

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnx.helper")
pytest.importorskip("onnx.numpy_helper")
safetensors_torch = pytest.importorskip("safetensors.torch")

from tensorrt_edgellm import external_weights


def _make_initializer(name, array):
    return onnx.numpy_helper.from_array(array, name=name)


def _make_float_value_info(name):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT,
                                              [1, 2])


def test_resolve_externalize_all_includes_nvfp4_moe():
    assert external_weights.EXTERNAL_WEIGHT_NVFP4_MOE in (
        external_weights.resolve_externalize_weights(
            external_weights.EXTERNAL_WEIGHT_ALL))


def test_externalize_nvfp4_moe_plugin_initializers(tmp_path):
    onnx_path = tmp_path / "model.onnx"
    plugin_inputs = [
        "router_logits",
        "hidden_states",
        "fc1_qweights",
        "fc1_blocks_scale",
        "fc1_alpha",
        "fc2_qweights",
        "fc2_blocks_scale",
        "fc2_alpha",
        "input_global_scale",
        "down_input_scale",
        "e_score_correction_bias",
    ]
    expected_external_names = plugin_inputs[2:]
    initializers = [
        _make_initializer("fc1_qweights",
                          np.arange(16, dtype=np.int8).reshape(2, 2, 4)),
        _make_initializer("fc1_blocks_scale",
                          np.arange(8, dtype=np.int8).reshape(2, 2, 2)),
        _make_initializer("fc1_alpha", np.ones((2, ), dtype=np.float32)),
        _make_initializer("fc2_qweights",
                          np.arange(16, dtype=np.int8).reshape(2, 4, 2)),
        _make_initializer("fc2_blocks_scale",
                          np.arange(8, dtype=np.int8).reshape(2, 2, 2)),
        _make_initializer("fc2_alpha", np.ones((2, ), dtype=np.float32)),
        _make_initializer("input_global_scale", np.ones((2, ),
                                                        dtype=np.float32)),
        _make_initializer("down_input_scale", np.ones((2, ),
                                                      dtype=np.float32)),
        _make_initializer("e_score_correction_bias",
                          np.zeros((2, ), dtype=np.float32)),
        _make_initializer("non_plugin_weight", np.ones((1, ),
                                                       dtype=np.float32)),
    ]
    node = onnx.helper.make_node(
        "Nvfp4MoePlugin",
        plugin_inputs,
        ["moe_output"],
        domain="trt_edgellm",
    )
    graph = onnx.helper.make_graph(
        [node],
        "nvfp4_moe_external_weight_test",
        [
            _make_float_value_info("hidden_states"),
            _make_float_value_info("router_logits"),
        ],
        [_make_float_value_info("moe_output")],
        initializers,
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 24),
            onnx.helper.make_opsetid("trt_edgellm", 1),
        ],
    )
    onnx.save_model(model, onnx_path)

    manifest = external_weights.externalize_model_weights(
        str(onnx_path), object(), externalize_weights=["nvfp4_moe"])

    assert manifest == [{
        "file": "external_nvfp4_moe_weights.safetensors",
        "kind": "nvfp4_moe_weights",
        "tensors": expected_external_names,
    }]
    saved_tensors = safetensors_torch.load_file(
        str(tmp_path / "external_nvfp4_moe_weights.safetensors"))
    assert set(saved_tensors) == set(expected_external_names)

    patched_model = onnx.load(onnx_path, load_external_data=False)
    graph_inputs = {
        graph_input.name
        for graph_input in patched_model.graph.input
    }
    remaining_initializers = {
        initializer.name
        for initializer in patched_model.graph.initializer
    }

    assert set(expected_external_names).issubset(graph_inputs)
    assert set(expected_external_names).isdisjoint(remaining_initializers)
    assert "non_plugin_weight" in remaining_initializers


def test_externalize_nvfp4_moe_geforce_plugin_initializers(tmp_path):
    onnx_path = tmp_path / "model_geforce.onnx"
    plugin_inputs = [
        "router_logits",
        "hidden_states",
        "geforce_fc1_qweights",
        "geforce_fc1_blocks_scale",
        "geforce_fc1_alpha",
        "geforce_fc2_qweights",
        "geforce_fc2_blocks_scale",
        "geforce_fc2_alpha",
        "geforce_input_global_scale",
        "geforce_down_input_scale",
        "geforce_e_score_correction_bias",
    ]
    expected_external_names = plugin_inputs[2:]
    initializers = [
        _make_initializer("geforce_fc1_qweights",
                          np.arange(16, dtype=np.int8).reshape(2, 2, 4)),
        _make_initializer("geforce_fc1_blocks_scale",
                          np.arange(8, dtype=np.int8).reshape(2, 2, 2)),
        _make_initializer("geforce_fc1_alpha", np.ones((2, ),
                                                       dtype=np.float32)),
        _make_initializer("geforce_fc2_qweights",
                          np.arange(16, dtype=np.int8).reshape(2, 4, 2)),
        _make_initializer("geforce_fc2_blocks_scale",
                          np.arange(8, dtype=np.int8).reshape(2, 2, 2)),
        _make_initializer("geforce_fc2_alpha", np.ones((2, ),
                                                       dtype=np.float32)),
        _make_initializer("geforce_input_global_scale",
                          np.ones((2, ), dtype=np.float32)),
        _make_initializer("geforce_down_input_scale",
                          np.ones((2, ), dtype=np.float32)),
        _make_initializer("geforce_e_score_correction_bias",
                          np.zeros((2, ), dtype=np.float32)),
        _make_initializer("non_plugin_weight", np.ones((1, ),
                                                       dtype=np.float32)),
    ]
    node = onnx.helper.make_node(
        "NvFP4MoEPluginGeforce",
        plugin_inputs,
        ["moe_output"],
        domain="trt_edgellm",
    )
    graph = onnx.helper.make_graph(
        [node],
        "nvfp4_moe_geforce_external_weight_test",
        [
            _make_float_value_info("hidden_states"),
            _make_float_value_info("router_logits"),
        ],
        [_make_float_value_info("moe_output")],
        initializers,
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 24),
            onnx.helper.make_opsetid("trt_edgellm", 1),
        ],
    )
    onnx.save_model(model, onnx_path)

    manifest = external_weights.externalize_model_weights(
        str(onnx_path), object(), externalize_weights=["nvfp4_moe"])

    assert manifest == [{
        "file": "external_nvfp4_moe_weights.safetensors",
        "kind": "nvfp4_moe_weights",
        "tensors": expected_external_names,
    }]
    saved_tensors = safetensors_torch.load_file(
        str(tmp_path / "external_nvfp4_moe_weights.safetensors"))
    assert set(saved_tensors) == set(expected_external_names)

    patched_model = onnx.load(onnx_path, load_external_data=False)
    graph_inputs = {
        graph_input.name
        for graph_input in patched_model.graph.input
    }
    remaining_initializers = {
        initializer.name
        for initializer in patched_model.graph.initializer
    }

    assert set(expected_external_names).issubset(graph_inputs)
    assert set(expected_external_names).isdisjoint(remaining_initializers)
    assert "non_plugin_weight" in remaining_initializers
