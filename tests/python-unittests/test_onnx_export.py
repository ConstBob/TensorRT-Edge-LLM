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

from tensorrt_edgellm.onnx.export import _fix_initializer_dtypes


def test_fix_initializer_dtypes_skips_external_data_load_when_no_changes(
        tmp_path, monkeypatch):
    onnx_path = tmp_path / "model.onnx"
    data_path = "model.onnx.data"
    initializer = onnx.numpy_helper.from_array(np.ones((2, ),
                                                       dtype=np.float32),
                                               name="fp32_elementwise_const")
    node = onnx.helper.make_node("Add", ["input", initializer.name],
                                 ["output"])
    graph = onnx.helper.make_graph(
        [node],
        "metadata_only_fixup_test",
        [
            onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT,
                                               [2])
        ],
        [
            onnx.helper.make_tensor_value_info("output",
                                               onnx.TensorProto.FLOAT, [2])
        ],
        [initializer],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 24)])
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path,
        size_threshold=0,
    )

    metadata_model = onnx.load(onnx_path, load_external_data=False)
    assert metadata_model.graph.initializer[0].external_data

    original_load = onnx.load
    load_external_data_args = []

    def guarded_load(*args, **kwargs):
        load_external_data_args.append(kwargs.get("load_external_data"))
        if kwargs.get("load_external_data") is not False:
            raise AssertionError("unexpected full external-data load")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(onnx, "load", guarded_load)

    _fix_initializer_dtypes(str(onnx_path),
                            dedup_dql_scales=True,
                            match_fp32_elementwise_initializers=True)

    assert load_external_data_args == [False]
