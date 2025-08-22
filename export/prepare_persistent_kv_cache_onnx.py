# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import shutil
from argparse import ArgumentParser

import onnx


def modify_onnx_model(input_dir, output_dir):
    """
    Modify ONNX model to add persistent KV cache support to AttentionPlugin nodes.
    
    For each AttentionPlugin:
    1. Add has_persistent_kv_cache = 1 attribute
    2. Add kvcache_start_index input (int32)
    
    Args:
        input_dir: Directory containing the input ONNX model
        output_dir: Directory to save the modified ONNX model
    """
    # Load the ONNX model
    input_path = os.path.join(input_dir, "model.onnx")
    model = onnx.load(input_path)
    graph = model.graph

    attention_plugin_count = 0

    # Add shared kvcache_start_index input to the graph inputs (only once)
    kvcache_start_index_input = "kvcache_start_index"
    kvcache_start_index_value_info = onnx.ValueInfoProto()
    kvcache_start_index_value_info.name = kvcache_start_index_input
    kvcache_start_index_value_info.type.tensor_type.elem_type = onnx.TensorProto.INT32
    # Dynamic shape with batch_size dimension
    dim = kvcache_start_index_value_info.type.tensor_type.shape.dim.add()
    dim.dim_param = "batch_size"
    graph.input.append(kvcache_start_index_value_info)

    # Modify the nodes in the graph
    for node in graph.node:
        if node.op_type == "AttentionPlugin":
            attention_plugin_count += 1

            # Add has_persistent_kv_cache attribute
            has_persistent_attr = onnx.AttributeProto()
            has_persistent_attr.name = "has_persistent_kv_cache"
            has_persistent_attr.type = onnx.AttributeProto.INT
            has_persistent_attr.i = 1
            node.attribute.append(has_persistent_attr)

            # Add shared kvcache_start_index input to the node
            node.input.append(kvcache_start_index_input)

    print(f"Modified {attention_plugin_count} AttentionPlugin nodes")

    # Save the modified model to a new file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model.onnx")
    onnx.save_model(model,
                    output_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=f"onnx_model.data",
                    convert_attribute=True)

    # Copy all other files from input_dir to output_dir except onnx files
    for filename in os.listdir(input_dir):
        if "onnx" in filename:
            continue
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)
        if os.path.isfile(src_path):
            shutil.copy(src_path, dst_path)

    print(f"Modified ONNX model saved to {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description=
        "Add persistent KV cache support to ONNX models with AttentionPlugin")
    parser.add_argument("--input_dir",
                        type=str,
                        required=True,
                        help="The path to input onnx directory.")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="The path to output onnx directory.")
    args = parser.parse_args()

    modify_onnx_model(args.input_dir, args.output_dir)
