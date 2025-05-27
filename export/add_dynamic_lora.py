# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import os
import shutil
import time

import onnx
import onnx_graphsurgeon as gs
from utils.lora import insert_dynamic_lora


def add_dynamic_lora_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path',
                        type=str,
                        help="Path to the existing ONNX model",
                        required=True)
    parser.add_argument('--output_dir',
                        type=str,
                        help="Directory to save the modified ONNX model",
                        required=True)
    parser.add_argument('--dtype',
                        type=str,
                        default="fp16",
                        choices=["fp16", "fp8", "int4", "nvfp4", "int4_ootb"],
                        help="The precision of the ONNX model")
    return parser


def main(args):
    start_time = time.time()
    print(f"Loading ONNX model from {args.onnx_path}...")
    graph = gs.import_onnx(onnx.load(args.onnx_path))

    # Insert dynamic LoRA patterns
    graph = insert_dynamic_lora(graph, args.dtype)

    # Save the modified model
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "model.onnx")
    onnx_model = gs.export_onnx(graph)
    onnx.save_model(onnx_model,
                    output_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location="onnx_model.data",
                    convert_attribute=True)

    # Copy config.json if it exists
    config_path = os.path.join(os.path.dirname(args.onnx_path), "config.json")
    if os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(args.output_dir, "config.json"))

    end_time = time.time()
    print(f"Dynamic LoRA added in {end_time - start_time}s.")
    print(f"Modified model saved to {output_path}")


if __name__ == '__main__':
    parser = add_dynamic_lora_arguments()
    args = parser.parse_args()
    main(args)
