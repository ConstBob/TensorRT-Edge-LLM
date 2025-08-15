import os
import shutil
from argparse import ArgumentParser

import onnx


def modify_onnx_model(input_dir, output_dir, kv_cache_capacity=8192):
    # Load the ONNX model
    input_path = os.path.join(input_dir, "model.onnx")
    model = onnx.load(input_path)
    graph = model.graph

    # Modify the nodes in the graph
    for node in graph.node:
        if node.op_type == "AttentionPlugin":
            for attr in node.attribute:
                if attr.name == "kv_cache_capacity":
                    attr.i = kv_cache_capacity  # Update the attribute value to 8192

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

    parser = ArgumentParser()
    parser.add_argument("--input_dir",
                        type=str,
                        required=True,
                        help="The path to input onnx directory.")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="The path to output onnx directory.")
    parser.add_argument("-kv",
                        "--kv_cache_capacity",
                        type=int,
                        default=8192,
                        help="The kv cache capacity for the attention plugin.")
    args = parser.parse_args()

    modify_onnx_model(args.input_dir, args.output_dir, args.kv_cache_capacity)
