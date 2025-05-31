import os
import shutil
from argparse import ArgumentParser

import onnx


def modify_onnx_model(input_path, output_path):
    # Load the ONNX model
    model = onnx.load(input_path)
    graph = model.graph

    # Modify the nodes in the graph
    for node in graph.node:
        if node.op_type == "AttentionPlugin":
            for attr in node.attribute:
                if attr.name == "kv_cache_capacity":
                    attr.i = 8192  # Update the attribute value to 8192

    # Save the modified model to a new file
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    onnx.save_model(model,
                    output_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=f"onnx_model.data",
                    convert_attribute=True)

    config_path = os.path.join(os.path.dirname(input_path), "config.json")
    if os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(output_dir, "config.json"))

    print(f"Modified ONNX model saved to {output_path}")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--input_path",
                        type=str,
                        required=True,
                        help="The path to input onnx file.")
    parser.add_argument("--output_path",
                        type=str,
                        required=True,
                        help="The path to output onnx file.")
    args = parser.parse_args()

    modify_onnx_model(args.input_path, args.output_path)
