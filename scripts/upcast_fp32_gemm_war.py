import onnx
import onnx_graphsurgeon as gs
from argparse import ArgumentParser
import os


# Helper function to create a Cast node
def insert_cast(graph, input_tensor, to_dtype, suffix):
    # Avoid redundant casting
    if input_tensor.dtype == to_dtype:
        return input_tensor

    cast_out = gs.Variable(
        name=f"{input_tensor.name}_{suffix}",
        dtype=to_dtype,
        shape=input_tensor.shape
    )
    cast_node = gs.Node(
        op="Cast",
        name=f"Cast_{input_tensor.name}_{suffix}",
        inputs=[input_tensor],
        outputs=[cast_out],
        attrs={"to": to_dtype}
    )
    graph.nodes.append(cast_node)
    return cast_out


def upcast_fp32_gemm(input_path, output_path):
    # Load ONNX model
    onnx_model = onnx.load(input_path)
    graph = gs.import_onnx(onnx_model)

    for node in graph.nodes:
        # FP16 overflow only happens in /blocks.31/mlp/down_proj
        if node.name == "/blocks.31/mlp/down_proj/Gemm":
            # Insert cast to FP32 before GEMM
            original_input = node.inputs[0]
            weight_input = node.inputs[1]
            cast_input_fp32 = insert_cast(graph, original_input, onnx.TensorProto.FLOAT, "to_fp32")
            cast_weights_input_fp32 = insert_cast(graph, weight_input, onnx.TensorProto.FLOAT, "to_fp32")
            node.inputs[0] = cast_input_fp32  # Replace input of GEMM with casted tensor
            node.inputs[1] = cast_weights_input_fp32

            if len(node.inputs) > 2:
                bias_input = node.inputs[2]
                bias_input_fp32 = insert_cast(graph, bias_input, onnx.TensorProto.FLOAT, "to_fp32")
                node.inputs[2] = bias_input_fp32

            original_output = node.outputs[0]
            gemm_output_fp32 = gs.Variable(
                name=f"{original_output.name}_fp32",
                dtype=onnx.TensorProto.FLOAT,
                shape=original_output.shape
            )
            node.outputs[0] = gemm_output_fp32  # GEMM outputs to intermediate FP32 tensor

            # Insert cast to FP32 for residual Add
            add_node = original_output.outputs[0]
            original_res = add_node.inputs[0]
            cast_res_fp32 = insert_cast(graph, original_res, onnx.TensorProto.FLOAT, "to_fp32")
            add_node.inputs[0] = cast_res_fp32
            add_node.inputs[1] = gemm_output_fp32

    # Cleanup and export modified ONNX
    graph.cleanup().toposort()
    onnx_model = gs.export_onnx(graph)

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    onnx.save_model(onnx_model,
                    output_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location="onnx_model.data",
                    convert_attribute=True)

    print(f"Modified ONNX model saved to {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="The path to input onnx file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The path to output onnx file."
    )
    args = parser.parse_args()

    upcast_fp32_gemm(args.input_path, args.output_path)