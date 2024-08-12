import argparse
from optimum.exporters.onnx import main_export
import onnx_graphsurgeon as gs
import onnx
import numpy as np
import os
import time

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help="The name or path of HF model ckpt", required=True)
    parser.add_argument('--output_dir', type=str, help="The directory to store the generated ONNX model", required=True)
    parser.add_argument('--dtype', type=str, default="fp16", help="The floating point precision to use for export")
    args = parser.parse_args()
    return args

def clear_inputs(node):
    for i in node.inputs:
        i.outputs.clear()
    node.inputs.clear()
    return node

def clear_outputs(node):
    for o in node.outputs:
        o.inputs.clear()
    node.outputs.clear()
    return node

def surgeon_graph(graph):
    # Look for kv cache inputs
    kv_inputs = {}
    removed_inputs = []
    for input in graph.inputs:
        # We need to remove them in the Graph
        if "attention_mask" in input.name or "position_ids" in input.name:
            removed_inputs.append(clear_outputs(input))
        if "past_key_values" in input.name:
            kv_inputs[input.name] = clear_outputs(input)
            # Remove kv inputs
            removed_inputs.append(kv_inputs[input.name])

    # Should not remove while iterating, so remove them separately
    for i in removed_inputs:
        graph.inputs.remove(i)

    context_length = gs.Variable("context_length", np.int64, [1])
    graph.inputs.append(context_length)

    # Look for kv cache outputs
    kv_outputs = {}
    for output in graph.outputs:
        if "present" in output.name:
            kv_outputs[output.name] = clear_inputs(output)
    
    # Remove kv outputs
    for name in kv_outputs:
        graph.outputs.remove(kv_outputs[name])

    # Look for qkv gemm and end of MHA nodes.
    q_outputs = {}
    k_outputs = {}
    v_outputs = {}
    attention_outputs = {}
    for node in graph.nodes:
        if "q_proj/MatMul" in node.name and node.op == "MatMul":
            assert len(node.outputs) == 1, "You did not reach the proper q_proj MatMul tensor!"
            q_output = node.outputs[0]
            q_outputs[q_output.name] = clear_outputs(q_output)
        if "k_proj/MatMul" in node.name and node.op == "MatMul":
            assert len(node.outputs) == 1, "You did not reach the proper k_proj MatMul tensor!"
            k_output = node.outputs[0]
            k_outputs[k_output.name] = clear_outputs(k_output)
        if "v_proj/MatMul" in node.name and node.op == "MatMul":
            assert len(node.outputs) == 1, "You did not reach the proper v_proj MatMul tensor!"
            v_output = node.outputs[0]
            v_outputs[v_output.name] = clear_outputs(v_output)
        if "self_attn/MatMul_1" in node.name and node.op == "MatMul":
            assert len(node.outputs) == 1, "You did not reach the proper MatMul tensor!"
            attention_output = node.outputs[0]
            attention_outputs[attention_output.name] = clear_inputs(attention_output)

    assert len(kv_inputs) % 2 == 0, "kv inputs should be multiples of 2"
    assert len(kv_outputs) % 2 == 0, "kv outputs should be multiples of 2"
    num_layers = len(kv_inputs) // 2
    print(f"Number of Layers is {num_layers}")
    num_layers_check_list = [len(kv_outputs) // 2, len(q_outputs), len(k_outputs), len(v_outputs), len(attention_outputs)]
    for i in num_layers_check_list:
        assert i == num_layers, f"Uneven number of I/O nodes detected: {num_layers_check_list}"

    # Create Plugin Layers
    for i in range(num_layers):
        q = q_outputs[f"/model/layers.{i}/self_attn/q_proj/MatMul_output_0"]
        k = k_outputs[f"/model/layers.{i}/self_attn/k_proj/MatMul_output_0"]
        v = v_outputs[f"/model/layers.{i}/self_attn/v_proj/MatMul_output_0"]
        # Merge qkv gemm into a single MatMul Op
        hidden_state = q.inputs[0].inputs[0]
        assert hidden_state == k.inputs[0].inputs[0]
        assert hidden_state == v.inputs[0].inputs[0]
        q_weight = clear_outputs(q.inputs[0].inputs[1])
        k_weight = clear_outputs(k.inputs[0].inputs[1])
        v_weight = clear_outputs(v.inputs[0].inputs[1])

        qkv_weight = gs.Constant(
            name = f"/model/layers.{i}/qkv_proj/weight",
            values=np.concatenate((q_weight.values, k_weight.values, v_weight.values), axis = 1)
        )

        qkv = gs.Variable(
            name = f"/model/layers.{i}/qkv_proj/MatMul_output",
            dtype = q.dtype
        )

        graph.layer(
            name=f"/model/layers.{i}/self_attn/qkv_proj",
            op="MatMul",
            inputs=[hidden_state, qkv_weight],
            outputs = [qkv],
        )

        k_cache = kv_inputs[f"past_key_values.{i}.key"]
        kv_input_shape = (k_cache.shape[0], 2, k_cache.shape[1], k_cache.shape[2], k_cache.shape[3])
        kv_input = gs.Variable(f"past_key_values.{i}", dtype=k_cache.dtype, shape=kv_input_shape)
        graph.inputs.append(kv_input)

        attn_output = attention_outputs[f"/model/layers.{i}/self_attn/MatMul_1_output_0"]
        attn_output.name = "/model/layers.{i}/self_attn/attention_output"
        k_cache_output = kv_outputs[f"present.{i}.key"]
        kv_output_shape = (k_cache_output.shape[0], 2, k_cache_output.shape[1], k_cache_output.shape[2], k_cache_output.shape[3])
        kv_output = gs.Variable(f"present_key_values.{i}", dtype=k_cache_output.dtype, shape=kv_output_shape)
        graph.outputs.append(kv_output)

        graph.layer(
            name=f"Attention-{i}",
            op="AttentionPlugin",
            inputs=[qkv,kv_input, context_length],
            outputs=[attn_output, kv_output],
        )

    graph.cleanup().toposort()
    graph.fold_constants()
    graph.cleanup().toposort()

    return graph


def main():
    t0 = time.time()
    args = parse_arguments()
    main_export(
        args.model_dir,
        task="text-generation-with-past",
        output=args.output_dir,
        opset=17,
        dtype=args.dtype,
        device="cuda",
        framework="pt",
        no_post_process=True,
        do_validation=False,
    )

    t1 = time.time()
    print(f"ONNX Initial Export takes {t1 - t0} seconds.")
    onnx_name = f"{args.output_dir}/model.onnx"
    graph = gs.import_onnx(onnx.load(onnx_name))
    graph = surgeon_graph(graph)
    model = gs.export_onnx(graph)
    folder = args.output_dir
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)

        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    t2 = time.time()
    print(f"ONNX Graphsurgeon takes {t2 - t1} seconds.")
    onnx.save_model(
        model,
        onnx_name,
        save_as_external_data=True,
        all_tensors_to_one_file = True,
        location=f"onnx_model.data",
        convert_attribute=True
    )

    t3 = time.time()
    print(f"ONNX Save takes {t3 - t2} seconds.")
    print(f"Total onnx export time: {t3 - t0}.")
    print(f"Model exported to {args.output_dir} with {args.dtype} precision successfully.")

if __name__ == '__main__':
    main()