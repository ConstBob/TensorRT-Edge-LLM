import argparse
import os
import time

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from optimum.exporters.onnx import main_export


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_dir',
                        type=str,
                        help="The name or path of HF PyTorch model ckpt",
                        required=False)
    parser.add_argument('--onnx_path',
                        type=str,
                        help="The path of original ONNX model",
                        required=False)
    parser.add_argument('--output_dir',
                        type=str,
                        help="The directory to store the generated ONNX model",
                        required=True)
    parser.add_argument('--dtype',
                        type=str,
                        default="fp16",
                        help="The floating point precision to use for export")
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
        
    context_lengths = gs.Variable("context_lengths", np.int32, ['batch_size'])

    # For context phase, last_token_ids should be context_length - 1; for generation phase, last_token_ids should be 0
    last_token_ids = gs.Variable("last_token_ids", np.int64, ['batch_size', 1])
    graph.inputs.append(context_lengths)
    graph.inputs.append(last_token_ids)

    # Look for kv cache outputs
    kv_outputs = {}
    logits = None
    for output in graph.outputs:
        if "present" in output.name:
            kv_outputs[output.name] = clear_inputs(output)
        if "logits" in output.name:
            logits = output

    assert logits, "There should be logits output"

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
            assert len(
                node.outputs
            ) == 1, "You did not reach the proper q_proj MatMul tensor!"
            q_output = node.outputs[0]
            q_outputs[q_output.name] = clear_outputs(q_output)
        if "k_proj/MatMul" in node.name and node.op == "MatMul":
            assert len(
                node.outputs
            ) == 1, "You did not reach the proper k_proj MatMul tensor!"
            k_output = node.outputs[0]
            k_outputs[k_output.name] = clear_outputs(k_output)
        if "v_proj/MatMul" in node.name and node.op == "MatMul":
            assert len(
                node.outputs
            ) == 1, "You did not reach the proper v_proj MatMul tensor!"
            v_output = node.outputs[0]
            v_outputs[v_output.name] = clear_outputs(v_output)
        if "self_attn/Transpose_4" in node.name and node.op == "Transpose":
            assert len(
                node.outputs
            ) == 1, "You did not reach the proper self_attn Transpose tensor!"
            attention_output = node.outputs[0]
            attention_outputs[attention_output.name] = clear_inputs(
                attention_output)

    assert len(kv_inputs) % 2 == 0, "kv inputs should be multiples of 2"
    assert len(kv_outputs) % 2 == 0, "kv outputs should be multiples of 2"
    num_layers = len(kv_inputs) // 2
    num_layers_check_list = [
        len(kv_outputs) // 2,
        len(q_outputs),
        len(k_outputs),
        len(v_outputs),
        len(attention_outputs)
    ]
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
            name=f"/model/layers.{i}/qkv_proj/weight",
            values=np.concatenate(
                (q_weight.values, k_weight.values, v_weight.values), axis=1))

        qkv = gs.Variable(name=f"/model/layers.{i}/qkv_proj/MatMul_output",
                          dtype=q.dtype)

        graph.layer(
            name=f"/model/layers.{i}/self_attn/qkv_proj",
            op="MatMul",
            inputs=[hidden_state, qkv_weight],
            outputs=[qkv],
        )

        k_cache = kv_inputs[f"past_key_values.{i}.key"]
        kv_input_shape = (k_cache.shape[0], 2, k_cache.shape[1],
                          k_cache.shape[2], k_cache.shape[3])
        kv_input = gs.Variable(f"past_key_values.{i}",
                               dtype=k_cache.dtype,
                               shape=kv_input_shape)
        graph.inputs.append(kv_input)

        attn_output = attention_outputs[
            f"/model/layers.{i}/self_attn/Transpose_4_output_0"]
        attn_output.name = f"/model/layers.{i}/self_attn/attention_output"
        k_cache_output = kv_outputs[f"present.{i}.key"]
        kv_output_shape = (k_cache_output.shape[0], 2, k_cache_output.shape[1],
                           k_cache_output.shape[2], k_cache_output.shape[3])
        kv_output = gs.Variable(f"present_key_values.{i}",
                                dtype=k_cache_output.dtype,
                                shape=kv_output_shape)
        graph.outputs.append(kv_output)

        graph.layer(
            name=f"Attention-{i}",
            op="AttentionPlugin",
            inputs=[qkv, kv_input, context_lengths],
            outputs=[attn_output, kv_output],
        )

    # Insert Gather node for lm_head's input.
    lm_head_matmul = clear_outputs(logits.inputs[0].inputs[0].inputs[0])
    assert lm_head_matmul.name == "/lm_head/MatMul", f"You did not reach lm_head, but you reached {lm_head_matmul.name}"
    lm_head_weight = lm_head_matmul.inputs[1]
    lm_head_weight.name = "/lm_head/MatMul/weight"
    lm_head_input = lm_head_matmul.inputs[0]

    gather_output = gs.Variable("/lm_head/Gather_Output")
    
    # lm_head_input shape: [batch_size, len, 4096]
    # last_token_ids shape: [batch_size, 1]
    # gather_output shape: [batch_size, 4096]
    graph.layer(
        name="/lm_head/GatherND",
        op="GatherND",
        inputs = [lm_head_input, last_token_ids],
        outputs = [gather_output],
        attrs = {"batch_dims": 1}
    )

    gather_output.outputs = [lm_head_matmul]
    lm_head_matmul.inputs = [gather_output, lm_head_weight]

    # Remove the last cast layer so logits are in fp16 instead of fp32
    logits = clear_inputs(logits)
    lm_head_matmul.outputs = [logits]
    logits.inputs = [lm_head_matmul]
    logits.dtype = np.float16
    # Force logits shape to be 1 for both context phase and generation phase
    logits.shape = [logits.shape[0], logits.shape[2]]
    graph.cleanup().toposort().fold_constants().cleanup().toposort()

    return graph


def main():
    t0 = time.time()
    args = parse_arguments()

    assert args.torch_dir or args.onnx_path, "You need to provide either --torch_dir or --onnx_path to process the export script"
    if args.torch_dir:
        print(f"Exporting ONNX from {args.torch_dir} to {args.output_dir}.")
        main_export(
            args.torch_dir,
            task="text-generation-with-past",
            output=args.output_dir,
            opset=17,
            dtype=args.dtype,
            device="cuda",
            framework="pt",
            no_post_process=True,
            do_validation=False,
        )
    else:
        print(f"ONNX path given. Importing ONNX from {args.onnx_path}")

    input_onnx_name = f"{args.output_dir}/model.onnx" if args.torch_dir else args.onnx_path
    graph = gs.import_onnx(onnx.load(input_onnx_name))
    t1 = time.time()
    print(
        f"ONNX export and load takes {t1 - t0}s. Using onnx_graphsurgeon to insert plugin."
    )

    graph = surgeon_graph(graph)

    t2 = time.time()
    print(f"onnx_graphsurgeon takes {t2 - t1}s. Export back to onnx model")
    model = gs.export_onnx(graph)
    t3 = time.time()
    print(f"gs.export(graph) takes {t3 - t2}s.")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"Saving ONNX files in {output_dir}. All existing ONNX in the folder will be overwritten."
    )
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                if ".json" not in file_path:
                    os.unlink(file_path)

        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    output_onnx_name = f"{args.output_dir}/model.onnx"
    onnx.save_model(model,
                    output_onnx_name,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=f"onnx_model.data",
                    convert_attribute=True)

    t4 = time.time()
    print(f"ONNX save takes {t4 - t3} seconds.")
    print(
        f"Model ONNX saved to {args.output_dir} with {args.dtype} precision in {t4 - t0}s."
    )


if __name__ == '__main__':
    main()
