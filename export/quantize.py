import argparse
import os
import time

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
from datasets import load_dataset
from modelopt.torch.quantization.utils import export_torch_mode
from torch.export._trace import _export
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_dir',
                        type=str,
                        help="The name or path of HF PyTorch model ckpt",
                        required=False)
    parser.add_argument('--onnx_path',
                        type=str,
                        help="The path of ONNX model to surgeon graph",
                        required=False)
    parser.add_argument('--dataset_dir',
                        type=str,
                        help="The path of dataset",
                        required=False)
    parser.add_argument('--output_dir',
                        type=str,
                        help="The directory to store the generated ONNX model",
                        required=True)
    args = parser.parse_args()
    return args


def get_calib_dataloader(dataset_name_or_dir="cnn_dailymail",
                         tokenizer=None,
                         batch_size=1,
                         calib_size=512,
                         block_size=512):
    print("Loading calibration dataset")
    if "cnn_dailymail" in dataset_name_or_dir:
        dataset = load_dataset(dataset_name_or_dir,
                               name="3.0.0",
                               split="train")
        dataset = dataset["article"][:calib_size]
    elif os.path.isdir(dataset_name_or_dir):
        print(
            f"Recognized local dataset repo {dataset_name_or_dir} for calibration; "
            "assuming the calibration data are in the train split and text column."
        )
        dataset = load_dataset(dataset_name_or_dir, split="train")
        dataset = dataset["text"][:calib_size]
    else:
        raise NotImplementedError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}."
        )

    batch_encoded = tokenizer.batch_encode_plus(dataset,
                                                return_tensors="pt",
                                                padding=True,
                                                truncation=True,
                                                max_length=block_size)

    calib_dataloader = DataLoader(batch_encoded["input_ids"],
                                  batch_size=batch_size,
                                  shuffle=False)

    return calib_dataloader


def get_quant_config():
    KV_CACHE_CFG = {
        "*proj.output_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True
        },
    }
    quant_cfg = mtq.FP8_DEFAULT_CFG
    quant_cfg["quant_cfg"].update(KV_CACHE_CFG)
    return quant_cfg


def quantize_model(model, calib_dataloader=None):

    # The calibration loop for the model can be setup using the modelopt API.
    #
    # Example usage:
    # from modelopt.torch.utils.dataset_utils import create_forward_loop
    # model = ...  # Initilaize the model
    # tokenizer = ...  # Initilaize the tokenizer
    # quant_cfg = ...  # Setup quantization configuration
    # forward_loop = create_forward_loop(model=model, dataset_name="cnn_dailymail", tokenizer=tokenizer)
    # mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    def calibrate_loop(model):
        """Adjusts weights and scaling factors based on selected algorithms."""
        for idx, data in enumerate(calib_dataloader):
            print(f"Calibrating batch {idx}")
            data = data.to(model.device)
            model(data)

    print("Starting quantization...")
    start_time = time.time()
    mtq.quantize(model, get_quant_config(), forward_loop=calibrate_loop)
    end_time = time.time()
    print(f"Quantization done. Total time used: {end_time - start_time}s")

    return model


def export_to_onnx(model, output_dir):
    config = model.config
    num_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    hidden_size = config.hidden_size
    hidden_size_per_layer = hidden_size // num_attention_heads

    dummy_bs = 2
    dummy_len = 10
    dummy_input_ids = torch.randint(100, (dummy_bs, dummy_len),
                                    dtype=torch.int64).cuda()
    input_names = ["input_ids"]
    output_names = ["logits"]
    dynamic_axes = {"input_ids": {0: "batch_size", 1: "seq_len"}}
    dummy_kv_cache = ()
    for i in range(num_layers):
        dummy_k = torch.rand(
            (dummy_bs, num_key_value_heads, dummy_len, hidden_size_per_layer),
            dtype=torch.float16).cuda()
        dummy_v = torch.rand(
            (dummy_bs, num_key_value_heads, dummy_len, hidden_size_per_layer),
            dtype=torch.float16).cuda()
        dummy_kv_cache = dummy_kv_cache + ((dummy_k, dummy_v), )
        input_names.extend(
            [f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
        output_names.extend(
            [f"present_key_values.{i}.key", f"present_key_values.{i}.value"])
        input_dynamic_axes = {0: "batch_size", 2: "past_len"}
        dynamic_axes[f"past_key_values.{i}.key"] = input_dynamic_axes
        dynamic_axes[f"past_key_values.{i}.value"] = input_dynamic_axes

    torch.onnx.export(
        model,
        (dummy_input_ids, {
            "past_key_values": dummy_kv_cache
        }),
        output_dir,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )


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

    context_length = gs.Variable("context_length", np.int32, [1])

    # For context phase, last_token_ids should be context_length - 1; for generation phase, last_token_ids should be 0
    last_token_ids = gs.Variable("last_token_ids", np.int64, [1])
    one_constant = gs.Constant(name="/lm_head/Slice/one_constant",
                               values=np.array([1], dtype=np.int64))
    slice_end = gs.Variable(name="/lm_head/Slice/end",
                            dtype=np.int64,
                            shape=[1])
    graph.layer(name="/lm_head/Slice/Add",
                op="Add",
                inputs=[last_token_ids, one_constant],
                outputs=[slice_end])

    graph.inputs.append(context_length)
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
        if "q_proj/output_quantizer/Cast_1" in node.name and node.op == "Cast":
            assert len(
                node.outputs
            ) == 1, "You did not reach the proper q_proj Quantizer Cast tensor!"
            q_output = node.outputs[0]
            q_outputs[q_output.name] = clear_outputs(q_output)
        if "k_proj/output_quantizer/Cast_1" in node.name and node.op == "Cast":
            assert len(
                node.outputs
            ) == 1, "You did not reach the proper k_proj Quantizer Cast tensor!"
            k_output = node.outputs[0]
            k_outputs[k_output.name] = clear_outputs(k_output)
        if "v_proj/output_quantizer/Cast_1" in node.name and node.op == "Cast":
            assert len(
                node.outputs
            ) == 1, "You did not reach the proper v_proj Quantizer Cast tensor!"
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
        q = q_outputs[
            f"/model/layers.{i}/self_attn/q_proj/output_quantizer/Cast_1_output_0"]
        k = k_outputs[
            f"/model/layers.{i}/self_attn/k_proj/output_quantizer/Cast_1_output_0"]
        v = v_outputs[
            f"/model/layers.{i}/self_attn/v_proj/output_quantizer/Cast_1_output_0"]

        qkv = gs.Variable(name=f"/model/layers.{i}/qkv_proj/Concat_output",
                          dtype=q.dtype)

        graph.layer(name=f"/model/layers.{i}/self_attn/qkv_proj/Concat",
                    op="Concat",
                    inputs=[q, k, v],
                    outputs=[qkv],
                    attrs={"axis": -1})

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
        k_cache_output = kv_outputs[f"present_key_values.{i}.key"]
        kv_output_shape = (k_cache_output.shape[0], 2, k_cache_output.shape[1],
                           k_cache_output.shape[2], k_cache_output.shape[3])
        kv_output = gs.Variable(f"present_key_values.{i}",
                                dtype=k_cache_output.dtype,
                                shape=kv_output_shape)
        graph.outputs.append(kv_output)

        graph.layer(
            name=f"Attention-{i}",
            op="AttentionPlugin",
            inputs=[qkv, kv_input, context_length],
            outputs=[attn_output, kv_output],
        )

    # Insert Slice node for lm_head's input.
    lm_head_matmul = clear_outputs(logits.inputs[0].inputs[0].inputs[0])
    assert lm_head_matmul.name == "/lm_head/MatMul", f"You did not reach lm_head, but you reached {lm_head.name}"
    lm_head_weight = lm_head_matmul.inputs[1]
    lm_head_weight.name = "/lm_head/MatMul/weight"
    lm_head_input = lm_head_matmul.inputs[0]

    # Insert a Slice node for lm_head_input
    slice_output = gs.Variable("/lm_head/Slice_Output")
    slice_axis = gs.Constant(name="/lm_head/Slice/axes",
                             values=np.array([1], dtype=np.int32))
    graph.layer(
        name="/lm_head/Slice",
        op="Slice",
        inputs=[lm_head_input, last_token_ids, slice_end, slice_axis],
        outputs=[slice_output],
    )

    slice_output.outputs = [lm_head_matmul]
    lm_head_matmul.inputs = [slice_output, lm_head_weight]

    # Remove the last cast layer so logits are in fp16 instead of fp32
    logits = clear_inputs(logits)
    lm_head_matmul.outputs = [logits]
    logits.inputs = [lm_head_matmul]
    logits.dtype = np.float16
    # Force logits shape to be 1 for both context phase and generation phase
    logits.shape = [logits.shape[0], 1, logits.shape[2]]
    graph.cleanup().toposort().fold_constants().cleanup().toposort()

    return graph


def main():
    t0 = time.time()
    args = parse_arguments()

    assert args.torch_dir or args.onnx_path, "You need to provide either --torch_dir or --onnx_path to process the export script"
    if args.torch_dir:
        model = AutoModelForCausalLM.from_pretrained(
            args.torch_dir, torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.torch_dir)
        if tokenizer.pad_token != "<unk>":
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dataset_dir = args.dataset_dir if args.dataset_dir else "cnn_dailymail"
        data_loader = get_calib_dataloader(dataset_name_or_dir=dataset_dir,
                                           tokenizer=tokenizer)
        quantized_model = quantize_model(model, data_loader)
        os.makedirs(args.output_dir, exist_ok=True)
        mtq.print_quant_summary(quantized_model)
        mto.save(quantized_model,
                 os.path.join(args.output_dir, "quantized_model.bin"))
        export_to_onnx(quantized_model,
                       os.path.join(args.output_dir, "quantized_model.onnx"))

    input_onnx_name = f"{args.output_dir}/quantized_model.onnx" if args.torch_dir else args.onnx_path
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

    output_onnx_name = f"{args.output_dir}/quantized_model.onnx"
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
