# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import re
import time
from enum import Enum
from typing import Union

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
from onnx_graphsurgeon.ir.tensor import LazyValues


class RopeType(Enum):
    kNone = 0
    kROPE_ROTATE_GPTJ = 1
    kROPE_ROTATE_NEOX = 2
    kMROPE = 3


def clear_inputs(node: Union[gs.Node, gs.Tensor]):
    """
    Clear all inputs for a node or tensor in ONNX
    """
    for i in node.inputs:
        i.outputs.clear()
    node.inputs.clear()
    return node


def clear_outputs(node: Union[gs.Node, gs.Tensor]):
    """
    Clear all outputs for a node or tensor in ONNX.
    """
    for o in node.outputs:
        o.inputs.clear()
    node.outputs.clear()
    return node


def extract_layer_id(name: str):
    """
    Extract layer id from certain ONNX layer name.

    Parameters:
        name: str
            The name of ONNX layer. e.g. /model/layer.0/q_proj/...

    Returns:
        The layer id for the layer as int. In the example above, it returns 0
    """
    match = re.search(r'layers\.(\d+)', name)
    if match:
        return int(match.group(1))
    raise Exception(f"{name} does not contain layer info!")


def no_none_elements(l):
    return all(i is not None for i in l)


def insert_gather_last_token(graph: gs.Graph):
    """
    Inserts GatherND for lm_head to only gather the last token.

    For regular ONNX, it outputs the logits for the entire model. However, in context phase, we only need logits
    for the last token in input_ids. Computing all logits will be not desirable. Therefore, we add a GatherND node
    before lm_head to ensure that only the hidden_states of the last token is passed to lm_head. An extra last_token_ids
    input is added to the ONNX. If Gather is already there, return the original graph.

    Parameters:
        graph: gs.Graph. The original ONNX graph for the LLM

    Returns:
        The graph after adding GatherND node.

    """

    start_time = time.time()
    print("Inserting GatherND to only compute logits for last token...")

    logits = None
    for output in graph.outputs:
        if "logits" in output.name:
            logits = output
            break
    assert logits, "Cannot find logits output in the graph!"

    lm_head_matmul = logits.inputs[0]
    for i in range(5):
        if "/lm_head/MatMul" in lm_head_matmul.name:
            lm_head_matmul = clear_outputs(lm_head_matmul)
            break
        if "Gather" in lm_head_matmul.name:
            end_time = time.time()
            print(
                f"Gather is already in the graph. No gather operation will be inserted. Function completed in {end_time - start_time}s."
            )
            return graph
        lm_head_matmul = lm_head_matmul.inputs[0].inputs[0]

    assert "/lm_head/MatMul" in lm_head_matmul.name and lm_head_matmul.op == "MatMul", f"You did not reach lm_head, but you reached {lm_head_matmul.name}"
    lm_head_weight = lm_head_matmul.inputs[1]
    lm_head_weight.name = "/lm_head/MatMul/weight"
    lm_head_input = lm_head_matmul.inputs[0]

    gather_output = gs.Variable("/lm_head/Gather_Output")
    # For context phase, last_token_ids should be context_length - 1; for generation phase, last_token_ids should be 0
    last_token_ids = gs.Variable("last_token_ids", np.int64, ['batch_size', 1])
    graph.inputs.append(last_token_ids)

    # lm_head_input shape: [batch_size, len, 4096]
    # last_token_ids shape: [batch_size, 1]
    # gather_output shape: [batch_size, 4096]
    graph.layer(name="/lm_head/GatherND",
                op="GatherND",
                inputs=[lm_head_input, last_token_ids],
                outputs=[gather_output],
                attrs={"batch_dims": 1})

    gather_output.outputs = [lm_head_matmul]
    lm_head_matmul.inputs = [gather_output, lm_head_weight]

    # Remove the last cast layer so logits are in fp16 instead of fp32
    logits = clear_inputs(logits)
    lm_head_matmul.outputs = [logits]
    logits.inputs = [lm_head_matmul]
    logits.dtype = np.float16
    # Force logits to have shape of [batch_size, vocab_size].
    logits.shape = [logits.shape[0], logits.shape[2]]

    graph.cleanup().toposort()
    end_time = time.time()
    print(f"GatherND inserted in {end_time - start_time}s.")
    return graph


def insert_attention_plugin(graph: gs.Graph,
                            config: dict,
                            rope_type: RopeType,
                            max_seq_length: int,
                            extra_inputs: list = None):
    """
    Insert AttentionPlugin for the graph. AttentionPlugin takes the following inputs and outputs:

    Inputs:
        qkv: [bs, seq_len, d_q+d_k+d_v]. Therefore qkv from q_proj, k_proj and v_proj will be concatenated
        kv_input: [bs, 2, num_head, max_kv_capacity, d_kv]
        context_lengths: [bs]
        extra_inputs:[]

    Outputs:
        attention_outputs: [bs, seq_len, h_q, d_q]
        kv_output: [bs, 2, num_head, max_kv_capacity, d_kv]

    The AttentionPlugin will perform all RoPE and MHA operations, and therefore requires an additional argument config to know the model parameters.

    Parameters:
        graph: gs.Graph
        config: dict. Converted from transformers.AutoConfig
        rope_type: RopeType.
        extra_inputs: list. Extra inputs is for VLM like QWen2-VL which will contains mrope_rotary_cos_sin and mrope_position_deltas, the mrope_rotary_cos_sin is the rotary cos/sin cache and mrope_position_deltas is the position deltas which are needed by Mrope.



    Returns:
        The graph after inserted AttentionPlugin
    """

    def set_with_warning(key, value):
        """
        Warns the user that particular field is not included in the dict, but is required for AttentionPlugin
        """
        if key not in config or config.get(key, value) is None:
            print(f"{key} does not exist. Set to {value}.")
            return value
        return config.get(key)

    start_time = time.time()
    print("Replacing MHA Pattern with AttentionPlugin...")
    # We do not have any optimization on long context and therefore rotary_scaling is always 1.0
    rotary_scaling = 1.0
    num_q_heads = set_with_warning("num_attention_heads", 32)
    num_kv_heads = set_with_warning("num_key_value_heads", 32)
    head_size = set_with_warning("hidden_size", 4096) // num_q_heads
    rotary_base_frequency = set_with_warning("rope_theta", 500000.0)
    rotary_embedding_max_positions = set_with_warning(
        "max_position_embeddings", 32768)

    attention_attrs = {
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "rotary_scaling": rotary_scaling,
        "rotary_base_frequency": rotary_base_frequency,
        "position_embedding_type": rope_type.value,
        "max_batch_size": 16,
        "kv_cache_capacity": max_seq_length,
        "rotary_embedding_max_positions": rotary_embedding_max_positions,
    }

    removed_inputs = []
    kv_shape = None
    for input in graph.inputs:
        if "past_key_values" in input.name:
            removed_inputs.append(input)
            if input.outputs[0].op == "AttentionPlugin":
                # The graph already contains AttentionPlugin. We should not do so.
                end_time = time.time()
                print(
                    f"Attention Plugin already inserted. No operation is done. Completed in {end_time-start_time}s."
                )
                return graph
            if kv_shape is None:
                kv_shape = input.shape

    assert kv_shape, "No KV Cache I/O Tensor detected! Make sure your kv cache tensor is named past_key_values."
    # Should not remove while iterating, so remove them separately
    for i in removed_inputs:
        graph.inputs.remove(clear_outputs(i))

    context_lengths = gs.Variable("context_lengths", np.int32, ['batch_size'])

    graph.inputs.append(context_lengths)

    for input in extra_inputs:
        graph.inputs.append(input)

    num_layers = (len(graph.outputs) - 1) // 2
    # Look for kv cache outputs and remove them from graph
    kv_outputs = []
    for output in graph.outputs:
        if "present" in output.name:
            kv_outputs.append(clear_inputs(output))

    for i in kv_outputs:
        graph.outputs.remove(i)

    # Look for qkv outputs and end of MHA nodes.
    q_outputs = [None] * num_layers
    k_outputs = [None] * num_layers
    v_outputs = [None] * num_layers
    attention_outputs = [None] * num_layers
    for node in graph.nodes:
        # Get the qkv proj output Tensors
        if node.op == "MatMul" and ("q_proj" in node.name or "k_proj"
                                    in node.name or "v_proj" in node.name):
            layer_id = extract_layer_id(node.name)

            # There are residual add or quantization that happens after MatMul for qkv projection
            def find_layer_output(layer_name):
                layer_output = node
                for i in range(15):
                    if layer_name not in layer_output.outputs[0].name:
                        return layer_output
                    layer_output = layer_output.outputs[0]
                raise Exception(
                    f"{layer_name} does not terminate. The last node is {layer_output.name}"
                )

            if "q_proj" in node.name:
                layer_output = find_layer_output("q_proj")
                q_outputs[layer_id] = layer_output
            if "k_proj" in node.name:
                layer_output = find_layer_output("k_proj")
                k_outputs[layer_id] = layer_output
            if "v_proj" in node.name:
                layer_output = find_layer_output("v_proj")
                v_outputs[layer_id] = layer_output

            layer_output = clear_outputs(layer_output)

        # In our current models, this is enough to signal the end of Attention blocks.
        if "self_attn/Transpose_4" in node.name and node.op == "Transpose":
            assert len(
                node.outputs
            ) == 1, "You did not reach the proper self_attn Transpose tensor!"
            layer_id = extract_layer_id(node.name)
            attention_outputs[layer_id] = clear_inputs(node.outputs[0])

    assert no_none_elements(q_outputs), f"q_outputs {q_outputs} contains None"
    assert no_none_elements(k_outputs), f"k_outputs {k_outputs} contains None"
    assert no_none_elements(v_outputs), f"v_outputs {v_outputs} contains None"
    assert no_none_elements(
        attention_outputs
    ), f"attention_outputs {attention_outputs} contains None"

    # Create Plugin Layers
    for i in range(num_layers):
        q = q_outputs[i]
        k = k_outputs[i]
        v = v_outputs[i]

        # Simply concat the QKV inputs. QKV Gemms will be fused into single GEMM by TensorRT.
        qkv = gs.Variable(name=f"/model/layers.{i}/qkv_proj/Concat_output",
                          dtype=q.dtype)

        graph.layer(name=f"/model/layers.{i}/self_attn/qkv_proj/Concat",
                    op="Concat",
                    inputs=[q, k, v],
                    outputs=[qkv],
                    attrs={"axis": -1})

        kv_input_shape = (kv_shape[0], 2, kv_shape[1], "past_len", kv_shape[3])
        kv_input = gs.Variable(f"past_key_values.{i}",
                               dtype=np.float16,
                               shape=kv_input_shape)
        graph.inputs.append(kv_input)

        attn_output = attention_outputs[i]
        attn_output.name = f"/model/layers.{i}/self_attn/attention_output"

        kv_output_shape = (kv_shape[0], 2, kv_shape[1],
                           attention_attrs["kv_cache_capacity"], kv_shape[3])
        kv_output = gs.Variable(f"present_key_values.{i}",
                                dtype=np.float16,
                                shape=kv_output_shape)
        graph.outputs.append(kv_output)

        graph.layer(name=f"Attention-{i}",
                    op="AttentionPlugin",
                    inputs=[qkv, kv_input, context_lengths] + extra_inputs,
                    outputs=[attn_output, kv_output],
                    attrs=attention_attrs)
    graph.cleanup().toposort()
    end_time = time.time()
    print(f"AttentionPlugin inserted in {end_time - start_time}s. ")

    return graph


def unpack_int4_weights(awq_weights: np.array, dtype: np.dtype):
    """
    The awq_weights have shape [N/2, K]. Reshape to a [K, N] matrix with expanded weights
    """
    awq_weights = awq_weights.T
    int4_weights = np.zeros((awq_weights.shape[0], awq_weights.shape[1], 2),
                            dtype=dtype)
    int4_weights[..., 0] = awq_weights & 0x0F
    int4_weights[..., 1] = (awq_weights >> 4) & 0x0F

    def toint(x):
        sign = x & 0x08
        value = ~((~x) & 0x07)
        return np.where(sign.astype(bool), sign | value, x)

    int4_weights = toint(int4_weights)
    int4_weights = int4_weights.reshape([int4_weights.shape[0], -1])
    return int4_weights


def insert_int4_dq(graph: gs.Graph, state_dict: dict):
    """
    Native INT4_AWQ ONNX export is not supported by modelopt. Therefore we need to quantize the PyTorch model and save the state_dict to get int4 weights and scales.
    This function starts with pure fp16 ONNX graph and replaces all the weights with DequantizeLinear for int4 weights.

    Parameters:
        graph: gs.Graph
            The original fp16 ONNX graph

        state_dict: dict[str, np.array]
            The state_dict of the int4_awq quantized PyTorch model.


    Returns:
        Graph with all fp16 weights replaced by DequantizeLinear layer.

    """
    import modelopt.onnx.quantization.qdq_utils as qdq
    start_time = time.time()
    print("Replacing all fp16 GEMM with int4 DequantizeLinear...")
    from modelopt.onnx.quantization.gs_patching import patch_gs_modules
    patch_gs_modules()
    state_keys = state_dict.keys()
    int4_weight_dict = {}
    pre_quant_scale_dict = {}
    weight_scale_dict = {}
    input_tensors_dict = {}

    for node in graph.nodes:
        if node.op == "MatMul" and "_proj" in node.name:
            # Need to insert DQ and pre_quant_scale
            hf_name = '.'.join(node.name.split("/")[1:-1])
            hf_weight_name = hf_name + ".weight"
            hf_pre_quant_scale_name = hf_name + ".input_quantizer._pre_quant_scale"
            hf_weight_scale_name = hf_name + ".weight_quantizer._amax"
            assert hf_weight_name in state_keys, f"{hf_weight_name} not in state_keys!"
            assert hf_weight_scale_name in state_keys, f"{hf_weight_scale_name} not in state_keys!"

            # Extract the tensors from HF state_dict
            hf_weight = state_dict[hf_weight_name].cpu().numpy()
            assert hf_weight.dtype == np.int8, f"Weight should be np.int8 type. You have {hf_weight.dtype}."
            # The weights from modelopt is a packed tensor with shape [N/2, K]. Need to unpack to shape [K, N]
            int4_weight = unpack_int4_weights(hf_weight, np.int8)
            # The scale factor has a shape [N, K/group_size]. Need to reshape to [K/group_size, N]
            # In addition, _amax is 7x the scale.
            hf_weight_scale = state_dict[hf_weight_scale_name].cpu().numpy(
            ).transpose(1, 0) / 7
            # Force weight scale in fp16 can help TensorRT GEMM fusion.
            hf_weight_scale = np.float16(hf_weight_scale)

            # MatMul in proj must be input and weights
            assert len(
                node.inputs
            ) == 2, f"You have more than 2 inputs for MatMul node {node.name}"
            weight_id = 0 if isinstance(node.inputs[0], gs.Constant) else 1
            assert isinstance(
                node.inputs[weight_id], gs.Constant
            ), f"Both inputs for node {node.name} are not Constant!"
            weight = node.inputs[weight_id]
            onnx_weight_name = weight.name
            assert list(weight.shape) == list(
                int4_weight.shape
            ), f"Provide quantized {hf_weight_name} weight shape {int4_weight.shape} is not same as original {weight.shape}"
            int4_weight_dict[onnx_weight_name] = int4_weight
            weight_scale_dict[onnx_weight_name] = hf_weight_scale
            if hf_pre_quant_scale_name in state_keys:
                input = node.inputs[1 - weight_id]
                hf_pre_quant_scale = state_dict[hf_pre_quant_scale_name].cpu(
                ).numpy()
                pre_quant_scale_dict[onnx_weight_name] = hf_pre_quant_scale
                input_tensors_dict[onnx_weight_name] = input.name

    assert len(int4_weight_dict) == len(
        weight_scale_dict
    ), f"{len(int4_weight_dict)} should be the same as {len(weight_scale_dict)}"
    # TODO: Make block_size dynamic
    dq_node_attributes = {"axis": 0, "block_size": 128}
    qdq.insert_dq_nodes(graph,
                        scales=weight_scale_dict,
                        quantized_weights=int4_weight_dict,
                        attributes=dq_node_attributes)
    if len(pre_quant_scale_dict) > 0:
        qdq.insert_pre_quant_scale_nodes(graph,
                                         input_tensors=input_tensors_dict,
                                         pre_quant_scale=pre_quant_scale_dict)
    graph.cleanup().toposort()
    end_time = time.time()
    print(f"Int4 DQ inserted in {end_time - start_time}s.")
    return graph


def interleave_int4_weights(naively_packed_weights, interleave=4, kstride=64):
    """
    This function is adapted from
    https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/qmodule.py#L26
    to prepare the packed int4 weights for int4 plugin to use.
    """
    # naively packed weights: int4 weights naively packed as int8 type with shape [N/2, K].
    naively_packed_weights = naively_packed_weights.cpu().numpy()
    # Convert from naively packed weights to unsigned int4 weights represented as int16.
    # Unpacked_qweight is int16 with shape [K, N]. The +8 is required to match the implementation of awq gemm kernels requirement.
    # In the plugin kernel, a -8 will be performed to convert back to original weights.
    unpacked_qweight = unpack_int4_weights(naively_packed_weights,
                                           np.int16) + 8
    # Transposed back to [N, K] with int16 weights representation
    unpacked_qweight = unpacked_qweight.transpose(1, 0)

    # unpacked_qwight: int16 [N, K]
    N = unpacked_qweight.shape[0]
    K = unpacked_qweight.shape[1]

    packed_kernel = unpacked_qweight.reshape(N, K // 32, 32)
    # np.arange(32).reshape(4, 4, 2).transpose(1, 0, 2) => [0, 1, 8, 9, 16, 17, 24, 25, ...]
    packed_kernel = packed_kernel.reshape(N, K // 32, 4, 4,
                                          2).transpose(0, 1, 3, 2, 4)
    packed_kernel = packed_kernel.reshape(N, K // 32, 32)

    # reorder each 8 weights for fast dequantization
    # [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7]
    packed_kernel = packed_kernel.reshape(N, K // 32, 4, 8)
    packed_kernel = packed_kernel.reshape(N, K // 32, 4, 4,
                                          2).transpose(0, 1, 2, 4, 3)
    packed_kernel = packed_kernel.reshape(N, K)

    # interleaving every four rows
    packed_kernel = packed_kernel.reshape(N // interleave, interleave,
                                          K // kstride, kstride)
    # N // 4, K // 64, 4, 64
    packed_kernel = packed_kernel.transpose(0, 2, 1, 3)
    packed_kernel = packed_kernel.reshape(N // interleave, K // kstride,
                                          kstride, interleave)
    # Packing -> (N // 4, K // 64, 64)
    packed_kernel = (packed_kernel[..., 0]
                     | (packed_kernel[..., 1] << 4)
                     | (packed_kernel[..., 2] << 8)
                     | (packed_kernel[..., 3] << 12))
    # reshape to (N // 4, K), FP16 format
    packed_kernel = packed_kernel.reshape(N // interleave, K)

    # reshape to (N // 2, K), int8 format
    qweight_int8 = packed_kernel.view(np.int8).reshape(N // 2, K)
    return qweight_int8


def insert_int4_gemm_plugin(graph: gs.Graph, state_dict: dict):
    """
    This function starts with pure fp16 ONNX graph and replaces all the weights with Int4GemmPlugin for int4 weights.

    Parameters:
        graph: gs.Graph
            The original fp16 ONNX graph

        state_dict: dict[str, np.array]
            The state_dict of the int4_awq quantized PyTorch model.

    Returns:
        Graph with all fp16 weights replaced by Int4GemmPlugin layer.

    """

    start_time = time.time()
    print("Replacing all fp16 weights with Int4GemmPlugin...")
    from modelopt.onnx.quantization.gs_patching import patch_gs_modules
    patch_gs_modules()
    state_keys = state_dict.keys()

    gemm_nodes = []
    for node in graph.nodes:
        if node.op == "MatMul" and "_proj" in node.name:
            gemm_nodes.append(node)

    for node in gemm_nodes:
        # MatMul in proj must be input and weights
        assert len(
            node.inputs
        ) == 2, f"You have more than 2 inputs for MatMul node {node.name}"
        weight_id = 0 if isinstance(node.inputs[0], gs.Constant) else 1
        assert isinstance(
            node.inputs[weight_id],
            gs.Constant), f"Both inputs for node {node.name} are not Constant!"
        gemm_name = node.name
        weight = node.inputs[weight_id]
        onnx_weight_name = weight.name

        # Should not clear outputs because qkv shares the same input
        input = node.inputs[1 - weight_id]
        input.outputs.remove(node)
        weight.outputs.remove(node)
        assert len(node.outputs) == 1, "Gemm should only have 1 output!"
        output = node.outputs[0]
        output.inputs.remove(node)
        graph.nodes.remove(node)

        # Need to insert DQ and pre_quant_scale
        hf_name = '.'.join(node.name.split("/")[1:-1])
        hf_weight_name = hf_name + ".weight"
        hf_pre_quant_scale_name = hf_name + ".input_quantizer._pre_quant_scale"
        hf_weight_scale_name = hf_name + ".weight_quantizer._amax"
        assert hf_weight_name in state_keys, f"{hf_weight_name} not in state_keys!"
        assert hf_weight_scale_name in state_keys, f"{hf_weight_scale_name} not in state_keys!"

        # Extract the tensors from HF state_dict
        hf_weight = state_dict[hf_weight_name]

        # The shape of packed int4 weight is [N/2, K] in numpy format
        hf_weight = interleave_int4_weights(hf_weight)

        assert hf_weight.dtype == np.int8, f"Weight should be np.int8 type. You have {hf_weight.dtype}."

        # The shape of weight scale is [N, K/group_size]
        hf_weight_scale = state_dict[hf_weight_scale_name].cpu().numpy(
        ).transpose(1, 0) / 7
        # Force weight scale in fp16
        hf_weight_scale = np.float16(hf_weight_scale)

        onnx_weights_int4 = gs.Constant(gemm_name + "/int8_packed_weights",
                                        hf_weight)
        onnx_scale = gs.Constant(gemm_name + "/fp16_scale", hf_weight_scale)
        gemm_n = hf_weight.shape[0] * 2
        assert gemm_n == hf_weight_scale.shape[
            1], f"Weights should have shape [N/2, K]. Scale should have shape [K/group_size, N]. You have weight shape {hf_weight.shape} and scale shape {hf_weight_scale.shape}."
        gemm_k = hf_weight.shape[1]
        group_size = gemm_k // hf_weight_scale.shape[0]
        gemm_attrs = {
            "gemm_n": gemm_n,
            "gemm_k": gemm_k,
            "group_size": group_size,
        }

        if hf_pre_quant_scale_name in state_keys:
            hf_pre_quant_scale = state_dict[hf_pre_quant_scale_name].cpu(
            ).numpy()
            smoothed_input = gs.Variable(name=f"{gemm_name}/smoothed_input",
                                         dtype=input.dtype,
                                         shape=input.shape)
            pre_quant_scale_weight = gs.Constant(
                gemm_name + "/pre_quant_scale",
                hf_pre_quant_scale,
            )

            graph.layer(
                name=f"{gemm_name}/SmoothMul",
                op="Mul",
                inputs=[input, pre_quant_scale_weight],
                outputs=[smoothed_input],
            )
        else:
            smoothed_input = input

        graph.layer(name=f"{gemm_name}Plugin",
                    op="Int4GroupwiseGemmPlugin",
                    inputs=[smoothed_input, onnx_weights_int4, onnx_scale],
                    outputs=[output],
                    attrs=gemm_attrs)

    graph.cleanup()
    # TODO: There is a bug in toposort()
    end_time = time.time()
    print(f"Int4 Gemm Plugin inserted in {end_time - start_time}s.")
    return graph


def fold_fp8_qdq_to_dq(graph: gs.Graph):
    """Convert FP32/FP16 weights of the given ONNX model to FP8 weights.

    Even though modelopt supports FP8 onnx export, the weights are represented in fp32 + QDQ. The storage is therefore very bad. In this function, Q nodes will get removed from the weights and have only DQ nodes with those converted FP8
    weights in the output model.

    Parameters:
        graph: gs.Graph.

    Returns:
        gs.Graph with only DQ nodes for weights and same QDQ nodes for activations.
    """

    start_time = time.time()
    print("Replacing all (fp32 weights + fp8 QDQ) with (fp8 weights + DQ)...")
    # Fold constants is required since the scale is not constant yet.
    graph.cleanup().toposort().fold_constants().cleanup()

    for node in graph.nodes:
        if node.op == "TRT_FP8QuantizeLinear":
            # Should not remove input QDQ
            if not isinstance(node.inputs[0], gs.Constant):
                continue

            weights = node.inputs[0]
            scale = node.inputs[1]
            torch_weights = torch.from_numpy(weights.values)
            torch_scale = torch.from_numpy(scale.values)
            quantizer_name = scale.name.rsplit('/', 1)[0]
            dq_op = node.outputs[0].outputs[0]
            assert dq_op.op == "TRT_FP8DequantizeLinear", f"QDQ does not occur in pairs. You reached {dq_op.op}"

            # Replace it with Dequantize with FP8 weights. This is a WAR because numpy does not support fp8.
            numpy_weights = (torch_weights / torch_scale).to(
                torch.float8_e4m3fn).view(torch.uint8).numpy()
            tensor = onnx.TensorProto()
            tensor.data_type = onnx.TensorProto.FLOAT8E4M3FN
            tensor.dims.extend(numpy_weights.shape)
            tensor.raw_data = numpy_weights.tobytes()
            values = LazyValues(tensor)
            onnx_weights_fp8 = gs.Constant(quantizer_name + "/fp8_weights",
                                           values)

            numpy_scale = torch_scale.to(torch.float16).numpy()
            onnx_scale = gs.Constant(quantizer_name + "/fp16_scale",
                                     numpy_scale)
            node.outputs.clear()
            # DQ Op is separated out
            dq_op.inputs = [onnx_weights_fp8, onnx_scale]
            dq_op.op = "DequantizeLinear"
            dq_op.outputs[0].dtype = np.float16

    graph.cleanup().toposort()
    end_time = time.time()
    print(
        f"fp8 qdq replaced with only dq completed in {end_time - start_time}s."
    )

    return graph


def insert_static_lora(graph: gs.Graph, lora_config, lora_weights, dtype: str):
    """
    Insert static LoRA pattern modifications for different precisions.

    Args:
        graph: gs.Graph
            The ONNX graph to modify
        lora_config: PeftConfig
            The LoRA adapter configuration
        lora_weights: dict
            The LoRA adapter weights
        dtype: str
            The precision type ("fp16", "fp8", "int4", "int4_ootb", "nvfp4")

    Returns:
        Modified graph with LoRA patterns inserted
    """
    start_time = time.time()
    print("Inserting dynamic LoRA patterns...")

    # Track all GEMM nodes that need LoRA
    lora_gemms = {}
    for node in graph.nodes:
        input_tensor = None
        if dtype == "int4" and node.op == "Int4GroupwiseGemmPlugin" and any(
                target in node.name for target in lora_config.target_modules):
            # For int4, the pattern is Mul -> Int4GroupwiseGemmPlugin. The real GEMM input is the input of the Mul node.
            lora_gemms[node.name] = {
                "input": node.inputs[0].inputs[0].inputs[0],
                "output": node.outputs[0],
                "node": node
            }
        elif node.op == "MatMul" and any(
                target in node.name for target in lora_config.target_modules):
            # Find the input tensor based on precision type
            if dtype == "fp16":
                # For FP16, input is direct
                input_tensor = node.inputs[1] if isinstance(
                    node.inputs[0], gs.Constant) else node.inputs[0]
            elif dtype == "fp8":
                # For FP8, need to find Cast node before TRT_FP8QuantizeLinear. FP8 GEMM pattern is Cast -> TRT_FP8QuantizeLinear -> TRT_FP8DequantizeLinear -> Cast -> MatMul
                for inp in node.inputs:
                    if inp.inputs[0].op != "Transpose":
                        curr = inp
                        while "TRT" in curr.inputs[0].op or curr.inputs[
                                0].op == "Cast":
                            curr = curr.inputs[0].inputs[0]
                        input_tensor = curr
            elif dtype == "int4_ootb":
                # For INT4 OOTB, find Mul node before DequantizeLinear. The real GEMM input is the input of the Mul node.
                for inp in node.inputs:
                    if inp.inputs[0].op != "DequantizeLinear":
                        input_tensor = inp.inputs[0].inputs[0]
                        break

            elif dtype == "nvfp4":
                # For NVFP4, find input to TRT_FP4DynamicQuantize. The real GEMM input is the input of the TRT_FP4DynamicQuantize node.
                # NVFP4 pattern is TRT_FP4DynamicQuantize -> DequantizeLinear -> DequantizeLinear -> Cast -> MatMul
                for inp in node.inputs:
                    previous_node = inp.inputs[0]
                    level = 0
                    while len(previous_node.inputs) > 0 and level < 8 and (
                            not isinstance(previous_node, gs.Node) or
                        (previous_node.op != "TRT_FP4DynamicQuantize")):
                        previous_node = previous_node.inputs[0]
                        level += 1
                    if isinstance(
                            previous_node, gs.Node
                    ) and previous_node.op == "TRT_FP4DynamicQuantize":
                        input_tensor = previous_node.inputs[0]
                        break
                # For NVFP4_AWQ recipe, the input is smoothed by a Mul node. The real GEMM input is the input of the Mul node.
                if input_tensor.inputs[0].op == "Mul":
                    if "input_quantizer" in input_tensor.inputs[0].name:
                        input_tensor = input_tensor.inputs[0].inputs[0]

            assert input_tensor is not None, f"Could not find real GEMM input tensor for {node.name}"

            # Get output tensor
            output_tensor = node.outputs[0]

            # Store for LoRA insertion
            lora_gemms[node.name] = {
                "input": input_tensor,
                "output": output_tensor,
                "node": node
            }

    # Insert LoRA patterns for each GEMM
    for gemm_name, tensors in lora_gemms.items():
        input_tensor = tensors["input"]
        output_tensor = tensors["output"]
        node = tensors["node"]

        # Get LoRA weights for this layer
        layer_name = ".".join(
            gemm_name.split("/")[:-1])  # Convert ONNX name to HF name
        if layer_name + ".lora_A.weight" not in lora_weights:
            layer_name = "base_model.model" + layer_name

        lora_a = lora_weights.get(f"{layer_name}.lora_A.weight")
        lora_b = lora_weights.get(f"{layer_name}.lora_B.weight")

        # Calculate scaling from LoRA config
        if hasattr(lora_config, 'r') and hasattr(lora_config, 'lora_alpha'):
            scaling = lora_config.lora_alpha / lora_config.r
        else:
            scaling = lora_weights.get(f"{layer_name}.scaling", 1.0)

        if lora_a is None or lora_b is None:
            print(
                f"Skipping LoRA insertion for {gemm_name} because lora_a or lora_b is None"
            )
            continue

        # Create LoRA computation graph
        lora_a_const = gs.Constant(f"{gemm_name}/lora_A",
                                   lora_a.cpu().half().T.numpy())
        # Fuse scaling into lora_B for better performance
        lora_b_const = gs.Constant(f"{gemm_name}/lora_B",
                                   lora_b.cpu().half().T.numpy() * scaling)

        # First MatMul: input @ lora_A
        lora_mid = gs.Variable(f"{gemm_name}/lora_mid", dtype=np.float16)
        graph.layer(name=f"{gemm_name}/lora_matmul_A",
                    op="MatMul",
                    inputs=[input_tensor, lora_a_const],
                    outputs=[lora_mid])

        # Second MatMul: (input @ lora_A) @ lora_B
        lora_out = gs.Variable(f"{gemm_name}/lora_gemm_out", dtype=np.float16)
        graph.layer(name=f"{gemm_name}/lora_matmul_B",
                    op="MatMul",
                    inputs=[lora_mid, lora_b_const],
                    outputs=[lora_out])

        # Add LoRA output to original output
        final_output = gs.Variable(f"{gemm_name}/lora_add_output",
                                   dtype=np.float16)
        final_output.outputs = output_tensor.outputs.copy()
        graph.layer(name=f"{gemm_name}/lora_add",
                    op="Add",
                    inputs=[output_tensor, lora_out],
                    outputs=[final_output])

        # Update the output connections
        for out_node in final_output.outputs:
            if final_output not in out_node.inputs:
                out_node.inputs.append(final_output)
            if output_tensor in out_node.inputs:
                out_node.inputs.remove(output_tensor)

    graph.cleanup().toposort().fold_constants().cleanup()
    end_time = time.time()
    print(f"Dynamic LoRA patterns inserted in {end_time - start_time}s.")
    return graph
