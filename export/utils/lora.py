# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import time

import numpy as np
import onnx_graphsurgeon as gs


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
    print("Inserting static LoRA patterns...")

    # Track all GEMM nodes that need LoRA
    lora_gemms = {}
    for node in graph.nodes:
        input_tensor = None
        # Only exclude lm_head from LoRA since we will pattern match with the safetensors later
        lora_applicable = "lm_head" not in node.name
        if dtype == "int4" and node.op == "Int4GroupwiseGemmPlugin" and lora_applicable:
            # For int4, the pattern is Mul -> Int4GroupwiseGemmPlugin. The real GEMM input is the input of the Mul node.
            lora_gemms[node.name] = {
                "input": node.inputs[0].inputs[0].inputs[0],
                "output": node.outputs[0],
                "node": node
            }
        elif node.op == "MatMul" and lora_applicable:
            # Find the input tensor based on precision type
            if dtype == "fp16":
                # For FP16, input is direct
                input_tensor = node.inputs[1] if isinstance(
                    node.inputs[0], gs.Constant) else node.inputs[0]
            elif dtype == "fp8":
                # For FP8, need to find Cast node before TRT_FP8QuantizeLinear. FP8 GEMM pattern is Cast -> TRT_FP8QuantizeLinear -> TRT_FP8DequantizeLinear -> Cast -> MatMul
                for inp in node.inputs:
                    if len(inp.inputs) > 0 and inp.inputs[0].op != "Transpose":
                        curr = inp
                        while "TRT" in curr.inputs[0].op or curr.inputs[
                                0].op == "Cast":
                            curr = curr.inputs[0].inputs[0]
                        input_tensor = curr
            elif dtype == "int4_ootb":
                # For INT4 OOTB, find Mul node before DequantizeLinear. The real GEMM input is the input of the Mul node.
                for inp in node.inputs:
                    if len(inp.inputs
                           ) > 0 and inp.inputs[0].op != "DequantizeLinear":
                        input_tensor = inp.inputs[0].inputs[0]
                        break

            elif dtype == "nvfp4":
                # For NVFP4, find input to TRT_FP4DynamicQuantize. The real GEMM input is the input of the TRT_FP4DynamicQuantize node.
                # NVFP4 pattern is TRT_FP4DynamicQuantize -> DequantizeLinear -> DequantizeLinear -> Cast -> MatMul
                for inp in node.inputs:
                    if len(inp.inputs) == 0:
                        continue
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
                if input_tensor is not None and input_tensor.inputs[
                        0].op == "Mul":
                    if "input_quantizer" in input_tensor.inputs[0].name:
                        input_tensor = input_tensor.inputs[0].inputs[0]

            if input_tensor is None:
                print(
                    f"Skipping LoRA insertion for {node.name} because input_tensor is None"
                )
                continue

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
    print(f"Static LoRA patterns inserted in {end_time - start_time}s.")
    return graph


def insert_dynamic_lora(graph: gs.Graph, dtype: str):
    """
    Insert dynamic LoRA pattern modifications for different precisions.
    Instead of using static weights, this function adds input tensors for LoRA weights.

    Args:
        graph: gs.Graph
            The ONNX graph to modify
        dtype: str
            The precision type ("fp16", "fp8", "int4", "int4_ootb", "nvfp4")

    Returns:
        Modified graph with dynamic LoRA patterns inserted
    """
    start_time = time.time()
    print("Inserting dynamic LoRA patterns...")

    # Track all GEMM nodes that need LoRA
    lora_gemms = {}
    for node in graph.nodes:
        input_tensor = None
        weight_shape = None
        if dtype == "int4" and node.op == "Int4GroupwiseGemmPlugin" and any(
                target in node.name for target in [
                    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                    "up_proj", "down_proj"
                ]):
            # For int4 plugin, get shape from attributes
            weight_shape = (node.attrs["gemm_k"], node.attrs["gemm_n"])
            input_tensor = node.inputs[0]
        elif node.op == "MatMul" and any(target in node.name for target in [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                "down_proj"
        ]):
            # Find the input tensor and weight shape based on precision type
            if dtype == "fp16":
                # For FP16, weight is direct input
                weight_id = 0 if isinstance(node.inputs[0], gs.Constant) else 1
                weight = node.inputs[weight_id]
                weight_shape = weight.shape
                input_tensor = node.inputs[1 - weight_id]
            elif dtype == "fp8":
                # For FP8, need to find Cast node before TRT_FP8QuantizeLinear. FP8 GEMM pattern is Cast -> TRT_FP8QuantizeLinear -> TRT_FP8DequantizeLinear -> Cast -> MatMul
                if node.inputs[0].inputs[0].op == "Transpose":
                    weight_id = 0
                else:
                    weight_id = 1

                # FP8 weights is DequantizeLinear->Cast->Transpose->Matmul
                weight_shape = node.inputs[weight_id].inputs[0].inputs[
                    0].inputs[0].inputs[0].inputs[0].inputs[0].shape[::-1]

                for inp in node.inputs:
                    if inp.inputs[0].op != "Transpose":
                        curr = inp
                        while "TRT" in curr.inputs[0].op or curr.inputs[
                                0].op == "Cast":
                            curr = curr.inputs[0].inputs[0]
                        input_tensor = curr
            elif dtype == "int4_ootb":
                # For INT4 OOTB, find Mul node before DequantizeLinear. The real GEMM input is the input of the Mul node.
                if node.inputs[0].inputs[0].op == "DequantizeLinear":
                    weight_id = 0
                else:
                    weight_id = 1

                input_tensor = node.inputs[1 - weight_id].inputs[0].inputs[0]
                # int4_ootb weights is DequantizeLinear->MatMul directly
                weight_shape = node.inputs[weight_id].inputs[0].inputs[0].shape

            elif dtype == "nvfp4":
                # For NVFP4, find input to TRT_FP4DynamicQuantize. The real GEMM input is the input of the TRT_FP4DynamicQuantize node.
                # NVFP4 pattern is TRT_FP4DynamicQuantize -> DequantizeLinear -> DequantizeLinear -> Cast -> MatMul
                for index, inp in enumerate(node.inputs):
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
                        weight_id = 1 - index
                        break
                # For NVFP4_AWQ recipe, the input is smoothed by a Mul node. The real GEMM input is the input of the Mul node.
                if input_tensor.inputs[0].op == "Mul":
                    if "input_quantizer" in input_tensor.inputs[0].name:
                        input_tensor = input_tensor.inputs[0].inputs[0]
                # NVFP4 weights is DequantizeLinear->DequantizeLinear->Transpose->Cast->MatMul
                weight_shape = node.inputs[weight_id].inputs[0].inputs[
                    0].inputs[0].inputs[0].inputs[0].inputs[0].shape[::-1]

            assert input_tensor is not None and weight_shape is not None, f"Could not find input tensor or weight shape for {node.name}"

        if input_tensor is not None and weight_shape is not None:
            # Get output tensor
            output_tensor = node.outputs[0]

            # Store for LoRA insertion
            lora_gemms[node.name] = {
                "input": input_tensor,
                "output": output_tensor,
                "node": node,
                "weight_shape": weight_shape
            }

    # Insert LoRA patterns for each GEMM
    for gemm_name, tensors in lora_gemms.items():
        input_tensor = tensors["input"]
        output_tensor = tensors["output"]
        node = tensors["node"]
        weight_shape = tensors["weight_shape"]
        k, n = weight_shape

        # Create dynamic input tensors for LoRA weights
        gemm_name_for_lora = gemm_name.replace("/", ".").rsplit(".", 1)[0][1:]

        lora_a = gs.Variable(f"{gemm_name_for_lora}.lora_A.weight",
                             dtype=np.float16,
                             shape=[k, f"{gemm_name_for_lora}.rank"])
        lora_b = gs.Variable(f"{gemm_name_for_lora}.lora_B.weight",
                             dtype=np.float16,
                             shape=[f"{gemm_name_for_lora}.rank", n])
        graph.inputs.extend([lora_a, lora_b])

        # First MatMul: input @ lora_A
        lora_mid = gs.Variable(f"{gemm_name}/lora_mid", dtype=np.float16)
        graph.layer(name=f"{gemm_name}/lora_matmul_A",
                    op="MatMul",
                    inputs=[input_tensor, lora_a],
                    outputs=[lora_mid])

        # Second MatMul: (input @ lora_A) @ lora_B
        lora_out = gs.Variable(f"{gemm_name}/lora_gemm_out", dtype=np.float16)
        graph.layer(name=f"{gemm_name}/lora_matmul_B",
                    op="MatMul",
                    inputs=[lora_mid, lora_b],
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
