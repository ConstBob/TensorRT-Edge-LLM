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

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
from packaging.version import Version
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from utils.export_utils import (WrapperEagleBaseModelForCausalLM,
                                WrapperEagleDraftModelForCausalLM,
                                WrapperModelForCausalLM, llm_to_onnx,
                                load_model_with_lora)
from utils.lora import insert_dynamic_lora, insert_static_lora
from utils.surgeon_utils import (RopeType, insert_attention_plugin,
                                 insert_gather_last_token,
                                 insert_gather_last_token_eagle)


def llm_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_dir',
                        type=str,
                        help="The folder of HF PyTorch model ckpt",
                        required=False)
    parser.add_argument('--dtype',
                        type=str,
                        default="fp16",
                        choices=["fp16", "fp8", "int4", "nvfp4", "int4_ootb"],
                        help="The precision of onnx export")
    parser.add_argument(
        '--eagle_torch_dir',
        type=str,
        help="The folder of HF PyTorch model ckpt for eagle draft",
        required=False)

    parser.add_argument(
        '--lm_head',
        type=str,
        default="fp16",
        choices=["fp16"],
        help=
        "The precision of lm_head. Currently only fp16 is tested and supported"
    )
    parser.add_argument('--output_dir',
                        type=str,
                        help="The directory to store the generated ONNX model",
                        required=True)

    parser.add_argument(
        '--onnx_path',
        type=str,
        help="Pass this option when you have existing onnx to surgeon",
        required=False)

    parser.add_argument(
        '--mode',
        type=str,
        default="plugin",
        choices=["plugin"],
        help=
        "Whether to insert AttentionPlugin to ONNX. In the future there might be other options to use native TensorRT Attention"
    )
    parser.add_argument(
        '--save_original',
        action='store_true',
        default=False,
        help=
        "Save the original ONNX from torch.onnx.export without any modification"
    )
    parser.add_argument('--dataset_dir',
                        type=str,
                        help="The path of dataset for quantization",
                        required=False)
    parser.add_argument(
        '--config_path',
        type=str,
        help=
        "The path of config.json, in case it is not with the PyTorch or ONNX file",
        default=None)
    parser.add_argument(
        '--max_seq_length',
        type=int,
        help=
        "Maximum sequence length as an attribute to AttentionPlugin. Change this when you need to run long context inference",
        default=4096)
    parser.add_argument('--lora_dir',
                        type=str,
                        help="The directory containing LoRA weights",
                        required=False)
    parser.add_argument(
        '--lora_mode',
        type=str,
        default="none",
        choices=["merged", "none", "static", "dynamic"],
        help=
        "LoRA mode. Currently merged mode (weights merged into base model), static mode (weights not merged) and dynamic mode (weights as inputs) are supported",
        required=False)
    parser.add_argument(
        '--eagle_base',
        type=bool,
        default=False,
        required=False,
    )
    parser.add_argument(
        '--eagle_draft',
        type=bool,
        default=False,
        required=False,
    )
    parser.add_argument(
        '--eagle3',
        type=bool,
        default=False,
        required=False,
    )
    return parser


def get_config_path(args):
    """
    Look for config.json. It is needed for AttentionPlugin insertion and is recommended to keep a copy per ONNX path
    """
    if args.config_path and os.path.exists(args.config_path):
        return args.config_path
    if args.eagle_torch_dir:
        eagle_config = os.path.join(args.eagle_torch_dir, "config.json")
        if os.path.exists(eagle_config):
            return eagle_config
        else:
            assert False, "Cannot find config.json in eagle_torch_dir. Please pass in --config_path."
    if args.torch_dir:
        torch_config = os.path.join(args.torch_dir, "config.json")
        if os.path.exists(torch_config):
            return torch_config
    if args.onnx_path:
        onnx_config = os.path.join(os.path.dirname(args.onnx_path),
                                   "config.json")
        if os.path.exists(onnx_config):
            return onnx_config
    print(
        "Warning: cannot find config.json. onnx_graphsurgeon requires HF Config to insert AttentionPlugin. Please pass in --config_path."
    )
    return None


def modelopt_quantized(torch_dir):
    return os.path.exists(os.path.join(torch_dir, "modelopt_state.pth"))


def export_raw_llm(model,
                   output_dir,
                   dtype,
                   config_path,
                   torch_dir,
                   lm_head_precision="fp16",
                   dataset_dir="",
                   wrapper_cls=WrapperModelForCausalLM,
                   extra_inputs={},
                   extra_dyn_axes={},
                   eagle3=False):
    """
    Export raw llm model to ONNX and perform quantization.

    Args:
        model: torch.nn.module
        output_dir: str
        dtype: str
        config_path: str
        torch_dir: str, Used for loading tokenizer for quantization
        dataset_dir: str, Used for quantization
    """
    os.makedirs(output_dir, exist_ok=True)

    if dtype == "fp16" or ("int4" in dtype):
        if "int4" in dtype:
            print(
                "int4 native onnx.export does not support. Need to surgeon from fp16 ONNX..."
            )
        else:
            print("Loading fp16 ONNX model...")

        if wrapper_cls in (WrapperEagleDraftModelForCausalLM,
                           WrapperEagleBaseModelForCausalLM):
            llm_to_onnx(wrapper_cls(model, eagle3=eagle3),
                        output_dir,
                        extra_inputs=extra_inputs,
                        extra_dyn_axes=extra_dyn_axes)
        else:
            llm_to_onnx(wrapper_cls(model),
                        output_dir,
                        extra_inputs=extra_inputs,
                        extra_dyn_axes=extra_dyn_axes)
        shutil.copy(config_path, os.path.join(output_dir, "config.json"))

    # Need to quantize model to fp8, int4 or nvfp4
    if dtype in ["fp8", "int4", "nvfp4", "int4_ootb"]:
        # Avoid import modelopt when no quantization is needed
        from modelopt.torch.export import export_hf_checkpoint
        from modelopt.torch.quantization.utils import is_quantized_linear
        from utils.quantization_utils import quantize
        tokenizer = AutoTokenizer.from_pretrained(torch_dir,
                                                  trust_remote_code=True)
        modelopt_state = os.path.join(torch_dir, "modelopt_state.pth")
        if not os.path.exists(modelopt_state):
            model = quantize(model, tokenizer, dtype, lm_head_precision,
                             dataset_dir)

            if dtype == "nvfp4":
                # This is required for nvfp4 ONNX export
                for module in model.modules():
                    assert not isinstance(
                        module, torch.nn.Linear) or is_quantized_linear(module)
                    if isinstance(module, torch.nn.Linear):
                        module.input_quantizer._trt_high_precision_dtype = "Half"
                        module.input_quantizer._onnx_quantizer_type = "dynamic"
                        module.weight_quantizer._onnx_quantizer_type = "static"

            if dtype == "fp8" or dtype == "nvfp4":
                print(
                    f"Exporting {dtype} ONNX model from quantized PyTorch model..."
                )
                llm_to_onnx(wrapper_cls(model),
                            output_dir,
                            extra_inputs=extra_inputs,
                            extra_dyn_axes=extra_dyn_axes)
                shutil.copy(config_path, os.path.join(output_dir,
                                                      "config.json"))

            # Compress weights
            quantized_model_dir = f"{output_dir}_{dtype}_quantized"
            os.makedirs(quantized_model_dir, exist_ok=True)
            with torch.inference_mode():
                export_hf_checkpoint(model,
                                     dtype=torch.float16,
                                     export_dir=quantized_model_dir)

    return model.state_dict()


def surgeon_llm(raw_onnx_path,
                output_dir,
                dtype,
                mode,
                config_path,
                state_dict,
                max_seq_length=4096,
                rope_type=RopeType.kROPE_ROTATE_NEOX,
                extra_plugin_inputs=[],
                extra_plugin_attributes={},
                lm_head_precision="fp16",
                lora_mode="none",
                lora_config=None,
                lora_weights=None,
                eagle_base=False,
                eagle_draft=False,
                eagle3=False):
    """
    Surgeon raw llm onnx to fit TRT.
    For example, insert attention plugin, insert quantization q/dq nodes.

    Args:
        raw_onnx_path: str
        output_dir: str
        dtype: str
        mode: str
        config_path: str
        state_dict: None or OrderedDict
        rope_type: RopeType.
        extra_plugin_inputs: list
        lora_config: PeftConfig, optional
            The LoRA adapter configuration for dynamic mode
        lora_weights: dict, optional
            The LoRA adapter weights for dynamic mode
    """
    if mode == "plugin":
        # AttentionPlugin requires knowledge of the model. Assume config file is in onnx_dir
        assert os.path.exists(
            config_path), "No config.json is found. Cannot run plugin mode."

        config = AutoConfig.from_pretrained(
            config_path,
            trust_remote_code=True,
        ).to_dict()
    else:
        print(
            "Warning: DriveOS LLM SDK currently does not support OOTB(Out-of-the-box) TensorRT so far."
        )

    t0 = time.time()
    graph = gs.import_onnx(onnx.load(raw_onnx_path))
    t1 = time.time()
    print(f"Importing ONNX graph takes {t1 - t0}s.")

    if mode == "plugin":
        graph = insert_attention_plugin(graph, config, rope_type,
                                        max_seq_length, extra_plugin_inputs,
                                        extra_plugin_attributes)
    if eagle_base:
        graph = insert_gather_last_token_eagle(graph, False)
    elif eagle_draft:
        graph = insert_gather_last_token_eagle(graph, True, eagle3)
    else:
        graph = insert_gather_last_token(graph)
    graph.fold_constants().cleanup().toposort()

    if dtype == "int4_ootb":
        from utils.surgeon_utils import insert_int4_dq
        graph = insert_int4_dq(graph, state_dict)

    elif dtype == "int4":
        from utils.surgeon_utils import insert_int4_gemm_plugin
        graph = insert_int4_gemm_plugin(graph, state_dict)

    elif dtype == "fp8" or lm_head_precision == "fp8":
        from utils.surgeon_utils import fold_fp8_qdq_to_dq
        graph = fold_fp8_qdq_to_dq(graph)

    os.makedirs(output_dir, exist_ok=True)
    t2 = time.time()

    onnx_model = gs.export_onnx(graph)

    if dtype == "nvfp4":
        t4 = time.time()
        from modelopt.onnx.quantization.qdq_utils import fp4qdq_to_2dq
        onnx_model = fp4qdq_to_2dq(onnx_model, verbose=True)
        t5 = time.time()
        print(f"nvfp4 qdq to 2 dqs inserted in {t5 - t4}.")

    # Insert static LoRA as the last step
    if lora_mode == "static" and lora_config is not None and lora_weights is not None:
        graph = gs.import_onnx(onnx_model)
        graph = insert_static_lora(graph, lora_config, lora_weights, dtype)
        onnx_model = gs.export_onnx(graph)
    elif lora_mode == "dynamic":
        graph = gs.import_onnx(onnx_model)
        graph = insert_dynamic_lora(graph, dtype)
        onnx_model = gs.export_onnx(graph)

    output_onnx_name = f"{output_dir}/model.onnx"
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

    onnx.save_model(onnx_model,
                    output_onnx_name,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=f"onnx_model.data",
                    convert_attribute=True)

    if os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(output_dir, "config.json"))

    t3 = time.time()
    print(f"Surgeon LLM completed in {t3 - t2}s.")


def check_dtype_support(args):
    """
    Check whether the dtype is supported by DriveOS LLM SDK. Returns False if it is not supported because of:
    1. Modelopt < 0.23.0 does not support nvfp4
    2. Modelopt > 0.19.0 has accuracy issues for int4
    """

    def get_modelopt_version():
        try:
            import modelopt
            return Version(modelopt.__version__)
        except Exception as e:
            print(f"Modelopt version cannot be parsed. Reason: {str(e)}")

    if (args.dtype == "nvfp4") and get_modelopt_version() < Version("0.23.0"):
        print(
            "nvfp4 is not supported by installed modelopt version. Please upgrade to 0.23.0 or above for nvfp4 export. Exit."
        )
        return False

    if ("int4" in args.dtype) and get_modelopt_version() > Version("0.19.0"):
        print(
            "int4 have accuracy issues with modelopt version > 0.19.0. Please use modelopt==0.19.0. Exit."
        )
        return False

    if args.dtype == "int4_ootb":
        print(
            "int4 ootb has performance issues. If you want to run int4_awq, it is recommended to export by default as plugin"
        )

    return True


def main(args):
    assert args.torch_dir or args.onnx_path, "You need to provide either --torch_dir or --onnx_path to process the export script"
    start_time = time.time()
    state_dict = None
    lora_config = None
    lora_weights = None

    if not check_dtype_support(args):
        return

    if args.onnx_path:
        raw_onnx_path = args.onnx_path

    if args.eagle_draft:
        assert args.eagle_torch_dir, "You need to provide --eagle_torch_dir when you want to export eagle draft"
        from eagle.ea_model import EagleModel
        onnx_dir = args.output_dir
        model = EagleModel.from_pretrained(
            base_model_path=args.torch_dir,
            ea_model_path=args.eagle_torch_dir,
            use_eagle3=args.eagle3,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu",
            local_files_only=True,
        )

        draft_model = model.ea_layer
        draft_model.to(torch.float16).to('cuda')
        dummy_len_input_hidden = 10
        dummy_bs = 1
        if args.eagle3:
            hidden_states_input = torch.randn(
                (dummy_bs, dummy_len_input_hidden,
                 model.config.hidden_size * 3),
                dtype=torch.float16).cuda()
            hidden_states_from_draft = torch.randn(
                (dummy_bs, dummy_len_input_hidden, model.config.hidden_size),
                dtype=torch.float16).cuda()

            extra_inputs = {
                "hidden_states_input": hidden_states_input,
                "hidden_states_from_draft": hidden_states_from_draft
            }
            extra_dyn_axes = {
                "hidden_states_input": {
                    0: "batch_size",
                    1: "seq_len",
                },
                "hidden_states_from_draft": {
                    0: "batch_size",
                    1: "seq_len",
                }
            }

        else:
            hidden_states_input = torch.randn(
                (dummy_bs, dummy_len_input_hidden, model.config.hidden_size),
                dtype=torch.float16).cuda()
            extra_inputs = {"hidden_states_input": hidden_states_input}
            extra_dyn_axes = {
                "hidden_states_input": {
                    0: "batch_size",
                    1: "seq_len"
                }
            }

        state_dict = export_raw_llm(
            draft_model,
            onnx_dir,
            args.dtype,
            args.config_path,
            args.torch_dir,
            args.lm_head,
            args.dataset_dir,
            wrapper_cls=WrapperEagleDraftModelForCausalLM,
            extra_inputs=extra_inputs,
            extra_dyn_axes=extra_dyn_axes,
            eagle3=args.eagle3)
        raw_onnx_path = f"{onnx_dir}/model.onnx"
    elif args.torch_dir:
        # Exporting ONNX from PyTorch model
        torch_dir = args.torch_dir
        # This can be useful for loading modelopt saved model
        if modelopt_quantized(torch_dir):
            assert "int4" in args.dtype, f"Currently only int4 supports quantized checkpoint, the precision is {args.dtype}"
            assert args.onnx_path is not None, f"Int4 quantized model only supports onnx graphsurgeon from existing fp16 ONNX"
            from safetensors import safe_open
            state_dict = {}
            for filename in os.listdir(torch_dir):
                if ".safetensors" == os.path.splitext(filename)[1]:
                    with safe_open(os.path.join(torch_dir, filename),
                                   framework="pt",
                                   device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
            raw_onnx_path = args.onnx_path
        else:
            model = AutoModelForCausalLM.from_pretrained(
                torch_dir, torch_dtype=torch.float16,
                trust_remote_code=True).cuda()

            # Handle LoRA weights if provided
            if args.lora_mode == "static":
                model, lora_config, lora_weights = load_model_with_lora(
                    model, args.lora_dir, args.lora_mode)
            elif args.lora_mode == "merged":
                model = load_model_with_lora(model, args.lora_dir,
                                             args.lora_mode)

            if args.save_original:
                onnx_dir = args.output_dir + "_raw"
            else:
                onnx_dir = args.output_dir

            if args.eagle_base:
                state_dict = export_raw_llm(
                    model,
                    onnx_dir,
                    args.dtype,
                    args.config_path,
                    args.torch_dir,
                    args.lm_head,
                    args.dataset_dir,
                    wrapper_cls=WrapperEagleBaseModelForCausalLM,
                    eagle3=args.eagle3)
            else:
                state_dict = export_raw_llm(
                    model,
                    onnx_dir,
                    args.dtype,
                    args.config_path,
                    args.torch_dir,
                    args.lm_head,
                    args.dataset_dir,
                    wrapper_cls=WrapperModelForCausalLM)

            # Surgeon graph based on precision and mode
            raw_onnx_path = f"{onnx_dir}/model.onnx"

    # Apply surgical operations
    extra_plugin_attributes = {}
    extra_plugin_inputs = []
    if args.eagle_base or args.eagle_draft:
        attention_mask = gs.Variable("attention_mask", np.int32,
                                     ['batch_size', 'q_len', 'q_len_aligned'])
        attention_pos_id = gs.Variable("attention_pos_id", np.int32,
                                       ['batch_size', 'q_len'])
        extra_plugin_inputs.append(attention_mask)
        extra_plugin_inputs.append(attention_pos_id)
        extra_plugin_attributes["enable_tree_attention"] = 1

    surgeon_llm(raw_onnx_path,
                args.output_dir,
                args.dtype,
                args.mode,
                args.config_path,
                state_dict,
                args.max_seq_length,
                lm_head_precision=args.lm_head,
                extra_plugin_inputs=extra_plugin_inputs,
                extra_plugin_attributes=extra_plugin_attributes,
                lora_mode=args.lora_mode,
                lora_config=lora_config,
                lora_weights=lora_weights,
                eagle_base=args.eagle_base,
                eagle_draft=args.eagle_draft,
                eagle3=args.eagle3)
    if args.eagle3 and args.eagle_draft:
        load_model_path = os.path.join(args.eagle_torch_dir,
                                       "pytorch_model.bin")
        ea_layer_state_dict = torch.load(load_model_path, weights_only=True)
        # When the  draft vocab size is not equal to the base vocab size, we need to map the token id from draft to base using d2t.bin
        d2t_tensor = ea_layer_state_dict['d2t']
        d2t_path = os.path.join(onnx_dir, "d2t.bin")
        with open(d2t_path, 'wb') as f:
            f.write(d2t_tensor.numpy().astype(np.int64).tobytes())
    end_time = time.time()
    print(
        f"LLM ONNX saved to {args.output_dir} with {args.dtype} precision in {end_time - start_time}s."
    )


if __name__ == '__main__':
    parser = llm_arguments()
    args = parser.parse_args()
    args.config_path = get_config_path(args)
    main(args)
