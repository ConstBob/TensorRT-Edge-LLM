import argparse
import os
import shutil
import time

import onnx
import onnx_graphsurgeon as gs
import torch
from modelopt.torch.export.unified_export_hf import export_hf_checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from utils.export_utils import WrapperModelForCausalLM, torch_to_onnx
from utils.quantization_utils import quantize
from utils.surgeon_utils import (fold_fp8_qdq_to_dq, insert_attention_plugin,
                                 insert_gather_last_token, insert_int4_dq)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_dir',
                        type=str,
                        help="The folder of HF PyTorch model ckpt",
                        required=False)
    parser.add_argument('--dtype',
                        type=str,
                        default="fp16",
                        choices=["fp16", "fp8", "int4"],
                        help="The precision of onnx export")
    parser.add_argument('--output_dir',
                        type=str,
                        help="The directory to store the generated ONNX model",
                        required=True)

    parser.add_argument(
        '--onnx_path',
        type=str,
        help="Pass this option when you have existing onnx to surgeon",
        required=False)

    parser.add_argument('--mode',
                        type=str,
                        default="plugin",
                        choices=["plugin", "ootb"],
                        help="Whether to insert AttentionPlugin to ONNX")
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
    parser.add_argument('--state_dict_path',
                        type=str,
                        help="The path of state dict path only for int4",
                        default=None)
    parser.add_argument(
        '--config_path',
        type=str,
        help=
        "The path of config.json, in case it is not with the PyTorch or ONNX file",
        default=None)
    args = parser.parse_args()
    return args


def get_config_path(args):
    """
    Look for config.json. It is needed for AttentionPlugin insertion and is recommended to keep a copy per ONNX path
    """
    if args.config_path and os.path.exists(args.config_path):
        return args.config_path
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


def main():
    start_time = time.time()
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    assert args.torch_dir or args.onnx_path, "You need to provide either --torch_dir or --onnx_path to process the export script"
    state_dict = None
    config_path = get_config_path(args)

    if args.torch_dir:
        # Exporting ONNX from PyTorch model
        torch_dir = args.torch_dir
        model = AutoModelForCausalLM.from_pretrained(
            torch_dir, torch_dtype=torch.float16).cuda()

        if args.save_original:
            onnx_dir = args.output_dir + "_raw"
        else:
            onnx_dir = args.output_dir

        os.makedirs(onnx_dir, exist_ok=True)

        if args.dtype == "fp16" or args.dtype == "int4":
            if args.dtype == "int4":
                print(
                    "int4 native onnx.export does not support. Need to surgeon from fp16 ONNX..."
                )
            else:
                print("Loading fp16 ONNX model...")
            torch_to_onnx(WrapperModelForCausalLM(model), onnx_dir)
            shutil.copy(os.path.join(torch_dir, "config.json"),
                        os.path.join(onnx_dir, "config.json"))

        # Need to quantize model to fp8 or int4
        if args.dtype == "fp8" or args.dtype == "int4":
            tokenizer = AutoTokenizer.from_pretrained(torch_dir)
            model = quantize(model, tokenizer, args.dtype, args.dataset_dir)

            if args.dtype == "fp8":
                print(
                    "Exporting fp8 ONNX model from quantized PyTorch model...")
                torch_to_onnx(WrapperModelForCausalLM(model), onnx_dir)
                shutil.copy(config_path, os.path.join(onnx_dir, "config.json"))

            # Compress weights
            quantized_model_dir = f"{args.output_dir}_{args.dtype}_quantized"
            os.makedirs(quantized_model_dir, exist_ok=True)
            export_hf_checkpoint(model, dtype=torch.float16)
            state_dict = model.state_dict()
            torch.save(state_dict,
                       os.path.join(quantized_model_dir, "model.pth"))
    else:
        print(f"ONNX path given. Importing ONNX from {args.onnx_path}")
        # Int4 requires knowledge of the int4 weights and scales.
        if args.dtype == "int4":
            assert args.state_dict_path, "You need to pass state_dict for int4 ONNX export"
            state_dict = torch.load(args.state_dict_path)

    # Surgeon graph based on precision and mode
    raw_onnx_path = f"{onnx_dir}/model.onnx" if args.torch_dir else args.onnx_path

    if args.mode == "plugin":
        # AttentionPlugin requires knowledge of the model. Assume config file is in onnx_dir
        assert os.path.exists(
            config_path), "No config.json is found. Cannot run plugin mode."

        config = AutoConfig.from_pretrained(
            config_path,
            trust_remote_code=True,
        ).to_dict()
    else:
        print(
            "Warning: DriveOS LLM SDK currently does not support OOTB TensorRT so far."
        )

    t0 = time.time()
    graph = gs.import_onnx(onnx.load(raw_onnx_path))
    t1 = time.time()
    print(f"Importing ONNX graph takes {t1 - t0}s.")

    if args.mode == "plugin":
        graph = insert_attention_plugin(graph, config)

    if args.dtype == "fp8":
        graph = fold_fp8_qdq_to_dq(graph)

    elif args.dtype == "int4":
        graph = insert_int4_dq(graph, state_dict)

    graph = insert_gather_last_token(graph)

    graph.cleanup().toposort().fold_constants().cleanup().toposort()

    output_dir = args.output_dir
    t2 = time.time()
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

    output_onnx_name = f"{output_dir}/model.onnx"
    onnx_model = gs.export_onnx(graph)

    onnx.save_model(onnx_model,
                    output_onnx_name,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=f"onnx_model.data",
                    convert_attribute=True)

    if os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(output_dir, "config.json"))
    t3 = time.time()
    print(f"ONNX export and save takes {t3 - t2}s.")
    end_time = time.time()
    print(
        f"Model ONNX saved to {output_dir} with {args.dtype} precision in {end_time - start_time}s."
    )


if __name__ == '__main__':
    main()
