'''
Export the model as binary files. info.json contains the shape info for each weight. This will be used in the future to implement models with TensorRT C++ API.
'''
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        type=str,
                        help="The name or path of HF model ckpt",
                        required=True)
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


def float_dtype_to_torch(dtype_str):
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    return torch.float32


def main():
    args = parse_arguments()
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(
        float_dtype_to_torch(args.dtype))
    shape_dict = {}
    os.makedirs(args.output_dir, exist_ok=True)
    for name, param in model.named_parameters():
        shape_dict[name] = param.shape
        np_array = param.detach().cpu().numpy()
        np_array.tofile(f"{args.output_dir}/{name}.bin")
        print(
            f"{name}: shape: {shape_dict[name]}; type: {param.dtype}; saved to {name}.bin"
        )
    with open(f"{args.output_dir}/info.json", 'w') as json_file:
        json.dump(shape_dict, json_file)

    print(
        f"Model binary exported to {args.output_dir} with {args.dtype} precision"
    )


if __name__ == '__main__':
    main()
