import argparse
from optimum.exporters.onnx import main_export
import unittest.mock as mock
import torch

# https://github.com/huggingface/optimum/issues/1952
def _unmask_unattended_patched(
    expanded_mask: torch.Tensor, attention_mask: torch.Tensor
):
    return expanded_mask

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help="The name or path of HF model ckpt", required=True)
    parser.add_argument('--output_dir', type=str, help="The directory to store the generated ONNX model", required=True)
    parser.add_argument('--dtype', type=str, default="fp16", help="The floating point precision to use for export")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    main_export(
        args.model_dir,
        task="text-generation-with-past",
        output=args.output_dir,
        opset=17,
        dtype=args.dtype,
        device="cuda",
        framework="pt",
        no_post_process=True
    )
    print(f"Model exported to {args.output_dir} with {args.dtype} precision")

if __name__ == '__main__':
    main()