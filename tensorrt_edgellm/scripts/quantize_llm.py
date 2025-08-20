"""
Command-line script for quantizing language models using TensorRT Edge-LLM.

This script provides a command-line interface for quantizing HuggingFace models
using various quantization schemes supported by NVIDIA ModelOpt.

Usage:
    python quantize_llm.py --torch_dir /path/to/model --output_dir /path/to/output --quantization fp8
"""

import argparse

from tensorrt_edgellm.quantization.llm_quantization import \
    quantize_and_save_model


def main() -> None:
    """
    Main function that parses command line arguments and quantizes the model.
    
    This function sets up argument parsing for the quantization script and calls
    the quantize_and_save_model function with the provided parameters.
    """
    parser = argparse.ArgumentParser(
        description="Quantize a model using NVIDIA ModelOpt")
    parser.add_argument("--torch_dir",
                        type=str,
                        required=True,
                        help="Path to the input model directory")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Path to save the quantized model")
    parser.add_argument(
        "--quantization",
        type=str,
        choices=[None, "fp8", "int4_awq", "nvfp4"],
        default=None,
        help="Quantization method to use (None for no quantization)")
    parser.add_argument("--torch_dtype",
                        type=str,
                        choices=["fp16", "bf16"],
                        default="fp16",
                        help="High precision dtype for model loading")
    parser.add_argument("--dataset_name_or_dir",
                        type=str,
                        default="cnn_dailymail",
                        help="Dataset directory or name for calibration")
    parser.add_argument("--lm_head_quantization",
                        type=str,
                        choices=[None, "fp8", "int4_awq", "nvfp4"],
                        default=None,
                        help="Quantization method for language model head")

    args = parser.parse_args()

    quantize_and_save_model(torch_dir=args.torch_dir,
                            output_dir=args.output_dir,
                            quantization=args.quantization,
                            torch_dtype=args.torch_dtype,
                            dataset_name_or_dir=args.dataset_name_or_dir,
                            lm_head_quantization=args.lm_head_quantization)


if __name__ == "__main__":
    main()
