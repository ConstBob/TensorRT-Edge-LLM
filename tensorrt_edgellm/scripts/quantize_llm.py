"""
Command-line script for quantizing language models using TensorRT Edge-LLM.

This script provides a command-line interface for quantizing HuggingFace models
using various quantization schemes supported by NVIDIA ModelOpt.

Usage:
    # Quantize with FP8 quantization
    python quantize_llm.py --model_dir /path/to/model --output_dir /path/to/output --quantization fp8
    
    # Quantize without quantization (default)
    python quantize_llm.py --model_dir /path/to/model --output_dir /path/to/output
    
    # Quantize with different quantization for LM head
    python quantize_llm.py --model_dir /path/to/model --output_dir /path/to/output --quantization fp8 --lm_head_quantization int4_awq
"""

import argparse
import sys
import traceback

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
    parser.add_argument("--model_dir",
                        type=str,
                        required=True,
                        help="Path to the input model directory")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Path to save the quantized model")
    parser.add_argument("--quantization",
                        type=str,
                        required=False,
                        choices=["fp8", "int4_awq", "nvfp4"],
                        default=None,
                        help="Quantization method to use")
    parser.add_argument("--torch_dtype",
                        type=str,
                        choices=["fp16", "bf16"],
                        required=False,
                        default="fp16",
                        help="High precision dtype for model loading")
    parser.add_argument("--dataset_dir",
                        type=str,
                        required=False,
                        default="cnn_dailymail",
                        help="Dataset directory or name for calibration")
    parser.add_argument("--lm_head_quantization",
                        type=str,
                        required=False,
                        choices=["fp8", "int4_awq", "nvfp4"],
                        default=None,
                        help="Quantization method for language model head")

    args = parser.parse_args()

    try:
        quantize_and_save_model(model_dir=args.model_dir,
                                output_dir=args.output_dir,
                                quantization=args.quantization,
                                torch_dtype=args.torch_dtype,
                                dataset_dir=args.dataset_dir,
                                lm_head_quantization=args.lm_head_quantization)
        print("Model quantization completed successfully!")
    except Exception as e:
        print(f"Error during model quantization: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
