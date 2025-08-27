"""
Command-line script for exporting visual models to ONNX format using TensorRT Edge-LLM.

This script provides a command-line interface for exporting visual components of
multimodal models (Qwen2-VL, Qwen2.5-VL, InternVL3) to ONNX format with optional
quantization support.

Usage:
    # Export without quantization
    python export_visual.py --model_dir /path/to/model --output_dir /path/to/output
    
    # Export with FP8 quantization
    python export_visual.py --model_dir /path/to/model --output_dir /path/to/output --quantization fp8
    
    # Export with specific device
    python export_visual.py --model_dir /path/to/model --output_dir /path/to/output --device cuda:1
    
"""

import argparse
import sys
import traceback

from tensorrt_edgellm.onnx_export.visual_export import visual_export


def main() -> None:
    """
    Main function that parses command line arguments and exports the visual model.
    
    This function sets up argument parsing for the visual export script and calls
    the visual_export function with the provided parameters.
    """
    parser = argparse.ArgumentParser(
        description="Export visual model to ONNX format using TensorRT Edge-LLM"
    )
    parser.add_argument("--model_dir",
                        type=str,
                        required=True,
                        help="Path to the input model directory")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Path to save the exported ONNX model")
    parser.add_argument("--dtype",
                        type=str,
                        required=False,
                        choices=["fp16"],
                        default="fp16",
                        help="Data type for export (only fp16 supported)")
    parser.add_argument(
        "--quantization",
        type=str,
        required=False,
        choices=["fp8"],
        default=None,
        help="Quantization method to use (fp8 for FP8 quantization)")
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help=
        "Device to load the model on (default: cuda, options: cpu, cuda, cuda:0, cuda:1, etc.)"
    )

    args = parser.parse_args()

    try:
        visual_export(model_dir=args.model_dir,
                      output_dir=args.output_dir,
                      dtype=args.dtype,
                      quantization=args.quantization,
                      device=args.device)
        print("Visual model export completed successfully!")
    except Exception as e:
        print(f"Error during visual model export: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
