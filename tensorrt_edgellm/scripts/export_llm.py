# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Command-line script for exporting LLM models to ONNX format using TensorRT Edge-LLM.

This script provides a command-line interface for exporting language models to ONNX format
with support for standard models, EAGLE base models, and EAGLE draft models.

Usage:
    # Standard model export
    python export_llm.py --model_dir /path/to/model --output_dir /path/to/output
    
    # EAGLE complete export (both base and draft)
    python export_llm.py --model_dir /path/to/base_model --draft_model_dir /path/to/draft_model --output_dir /path/to/output --eagle2
"""

import argparse
import sys
import traceback

from tensorrt_edgellm.onnx_export.llm_export import llm_export


def main() -> None:
    """
    Main function that parses command line arguments and exports the LLM model.
    
    This function sets up argument parsing for the LLM export script and calls
    the llm_export function with the provided parameters. For EAGLE models with
    both base and draft models, it exports both models to separate subdirectories.
    """
    parser = argparse.ArgumentParser(
        description="Export LLM model to ONNX format using TensorRT Edge-LLM")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the input model directory (base model for EAGLE)")
    parser.add_argument(
        "--draft_model_dir",
        type=str,
        required=False,
        help=
        "Path to the draft model directory (required for EAGLE complete export)"
    )
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Path to save the exported ONNX model")
    parser.add_argument(
        "--eagle2",
        required=False,
        action='store_true',
        help="Whether this is EAGLE2 (True) or EAGLE3 (False) for EAGLE models"
    )
    parser.add_argument(
        "--max_position_embeddings",
        type=int,
        required=False,
        default=4096,
        help="Maximum positional embedding length (default: 4096)")
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
        # Determine EAGLE version
        eagle2 = args.eagle2

        # Export model(s)
        llm_export(model_dir=args.model_dir,
                   output_dir=args.output_dir,
                   eagle2=eagle2,
                   draft_model_dir=args.draft_model_dir,
                   max_position_embeddings=args.max_position_embeddings,
                   device=args.device)

        print("LLM model export completed successfully!")

    except Exception as e:
        print(f"Error during LLM model export: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
