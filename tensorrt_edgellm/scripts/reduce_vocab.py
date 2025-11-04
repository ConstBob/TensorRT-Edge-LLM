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
This script provides a command-line interface for reducing vocabulary size
based on token frequency analysis in a calibration dataset.

Usage:
    # Reduce vocabulary to 16k tokens with default cnn_dailymail dataset
    python reduce_vocab.py --model_dir /path/to/model --output_dir /path/to/output --reduced_vocab_size 16384
    
    # Use custom dataset for analysis
    python reduce_vocab.py --model_dir /path/to/model --output_dir /path/to/output --reduced_vocab_size 8192 --dataset_dir /path/to/dataset
"""

import argparse
import json
import os
import sys
import traceback

from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

from tensorrt_edgellm.vocab_reduction.vocab_reduction import reduce_vocab_size


def main() -> None:
    """
    Main function that parses command line arguments and reduces vocabulary.
    
    This function sets up argument parsing for the vocabulary reduction script,
    loads the model tokenizer and config, processes the dataset, and saves the
    token mapping and vocabulary information.
    """
    parser = argparse.ArgumentParser(
        description="Reduce vocabulary size based on token frequency analysis")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the input model directory containing tokenizer and config"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the token mapping and vocabulary info")
    parser.add_argument(
        "--reduced_vocab_size",
        type=int,
        required=True,
        help=
        "Target reduced vocabulary size (must be less than original vocab size)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=False,
        default="cnn_dailymail",
        help="Dataset name or path for token frequency analysis")
    parser.add_argument("--dataset_split",
                        type=str,
                        required=False,
                        default="train",
                        help="Dataset split to use (default: train)")
    parser.add_argument(
        "--max_samples",
        type=int,
        required=False,
        default=50000,
        help="Maximum number of samples to use from dataset (default: 50000)")

    args = parser.parse_args()

    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        print(f"Loading tokenizer and config from {args.model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        config = AutoConfig.from_pretrained(args.model_dir)

        print(f"Original vocabulary size: {config.vocab_size}")
        print(f"Target reduced vocabulary size: {args.reduced_vocab_size}")

        # Load dataset
        print(f"Loading dataset: {args.dataset_dir}")
        if args.dataset_dir == "cnn_dailymail":
            dataset = load_dataset(args.dataset_dir,
                                   "3.0.0",
                                   split=args.dataset_split)
            # CNN/DailyMail uses 'article' field
            dataset = dataset.select(range(min(args.max_samples,
                                               len(dataset))))
            # Rename 'article' to 'text' for consistency
            dataset = dataset.map(
                lambda x: {"text": x["article"]},
                remove_columns=["article", "highlights", "id"])
        else:
            # Try to load as a HuggingFace dataset or local path
            dataset = load_dataset(args.dataset_dir, split=args.dataset_split)
            dataset = dataset.select(range(min(args.max_samples,
                                               len(dataset))))

        print(f"Using {len(dataset)} samples for vocabulary analysis")

        # Reduce vocabulary
        print("Analyzing token frequencies and reducing vocabulary...")
        token_map = reduce_vocab_size(
            tokenizer=tokenizer,
            config=config,
            dataset=dataset,
            reduced_vocab_size=args.reduced_vocab_size)

        # Get actual reduced vocabulary size from token_map
        actual_reduced_vocab_size = len(token_map)

        # Save token map as safetensors
        token_map_path = os.path.join(args.output_dir, "token_map.safetensors")
        print(f"Saving token map to {token_map_path}...")
        save_file({"token_map": token_map}, str(token_map_path))

        # Save vocabulary info as JSON
        vocab_info = {
            "vocab_size": config.vocab_size,
            "reduced_vocab_size": actual_reduced_vocab_size
        }
        vocab_info_path = os.path.join(args.output_dir, "reduced_vocab.json")
        print(f"Saving vocabulary info to {vocab_info_path}...")
        with open(vocab_info_path, "w") as f:
            json.dump(vocab_info, f, indent=2)

        print("Vocabulary reduction completed successfully!")
        print(f"Output files saved to: {args.output_dir}")
        print(
            f"  - token_map.safetensors: Token mapping tensor [{actual_reduced_vocab_size}]"
        )
        print(f"  - reduced_vocab.json: Vocabulary size information")

    except Exception as e:
        print(f"Error during vocabulary reduction: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
