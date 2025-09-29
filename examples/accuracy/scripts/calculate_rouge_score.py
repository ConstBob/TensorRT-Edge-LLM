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

import argparse
import json

import evaluate


def calculate_rouge_score(predictions, references):
    """
    Compute Rouge score between predictions and references.
    Args:
        predictions: List of predictions.
        references: List of references.
    Returns:
        Rouge score. Format: {
            "rouge1": float,
            "rouge2": float,
            "rougeL": float,
            "rougeLsum": float
        }
    """
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references)


def main():
    """Main function to calculate Rouge score from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate Rouge score between two JSON files")
    parser.add_argument("--predictions_file",
                        type=str,
                        required=True,
                        help="Path to predictions JSON file")
    parser.add_argument("--references_file",
                        type=str,
                        required=True,
                        help="Path to references JSON file")

    args = parser.parse_args()

    # Load JSON files
    with open(args.predictions_file, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)

    with open(args.references_file, 'r', encoding='utf-8') as f:
        references_data = json.load(f)

    # Extract predictions and references
    predictions = []
    references = []

    for response in predictions_data["responses"]:
        predictions.append(response["output_text"])

    for message in references_data["messages"]:
        references.append(message["reference"])

    # Calculate and print Rouge score
    assert len(predictions) == len(
        references), "Predictions and references must have the same length"
    rouge_score_result = calculate_rouge_score(predictions, references)

    print("Rouge Score Results:")
    print(f"Rouge-1:  {rouge_score_result['rouge1']:.4f}")
    print(f"Rouge-2:  {rouge_score_result['rouge2']:.4f}")
    print(f"Rouge-L:  {rouge_score_result['rougeL']:.4f}")
    print(f"Rouge-Lsum: {rouge_score_result['rougeLsum']:.4f}")

    return rouge_score_result


if __name__ == "__main__":
    main()
