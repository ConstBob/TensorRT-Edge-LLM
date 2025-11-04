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
Vocabulary reduction utilities for TensorRT Edge-LLM.

This module provides functionality to reduce vocabulary based on token frequency.
"""

from collections import Counter

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer


def reduce_vocab_size(tokenizer: AutoTokenizer, config: AutoConfig,
                      dataset: Dataset,
                      reduced_vocab_size: int) -> torch.Tensor:
    """
    Truncate vocabulary based on token frequency in the dataset.
    
    This function analyzes token frequency in the provided dataset and creates
    a reduced vocabulary containing the most frequent tokens plus the EOS token.
    
    Args:
        tokenizer: HuggingFace AutoTokenizer instance
        config: HuggingFace AutoConfig instance
        dataset: Dataset to analyze for token frequency
        reduced_vocab_size: Target vocabulary size (must be < config.vocab_size)
        
    Returns:
        token_map: torch.Tensor of shape (reduced_vocab_size,) mapping 
                   reduced token IDs to original token IDs (int32)
                   i.e., token_map[i] = original_token_id
            
    Raises:
        ValueError: If reduced_vocab_size >= config.vocab_size
    """
    # Validate input
    if reduced_vocab_size >= config.vocab_size:
        raise ValueError(
            f"reduced_vocab_size ({reduced_vocab_size}) must be less than "
            f"config.vocab_size ({config.vocab_size})")

    # Get EOS token ID
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.pad_token_id
        if eos_token_id is None:
            raise ValueError(
                "Tokenizer does not have an eos_token_id or pad_token_id")

    # Count token frequencies across the dataset (excluding special tokens)
    print(
        f"Analyzing token frequencies in dataset with {len(dataset)} samples..."
    )
    token_counter = Counter()

    for sample in tqdm(dataset, desc="Tokenizing and counting tokens"):
        if isinstance(sample, dict):
            text = sample.get(
                'text', sample.get('content', sample.get('sentence', '')))
        else:
            text = str(sample)

        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_counter.update(tokens)

    print(f"Found {len(token_counter)} unique tokens in dataset")

    # Handle case where dataset has fewer unique tokens than requested
    num_available_tokens = len(token_counter) + 1  # +1 for EOS token
    if num_available_tokens < reduced_vocab_size:
        print(
            f"WARNING: Dataset only contains {num_available_tokens} unique tokens "
            f"(including EOS), but {reduced_vocab_size} were requested. Using {num_available_tokens} as the reduced vocabulary size instead."
        )
        reduced_vocab_size = num_available_tokens

    # Get top (N-1) most frequent tokens and add EOS token
    top_tokens = [
        token_id
        for token_id, _ in token_counter.most_common(reduced_vocab_size - 1)
    ]

    # EOS tokens or any special tokens should not be in the top tokens.
    assert eos_token_id not in top_tokens, "EOS token should not be in the top tokens"
    top_tokens.append(eos_token_id)

    return torch.tensor(sorted(top_tokens), dtype=torch.int32)
