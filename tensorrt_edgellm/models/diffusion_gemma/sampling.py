# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Reference DiffusionGemma block sampler used by correctness tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EntropyBoundSamplerConfig:
    entropy_threshold: float = 0.005
    entropy_bound: float = 0.1
    stability_window: int = 2


def entropy_bound_accept_mask(
    logits: torch.Tensor,
    previous_tokens: torch.Tensor | None,
    stable_counts: torch.Tensor | None,
    config: EntropyBoundSamplerConfig,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return accepted tokens/mask for one deterministic denoise step.

    Args:
        logits: ``[B, C, V]`` full-canvas logits.
        previous_tokens: Previous argmax tokens ``[B, C]`` or ``None``.
        stable_counts: Previous same-token run lengths ``[B, C]`` or ``None``.
        config: Entropy/stability thresholds.
        temperature: Softmax temperature used for entropy and acceptance.

    Returns:
        ``(tokens, accept_mask, next_stable_counts, entropy)``.
    """
    if logits.ndim != 3:
        raise ValueError(
            "entropy_bound_accept_mask expects logits with shape [B, C, V].")
    scaled_logits = logits.float() / max(float(temperature), 1e-6)
    probs = torch.softmax(scaled_logits, dim=-1)
    tokens = torch.argmax(probs, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-20))).sum(dim=-1)

    if previous_tokens is None:
        previous_tokens = torch.full_like(tokens, -1)
    if stable_counts is None:
        stable_counts = torch.zeros_like(tokens, dtype=torch.int32)

    same_as_previous = tokens == previous_tokens
    next_stable_counts = torch.where(same_as_previous, stable_counts + 1,
                                     torch.ones_like(stable_counts))

    sorted_entropy, sorted_idx = torch.sort(entropy, dim=-1)
    cumsum_entropy = torch.cumsum(sorted_entropy, dim=-1)
    cummax_entropy = torch.cummax(sorted_entropy, dim=-1).values
    sorted_accept = ((cumsum_entropy - cummax_entropy)
                     <= float(config.entropy_bound))
    accept_mask = torch.zeros_like(sorted_accept, dtype=torch.bool)
    accept_mask.scatter_(1, sorted_idx, sorted_accept)
    return tokens, accept_mask, next_stable_counts, entropy


def soft_token_embeds(logits: torch.Tensor,
                      embedding_weight: torch.Tensor,
                      temperature: float = 1.0) -> torch.Tensor:
    """Project full logits back to embedding space for self-conditioning."""
    scaled_logits = logits.float() / max(float(temperature), 1e-6)
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.matmul(probs.to(embedding_weight.dtype), embedding_weight)
