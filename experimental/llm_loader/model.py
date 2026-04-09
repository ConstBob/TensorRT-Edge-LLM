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
Auto-dispatch model factory and parameter utilities.

``AutoModel.from_pretrained`` reads a checkpoint config, picks the right model
class, constructs it, and loads weights — the primary entry point for callers.

Custom model classes can be registered via :func:`register_model` to override
the default :class:`~models.default.modeling_default.CausalLM` for a given
``model_type`` string.
"""

from typing import Dict, Type

import torch.nn as nn

from .checkpoint.loader import load_weights
from .config import ModelConfig

__all__ = ["AutoModel", "register_model", "dtype_summary", "param_count"]

_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(model_type: str, model_class: Type[nn.Module]) -> None:
    """Register *model_class* as the handler for *model_type*.

    When :meth:`AutoModel.from_pretrained` encounters a checkpoint whose
    ``model_type`` field equals *model_type*, it instantiates *model_class*
    instead of the built-in :class:`~models.default.modeling_default.CausalLM`.

    Args:
        model_type:  Value of ``model_type`` in the checkpoint ``config.json``.
        model_class: ``nn.Module`` subclass; must accept a single
                     :class:`~config.ModelConfig` as its constructor argument.
    """
    _MODEL_REGISTRY[model_type] = model_class


class AutoModel:
    """HuggingFace-style factory that dispatches on ``model_type``."""

    @classmethod
    def from_pretrained(cls,
                        model_dir: str,
                        device: str = "cpu",
                        key_remap=None,
                        key_prefix: "str | None" = None) -> nn.Module:
        """Construct and load a model from *model_dir*.

        Reads ``config.json`` via :class:`~config.ModelConfig`, looks up the
        model class in the registry (falling back to the built-in
        :class:`~models.default.modeling_default.CausalLM`), instantiates it,
        moves it to *device*, and loads safetensors weights.

        Args:
            model_dir:  Local HF checkpoint directory.
            device:     Target device (e.g. ``"cpu"``, ``"cuda:0"``).
            key_remap:  Optional callable ``(key: str) -> Optional[str]``.
                        Passed through to :func:`load_weights` for checkpoint
                        key remapping (e.g. TTS talker ``codec_embedding``
                        → ``embed_tokens``).
            key_prefix: Explicit checkpoint key prefix to strip (e.g.
                        ``"talker."``).  Passed through to :func:`load_weights`.

        Returns:
            Loaded ``nn.Module`` in eval mode.
        """
        from .models.default.modeling_default import CausalLM

        config = ModelConfig.from_pretrained(model_dir)
        model_class = _MODEL_REGISTRY.get(config.model_type, CausalLM)
        model = model_class(config)
        model.to(device)
        load_weights(model,
                     model_dir,
                     device=device,
                     key_remap=key_remap,
                     key_prefix=key_prefix)
        return model


def param_count(model: nn.Module) -> int:
    """Return total parameter element count (trainable and frozen)."""
    return sum(p.numel() for p in model.parameters())


def dtype_summary(model: nn.Module) -> Dict[str, int]:
    """Map dtype name -> number of parameter elements."""
    out: Dict[str, int] = {}
    for p in model.parameters():
        name = str(p.dtype).replace("torch.", "")
        out[name] = out.get(name, 0) + p.numel()
    return dict(sorted(out.items(), key=lambda x: -x[1]))
