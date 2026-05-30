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

import os
from typing import Dict, Type

import torch.nn as nn

from .checkpoint.loader import load_weights
from .config import (ModelConfig, make_dflash_draft_config,
                     make_mtp_draft_config)

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
                        key_prefix: "str | None" = None,
                        eagle_base: bool = False,
                        reduced_vocab_dir: "str | None" = None,
                        nvfp4_moe_backend: "str | None" = None,
                        mtp_base: bool = False,
                        mtp_draft: bool = False,
                        tp_size: int = 1,
                        tp_rank: int = 0,
                        dflash_base: bool = False,
                        dflash_draft: bool = False,
                        dflash_draft_dir: "str | None" = None) -> nn.Module:
        """Construct and load a model from *model_dir*.

        Reads ``config.json`` via :class:`~config.ModelConfig`, looks up the
        model class in the registry (falling back to the built-in
        :class:`~models.default.modeling_default.CausalLM`), instantiates it,
        moves it to *device*, and loads safetensors weights.

        Args:
            model_dir:      Local HF checkpoint directory.
            device:         Target device (e.g. ``"cpu"``, ``"cuda:0"``).
            key_remap:      Optional callable ``(key: str) -> Optional[str]``.
                            Passed through to :func:`load_weights` for checkpoint
                            key remapping (e.g. TTS talker ``codec_embedding``
                            → ``embed_tokens``).
            key_prefix:     Explicit checkpoint key prefix to strip (e.g.
                            ``"talker."``).  Passed through to :func:`load_weights`.
            eagle_base:     When True, export as EAGLE3 base model with extra
                            tree-attention inputs and hidden_states output.
            reduced_vocab_dir:
                            Optional directory containing ``vocab_map.safetensors``.
            nvfp4_moe_backend:
                            Optional export-time override for Qwen3 NVFP4 MoE
                            backend selection (``"thor"`` or ``"geforce"``).
            mtp_base:       When True, export the standard Qwen3.5 text model as
                            the dense MTP base variant.
            mtp_draft:      When True, build the dedicated Qwen3.5 dense MTP
                            draft model from the base checkpoint config.
            tp_size:        Tensor-parallel world size.  When >1 the config
                            is reduced to per-rank shapes via
                            :meth:`ModelConfig.for_rank`, and weights
                            are sharded on assignment.  Default 1 = no TP.
            tp_rank:        This rank's index in [0, tp_size).
            dflash_base:    When True, export as DFlash base model.
            dflash_draft:   When True, build the DFlash draft model.
            dflash_draft_dir:
                            Path to the DFlash draft checkpoint directory.

        Returns:
            Loaded ``nn.Module`` in eval mode.
        """
        from .models.default.modeling_default import CausalLM

        config = ModelConfig.from_pretrained(model_dir)
        if nvfp4_moe_backend is not None:
            config.quant.nvfp4_moe_backend = nvfp4_moe_backend
            config.quant.__post_init__()
        if eagle_base:
            config.eagle_base = True
        if mtp_base or config.mtp_base:
            config.mtp_base = True
        if dflash_base:
            config.dflash_base = True
            # Read target_layer_ids from DFlash draft checkpoint if provided
            if not config.dflash_target_layer_ids and dflash_draft_dir:
                import json
                draft_cfg_path = os.path.join(dflash_draft_dir, "config.json")
                if os.path.isfile(draft_cfg_path):
                    with open(draft_cfg_path) as f:
                        draft_cfg = json.load(f)
                    dflash_cfg = draft_cfg.get("dflash_config", {})
                    config.dflash_target_layer_ids = dflash_cfg.get(
                        "target_layer_ids", [1, 8, 15, 22, 29])
                    config.dflash_block_size = dflash_cfg.get("block_size", 16)
                    config.dflash_mask_token_id = dflash_cfg.get(
                        "mask_token_id", 248070)
            if not config.dflash_target_layer_ids:
                config.dflash_target_layer_ids = [1, 8, 15, 22, 29]
        if tp_size > 1:
            config = config.for_rank(tp_rank, tp_size)

        variant = _resolve_model_variant(config,
                                         eagle_base=eagle_base,
                                         mtp_base=config.mtp_base,
                                         mtp_draft=mtp_draft,
                                         dflash_base=config.dflash_base,
                                         dflash_draft=dflash_draft)

        # EAGLE3 draft: auto-detect from draft_vocab_size
        if variant == "eagle3_draft":
            from .models.eagle3.modeling_eagle3_draft import Eagle3DraftModel
            model_class = Eagle3DraftModel
            # Set up key remapping: midlayer -> layers.0, skip t2d
            if key_remap is None:
                key_remap = _eagle3_key_remap
        elif variant == "mtp_draft":
            # TODO: support other model types
            if config.model_type != "qwen3_5_text":
                raise NotImplementedError(
                    "MTP draft is only supported for qwen3_5_text checkpoints."
                )
            from .models.qwen3_5 import Qwen3_5MtpDraftModel
            tie_word_embeddings = config.tie_word_embeddings
            config = make_mtp_draft_config(config)
            model_class = Qwen3_5MtpDraftModel
            if key_remap is None:
                key_remap = lambda key: _mtp_key_remap(
                    key, tie_word_embeddings=tie_word_embeddings)
        elif variant == "dflash_draft":
            if dflash_draft_dir is None:
                raise ValueError(
                    "dflash_draft requires dflash_draft_dir to be set.")
            from .models.dflash.modeling_dflash_draft import DFlashDraftModel
            base_model_dir = model_dir
            base_tie_word_embeddings = config.tie_word_embeddings
            config = make_dflash_draft_config(dflash_draft_dir)
            model_class = DFlashDraftModel
            model_dir = dflash_draft_dir
            if key_remap is None:
                key_remap = _dflash_key_remap
        else:
            if variant == "mtp_base" and config.model_type != "qwen3_5_text":
                raise NotImplementedError(
                    "Qwen3.5 dense MTP base is only supported for qwen3_5_text checkpoints."
                )
            # DFlash base is supported for both Qwen3.5 hybrid (qwen3_5_text) and
            # dense Qwen3 (default CausalLM). Dense models use the Transformer's
            # dflash_target_layer_ids parameter to collect target-layer hidden states.
            model_class = _MODEL_REGISTRY.get(config.model_type, CausalLM)

        model = model_class(config)
        model.to(device)

        pre_repack_hook = None
        if reduced_vocab_dir is not None:
            from .vocab_reduction.onnx_export import (
                apply_reduced_vocab, load_reduced_vocab_map,
                should_apply_reduced_vocab_before_repacking)
            vocab_map = load_reduced_vocab_map(reduced_vocab_dir,
                                               vocab_size=config.vocab_size,
                                               device=device)
            if should_apply_reduced_vocab_before_repacking(model):

                def _apply_pre_repack_reduced_vocab(loaded_model: nn.Module):
                    apply_reduced_vocab(loaded_model, vocab_map)
                    loaded_model._reduced_vocab_dir = reduced_vocab_dir

                pre_repack_hook = _apply_pre_repack_reduced_vocab

        load_weights(model,
                     model_dir,
                     device=device,
                     key_remap=key_remap,
                     key_prefix=key_prefix,
                     pre_repack_hook=pre_repack_hook,
                     mapping=config.mapping)
        if variant == "dflash_draft":
            _load_dflash_lm_head(model,
                                 base_model_dir,
                                 device,
                                 tie_word_embeddings=base_tie_word_embeddings)
        if reduced_vocab_dir is not None and pre_repack_hook is None:
            from .vocab_reduction.onnx_export import \
                apply_reduced_vocab_from_dir
            apply_reduced_vocab_from_dir(model, reduced_vocab_dir)

        # Post-load optimisation: fuse GDN input projections for Qwen3.5 / Qwen3.5-MoE.
        if (config.model_type in ("qwen3_5_text", "qwen3_5_moe_text")
                and not mtp_draft):
            from .models.qwen3_5 import fuse_gdn_input_projections
            fuse_gdn_input_projections(model)

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


# ---------------------------------------------------------------------------
# EAGLE3 helpers
# ---------------------------------------------------------------------------


def _resolve_model_variant(config: ModelConfig,
                           *,
                           eagle_base: bool,
                           mtp_base: bool,
                           mtp_draft: bool,
                           dflash_base: bool = False,
                           dflash_draft: bool = False) -> str:
    """Resolve the requested model variant while keeping EAGLE3 behavior intact."""
    if eagle_base and mtp_base:
        raise ValueError("eagle_base and mtp_base cannot both be enabled.")
    if eagle_base and mtp_draft:
        raise ValueError("eagle_base and mtp_draft cannot both be enabled.")
    if mtp_base and mtp_draft:
        raise ValueError("mtp_base and mtp_draft cannot both be enabled.")
    if dflash_base and dflash_draft:
        raise ValueError(
            "dflash_base and dflash_draft cannot both be enabled.")
    if dflash_base and (eagle_base or mtp_base or mtp_draft):
        raise ValueError(
            "dflash_base cannot be combined with eagle/mtp variants.")
    if dflash_draft and (eagle_base or mtp_base or mtp_draft):
        raise ValueError(
            "dflash_draft cannot be combined with eagle/mtp variants.")
    if config.is_eagle3_draft:
        if mtp_base or mtp_draft:
            raise ValueError(
                "EAGLE3 draft checkpoints cannot be loaded as Qwen3.5 MTP variants."
            )
        return "eagle3_draft"
    if dflash_draft:
        return "dflash_draft"
    if dflash_base:
        return "dflash_base"
    if mtp_draft:
        return "mtp_draft"
    if mtp_base:
        return "mtp_base"
    if eagle_base:
        return "eagle_base"
    return "llm"


def _eagle3_key_remap(key: str) -> "str | None":
    """Remap EAGLE3 draft checkpoint keys.

    Handles all known EAGLE3 draft checkpoint variations:
    - ``t2d`` keys are skipped (but ``d2t`` is kept).
    - ``target_model.*`` keys are skipped (multi-target training artifact).
    - ``midlayer.*`` -> ``layers.0.*``
    - ``qkv_proj.{q,k,v}_proj`` -> ``{q,k,v}_proj`` (flatten old pipeline
      ``EdgeLLMAttention`` wrapper nesting, used by quantized checkpoints).
    - ``._pre_quant_scale`` -> ``.pre_quant_scale`` (modelopt internal naming;
      normally stripped by ``postprocess_state_dict()`` but not by per-module
      export via ``_export_quantized_weight()``).
    """
    if "t2d" in key and "d2t" not in key:
        return None  # skip t2d
    if key.startswith("target_model."):
        return None  # skip multi-target training artifact
    key = key.replace("midlayer.", "layers.0.")
    key = key.replace("qkv_proj.q_proj", "q_proj")
    key = key.replace("qkv_proj.k_proj", "k_proj")
    key = key.replace("qkv_proj.v_proj", "v_proj")
    key = key.replace("._pre_quant_scale", ".pre_quant_scale")
    return key


def _load_dflash_lm_head(model: nn.Module,
                         base_model_dir: str,
                         device: str,
                         *,
                         tie_word_embeddings: bool = True) -> None:
    """Load the DFlash draft lm_head from the base model checkpoint.

    Deterministic source selection (matching MTP pattern):
      1. Explicit ``lm_head.weight`` from the base checkpoint.
      2. Embedding fallback *only* when ``tie_word_embeddings=True``.
      3. Otherwise fail loudly — untied models must not use embeddings.

    Quantized draft checkpoints own their packed lm_head buffers.  The generic
    checkpoint loader has already copied them before this helper runs, so only
    FP16 draft heads are overwritten from the original base checkpoint.
    """
    import logging
    import os

    import torch
    from safetensors import safe_open

    from .models.linear import FP16Linear, is_nvfp4_linear

    logger = logging.getLogger(__name__)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        logger.warning("DFlash draft model has no lm_head; skipping.")
        return

    if is_nvfp4_linear(lm_head):
        required = ("weight", "weight_scale", "weight_scale_2", "input_scale")
        missing = [name for name in required if not hasattr(lm_head, name)]
        if missing:
            raise ValueError(
                "DFlash NVFP4 lm_head is missing quantized buffers: "
                f"{missing}")
        logger.info(
            "DFlash lm_head source: quantized draft checkpoint buffers")
        return

    from .checkpoint.loader import _build_shard_map
    shard_map = _build_shard_map(base_model_dir)

    # --- Determine source key with strict priority ---
    lm_head_candidates = [
        "lm_head.weight",
        "model.lm_head.weight",
        "language_model.lm_head.weight",
        "model.language_model.lm_head.weight",
    ]
    embed_candidates = [
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
    ]

    # Priority 1: explicit lm_head.weight from base checkpoint
    source_key = None
    source_type = None
    for cand in lm_head_candidates:
        if cand in shard_map:
            source_key = cand
            source_type = "lm_head"
            break

    # Priority 2: embedding fallback only if tie_word_embeddings
    if source_key is None:
        if tie_word_embeddings:
            for cand in embed_candidates:
                if cand in shard_map:
                    source_key = cand
                    source_type = "tied_embedding"
                    break
        else:
            raise ValueError(
                "DFlash lm_head: base model at %s has "
                "tie_word_embeddings=False but no lm_head.weight found. "
                "Cannot safely fall back to embed_tokens." % base_model_dir)

    if source_key is None:
        raise ValueError(
            "Cannot find lm_head.weight or embed_tokens.weight in "
            "base model at %s." % base_model_dir)

    shard_path = shard_map[source_key]
    if source_type == "tied_embedding":
        logger.info("DFlash lm_head source: %s (tied fallback) from %s",
                    source_key, os.path.basename(shard_path))
    else:
        logger.info("DFlash lm_head source: %s from %s", source_key,
                    os.path.basename(shard_path))

    with safe_open(shard_path, framework="pt", device=device) as f:
        source_weight = f.get_tensor(source_key)

    # --- Copy weight into model's lm_head ---
    if isinstance(lm_head, FP16Linear):
        if source_weight.shape != lm_head.weight.shape:
            raise ValueError(
                f"DFlash lm_head shape mismatch: source={source_weight.shape} "
                f"vs lm_head={lm_head.weight.shape}")
        with torch.no_grad():
            lm_head.weight.copy_(source_weight.to(lm_head.weight.dtype))
    else:
        # Generic fallback
        if source_weight.shape != lm_head.weight.shape:
            raise ValueError(
                f"DFlash lm_head shape mismatch: source={source_weight.shape} "
                f"vs lm_head={lm_head.weight.shape}")
        with torch.no_grad():
            lm_head.weight.copy_(source_weight.to(lm_head.weight.dtype))


def _dflash_key_remap(key: str) -> "str | None":
    """Remap DFlash draft checkpoint keys."""
    if "rotary_emb" in key:
        return None
    return key


def _mtp_key_remap(key: str, *, tie_word_embeddings: bool) -> "str | None":
    """Remap MTP checkpoint keys for the draft model.

    The embedding table is only a valid LM-head fallback when the source
    checkpoint ties word embeddings.
    """
    if key.startswith("mtp."):
        return key[len("mtp."):]
    if key == "lm_head.weight":
        return key
    if tie_word_embeddings and key in (
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight"):
        return "lm_head.weight"
    return None
