# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Qwen3-Omni Thinker text-MoE NVFP4 quantization driver.

ModelOpt only quantizes ``nn.Linear`` modules. The HF
``Qwen3OmniMoeThinkerTextExperts`` block stores all 128 experts as two
fused ``nn.Parameter`` tensors (``gate_up_proj`` ``[E, 2*I, H]``,
``down_proj`` ``[E, H, I]``) consumed via ``F.linear`` — so out of the
box ModelOpt would skip the experts entirely. This driver:

1. Patches every ``Qwen3OmniMoeThinkerTextExperts`` to expose per-expert
   ``nn.Linear`` ``gate_proj`` / ``up_proj`` / ``down_proj`` (with the
   weights copied from the fused tensors). Forward semantics preserved.
2. Runs ``mtq.quantize`` with a NVFP4 recipe scoped to the Thinker text
   path; audio_tower / visual / talker / code2wav are disabled.
3. Calibrates by forwarding text-only inputs through ``model.thinker``
   (audio/vision encoders stay quiescent).
4. Saves only the Thinker text subgraph as a standalone HF checkpoint
   (``model.layers.*``, ``lm_head.*``, ``embed_tokens.*``) with
   ``model_type="qwen3_omni_moe_text"``, consumable by
   :class:`llm_loader.Qwen3MoeNvfp4CutedslCausalLM`.

This Thinker-only path is an internal helper (no CLI subcommand); the
Thinker NVFP4 checkpoint is normally produced by
``tensorrt-edgellm-quantize qwen3-omni`` as part of the joint
Thinker+Talker pass.
"""

import json
import os
import time
from typing import Optional

import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.utils import is_quantized
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from .quantization_configs import build_quant_config

# ---------------------------------------------------------------------------
# Per-expert nn.Linear patching
# ---------------------------------------------------------------------------


class Qwen3OmniThinkerExpertsLinear(nn.Module):
    """Drop-in replacement for ``Qwen3OmniMoeThinkerTextExperts``.

    Stores each expert as three ``nn.Linear`` modules (``gate_proj`` /
    ``up_proj`` / ``down_proj``) so ModelOpt can quantize them.
    Forward signature matches the HF original: ``(hidden_states,
    top_k_index, top_k_weights) -> final_hidden_states``.
    """

    def __init__(self, num_experts: int, hidden_dim: int,
                 intermediate_dim: int, act_fn: nn.Module, dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = act_fn
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = nn.Module()
            expert.gate_proj = nn.Linear(hidden_dim,
                                         intermediate_dim,
                                         bias=False,
                                         dtype=dtype,
                                         device=device)
            expert.up_proj = nn.Linear(hidden_dim,
                                       intermediate_dim,
                                       bias=False,
                                       dtype=dtype,
                                       device=device)
            expert.down_proj = nn.Linear(intermediate_dim,
                                         hidden_dim,
                                         bias=False,
                                         dtype=dtype,
                                         device=device)
            self.experts.append(expert)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        # Mirrors HF Qwen3OmniMoeThinkerTextExperts.forward: dispatch tokens
        # to hit experts only.
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)),
                                       0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = int(expert_idx.item())
            if expert_idx >= self.num_experts:
                continue
            expert = self.experts[expert_idx]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate = expert.gate_proj(current_state)
            up = expert.up_proj(current_state)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = expert.down_proj(current_hidden_states)
            current_hidden_states = current_hidden_states * top_k_weights[
                token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx,
                current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


def _patch_thinker_experts(model: nn.Module) -> int:
    """Replace every fused-Parameter Qwen3OmniMoeThinkerTextExperts with a
    per-expert nn.Linear equivalent. Weights are copied from the fused
    tensors. Returns the number of MoE blocks patched.
    """
    # Lazy import to avoid hard transformers dependency at module-load time.
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerTextExperts, Qwen3OmniMoeThinkerTextSparseMoeBlock)

    patched = 0
    for moe_block in model.modules():
        if not isinstance(moe_block, Qwen3OmniMoeThinkerTextSparseMoeBlock):
            continue
        old: Qwen3OmniMoeThinkerTextExperts = moe_block.experts
        if isinstance(old, Qwen3OmniThinkerExpertsLinear):
            continue  # already patched
        E = old.num_experts
        H = old.hidden_dim
        I = old.intermediate_dim
        device = old.gate_up_proj.device
        dtype = old.gate_up_proj.dtype
        new = Qwen3OmniThinkerExpertsLinear(E, H, I, old.act_fn, dtype, device)
        with torch.no_grad():
            # gate_up_proj: [E, 2*I, H]; HF chunk(2, dim=-1) on gate_up = (gate, up).
            # F.linear(x, W) computes x @ W.T -> output [..., 2*I]; chunk along dim=-1
            # gives gate=output[..., :I], up=output[..., I:]. So rows 0..I-1 of
            # gate_up_proj[e] are gate, rows I..2I-1 are up.
            gu = old.gate_up_proj  # [E, 2*I, H]
            dn = old.down_proj  # [E, H, I]
            for e in range(E):
                exp = new.experts[e]
                exp.gate_proj.weight.copy_(gu[e, :I, :])
                exp.up_proj.weight.copy_(gu[e, I:2 * I, :])
                exp.down_proj.weight.copy_(dn[e])
        moe_block.experts = new.to(device=device, dtype=dtype)
        patched += 1
    return patched


# ---------------------------------------------------------------------------
# NVFP4 quantization config — Thinker text only
# ---------------------------------------------------------------------------


def _build_thinker_text_only_quant_cfg(
        lm_head_quantization: Optional[str],
        kv_cache_quantization: Optional[str]) -> dict:
    """Build a NVFP4 quant_cfg that only targets the Thinker text path.

    Disables: visual encoder, audio encoder, talker, code2wav, plus all
    routers and all non-text shared-expert gates.
    """
    cfg = build_quant_config("nvfp4", lm_head_quantization,
                             kv_cache_quantization)
    disable_globs = [
        "*visual.*",
        "*audio_tower.*",
        "*talker.*",
        "*code2wav.*",
        # Thinker-internal: keep router weight unquantized.
        "*thinker*mlp.gate.*",
    ]
    for g in disable_globs:
        cfg["quant_cfg"][g] = {"enable": False}
    return cfg


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _text_calib_loader(tokenizer, dataset_name: str, batch_size: int,
                       num_samples: int, max_length: int):
    """Tokenized DataLoader of input_ids for text-only calibration."""
    if "cnn_dailymail" in dataset_name:
        ds = load_dataset(dataset_name, name="3.0.0", split="train")
        texts = ds["article"][:num_samples]
    elif os.path.isdir(dataset_name):
        ds = load_dataset(dataset_name, split="train")
        texts = ds["text"][:num_samples]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    enc = tokenizer(texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length)
    return DataLoader(enc["input_ids"], batch_size=batch_size, shuffle=False)


def _calib_thinker_text(model, dataloader):
    """Forward text inputs through ``model.thinker`` (text-only path).

    Audio/vision encoders stay idle: with no audio/image inputs the
    Thinker treats the input as plain text, so all 128 MoE experts in
    every layer see realistic activation distributions.
    """
    device = next(model.thinker.parameters()).device
    for input_ids in tqdm(dataloader, desc="Calibrating Thinker (text)"):
        input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)
        model.thinker(input_ids=input_ids, attention_mask=attention_mask)


# ---------------------------------------------------------------------------
# Standalone Thinker-text checkpoint extraction
# ---------------------------------------------------------------------------

# Token-id fields that live at ``thinker_config`` top-level (NOT in
# ``thinker_config.text_config``) and that the downstream multimodal runtime /
# chat-template builder need in order to insert ``<|audio_pad|>`` /
# ``<|image_pad|>`` / ``<|video_pad|>`` placeholders into prompts.  Dropping
# them produces a "looks like text-only LLM" config that silently breaks
# multimodal OmniBench evaluation (issue: empty content_types map at runtime).
_THINKER_MULTIMODAL_TOKEN_FIELDS = (
    "audio_token_id",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
    "audio_start_token_id",
    "audio_end_token_id",
    "user_token_id",
    "position_id_per_seconds",
    "seconds_per_chunk",
)


def _write_standalone_thinker_text_config(thinker_text_dict: dict,
                                          output_dir: str,
                                          thinker_top_dict: dict = None,
                                          root_top_dict: dict = None) -> None:
    """Drop a standalone ``config.json`` with ``model_type=qwen3_omni_moe_text``.

    Strips multimodal sub-configs (audio_config, vision_config, ...) so
    :class:`ModelConfig` parses just the LLM portion, but PROMOTES the
    multimodal token-id fields from ``thinker_config`` top-level into the
    standalone config so the runtime tokenizer / chat-template builder can
    still emit ``<|audio_pad|>`` / ``<|image_pad|>`` / ``<|video_pad|>`` for
    multimodal benchmarks (OmniBench etc.).

    Additionally promotes ``accept_hidden_layer`` from the root
    ``talker_config`` into the standalone Thinker config so the llm_loader
    ONNX export can emit the correct hidden-states layer for the Talker
    consumes them.  HF stores this value under ``root.talker_config.accept_hidden_layer``;
    once the Thinker is detached into its own checkpoint that reference is
    gone, so we copy the integer up here.
    """
    cfg = dict(thinker_text_dict)
    cfg["model_type"] = "qwen3_omni_moe_text"
    cfg["architectures"] = ["Qwen3OmniMoeThinkerCausalLM"]
    if thinker_top_dict is not None:
        for k in _THINKER_MULTIMODAL_TOKEN_FIELDS:
            if k in thinker_top_dict and k not in cfg:
                cfg[k] = thinker_top_dict[k]
    if root_top_dict is not None and "accept_hidden_layer" not in cfg:
        talker_cfg = root_top_dict.get("talker_config") or {}
        ahl = talker_cfg.get("accept_hidden_layer")
        if ahl is None:
            ahl = root_top_dict.get("accept_hidden_layer")
        if ahl is not None:
            cfg["accept_hidden_layer"] = int(ahl)
    out_path = os.path.join(output_dir, "config.json")
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)


def _extract_and_save_thinker_text(model_dir: str, full_export_dir: str,
                                   output_dir: str) -> None:
    """Filter a full-multimodal NVFP4 export down to the Thinker text subgraph.

    Reads every safetensors shard in ``full_export_dir``, keeps only keys
    starting with ``thinker.model.`` or ``thinker.lm_head.``, strips the
    ``thinker.`` prefix, and writes a single consolidated
    ``model.safetensors`` to ``output_dir``. Copies ``hf_quant_config.json``
    after filtering its ``exclude_modules`` / ``quant_method`` lists to
    match the new key namespace.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)

    # 1) Gather + rekey tensors.
    # Since we export model.thinker (not the full Qwen3-Omni wrapper), keys
    # are already prefixed with "model." (text-model) / "lm_head." or with
    # the audio/visual sub-encoders. Keep only the text-model + lm_head;
    # drop audio_tower/visual.
    SKIP_PREFIXES = ("audio_tower.", "visual.", "audio.", "vision.")
    keep_state: dict = {}
    for fname in sorted(os.listdir(full_export_dir)):
        if not fname.endswith(".safetensors"):
            continue
        fpath = os.path.join(full_export_dir, fname)
        with safe_open(fpath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("thinker.model."):
                    new_key = key[len("thinker."):]
                elif key.startswith("thinker.lm_head."):
                    new_key = key[len("thinker."):]
                elif key.startswith(SKIP_PREFIXES):
                    continue
                elif key.startswith("model.") or key.startswith("lm_head."):
                    new_key = key
                else:
                    continue
                keep_state[new_key] = f.get_tensor(key)
    if not keep_state:
        raise RuntimeError(f"No 'model.*' / 'lm_head.*' tensors found under "
                           f"{full_export_dir}")
    print(f"[extract] keeping {len(keep_state)} tensors from full export")
    save_file(keep_state, os.path.join(output_dir, "model.safetensors"))

    # 2) Standalone config.json (Thinker text only).
    full_cfg = AutoConfig.from_pretrained(model_dir,
                                          trust_remote_code=True).to_dict()
    thinker_cfg = full_cfg["thinker_config"]
    text_cfg = thinker_cfg["text_config"]
    _write_standalone_thinker_text_config(text_cfg,
                                          output_dir,
                                          thinker_top_dict=thinker_cfg,
                                          root_top_dict=full_cfg)

    # 3) hf_quant_config.json: copy + rewrite key prefixes if present.
    src_hf = os.path.join(full_export_dir, "hf_quant_config.json")
    if os.path.isfile(src_hf):
        with open(src_hf) as f:
            hfq = json.load(f)

        def _rewrite_pattern(p: str) -> Optional[str]:
            # Drop any pattern entirely outside the thinker text path.
            if any(skip in p for skip in ("audio_tower", "visual", "talker",
                                          "code2wav")):
                return None
            return p.replace("thinker.model.",
                             "model.").replace("thinker.lm_head.", "lm_head.")

        for list_key in ("exclude_modules", ):
            if list_key in hfq and isinstance(hfq[list_key], list):
                hfq[list_key] = [
                    p for p in (_rewrite_pattern(x) for x in hfq[list_key])
                    if p is not None
                ]
        with open(os.path.join(output_dir, "hf_quant_config.json"), "w") as f:
            json.dump(hfq, f, indent=2)

    # 4) chat_template.json. HF Qwen3-Omni keeps its Jinja chat template in
    #    a dedicated file (not in tokenizer_config.json), so
    #    ``tokenizer.save_pretrained`` would lose it and downstream
    #    ``apply_chat_template`` would fall back to the generic
    #    ``User:/Assistant:`` template. Copy the source file verbatim.
    src_tpl = os.path.join(model_dir, "chat_template.json")
    if os.path.isfile(src_tpl):
        import shutil
        shutil.copyfile(src_tpl, os.path.join(output_dir,
                                              "chat_template.json"))


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def quantize_qwen3_omni_thinker(
    model_dir: str,
    output_dir: str,
    lm_head_quantization: Optional[str] = None,
    kv_cache_quantization: Optional[str] = None,
    dtype: str = "fp16",
    device: str = "cuda",
    dataset: str = "cnn_dailymail",
    num_samples: int = 512,
    max_length: int = 512,
    keep_full_export: bool = False,
) -> str:
    """Quantize the Thinker text-MoE subgraph to NVFP4 and save standalone."""
    t0 = time.time()
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16

    # 1. Load multimodal model.
    from transformers import Qwen3OmniMoeForConditionalGeneration
    print(f"[load] {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch_dtype,
        trust_remote_code=True).to(device).eval()

    # 2. Patch Thinker MoE experts (fused gate_up_proj Parameter ->
    #    per-expert nn.Linear) so ModelOpt sees them.
    n_patched = _patch_thinker_experts(model)
    print(f"[patch] replaced {n_patched} Thinker MoE blocks with per-expert "
          f"nn.Linear")

    # 3. NVFP4 quantize.
    if not is_quantized(model):
        cfg = _build_thinker_text_only_quant_cfg(lm_head_quantization,
                                                 kv_cache_quantization)
        loader = _text_calib_loader(tokenizer,
                                    dataset,
                                    batch_size=1,
                                    num_samples=num_samples,
                                    max_length=max_length)
        mtq.quantize(model.thinker,
                     cfg,
                     forward_loop=lambda m: _calib_thinker_text(model, loader))
        mtq.print_quant_summary(model.thinker)
    print(f"[quant] {time.time() - t0:.1f}s")

    # 4. Export full NVFP4 checkpoint to a staging directory.
    # Export only model.thinker so modelopt's resmooth dummy-forward sees a
    # plain CausalLM forward. Detach audio_tower/visual to avoid the
    # multimodal dummy-forward path (which requires audio/image tensors).
    full_dir = output_dir + "_full" if keep_full_export else (output_dir +
                                                              ".staging")
    os.makedirs(full_dir, exist_ok=True)

    thinker = model.thinker
    if thinker.config.architectures is None:
        thinker.config.architectures = ["Qwen3MoeForCausalLM"]
    saved_audio = getattr(thinker, "audio_tower", None)
    saved_visual = getattr(thinker, "visual", None)
    if saved_audio is not None:
        thinker.audio_tower = None
    if saved_visual is not None:
        thinker.visual = None
    saved_vision_cfg = getattr(thinker.config, "vision_config", None)
    saved_audio_cfg = getattr(thinker.config, "audio_config", None)
    if saved_vision_cfg is not None:
        del thinker.config.vision_config
    if saved_audio_cfg is not None:
        del thinker.config.audio_config

    # Some modelopt code paths inspect is_multimodal_model() which checks for
    # vision_config / audio_processor attrs; force a non-multimodal classification.
    from modelopt.torch.export import model_utils as _mu
    _orig_is_mm = _mu.is_multimodal_model
    _mu.is_multimodal_model = lambda *a, **kw: False
    try:
        with torch.inference_mode():
            export_hf_checkpoint(thinker, export_dir=full_dir)
    finally:
        _mu.is_multimodal_model = _orig_is_mm
        if saved_audio is not None:
            thinker.audio_tower = saved_audio
        if saved_visual is not None:
            thinker.visual = saved_visual
        if saved_vision_cfg is not None:
            thinker.config.vision_config = saved_vision_cfg
        if saved_audio_cfg is not None:
            thinker.config.audio_config = saved_audio_cfg
    print(f"[export-full] {full_dir}")

    # 5. Extract Thinker text subgraph as a standalone HF ckpt.
    _extract_and_save_thinker_text(model_dir, full_dir, output_dir)
    tokenizer.save_pretrained(output_dir)

    if not keep_full_export:
        import shutil
        shutil.rmtree(full_dir, ignore_errors=True)

    print(f"[done] {output_dir} (total {time.time() - t0:.1f}s)")
    return output_dir
