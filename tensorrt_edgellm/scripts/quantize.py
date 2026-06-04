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
"""CLI for standalone quantization.

Usage::

    # LLM quantization
    tensorrt-edgellm-quantize llm \\
        --model_dir /path/to/model \\
        --output_dir /path/to/output \\
        --quantization nvfp4 \\
        --lm_head_quantization nvfp4

    # Speculative draft quantization (Eagle3 or DFlash, auto-detected)
    tensorrt-edgellm-quantize draft \\
        --base_model_dir /path/to/base \\
        --draft_model_dir /path/to/draft \\
        --output_dir /path/to/output \\
        --quantization fp8
"""

import argparse
import json
import os


def _add_common_args(parser):
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--quantization",
        default=None,
        choices=["fp8", "int4_awq", "nvfp4", "mxfp8", "int8_sq"],
        help="Backbone quantization method.")
    parser.add_argument("--lm_head_quantization",
                        default=None,
                        choices=["fp8", "int4_awq", "nvfp4", "mxfp8"],
                        help="LM-head quantization method.")
    parser.add_argument(
        "--visual_quantization",
        default=None,
        choices=["fp8"],
        help=("Quantize the visual tower (visual / vision_tower / "
              "multi_modal_projector). Only fp8 is exposed today; other "
              "precisions are deferred until per-recipe accuracy is "
              "validated. When unset the visual tower stays at fp16 "),
    )
    parser.add_argument(
        "--audio_quantization",
        default=None,
        choices=["fp8"],
        help=("Quantize the audio tower (audio_tower / audio_embed). Only "
              "fp8 is exposed today; lower-bit recipes are deferred until "
              "WER-validated. For Qwen3-ASR this triggers an ASR-shaped "
              "multimodal calibration loop (LibriSpeech audio + transcript "
              "pairs streamed through audio_tower + text decoder). When "
              "unset the audio tower stays at fp16."),
    )
    parser.add_argument("--kv_cache_quantization",
                        default=None,
                        choices=["fp8"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", default="cnn_dailymail")
    parser.add_argument("--num_samples", type=int, default=512)


def main():
    parser = argparse.ArgumentParser(
        description="TensorRT Edge-LLM standalone quantization")
    sub = parser.add_subparsers(dest="command", required=True)

    llm_parser = sub.add_parser("llm", help="Quantize an LLM")
    llm_parser.add_argument("--model_dir", required=True)
    _add_common_args(llm_parser)

    draft_parser = sub.add_parser(
        "draft", help="Quantize an Eagle3 or DFlash draft model")
    draft_parser.add_argument("--base_model_dir", required=True)
    draft_parser.add_argument("--draft_model_dir", required=True)
    _add_common_args(draft_parser)

    args = parser.parse_args()

    if args.command == "llm":
        from ..quantization.quantize import quantize_and_export
        quantize_and_export(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            quantization=args.quantization,
            lm_head_quantization=args.lm_head_quantization,
            visual_quantization=args.visual_quantization,
            audio_quantization=args.audio_quantization,
            kv_cache_quantization=args.kv_cache_quantization,
            dtype=args.dtype,
            device=args.device,
            dataset=args.dataset,
            num_samples=args.num_samples,
        )
    elif args.command == "draft":
        if _is_dflash_draft(args.draft_model_dir):
            _validate_dflash_quant_args(parser, args)
            from ..quantization.models.dflash_draft import \
                quantize_and_export_dflash_draft
            quantize_and_export_dflash_draft(
                base_model_dir=args.base_model_dir,
                draft_model_dir=args.draft_model_dir,
                output_dir=args.output_dir,
                quantization=args.quantization or "nvfp4",
                lm_head_quantization=args.lm_head_quantization,
                kv_cache_quantization=args.kv_cache_quantization,
                dtype=args.dtype,
                device=args.device,
                dataset=args.dataset,
                num_samples=args.num_samples,
            )
        else:
            from ..quantization.models.eagle3_draft import \
                quantize_and_export_draft
            quantize_and_export_draft(
                base_model_dir=args.base_model_dir,
                draft_model_dir=args.draft_model_dir,
                output_dir=args.output_dir,
                quantization=args.quantization or "fp8",
                lm_head_quantization=args.lm_head_quantization,
                kv_cache_quantization=args.kv_cache_quantization,
                dtype=args.dtype,
                device=args.device,
                dataset=args.dataset,
                num_samples=args.num_samples,
            )


def _is_dflash_draft(draft_model_dir: str) -> bool:
    cfg_path = os.path.join(draft_model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    with open(cfg_path, encoding="utf-8") as f:
        return bool(json.load(f).get("dflash_config"))


def _validate_dflash_quant_args(parser, args) -> None:
    if args.kv_cache_quantization is not None:
        parser.error(
            "DFlash draft KV-cache quantization is not validated yet.")


if __name__ == "__main__":
    main()
