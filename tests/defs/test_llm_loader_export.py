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
"""LLM Loader export test suite for pre-quantized models.

Uses the experimental llm_loader.export_all_cli to export pre-quantized
models (e.g. Nemotron 4B NVFP4, Phi-4) to ONNX format. The exported ONNX
is placed in the same ONNX directory structure used by the standard export
pipeline, so downstream engine build and inference tests can consume it.
"""

import os
import shutil
import tempfile

import pytest
from conftest import EnvironmentConfig
from pytest_helpers import run_command, timer_context

from .config import ModelType, TaskType, TestConfig


def test_llm_loader_export(test_param: str, test_logger,
                           env_config: EnvironmentConfig):
    """Export a pre-quantized model via llm_loader.export_all_cli."""

    config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                          TaskType.EXPORT, env_config)

    # Locate source model checkpoint
    torch_dir = config.get_torch_model_dir()
    if not os.path.exists(torch_dir):
        raise FileNotFoundError(f"Model checkpoint not found: {torch_dir}")

    # Determine the final ONNX output directory (same layout as standard export)
    llm_onnx_dir = config.get_llm_onnx_dir()
    os.makedirs(llm_onnx_dir, exist_ok=True)

    # Use a temporary directory for the raw export output
    tmp_dir = tempfile.mkdtemp(prefix="llm_loader_export_")

    try:
        # Build the export command
        # PYTHONPATH must include experimental/ so `python -m llm_loader.export_all_cli` works.
        # The CI job sets this via PYTHONPATH=$LLM_SDK_DIR/experimental:$PYTHONPATH.
        export_cmd = [
            "python3",
            "-m",
            "llm_loader.export_all_cli",
            torch_dir,
            tmp_dir,
            "--device",
            "cpu",
        ]

        with timer_context(f"Exporting {config.model_name} via llm_loader",
                           test_logger):
            result = run_command(export_cmd,
                                 timeout=600,
                                 remote_config=None,
                                 logger=test_logger)
            if not result['success']:
                pytest.fail(
                    f"llm_loader export failed: {result.get('error', 'Unknown error')}"
                )

        # Move the LLM ONNX output to the expected directory
        llm_output = os.path.join(tmp_dir, "llm")
        if not os.path.isdir(llm_output):
            pytest.fail(
                f"llm_loader did not produce llm/ output directory in {tmp_dir}"
            )

        # Copy all files from tmp_dir/llm/ into the final ONNX directory
        shutil.copytree(llm_output, llm_onnx_dir, dirs_exist_ok=True)

        # If the model also has a visual encoder output, move that too
        visual_output = os.path.join(tmp_dir, "visual")
        if os.path.isdir(visual_output):
            visual_onnx_dir = os.path.join(config.get_onnx_base_dir(),
                                           "visual-fp16")
            shutil.copytree(visual_output, visual_onnx_dir, dirs_exist_ok=True)

    finally:
        # Clean up temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Validate the exported ONNX model exists
    onnx_model = os.path.join(llm_onnx_dir, "model.onnx")
    if not os.path.exists(onnx_model):
        pytest.fail(f"LLM ONNX model not found after export: {onnx_model}")


def _run_llm_loader_export(cmd, timeout, test_logger, label):
    """Run an llm_loader export command and fail on error."""
    with timer_context(label, test_logger):
        result = run_command(cmd,
                             timeout=timeout,
                             remote_config=None,
                             logger=test_logger)
        if not result['success']:
            pytest.fail(
                f"{label} failed: {result.get('error', 'Unknown error')}")


def test_llm_loader_eagle_export(test_param: str, test_logger,
                                 env_config: EnvironmentConfig):
    """Export EAGLE base + draft models via llm_loader.export_all_cli.

    Uses pre-quantized unified checkpoints for both base and draft models
    so that no on-the-fly quantization is needed during CI.

    Two export_all_cli invocations:
      1. Base model with --eagle-base  -> llm-base ONNX (+ visual for VLMs)
      2. Draft model -> draft ONNX
    """

    config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                          TaskType.EXPORT, env_config)

    # Locate pre-quantized base model checkpoint
    base_torch_dir = config.get_torch_model_dir()
    if not os.path.exists(base_torch_dir):
        raise FileNotFoundError(
            f"Base model checkpoint not found: {base_torch_dir}")

    # Locate pre-quantized draft model checkpoint
    draft_torch_dir = config.get_draft_model_dir()
    if not os.path.exists(draft_torch_dir):
        raise FileNotFoundError(
            f"Draft model checkpoint not found: {draft_torch_dir}")

    # Strip quantization suffix from model_name so ONNX paths match downstream
    # build/inference tests.  Export test params include the quantization type
    # in the model name (e.g. "Qwen3-1.7B-NVFP4") to locate the pre-quantized
    # checkpoint, but downstream tests use the base model name ("Qwen3-1.7B").
    _QUANT_SUFFIXES = ("-NVFP4", "-FP8", "-FP8-KV", "-INT8-SQ", "-INT4-AWQ")
    for suffix in _QUANT_SUFFIXES:
        if config.model_name.endswith(suffix):
            config.model_name = config.model_name[:-len(suffix)]
            break

    # Output directories
    llm_onnx_dir = config.get_llm_onnx_dir()
    draft_onnx_dir = config.get_draft_onnx_dir()
    os.makedirs(llm_onnx_dir, exist_ok=True)
    os.makedirs(draft_onnx_dir, exist_ok=True)

    tmp_base = tempfile.mkdtemp(prefix="eagle_base_export_")
    tmp_draft = tempfile.mkdtemp(prefix="eagle_draft_export_")

    try:
        # --- Export base model with --eagle-base ---
        base_cmd = [
            "python3",
            "-m",
            "llm_loader.export_all_cli",
            base_torch_dir,
            tmp_base,
            "--eagle-base",
            "--device",
            "cpu",
        ]
        _run_llm_loader_export(
            base_cmd, 600, test_logger,
            f"Exporting EAGLE base {config.model_name} via llm_loader")

        # Copy base LLM ONNX
        base_llm_out = os.path.join(tmp_base, "llm")
        if not os.path.isdir(base_llm_out):
            pytest.fail(f"Base export did not produce llm/ in {tmp_base}")
        shutil.copytree(base_llm_out, llm_onnx_dir, dirs_exist_ok=True)

        # Copy visual encoder if present (VLM models)
        base_vis_out = os.path.join(tmp_base, "visual")
        if os.path.isdir(base_vis_out):
            visual_onnx_dir = os.path.join(config.get_onnx_base_dir(),
                                           "visual-fp16")
            shutil.copytree(base_vis_out, visual_onnx_dir, dirs_exist_ok=True)

        # --- Export draft model ---
        draft_cmd = [
            "python3",
            "-m",
            "llm_loader.export_all_cli",
            draft_torch_dir,
            tmp_draft,
            "--device",
            "cpu",
        ]
        _run_llm_loader_export(
            draft_cmd, 600, test_logger,
            f"Exporting EAGLE draft {config.draft_model_id} via llm_loader")

        # Copy draft ONNX
        draft_llm_out = os.path.join(tmp_draft, "llm")
        if not os.path.isdir(draft_llm_out):
            pytest.fail(f"Draft export did not produce llm/ in {tmp_draft}")
        shutil.copytree(draft_llm_out, draft_onnx_dir, dirs_exist_ok=True)

    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)
        shutil.rmtree(tmp_draft, ignore_errors=True)

    # Validate outputs
    base_onnx = os.path.join(llm_onnx_dir, "model.onnx")
    if not os.path.exists(base_onnx):
        pytest.fail(f"Base ONNX model not found: {base_onnx}")

    draft_onnx = os.path.join(draft_onnx_dir, "model.onnx")
    if not os.path.exists(draft_onnx):
        pytest.fail(f"Draft ONNX model not found: {draft_onnx}")
