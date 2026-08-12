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
"""Build an engine and run one prompt using an installed EdgeLLM wheel."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

import experimental.builder
import tensorrt_edgellm
import tensorrt_edgellm.runtime as runtime_api
from tensorrt_edgellm._native.load import resolve_payload


def _arguments(values: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("engine_dir", type=Path)
    parser.add_argument("--prompt", default="Please introduce NVIDIA.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--expected-variant")
    parser.add_argument("--result", type=Path)
    return parser.parse_args(values)


def _require_installed_package() -> Path:
    package = Path(tensorrt_edgellm.__file__).resolve(strict=True)
    environment = Path(sys.prefix).resolve(strict=True)
    if not package.is_relative_to(environment):
        raise RuntimeError(
            f"Imported tensorrt_edgellm from {package}, outside {environment}."
        )
    builder = Path(experimental.builder.__file__).resolve(strict=True)
    if not builder.is_relative_to(environment):
        raise RuntimeError(
            f"Imported experimental.builder from {builder}, outside {environment}."
        )
    return package


def _build_engine(model_dir: Path, engine_dir: Path) -> None:
    command = Path(sys.executable).with_name("tensorrt-edgellm-build")
    if not command.is_file():
        discovered = shutil.which("tensorrt-edgellm-build")
        if discovered:
            command = Path(discovered)
    if not command.is_file():
        raise RuntimeError(
            "The tensorrt-edgellm-build entry point is missing.")
    shutil.rmtree(engine_dir, ignore_errors=True)
    subprocess.run([
        str(command),
        "--model-dir",
        str(model_dir),
        "--engine-dir",
        str(engine_dir),
        "--components",
        "all",
        "--max-input-len",
        "128",
        "--max-kv-cache-capacity",
        "256",
        "--max-batch-size",
        "1",
    ],
                   check=True)
    engines = list(engine_dir.rglob("*.engine"))
    if not engines or any(path.stat().st_size == 0 for path in engines):
        raise RuntimeError("The installed builder produced no usable engine.")


def _request(runtime_module, prompt: str, max_tokens: int):
    message = runtime_module.create_text_message("user", prompt)
    request = runtime_module.LLMGenerationRequest()
    request.requests = [runtime_module.Request(messages=[message])]
    request.temperature = 0.7
    request.top_p = 0.9
    request.top_k = 50
    request.max_generate_length = max_tokens
    request.apply_chat_template = True
    request.add_generation_prompt = True
    request.enable_thinking = False
    request.disable_spec_decode = False
    return request


def main(values: Optional[Sequence[str]] = None) -> int:
    args = _arguments(values)
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive.")
    package = _require_installed_package()
    model_dir = args.model_dir.resolve(strict=True)
    engine_dir = args.engine_dir.resolve()
    payload = resolve_payload()
    if args.expected_variant and payload.variant_id != args.expected_variant:
        raise RuntimeError(
            f"Selected {payload.variant_id}, expected {args.expected_variant}."
        )

    _build_engine(model_dir, engine_dir)
    runtime_module = runtime_api.load()
    runtime = runtime_module.LLMRuntime(str(engine_dir), "", {})
    runtime.capture_decoding_cuda_graph()
    response = runtime.handle_request(
        _request(runtime_module, args.prompt, args.max_tokens))
    if not response.output_texts or not response.output_ids:
        raise RuntimeError("EdgeLLM returned no generated output.")

    result = {
        "package": str(package),
        "variant_id": payload.variant_id,
        "extension": str(payload.extension),
        "plugin": str(payload.plugin),
        "engine_dir": str(engine_dir),
        "output_text": response.output_texts[0],
        "output_token_count": len(response.output_ids[0]),
    }
    if args.result:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, sort_keys=True))
    print("INSTALLED_WHEEL_BUILD_AND_INFERENCE_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
