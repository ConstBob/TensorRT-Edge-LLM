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
"""
Server-side audio content resolution.

The HTTP server accepts three OpenAI-compatible audio content forms in chat
messages:

  - ``{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}``
  - ``{"type": "audio_url", "audio_url": {"url": "file:///abs/path | data:audio/...;base64,..."}}``
  - ``{"type": "audio", "audio": "<local path>"}``  (legacy shorthand)

``http(s)://`` audio URLs are rejected by design — host the audio locally and
use ``file://``. There is no remote-fetch on the server.

This module resolves each accepted form to raw audio bytes and hands them
to the C++ runtime via ``_edgellm_runtime.load_audio_buffer_from_bytes``.
Container decode + resample run in C++ (miniaudio); the audio runner
auto-selects its mel feature extractor (``whisper`` for Qwen3-Omni /
Qwen3-ASR, ``parakeet`` for Nemotron-Omni) from
``<engine>/audio/config.json::model_type`` at init. The Python side
never touches a feature extractor or PCM samples.
"""

import base64
import logging
from typing import Any, Dict, List
from urllib.parse import unquote, urlparse

logger = logging.getLogger("edgellm.audio")


def _decode_data_url(url: str) -> bytes:
    """Decode ``data:audio/...;base64,<payload>`` URLs.

    Non-base64 ``data:`` URLs are rejected — audio payloads are binary and
    always base64-encoded in practice.
    """
    head, _, payload = url.partition(",")
    if ";base64" not in head:
        raise ValueError("data: audio URLs must use base64 encoding")
    try:
        return base64.b64decode(payload, validate=False)
    except Exception as exc:
        raise ValueError(f"Malformed base64 data URL: {exc}") from exc


def _resolve_file_url(url: str) -> str:
    """Convert a ``file://`` URL to a local filesystem path.

    Callers guard with ``url.startswith("file:")`` before calling, so this
    handles only ``file:///abs/path`` / ``file://localhost/abs/path``;
    non-local ``netloc`` and empty paths raise ``ValueError``.
    """
    parsed = urlparse(url)
    # Reject anything that isn't local. `urlparse('file:///x').netloc` is ''.
    if parsed.netloc and parsed.netloc not in ("", "localhost"):
        raise ValueError(
            f"file:// URL must be local (no host); got netloc={parsed.netloc!r}"
        )
    path = unquote(parsed.path)
    if not path:
        raise ValueError("file:// URL has empty path")
    return path


def resolve_audio_message(item: Dict[str, Any]):
    """Resolve one audio content item to its underlying source.

    Returns one of:
      * ``str`` — a local filesystem path (caller will open it)
      * ``bytes`` — already-decoded audio bytes (wav/mp3/flac container)

    Raises ``ValueError`` for anything we don't accept:
      * ``http(s)://`` URLs (no remote fetch)
      * non-base64 ``data:`` URLs
      * Missing required fields
    """
    content_type = item.get("type")

    if content_type == "input_audio":
        # OpenAI canonical form. We don't validate `format` against soundfile's
        # supported list — soundfile will raise a clear error on read.
        payload = item.get("input_audio") or {}
        data_b64 = payload.get("data")
        if not data_b64:
            raise ValueError("input_audio.data is required (base64 string)")
        try:
            return base64.b64decode(data_b64, validate=False)
        except Exception as exc:
            raise ValueError(
                f"Malformed base64 in input_audio.data: {exc}") from exc

    if content_type == "audio_url":
        url = ((item.get("audio_url") or {}).get("url") or "").strip()
        if not url:
            raise ValueError("audio_url.url is required")
        if url.startswith("http://") or url.startswith("https://"):
            # Hard rejection — see module docstring.
            raise ValueError(
                "Remote audio URLs (http/https) are not supported. "
                "Host the file locally and use file:// or pass input_audio.data."
            )
        if url.startswith("data:"):
            return _decode_data_url(url)
        if url.startswith("file:"):
            return _resolve_file_url(url)
        # Bare path tolerated for ergonomics — same as shorthand `audio`.
        return url

    if content_type == "audio":
        # Legacy shorthand: just a local path string.
        path = (item.get("audio") or "").strip()
        if not path:
            raise ValueError("audio field is required")
        if path.startswith("http://") or path.startswith("https://"):
            raise ValueError(
                "Remote audio URLs (http/https) are not supported.")
        if path.startswith("file:"):
            return _resolve_file_url(path)
        if path.startswith("data:"):
            return _decode_data_url(path)
        return path

    raise ValueError(f"Unsupported audio content type: {content_type!r}")


def load_audio_buffers(
    rt_module,
    messages: List[Dict[str, Any]],
) -> list:
    """Walk messages and return a list of ``rt.AudioData`` for the runtime.

    Mirrors ``engine._load_image_buffers``: hand raw encoded bytes to the
    C++ runtime via ``load_audio_buffer_from_bytes``; container decode runs
    in C++ (miniaudio) and the audio runner extracts mel internally per its
    ``audio/config.json``. Python touches neither PCM samples nor mel.
    """
    audios = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") not in ("input_audio", "audio_url", "audio"):
                continue
            source = resolve_audio_message(item)
            # ``resolve_audio_message`` returns either a path (str) or already
            # -decoded container bytes; normalise to bytes here so the runtime
            # API surface is single-typed.
            if isinstance(source, (bytes, bytearray)):
                audio_bytes = bytes(source)
            else:
                with open(source, "rb") as f:
                    audio_bytes = f.read()
            audios.append(rt_module.load_audio_buffer_from_bytes(audio_bytes))
    return audios
