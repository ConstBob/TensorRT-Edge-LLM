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
OpenAI-compatible HTTP server for TensorRT Edge-LLM.

Endpoints:
    GET  /health                  - Health check
    GET  /v1/models               - List available models
    POST /v1/chat/completions     - Chat completion (OpenAI-compatible)

Usage (standalone)::

    # --model takes a HuggingFace id or a local path; a local path is
    # auto-detected as a prebuilt engine dir, a prebuilt ONNX dir, or a
    # checkpoint (exports ONNX + builds an engine).
    python -m experimental.server --model Qwen/Qwen3-1.7B --port 8000
    python -m experimental.server --model /path/to/engine --port 8000

Usage (from LLM object)::

    from experimental.server import LLM
    llm = LLM(model="Qwen/Qwen3-1.7B")
    llm.serve(port=8000)
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .engine import (SamplingParams, _normalize_logit_bias,
                     _validate_logit_bias_spec_decode, finish_reason_name)
from .tool_calling import (ToolConfig, parse_assistant_output,
                           validate_tool_request)

logger = logging.getLogger("edgellm.api_server")

# Whole-file uploads are buffered in memory (and copied again as base64),
# and compressed audio expands further when decoded (the C++ loader also
# caps the decoded duration); 25 MiB matches the OpenAI/vLLM limit.
MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024

# Qwen3-ASR language normalization (mirrors the HF processor's
# resolve_language): ISO codes or full names -> the canonical full name the
# model expects in its system turn.
_ASR_LANGUAGES = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
}
_ASR_LANGUAGES.update({v.lower(): v for v in list(_ASR_LANGUAGES.values())})


def _parse_content_length(value):
    """None when absent, -1 when malformed (proxies can inject non-integer
    values), else the parsed byte count."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return -1


class _ServerBusy(Exception):
    """Admission gate unavailable; mapped to HTTP 429."""


def _release_once(sem):
    """Idempotent release: the gate is handed to both the SSE generator's
    finally and the response's ASGI-call finally; only one may fire it."""
    import threading
    lock = threading.Lock()
    fired = [False]

    def _release():
        with lock:
            if fired[0]:
                return
            fired[0] = True
        sem.release()

    return _release


class _AdmissionHandoff:
    """Gate ownership for streams: HTTP releases only while no worker has
    started; once the worker starts, only its exit releases (a join timeout
    must not free the gate while the C++ call still runs)."""

    def __init__(self, sem):
        self._fire = _release_once(sem)
        self._started = False

    def worker_started(self):
        self._started = True

    def release(self):
        self._fire()

    def release_if_unstarted(self):
        if not self._started:
            self._fire()


def _releasing_streaming_response(content, release, **kw):
    """StreamingResponse that releases on ASGI-call exit: a client that
    disconnects before the first body iteration never starts the generator,
    so its finally cannot run."""
    from fastapi.responses import StreamingResponse

    class _Resp(StreamingResponse):

        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                release()

    return _Resp(content, **kw)


def _busy_response():
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=429,
        content={"error": "server busy: another request is in progress"},
        headers={"Retry-After": "1"})


def _is_asr_model(llm_instance, audio_dir: str) -> bool:
    """The transcription protocol is Qwen3-ASR specific: accept when the LLM
    or the audio encoder identifies as an ASR model type."""
    if "asr" in str(getattr(llm_instance, "_model_type", "") or "").lower():
        return True
    try:
        with open(os.path.join(audio_dir, "config.json"),
                  encoding="utf-8") as f:
            return "asr" in str(json.load(f).get("model_type", "")).lower()
    except (OSError, ValueError):
        return False


THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"
IM_END_TOKEN = "<|im_end|>"


def _split_reasoning_and_content(text: str):
    """Split model output into (reasoning_content, content) around <think> tags."""
    think_open = text.find(THINK_OPEN_TAG)
    think_close = text.find(THINK_CLOSE_TAG)
    if think_open != -1 and think_close != -1 and think_close > think_open:
        reasoning = text[think_open + len(THINK_OPEN_TAG):think_close].strip()
        content = text[think_close + len(THINK_CLOSE_TAG):].strip()
        return reasoning, content or None
    return None, text.strip() if text.strip() else None


def _create_app(llm_instance):
    """Create a FastAPI app backed by the given LLM instance."""
    try:
        from fastapi import FastAPI, File, Form, UploadFile
        from fastapi.responses import JSONResponse, PlainTextResponse
    except ImportError as exc:
        raise RuntimeError("FastAPI is required for the server. "
                           "Install: pip install fastapi uvicorn") from exc

    app = FastAPI(
        title="TensorRT Edge-LLM Server",
        version="0.1.0",
        description=
        "OpenAI-compatible inference server powered by TensorRT Edge-LLM",
    )

    @app.middleware("http")
    async def _cap_upload_body(request, call_next):
        # Starlette spools multipart uploads to disk before the handler runs, so
        # cap at the transport layer: require a Content-Length on the upload
        # route and bound it (margin for the multipart framing).
        if request.url.path == "/v1/audio/transcriptions":
            length = _parse_content_length(
                request.headers.get("content-length"))
            if length is None:
                return JSONResponse(
                    status_code=411,
                    content={"error": "Content-Length required"})
            if length < 0:
                return JSONResponse(
                    status_code=400,
                    content={"error": "invalid Content-Length header"})
            if length > MAX_AUDIO_UPLOAD_BYTES + 1024 * 1024:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error":
                        f"audio upload exceeds the supported "
                        f"maximum of {MAX_AUDIO_UPLOAD_BYTES} bytes"
                    })
        return await call_next(request)

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "model": llm_instance.model_dir,
            "speculative_decoding": llm_instance.has_draft_model,
        }

    @app.get("/v1/models")
    def list_models():
        return {
            "object":
            "list",
            "data": [{
                "id": llm_instance._model_id,
                "object": "model",
                "owned_by": "tensorrt-edgellm",
            }],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(body: Dict[str, Any]):
        messages = body.get("messages", [])
        if not messages:
            return JSONResponse(status_code=400,
                                content={"error": "messages required"})

        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        top_k = body.get("top_k", 50)
        max_tokens = body.get("max_tokens", 2048)
        stream = body.get("stream", False)
        enable_thinking = body.get("enable_thinking", False)
        disable_spec_decode = body.get("disable_spec_decode", False)
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")
        try:
            logit_bias = _normalize_logit_bias(body.get("logit_bias"))
            _validate_logit_bias_spec_decode(
                logit_bias,
                disable_spec_decode=disable_spec_decode,
                has_draft_model=llm_instance.has_draft_model,
            )
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})

        # OpenAI logprobs: "logprobs": bool, "top_logprobs": int (0-50, mirrors
        # kMaxLogprobsK). "logprobs": true alone returns the chosen token's
        # logprob per step (num_logprobs=1); top_logprobs raises K. Absent or
        # false disables regardless of top_logprobs.
        req_logprobs_flag = body.get("logprobs", False)
        req_top_logprobs = body.get("top_logprobs")
        num_logprobs = 0
        if req_logprobs_flag:
            if req_top_logprobs is None:
                num_logprobs = 1
            elif (isinstance(req_top_logprobs, bool)
                  or not isinstance(req_top_logprobs, int)
                  or not 0 <= req_top_logprobs <= 50):
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "'top_logprobs' must be an integer in [0, 50]"
                    })
            else:
                num_logprobs = max(1, req_top_logprobs)

        # OpenAI-compatible "stop": null | str | list[str]. Reject other types with 400.
        stop_raw = body.get("stop")
        stop: List[str] = []
        if stop_raw is None:
            pass
        elif isinstance(stop_raw, str):
            stop = [stop_raw]
        elif isinstance(stop_raw, list) and all(
                isinstance(s, str) for s in stop_raw):
            stop = stop_raw
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "'stop' must be a string or array of strings"
                })

        try:
            tool_config = validate_tool_request(messages, tools, tool_choice)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": str(exc)},
            )

        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            disable_spec_decode=disable_spec_decode,
            stop=stop,
            logit_bias=logit_bias,
            num_logprobs=num_logprobs,
        )

        # OpenAI: "logprobs": true without "top_logprobs" returns the chosen
        # token's logprob with an empty top_logprobs list.
        include_top_logprobs = req_top_logprobs is not None

        if stream:
            # Prebuild before the SSE response (bad input stays a 400; media
            # decodes once). Non-blocking acquire: a parked pool thread would
            # starve the SSE generator that releases the gate (busy -> 429).
            sem = llm_instance._admission()
            if not sem.acquire(blocking=False):
                return _busy_response()
            try:
                prebuilt_request = llm_instance._make_generation_request(
                    messages,
                    params,
                    tools=tool_config.tools,
                    tool_choice=tool_config.tool_choice)
            except (ValueError, KeyError) as exc:
                sem.release()
                return JSONResponse(status_code=400,
                                    content={"error": str(exc)})
            except BaseException:
                sem.release()
                raise
            handoff = _AdmissionHandoff(sem)
            return _releasing_streaming_response(
                _generate_stream_sse(
                    llm_instance,
                    messages,
                    params,
                    response_id,
                    enable_thinking,
                    tool_config=tool_config,
                    include_top_logprobs=include_top_logprobs,
                    prebuilt_request=prebuilt_request,
                    handoff=handoff,
                ),
                handoff.release_if_unstarted,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        sem = llm_instance._admission()
        if not sem.acquire(blocking=False):
            return _busy_response()
        try:
            try:
                request = llm_instance._make_generation_request(
                    messages,
                    params,
                    tools=tool_config.tools,
                    tool_choice=tool_config.tool_choice,
                    tool_config=tool_config,
                )
                response = llm_instance._handle_request(request)
            finally:
                sem.release()
        except (ValueError, KeyError) as exc:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid messages: {exc}"},
            )
        except Exception as exc:
            # Input longer than the engine's built max_input_len: the C++ runtime
            # raises with an EDGELLM_INPUT_TOO_LONG marker. Surface it as 413 with a
            # clear message instead of an opaque 500 (rebuild engine with larger
            # --maxInputLen to accept longer prompts / larger tool lists).
            if "EDGELLM_INPUT_TOO_LONG" in str(exc):
                return JSONResponse(status_code=413,
                                    content={"error": str(exc)})
            logger.exception("Inference failed")
            return JSONResponse(status_code=500, content={"error": str(exc)})

        raw_text = response.output_texts[0] if response.output_texts else ""
        output_text = raw_text.replace(IM_END_TOKEN, "")
        output_ids = response.output_ids[0] if response.output_ids else []
        completion_tokens = len(output_ids)

        message_body, has_tool_calls = _build_message_body(
            output_text, tool_config, llm_instance.model_dir)

        finish_reason = (finish_reason_name(llm_instance._rt,
                                            response.finish_reasons[0])
                         if response.finish_reasons else "stop")
        if has_tool_calls:
            finish_reason = "tool_calls"

        # ``prompt_tokens`` is reported as 0 because the runtime response does
        # not expose tokenised prompt ids; ``total_tokens`` is then equal to
        # ``completion_tokens``. SDKs that validate the schema (existence of
        # the three fields) succeed; consumers that compute cost from
        # ``prompt_tokens`` will see 0 until the runtime is extended.
        logprobs_obj = _format_logprobs(
            response,
            include_top=include_top_logprobs) if num_logprobs > 0 else None
        return {
            "id":
            response_id,
            "object":
            "chat.completion",
            "created":
            int(time.time()),
            "model":
            os.path.basename(llm_instance.model_dir) or llm_instance.model_dir,
            "choices": [{
                "index": 0,
                "message": message_body,
                "logprobs": logprobs_obj,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": completion_tokens,
                "total_tokens": completion_tokens,
            },
        }

    @app.post("/v1/audio/transcriptions")
    async def audio_transcriptions(
            file: UploadFile = File(...),
            model: str = Form(""),
            prompt: str = Form(""),
            language: str = Form(""),
            response_format: str = Form("json"),
            temperature: float = Form(0.0),
    ):
        """OpenAI-compatible ASR endpoint (Whisper SDK): routes the upload through
        the ``input_audio`` chat path (C++ extracts mel + transcribes).
        ``model``/``language`` accepted for SDK compatibility."""
        audio_dir = os.path.join(
            getattr(llm_instance, "_multimodal_engine_dir", "") or "", "audio")
        if not os.path.isdir(audio_dir):
            return JSONResponse(status_code=400,
                                content={
                                    "error":
                                    "the loaded engine has no audio encoder; "
                                    "transcription is unsupported"
                                })
        # The prompt/output protocol below is Qwen3-ASR specific; other
        # audio-capable families (Omni) do chat-audio, not transcription.
        if not _is_asr_model(llm_instance, audio_dir):
            return JSONResponse(status_code=400,
                                content={
                                    "error":
                                    "the loaded model is not an ASR model; "
                                    "transcription is unsupported"
                                })
        if response_format not in ("json", "text"):
            return JSONResponse(status_code=400,
                                content={
                                    "error":
                                    f"unsupported response_format "
                                    f"{response_format!r}; use json or text"
                                })
        # Bounded read: one extra byte detects oversize without buffering
        # an unbounded upload (the base64 copy would double it again).
        raw = await file.read(MAX_AUDIO_UPLOAD_BYTES + 1)
        if len(raw) > MAX_AUDIO_UPLOAD_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "error":
                    f"audio upload exceeds the supported "
                    f"maximum of {MAX_AUDIO_UPLOAD_BYTES} bytes"
                })
        if not raw:
            return JSONResponse(status_code=400,
                                content={"error": "empty audio file"})
        ext = (os.path.splitext(file.filename or "")[1].lstrip(".").lower()
               or "wav")
        content: List[Dict[str, Any]] = [{
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(raw).decode(),
                "format": ext,
            },
        }]
        if prompt:
            content.append({"type": "text", "text": prompt})
        messages = []
        if language:
            # HF Qwen3-ASR protocol: the normalized language name is a
            # system turn ("en" -> "English").
            lang = _ASR_LANGUAGES.get(language.strip().lower())
            if not lang:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"unsupported language {language!r}"})
            messages.append({"role": "system", "content": lang})
        messages.append({"role": "user", "content": content})
        params = SamplingParams(temperature=temperature,
                                top_p=1.0,
                                top_k=1,
                                max_tokens=4096)

        def _prepare_and_infer():
            sem = llm_instance._admission()
            if not sem.acquire(blocking=False):
                raise _ServerBusy()
            try:
                request = llm_instance._make_generation_request(
                    messages, params)
                return llm_instance._handle_request(request)
            finally:
                sem.release()

        try:
            # Request construction includes the C++ audio decode; run the
            # whole prepare+infer off the event loop.
            response = await asyncio.get_running_loop().run_in_executor(
                None, _prepare_and_infer)
        except _ServerBusy:
            return _busy_response()
        except (ValueError, KeyError, RuntimeError) as exc:
            # ValueError/KeyError: malformed request; RuntimeError from
            # the C++ audio loader: undecodable or over-long audio.
            if "EDGELLM_INPUT_TOO_LONG" in str(exc):
                return JSONResponse(status_code=413,
                                    content={"error": str(exc)})
            return JSONResponse(status_code=400,
                                content={"error": f"Invalid audio: {exc}"})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Transcription failed")
            return JSONResponse(status_code=500, content={"error": str(exc)})
        text = (response.output_texts[0] if response.output_texts else
                "").replace(IM_END_TOKEN, "").strip()
        # Qwen3-ASR output protocol: "language <LANG><asr_text><text>";
        # split on the delimiter (HF _parse_single_output semantics).
        detected = ""
        if "<asr_text>" in text:
            prefix, text = text.split("<asr_text>", 1)
            text = text.strip()
            m = re.search(r"language[ :]*([A-Za-z_\- ]+)", prefix)
            if m:
                detected = m.group(1).strip()
        if response_format == "text":
            return PlainTextResponse(text)
        body = {"text": text}
        if detected:
            body["language"] = detected
        return body

    return app


class _ThinkingStateMachine:
    """Tracks <think>...</think> boundaries across streaming deltas."""

    def __init__(self, thinking_enabled: bool):
        self._enabled = thinking_enabled
        self._in_think = False
        self._think_opened = False
        self._buf = ""

    def feed(self, text: str):
        """Yield (field, text) pairs: field is 'reasoning' or 'content'."""
        if not self._enabled:
            yield "content", text
            return

        self._buf += text
        while self._buf:
            if not self._in_think:
                idx = self._buf.find(THINK_OPEN_TAG)
                if idx == -1:
                    if len(self._buf) > len(THINK_OPEN_TAG):
                        safe = self._buf[:-len(THINK_OPEN_TAG)]
                        self._buf = self._buf[len(safe):]
                        if safe and self._think_opened:
                            yield "content", safe
                        elif safe:
                            yield "content", safe
                    break
                if idx > 0 and self._think_opened:
                    yield "content", self._buf[:idx]
                elif idx > 0:
                    yield "content", self._buf[:idx]
                self._buf = self._buf[idx + len(THINK_OPEN_TAG):]
                self._in_think = True
                self._think_opened = True
            else:
                idx = self._buf.find(THINK_CLOSE_TAG)
                if idx == -1:
                    if len(self._buf) > len(THINK_CLOSE_TAG):
                        safe = self._buf[:-len(THINK_CLOSE_TAG)]
                        self._buf = self._buf[len(safe):]
                        if safe:
                            yield "reasoning", safe
                    break
                if idx > 0:
                    yield "reasoning", self._buf[:idx]
                self._buf = self._buf[idx + len(THINK_CLOSE_TAG):]
                self._in_think = False

    def flush(self):
        """Flush remaining buffer at end of stream."""
        if self._buf:
            field = "reasoning" if self._in_think else "content"
            yield field, self._buf
            self._buf = ""


def _sse_error(message: str) -> str:
    """Terminal SSE event before ``[DONE]``: streaming clients get an actionable
    failure instead of a silent ``finish_reason=error``."""
    return "data: " + json.dumps({"error": {"message": message}}) + "\n\n"


def _generate_stream_sse(llm_instance,
                         messages,
                         params,
                         response_id,
                         enable_thinking,
                         tool_config: Optional[ToolConfig] = None,
                         include_top_logprobs: bool = True,
                         prebuilt_request=None,
                         handoff=None):
    """Yield SSE chunks via StreamChannel streaming. ``handoff`` carries the
    admission gate; it is only released here while no worker owns it."""
    try:
        yield from _generate_stream_sse_inner(llm_instance, messages, params,
                                              response_id, enable_thinking,
                                              tool_config,
                                              include_top_logprobs,
                                              prebuilt_request, handoff)
    finally:
        if handoff is not None:
            handoff.release_if_unstarted()


def _generate_stream_sse_inner(llm_instance,
                               messages,
                               params,
                               response_id,
                               enable_thinking,
                               tool_config: Optional[ToolConfig] = None,
                               include_top_logprobs: bool = True,
                               prebuilt_request=None,
                               handoff=None):
    """Yield SSE chunks via StreamChannel streaming."""
    yield _sse_chunk(response_id, {"role": "assistant"})

    if tool_config is not None and tool_config.parse_output:
        yield from _generate_tool_stream_sse(llm_instance, messages, params,
                                             response_id, tool_config,
                                             prebuilt_request, handoff)
        return

    sm = _ThinkingStateMachine(enable_thinking)
    finish_reason: Optional[str] = None
    error_message: Optional[str] = None
    stream_tools = tool_config.tools if tool_config else None
    stream_tool_choice = tool_config.tool_choice if tool_config else None

    try:
        for delta in llm_instance.generate_stream(
                messages,
                params,
                tools=stream_tools,
                tool_choice=stream_tool_choice,
                prebuilt_request=prebuilt_request,
                admission_handoff=handoff):
            lp_obj: Optional[Dict[str, Any]] = None
            if delta.logprobs:
                # OpenAI streaming schema: choices[0].logprobs = {"content": [...]},
                # one entry per generated token, mirroring the non-streaming _format_logprobs.
                content = []
                for token_id, step in zip(delta.token_ids, delta.logprobs):
                    top = [{
                        "token": e.token,
                        "token_id": e.token_id,
                        "bytes": e.bytes,
                        "logprob": float(e.logprob),
                    } for e in step]
                    chosen = next(
                        (t for t in top if t["token_id"] == token_id), None)
                    content.append({
                        "token":
                        chosen["token"] if chosen else "",
                        "token_id":
                        token_id,
                        "bytes":
                        chosen["bytes"] if chosen else [],
                        "logprob":
                        chosen["logprob"] if chosen else None,
                        "top_logprobs":
                        top if include_top_logprobs else [],
                    })
                lp_obj = {"content": content}
            if delta.text:
                for field, text in sm.feed(delta.text):
                    yield _sse_chunk(response_id, {field: text},
                                     logprobs=lp_obj)
                    lp_obj = None  # logprobs only on the first chunk per delta
            if lp_obj is not None:
                yield _sse_chunk(response_id, {}, logprobs=lp_obj)
            if delta.finished:
                finish_reason = delta.finish_reason or "stop"
    except Exception as exc:
        logger.exception("Streaming inference failed")
        finish_reason = "error"
        error_message = str(exc)

    for field, text in sm.flush():
        yield _sse_chunk(response_id, {field: text})

    if error_message and "EDGELLM_INPUT_TOO_LONG" in error_message:
        yield _sse_error(error_message)
    yield _sse_chunk(response_id, {}, finish_reason=finish_reason or "stop")
    yield "data: [DONE]\n\n"


def _generate_tool_stream_sse(llm_instance,
                              messages,
                              params,
                              response_id,
                              tool_config: ToolConfig,
                              prebuilt_request=None,
                              handoff=None):
    text_parts: List[str] = []
    finish_reason: Optional[str] = None
    error_message: Optional[str] = None
    try:
        for delta in llm_instance.generate_stream(
                messages,
                params,
                prebuilt_request=prebuilt_request,
                admission_handoff=handoff,
                tools=tool_config.tools,
                tool_choice=tool_config.tool_choice):
            if delta.text:
                text_parts.append(delta.text)
            if delta.finished:
                finish_reason = delta.finish_reason or "stop"
    except Exception as exc:
        logger.exception("Streaming inference failed")
        finish_reason = "error"
        error_message = str(exc)

    output_text = "".join(text_parts).replace(IM_END_TOKEN, "")
    parsed = parse_assistant_output(output_text, tool_config,
                                    llm_instance.model_dir)
    tool_index = 0
    for event in parsed.events:
        if event["type"] == "reasoning" and event["text"]:
            yield _sse_chunk(response_id, {"reasoning": event["text"]})
        elif event["type"] == "content" and event["text"]:
            yield _sse_chunk(response_id, {"content": event["text"]})
        elif event["type"] == "tool_call":
            call = event["tool_call"]
            yield _sse_chunk(
                response_id, {
                    "tool_calls": [{
                        "index": tool_index,
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": "",
                        },
                    }]
                })
            if call.arguments:
                yield _sse_chunk(
                    response_id, {
                        "tool_calls": [{
                            "index": tool_index,
                            "function": {
                                "arguments": call.arguments,
                            },
                        }]
                    })
            tool_index += 1

    finish = "tool_calls" if tool_index else finish_reason or "stop"
    if error_message and "EDGELLM_INPUT_TOO_LONG" in error_message:
        yield _sse_error(error_message)
    yield _sse_chunk(response_id, {}, finish_reason=finish)
    yield "data: [DONE]\n\n"


def _entry_to_openai(entry) -> Dict[str, Any]:
    """Convert one native (pybind) LogprobEntry into an OpenAI-style dict.

    ``entry.piece`` is raw token bytes: ``token`` is the UTF-8-sanitized string
    (invalid/partial bytes -> U+FFFD) and ``bytes`` carries the raw bytes so
    clients can losslessly reconstruct tokens split across multi-byte chars.
    """
    piece = entry.piece  # bytes
    return {
        "token": piece.decode("utf-8", "replace"),
        "token_id": entry.token_id,
        "bytes": list(piece),
        "logprob": float(entry.logprob),
    }


def _format_logprobs(response,
                     include_top: bool = True) -> Optional[Dict[str, Any]]:
    """Format C++ logprobs into an OpenAI-compatible logprobs object, or None.

    Each entry carries ``token`` (decoded string), ``bytes`` (raw token bytes),
    ``token_id`` (kept as a superset for internal callers) and ``logprob``.
    ``include_top=False`` emits ``top_logprobs: []`` per entry — OpenAI's shape
    for requests with ``logprobs: true`` but no ``top_logprobs``.

    Note: with non-greedy sampling the sampled token may not appear in the
    top-K candidates (logprobs are computed at temperature=1.0). In that case
    ``token``/``bytes`` are empty and ``logprob`` is ``null`` for the chosen
    token; ``top_logprobs`` still lists the candidates. Greedy is unaffected.
    """
    if not response.logprobs:
        return None
    output_ids = response.output_ids[0] if response.output_ids else []
    step_logprobs = response.logprobs[0]
    if not step_logprobs:
        return None

    content = []
    for token_id, step_topk in zip(output_ids, step_logprobs):
        top = [_entry_to_openai(e) for e in step_topk]
        chosen = next((d for d in top if d["token_id"] == token_id), None)
        content.append({
            "token": chosen["token"] if chosen else "",
            "token_id": token_id,
            "bytes": chosen["bytes"] if chosen else [],
            "logprob": chosen["logprob"] if chosen else None,
            "top_logprobs": top if include_top else [],
        })
    return {"content": content}


def _build_message_body(output_text: str, tool_config: ToolConfig,
                        model_dir: str) -> Tuple[Dict[str, Any], bool]:
    if not tool_config.parse_output:
        reasoning, answer = _split_reasoning_and_content(output_text)
        message_body: Dict[str, Any] = {"role": "assistant"}
        if reasoning is not None:
            message_body["reasoning"] = reasoning
        message_body["content"] = (
            (answer if answer is not None else reasoning) or "")
        return message_body, False

    parsed = parse_assistant_output(output_text, tool_config, model_dir)
    tool_calls = [call.to_openai() for call in parsed.tool_calls]
    message_body: Dict[str, Any] = {"role": "assistant"}
    if parsed.reasoning:
        message_body["reasoning"] = parsed.reasoning
    content = parsed.content.strip()
    message_body["content"] = content if content or not tool_calls else None
    if tool_calls:
        message_body["tool_calls"] = tool_calls
    return message_body, bool(tool_calls)


def _sse_chunk(response_id: str,
               delta: dict,
               finish_reason: Optional[str] = None,
               logprobs: Optional[Dict[str, Any]] = None):
    choice: Dict[str, Any] = {"delta": delta, "index": 0}
    if finish_reason:
        choice["finish_reason"] = finish_reason
    if logprobs:
        choice["logprobs"] = logprobs
    payload = {"id": response_id, "choices": [choice]}
    return f"data: {json.dumps(payload)}\n\n"


def run_server(llm_instance, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the OpenAI-compatible server."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required. Install: pip install uvicorn") from exc

    app = _create_app(llm_instance)
    logger.info("Starting server on %s:%d ...", host, port)
    uvicorn.run(app, host=host, port=port)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="TensorRT Edge-LLM OpenAI-compatible server")
    # Single model source. A local path is auto-detected as a prebuilt engine
    # dir, a prebuilt ONNX dir, or a checkpoint, then routed to the matching one
    # of the LLM class's three init modes.
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID or local path. A local path is auto-detected "
        "as a prebuilt engine dir, a prebuilt ONNX dir, or a checkpoint (exports "
        "ONNX + builds an engine).",
    )
    parser.add_argument(
        "--multimodal-engine-dir",
        "--visual-engine-dir",
        dest="multimodal_engine_dir",
        default="",
        help="Pre-built multimodal engine directory (visual and/or audio "
        "encoders) for a prebuilt --model engine dir",
    )
    parser.add_argument(
        "--visual-onnx-dir",
        dest="visual_onnx_dir",
        default="",
        help="Pre-built visual ONNX directory for a VLM "
        "(use when --model is a prebuilt ONNX dir)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--max-input-len",
        type=int,
        default=4096,
        help="Max input sequence length",
    )
    parser.add_argument("--max-batch-size",
                        type=int,
                        default=1,
                        help="Max batch size")
    parser.add_argument(
        "--max-kv-cache-capacity",
        type=int,
        default=8192,
        help="Max KV cache capacity",
    )
    parser.add_argument(
        "--spec-decode-engine-dir",
        dest="spec_decode_engine_dir",
        default="",
        help=
        "Pre-built speculative decoding engine dir (EAGLE, MTP, or DFlash)",
    )
    parser.add_argument("--draft-top-k",
                        type=int,
                        default=10,
                        help="Speculative decoding: tokens per predecessor")
    parser.add_argument("--draft-step",
                        type=int,
                        default=6,
                        help="Speculative decoding: number of draft steps")
    parser.add_argument("--verify-tree-size",
                        type=int,
                        default=60,
                        help="Speculative decoding: verification tree size")
    args = parser.parse_args()

    from .engine import LLM
    from .engine_layout import classify_model_source

    # Auto-classify the --model path (prebuilt engine / prebuilt ONNX /
    # checkpoint) and route it to the matching one of LLM's three init modes.
    # Build params are ignored for the prebuilt-engine mode.
    model_arg = args.model
    onnx_dir = ""
    engine_dir = ""
    mode = classify_model_source(model_arg)
    if mode == "engine_dir":
        engine_dir, model_arg = model_arg, ""
    elif mode == "onnx_dir":
        onnx_dir, model_arg = model_arg, ""

    llm = LLM(
        model=model_arg,
        onnx_dir=onnx_dir,
        visual_onnx_dir=args.visual_onnx_dir,
        engine_dir=engine_dir,
        multimodal_engine_dir=args.multimodal_engine_dir,
        max_input_len=args.max_input_len,
        max_batch_size=args.max_batch_size,
        max_kv_cache_capacity=args.max_kv_cache_capacity,
        eagle_engine_dir=args.spec_decode_engine_dir,
        draft_top_k=args.draft_top_k,
        draft_step=args.draft_step,
        verify_tree_size=args.verify_tree_size,
    )
    llm.serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
