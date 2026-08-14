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
"""Anthropic Messages serving over the shared Edge-LLM engine client."""

import asyncio
import contextlib
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict

from ..runtime.engine_client import EngineClient, PreparedRequest
from . import anthropic_compat as protocol
from .errors import ServerError
from .protocol import ChatCompletionRequest
from .serving_chat import IM_END_TOKEN, OpenAIServingChat, PreparedChatRequest


@dataclass(frozen=True)
class PreparedAnthropicStream:
    request: ChatCompletionRequest
    chat: PreparedChatRequest
    engine: PreparedRequest
    message_id: str


class AnthropicServingMessages:
    """Translate Anthropic protocol data without duplicating inference."""

    def __init__(self, client: EngineClient, chat: OpenAIServingChat) -> None:
        self._client = client
        self._chat = chat

    def _request(self,
                 body: Dict[str, Any],
                 *,
                 require_max_tokens: bool = True) -> ChatCompletionRequest:
        messages, tools, tool_choice, sampling = protocol.convert_request(
            body, require_max_tokens=require_max_tokens)
        raw_choice = body.get("tool_choice") or {}
        return ChatCompletionRequest(
            model=body.get("model"),
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=not bool(
                raw_choice.get("disable_parallel_tool_use", False)),
            stream=False,
            **sampling,
        )

    async def create_message(self, body: Dict[str, Any]) -> Dict[str, Any]:
        request = self._request(body)
        response = await self._chat.create_chat_completion(request)
        choice = response.choices[0]
        message = choice.message
        return {
            "id":
            f"msg_{uuid.uuid4().hex}",
            "type":
            "message",
            "role":
            "assistant",
            "model":
            response.model,
            "content":
            protocol.build_content_blocks(
                message.content,
                message.tool_calls or [],
                message.reasoning_content,
            ),
            "stop_reason":
            protocol.convert_stop_reason(choice.finish_reason),
            "stop_sequence":
            None,
            "usage":
            protocol.usage(response.usage.prompt_tokens,
                           response.usage.completion_tokens),
        }

    async def count_tokens(self, body: Dict[str, Any]) -> int:
        request = self._request(body, require_max_tokens=False)
        chat = self._chat.prepare_request(request)
        count = await self._client.count_prompt_tokens(
            request.messages,
            tool_config=chat.tool_config,
            enable_thinking=request.enable_thinking,
        )
        return count or 0

    async def prepare_stream(self, body: Dict[str,
                                              Any]) -> PreparedAnthropicStream:
        request = self._request(body)
        chat = self._chat.prepare_request(request)
        engine = await self._chat.prepare_engine_request(request, chat)
        return PreparedAnthropicStream(
            request=request,
            chat=chat,
            engine=engine,
            message_id=f"msg_{uuid.uuid4().hex}",
        )

    async def stream_message(
        self,
        prepared: PreparedAnthropicStream,
    ) -> AsyncGenerator[str, None]:
        prompt_tokens = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(
            self._collect_stream(prepared, prompt_tokens))
        try:
            exact_prompt_tokens = await prompt_tokens
            for chunk in protocol.message_start_events(prepared.message_id,
                                                       self._client.model_name,
                                                       exact_prompt_tokens):
                yield chunk
            while not task.done():
                done, _ = await asyncio.wait({task}, timeout=5.0)
                if not done:
                    yield protocol.event("ping", {})
            blocks, finish_reason, completion_tokens = await task
        except asyncio.CancelledError:
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
            raise
        except ServerError as exc:
            with contextlib.suppress(BaseException):
                await task
            status = 529 if exc.status_code == 429 else exc.status_code
            yield protocol.event("error",
                                 protocol.error_payload(status, str(exc)))
            return
        except Exception as exc:
            with contextlib.suppress(BaseException):
                await task
            yield protocol.event("error",
                                 protocol.error_payload(500, str(exc)))
            return

        for chunk in protocol.content_tail_events(
                blocks, protocol.convert_stop_reason(finish_reason),
                completion_tokens):
            yield chunk

    async def _collect_stream(self, prepared: PreparedAnthropicStream,
                              prompt_tokens: asyncio.Future):
        text_parts = []
        completion_tokens = 0
        finish_reason = "stop"
        try:
            async for delta in self._client.stream(
                    prepared.request.messages,
                    prepared.chat.sampling,
                    tools=prepared.chat.tool_config.tools,
                    tool_choice=prepared.chat.tool_config.tool_choice,
                    prepared=prepared.engine):
                if (not prompt_tokens.done()
                        and delta.prompt_tokens is not None):
                    prompt_tokens.set_result(delta.prompt_tokens)
                completion_tokens += len(delta.token_ids)
                if delta.text:
                    text_parts.append(delta.text)
                if delta.finished:
                    finish_reason = delta.finish_reason or "stop"
        except BaseException as exc:
            if not prompt_tokens.done():
                prompt_tokens.set_exception(exc)
            raise
        if not prompt_tokens.done():
            prompt_tokens.set_result(None)

        parsed = self._chat.parse_output(
            "".join(text_parts).replace(IM_END_TOKEN, ""), prepared.chat)
        tool_calls = [call.to_openai() for call in parsed.tool_calls]
        if tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"
        blocks = protocol.build_content_blocks(parsed.content, tool_calls,
                                               parsed.reasoning)
        return blocks, finish_reason, completion_tokens
