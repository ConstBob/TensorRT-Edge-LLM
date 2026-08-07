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
"""Tests for OpenAI-compatible logit_bias request validation."""

import math
from types import SimpleNamespace

import pytest

from experimental.server.engine import (_MAX_LOGIT_BIAS_TOKENS, LLM,
                                        SamplingParams, _normalize_logit_bias)

_INT32_MAX = 2**31 - 1


def test_normalize_logit_bias_accepts_integer_like_keys_and_boundaries():
    assert _normalize_logit_bias(None) == {}
    assert _normalize_logit_bias({
        "1": -100,
        2: 0.5,
        "3": 100.0,
        str(_INT32_MAX): 0,
    }) == {
        1: -100.0,
        2: 0.5,
        3: 100.0,
        _INT32_MAX: 0.0,
    }


def test_normalize_logit_bias_rejects_too_many_entries():
    too_many_entries = {str(i): 0.0 for i in range(_MAX_LOGIT_BIAS_TOKENS + 1)}

    with pytest.raises(ValueError, match="max is"):
        _normalize_logit_bias(too_many_entries)


@pytest.mark.parametrize("token_id", [True, 1.25, object(), "not-an-int"])
def test_normalize_logit_bias_rejects_non_integer_token_ids(token_id):
    with pytest.raises(ValueError, match="not an integer"):
        _normalize_logit_bias({token_id: 0.0})


@pytest.mark.parametrize(
    "token_id",
    [-1, "-1", _INT32_MAX + 1, str(_INT32_MAX + 1)])
def test_normalize_logit_bias_rejects_token_ids_outside_nonnegative_int32(
        token_id):
    with pytest.raises(ValueError, match="token ID"):
        _normalize_logit_bias({token_id: 0.0})


@pytest.mark.parametrize("bias", [
    True, "1.0", math.nan, math.inf, -100.1, 100.1,
    pytest.param(10**400, id="overflowing-int")
])
def test_normalize_logit_bias_rejects_invalid_bias_values(bias):
    with pytest.raises(ValueError):
        _normalize_logit_bias({"1": bias})


def test_hlapi_generate_accepts_logit_bias_with_active_spec_decode():

    class FakeAdmission:

        @staticmethod
        def __enter__():
            return None

        @staticmethod
        def __exit__(*args):
            return False

    llm = object.__new__(LLM)
    llm._rt = object()
    llm._runtime = SimpleNamespace(has_draft_model=lambda: True)
    llm._admission = lambda: FakeAdmission()
    llm._make_generation_request = lambda *args, **kwargs: object()
    llm._handle_request = lambda request: SimpleNamespace(
        output_texts=[""], output_ids=[[]], finish_reasons=[], logprobs=[])

    outputs = llm.generate("hello", SamplingParams(logit_bias={1: 1.0}))

    assert len(outputs) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_api_accepts_logit_bias_with_active_spec_decode(stream):
    TestClient = pytest.importorskip("fastapi.testclient").TestClient

    from experimental.server.api_server import _create_app

    class FakeRuntime:

        @staticmethod
        def handle_request(_request):

            class Response:
                output_texts = [""]
                output_ids = [[]]
                finish_reasons = []
                logprobs = []

            return Response()

    class FakeAdmission:

        @staticmethod
        def acquire(blocking=True):
            return True

        @staticmethod
        def release():
            pass

    class FakeLLM:
        _model_id = "test-model"
        has_draft_model = True
        model_dir = ""
        _rt = object()
        _runtime = FakeRuntime()

        @staticmethod
        def _admission():
            return FakeAdmission()

        @staticmethod
        def _make_generation_request(*args, **kwargs):
            return object()

        @staticmethod
        def count_prompt_tokens(*args, **kwargs):
            return 1

        @staticmethod
        def generate_stream(*args, **kwargs):
            return iter(())

    response = TestClient(_create_app(FakeLLM())).post(
        "/v1/chat/completions",
        json={
            "messages": [{
                "role": "user",
                "content": "hello"
            }],
            "logit_bias": {
                "1": 1.0
            },
            "stream": stream,
        },
    )

    assert response.status_code == 200


@pytest.mark.parametrize("stream", [False, True])
def test_api_rejects_overflowing_logit_bias(stream):
    TestClient = pytest.importorskip("fastapi.testclient").TestClient

    from experimental.server.api_server import _create_app

    class FakeLLM:
        _model_id = "test-model"
        has_draft_model = False

    response = TestClient(_create_app(FakeLLM())).post(
        "/v1/chat/completions",
        json={
            "messages": [{
                "role": "user",
                "content": "hello"
            }],
            "logit_bias": {
                "1": 10**400
            },
            "stream": stream,
        },
    )

    assert response.status_code == 400
    assert "logit_bias" in response.json()["error"]
