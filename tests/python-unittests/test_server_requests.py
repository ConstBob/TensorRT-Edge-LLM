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
Server request-parsing coverage for every media modality, no engine / C++
runtime: OpenAI chat requests flow through the real HTTP endpoint and engine
conversion with stubs only at the C++-binding and video-decode boundaries,
plus the ``/v1/audio/transcriptions`` endpoint (FastAPI TestClient).
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                         ".."))

# ---------------------------------------------------------------------------
# Stub runtime module (no C++): only what engine.py's message/buffer helpers use
# ---------------------------------------------------------------------------


class _FakeBuffer(tuple):
    """Tuple-comparable stub buffer that also accepts attribute writes (e.g. do_resize)."""

    def __new__(cls, *items):
        return super().__new__(cls, items)


class _Content:

    def __init__(self, ctype, data=""):
        self.type = ctype
        self.data = data


class _Message:

    def __init__(self):
        self.role = ""
        self.contents = []


class _StubRt:
    """Records which load_* binding each visual buffer used."""

    def MessageContent(self, ctype, data=""):
        return _Content(ctype, data)

    def Message(self):
        return _Message()

    def load_image_from_path(self, path):
        return _FakeBuffer("image", path)

    def load_video_from_array(self, frames, fps, timestamps=()):
        return _FakeBuffer("video_array", fps)

    def load_video_from_paths(self, paths, fps):
        return _FakeBuffer("video_paths", list(paths), fps)


def _engine():
    try:
        from experimental.server import engine
    except ImportError as exc:  # optional deps (pybind runtime chain)
        pytest.skip(f"engine import unavailable: {exc}")
    return engine


# ---------------------------------------------------------------------------
# engine.py: video content -> MessageContent("video") + ordered image_buffers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# OpenAI HTTP layer (FastAPI TestClient + stub LLM)
# ---------------------------------------------------------------------------
def _make_stub_llm():

    class _Resp:
        output_texts = ["a caption"]
        output_ids = [[1, 2, 3]]
        finish_reasons = []  # -> endpoint defaults finish_reason to "stop"

    class _RT:
        _resp_text = None  # tests may override the canned output

        def handle_request(self, request):
            resp = _Resp()
            if self._resp_text is not None:
                resp.output_texts = [self._resp_text]
            return resp

    class _LLM:
        model_dir = "/fake/qwen3_vl"
        _model_id = "qwen3-vl"
        _rt = None
        has_draft_model = False

        def __init__(self):
            self._runtime = _RT()
            self.captured = None
            # Advertise ASR capability (the endpoint probes for an audio/
            # engine subdir with an ASR-typed config).
            self._multimodal_engine_dir = tempfile.mkdtemp()
            audio_dir = os.path.join(self._multimodal_engine_dir, "audio")
            os.makedirs(audio_dir, exist_ok=True)
            with open(os.path.join(audio_dir, "config.json"),
                      "w",
                      encoding="utf-8") as f:
                f.write('{"model_type": "qwen3_asr"}')

        def _handle_request(self, request):
            return self._runtime.handle_request(request)

        def _admission(self):
            import threading
            sem = self.__dict__.get("_admission_sem")
            if sem is None:
                sem = self.__dict__.setdefault("_admission_sem",
                                               threading.Semaphore(1))
            return sem

        def _make_generation_request(self,
                                     messages,
                                     params,
                                     *,
                                     tools=None,
                                     tool_choice=None,
                                     tool_config=None):
            self.captured = messages
            return object()

        def count_prompt_tokens(self, messages, **kw):
            return 7

    return _LLM()


@pytest.fixture
def client_and_llm():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")  # fastapi.testclient needs httpx
    try:
        from fastapi.testclient import TestClient

        from experimental.server.api_server import _create_app
    except ImportError as exc:  # optional deps / pybind runtime chain
        pytest.skip(f"api_server / fastapi TestClient unavailable: {exc}")
    llm = _make_stub_llm()
    return TestClient(_create_app(llm)), llm


@pytest.mark.parametrize("response_format,expect_json", [("json", True),
                                                         ("text", False)])
def test_audio_transcriptions_endpoint(client_and_llm, response_format,
                                       expect_json):
    client, llm = client_and_llm
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        data={
            "model": "asr",
            "response_format": response_format
        })
    assert resp.status_code == 200, resp.text
    if expect_json:
        assert resp.json()["text"] == "a caption"
    else:
        assert resp.text == "a caption"
    forwarded = llm.captured[0]["content"]
    assert any(c.get("type") == "input_audio" for c in forwarded)


def test_audio_transcriptions_rejects_empty(client_and_llm):
    client, _ = client_and_llm
    resp = client.post("/v1/audio/transcriptions",
                       files={"file": ("e.wav", b"", "audio/wav")},
                       data={"model": "asr"})
    assert resp.status_code == 400


def test_audio_transcriptions_protocol(client_and_llm):
    # Qwen3-ASR protocol: normalized language ("en" -> "English") goes into a
    # system turn, and the "language <LANG><asr_text><text>" output splits on
    # the delimiter.
    client, llm = client_and_llm
    llm._runtime._resp_text = "language English<asr_text>Hello world"
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        data={
            "model": "asr",
            "language": "en"
        })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["text"] == "Hello world"
    assert body["language"] == "English"
    assert llm.captured[0]["role"] == "system"
    assert llm.captured[0]["content"] == "English"
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        data={
            "model": "asr",
            "language": "klingon"
        })
    assert resp.status_code == 400


def test_content_length_parsing():
    # Malformed Content-Length (proxy-injected lists, floats) must map to a
    # clean 400, never an unhandled ValueError in the middleware.
    from experimental.server.api_server import _parse_content_length
    assert _parse_content_length(None) is None
    assert _parse_content_length("1024") == 1024
    assert _parse_content_length("abc") == -1
    assert _parse_content_length("1.5") == -1
    assert _parse_content_length("10, 20") == -1


def test_audio_transcriptions_requires_audio_engine(client_and_llm):
    # A model without an audio encoder must reject transcription up front.
    client, llm = client_and_llm
    saved = llm._multimodal_engine_dir
    llm._multimodal_engine_dir = ""
    try:
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
            data={"model": "asr"})
        assert resp.status_code == 400
        assert "audio" in resp.json()["error"]
    finally:
        llm._multimodal_engine_dir = saved


def test_audio_transcriptions_limits(client_and_llm, monkeypatch):
    # Uploads are buffered in memory (plus a base64 copy): oversize is 413,
    # and unsupported response_format values are rejected instead of being
    # silently served as JSON.
    client, _ = client_and_llm
    from experimental.server import api_server
    monkeypatch.setattr(api_server, "MAX_AUDIO_UPLOAD_BYTES", 16)
    resp = client.post("/v1/audio/transcriptions",
                       files={"file": ("big.wav", b"x" * 17, "audio/wav")},
                       data={"model": "asr"})
    assert resp.status_code == 413
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        data={
            "model": "asr",
            "response_format": "srt"
        })
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Model-family detection
# ---------------------------------------------------------------------------


def test_video_model_family_resolution(tmp_path):
    """All model_type -> family branches of LLM._video_model_family:
    accepted families, rejected model types, and the nested-layout priority
    (the C++ runtime loads <root>/visual first)."""
    eng = _engine()

    def family_for(model_type, nested=None, visual=True, sub=None):
        root = tmp_path / (sub or model_type or "empty")
        root.mkdir(exist_ok=True)
        if visual:
            (root / "visual").mkdir(exist_ok=True)
        if model_type is not None:
            (root / "config.json").write_text('{"model_type": "%s"}' %
                                              model_type)
        if nested is not None:
            (root / "visual" / "config.json").write_text(
                '{"model_type": "%s"}' % nested)
        llm = eng.LLM.__new__(eng.LLM)
        llm._multimodal_engine_dir = str(root)
        return llm._video_model_family()

    assert family_for("internvl") == "internvl"
    assert family_for("qwen3_vl") == "qwen"
    # Nested <root>/visual/config.json wins over a legacy flat config.json.
    assert family_for("qwen3_vl", nested="internvl_chat",
                      sub="nested") == "internvl"
    # No visual engine at all: video must be rejected, not defaulted to qwen.
    with pytest.raises(ValueError, match="not supported"):
        family_for(None, visual=False, sub="none")
    # Model families without a video preprocessing path (phi4mm reads only
    # the first frame) must be rejected up front.
    with pytest.raises(ValueError, match="not supported"):
        family_for("phi4mm")
    # Audio-side omni/asr types share the qwen prefix but have no video path.
    for mt in ("qwen3_omni_audio_encoder", "qwen3_omni_code2wav", "qwen3_asr"):
        with pytest.raises(ValueError, match="not supported"):
            family_for(mt)


def test_load_image_buffers_request_wide_internvl_minimum():
    # Two 2-frame videos jointly reach the 4-block engine minimum; one alone
    # must be rejected AFTER accumulation (the bound is request-wide).
    eng = _engine()
    limits = {
        "model_type": "internvl",
        "min_image_tokens": 1024,
        "max_image_tokens": 4096,
        "max_image_tokens_per_image": 512,
    }

    def fake_load_video_buffer(rt,
                               item,
                               family,
                               frame_limits=None,
                               budget=None,
                               pixel_budget=None,
                               cu_budget=None):
        return ("video", item["video"]), 2 * 256, 0, 2  # 2 frames = 2 blocks

    import experimental.server.video_sampling as vs_mod
    orig = vs_mod.load_video_buffer
    vs_mod.load_video_buffer = fake_load_video_buffer
    try:
        two = eng._load_image_buffers(None, [{
            "role":
            "user",
            "content": [{
                "type": "video",
                "video": "a.mp4"
            }, {
                "type": "video",
                "video": "b.mp4"
            }]
        }], lambda: "internvl", lambda: limits)
        assert len(two) == 2
        # Missing image files are skipped and must NOT count toward the
        # minimum (they produce no buffer, so no blocks).
        with pytest.raises(ValueError, match="at least 1024"):
            eng._load_image_buffers(None, [{
                "role":
                "user",
                "content": [{
                    "type": "video",
                    "video": "a.mp4"
                }] + [{
                    "type": "image",
                    "image": f"/no/such/img{i}.jpg"
                } for i in range(3)]
            }], lambda: "internvl", lambda: limits)
        with pytest.raises(ValueError, match="at least 1024"):
            eng._load_image_buffers(
                None, [{
                    "role": "user",
                    "content": [{
                        "type": "video",
                        "video": "a.mp4"
                    }]
                }], lambda: "internvl", lambda: limits)
    finally:
        vs_mod.load_video_buffer = orig


def test_load_image_buffers_request_wide_qwen_minimum(tmp_path, monkeypatch):
    # The engine minimum is request-wide for Qwen too: resized media are
    # floored per item, but do_resize=false media can undershoot and the C++
    # runner checks the accumulated totalSeqLength against the profile MIN.
    eng = _engine()
    limits = {
        "model_type": "qwen3_vl",
        "min_image_tokens": 128,
        "max_image_tokens": 4096,
        "max_image_tokens_per_image": 4096,
        "patch_size": 16,
        "merge_size": 2,
        "temporal_patch_size": 2,
    }
    est_by_source = {"tiny.mp4": 8, "a.mp4": 64, "b.mp4": 64}

    def fake_load_video_buffer(rt,
                               item,
                               family,
                               frame_limits=None,
                               budget=None,
                               pixel_budget=None,
                               cu_budget=None):
        est = est_by_source[item["video"]]
        return ("video", item["video"]), est, 0, 1

    import experimental.server.video_sampling as vs_mod
    monkeypatch.setattr(vs_mod, "load_video_buffer", fake_load_video_buffer)

    def _load(content):
        return eng._load_image_buffers(_StubRt(), [{
            "role": "user",
            "content": content
        }], lambda: "qwen", lambda: limits)

    # A single raw video far below the minimum -> 400 up front, not a C++
    # runtime failure.
    with pytest.raises(ValueError, match="at least 128"):
        _load([{"type": "video", "video": "tiny.mp4"}])
    # Two raw media jointly reaching the minimum pass.
    assert len(
        _load([{
            "type": "video",
            "video": "a.mp4"
        }, {
            "type": "video",
            "video": "b.mp4"
        }])) == 2
    # image + video still short of the minimum -> 400.
    img = tmp_path / "small.jpg"
    img.write_bytes(b"x")
    monkeypatch.setattr(vs_mod,
                        "estimate_image_tokens",
                        lambda path, family, lim, do_resize=True: 32)
    with pytest.raises(ValueError, match="at least 128"):
        _load([{
            "type": "image",
            "image": str(img)
        }, {
            "type": "video",
            "video": "a.mp4"
        }])


def test_load_image_buffers_budget_order_independent(tmp_path, monkeypatch):
    # The video sampler must see the same remaining budget whether the image
    # appears before or after the video, so both orders produce identical
    # buffers (images are probed and reserved up front, phase 1).
    eng = _engine()
    limits = {
        "model_type": "qwen2_5_vl",
        "min_image_tokens": 4,
        "max_image_tokens": 4096,
        "max_image_tokens_per_image": 4096,
        "patch_size": 14,
        "merge_size": 2,
        "temporal_patch_size": 2,
    }
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    img = tmp_path / "a.mp4"
    container = av.open(str(img), mode="w")
    stream = container.add_stream("mpeg4", rate=1)
    stream.width = stream.height = 64
    stream.pix_fmt = "yuv420p"
    frame = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), dtype=np.uint8),
                                       format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    # The REAL estimator is used for the phase-1 reservation (it must accept
    # do_resize=...; a dropped parameter once surfaced as TypeError).
    import experimental.server.video_sampling as vs_mod
    image_est = vs_mod.estimate_image_tokens(str(img), "qwen", limits)
    assert image_est > 0
    seen_budgets = []

    def fake_load_video_buffer(rt,
                               item,
                               family,
                               frame_limits=None,
                               budget=None,
                               pixel_budget=None,
                               cu_budget=None):
        # The sampler clamps to the remaining budget: consume min(cap, need).
        seen_budgets.append(budget)
        need = 4000
        if budget is not None and budget < need:
            return ("video", budget), budget, 0, 1
        return ("video", need), need, 0, 1

    monkeypatch.setattr(vs_mod, "load_video_buffer", fake_load_video_buffer)

    def items(order):
        content = [{
            "type": "image",
            "image": str(img)
        }, {
            "type": "video",
            "video": "v.mp4"
        }]
        return [{
            "role": "user",
            "content": content if order == "iv" else content[::-1]
        }]

    rt = _StubRt()
    results = {}
    for order in ("iv", "vi"):
        buffers = eng._load_image_buffers(rt, items(order), lambda: "qwen",
                                          lambda: limits)
        video_buf = next(b for b in buffers if b[0] == "video")
        results[order] = video_buf
    # Identical remaining budget in both orders -> identical video buffer.
    assert seen_budgets[0] == seen_budgets[1] == 4096 - image_est
    assert results["iv"] == results["vi"]


def test_stream_disconnect_before_first_byte_releases_admission(
        client_and_llm):
    # A disconnect before the first body frame means the SSE generator never
    # starts, so its finally cannot run; the ASGI-call finally must release.
    client, llm = client_and_llm
    app = client.app
    body = json.dumps({
        "messages": [{
            "role": "user",
            "content": "hi"
        }],
        "stream": True
    }).encode()
    scope = {
        "type":
        "http",
        "http_version":
        "1.1",
        "method":
        "POST",
        "path":
        "/v1/chat/completions",
        "raw_path":
        b"/v1/chat/completions",
        "root_path":
        "",
        "scheme":
        "http",
        "query_string":
        b"",
        "headers": [(b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode())],
        "client": ("test", 1),
        "server": ("test", 80),
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    class _Disconnect(Exception):
        pass

    async def send(message):
        raise _Disconnect()  # abort on the headers frame

    async def run():
        try:
            await app(scope, receive, send)
        except _Disconnect:
            pass

    asyncio.run(run())
    assert llm._admission().acquire(blocking=False), "admission gate leaked"
    llm._admission().release()


def test_admission_busy_returns_503(client_and_llm):
    # A held runtime slot must fail fast (503 overloaded + Retry-After), never
    # park a server pool thread on acquire: a parked thread starves the sync
    # SSE generator that has to release the slot.
    client, llm = client_and_llm
    assert llm._admission().acquire(blocking=False)
    try:
        body = {"messages": [{"role": "user", "content": "hi"}]}
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 503
        assert resp.headers.get("retry-after") == "1"
        resp = client.post("/v1/chat/completions",
                           json={
                               **body, "stream": True
                           })
        assert resp.status_code == 503
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("c.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
            data={"model": "asr"})
        assert resp.status_code == 503
    finally:
        llm._admission().release()
    # Gate released: the same request now succeeds.
    resp = client.post("/v1/chat/completions",
                       json={"messages": [{
                           "role": "user",
                           "content": "hi"
                       }]})
    assert resp.status_code == 200


def test_admission_spans_decode_and_infer():
    # The admission gate must already be held while media decodes, not just
    # around handle_request: concurrent requests may not stack decoded
    # buffers while queueing on the inference lock.
    eng = _engine()
    llm = eng.LLM.__new__(eng.LLM)
    llm._rt = None
    seen = {}

    def fake_build(messages, params, **kw):
        seen["held_during_build"] = llm._admission()._value == 0
        return "req"

    def fake_handle(request):
        seen["held_during_infer"] = llm._admission()._value == 0

        class _R:
            output_texts = ["ok"]
            output_ids = [[1]]
            finish_reasons = []
            logprobs = []

        return _R()

    llm._make_generation_request = fake_build
    llm._handle_request = fake_handle
    llm._parse_generation_output = (
        lambda text, ids, reason, cfg: eng.CompletionOutput(
            text=text, token_ids=ids, finish_reason=reason))
    out = llm.generate(["hi"], eng.SamplingParams(max_tokens=4))
    assert out and seen["held_during_build"] and seen["held_during_infer"]
    assert llm._admission()._value == 1  # released afterwards


def test_stream_disconnect_keeps_gate_until_worker_exits(monkeypatch):
    # A join timeout after disconnect must NOT release admission: the C++
    # worker may still be inside prefill holding the inference lock. The
    # gate transfers to the worker and frees only when it really exits.
    eng = _engine()
    monkeypatch.setattr(eng, "_STREAM_JOIN_TIMEOUT_S", 0.05)
    llm = eng.LLM.__new__(eng.LLM)
    import threading as _t
    worker_may_exit = _t.Event()
    state = {"cancelled": False}

    class _Chunk:
        text = "t"
        token_ids = [1]
        finished = False
        reason = None
        logprobs = []

    class _Channel:

        def set_skip_special_tokens(self, flag):
            pass

        def wait_pop(self, timeout_ms=0):
            return None if state["cancelled"] else _Chunk()

        def is_finished(self):
            return False

        def is_cancelled(self):
            return state["cancelled"]

        def cancel(self):
            state["cancelled"] = True

    class _RT:

        class StreamChannel:

            @staticmethod
            def create():
                return _Channel()

    llm._rt = _RT()
    llm._handle_request = lambda request: worker_may_exit.wait(timeout=10)

    sem = llm._admission()
    assert sem.acquire(blocking=False)
    from experimental.server import api_server
    handoff = api_server._AdmissionHandoff(sem)

    class _Req:
        stream_channels = None

    gen = llm.generate_stream([],
                              eng.SamplingParams(),
                              prebuilt_request=_Req(),
                              admission_handoff=handoff)
    next(gen)  # worker started
    gen.close()  # disconnect: cancel + join times out; worker still runs
    assert not sem.acquire(blocking=False), "gate released while worker alive"
    worker_may_exit.set()
    deadline = time.time() + 5
    while time.time() < deadline:
        if sem.acquire(blocking=False):
            sem.release()
            break
        time.sleep(0.01)
    else:
        raise AssertionError("worker exit did not release the gate")


def test_stream_close_cancels_channel():
    # Closing the generator early (client disconnect) must cancel the
    # channel so the worker releases the inference lock.
    eng = _engine()
    llm = eng.LLM.__new__(eng.LLM)
    state = {"cancelled": False}

    class _Chunk:
        text = "t"
        token_ids = [1]
        finished = False
        reason = None
        logprobs = []

    class _Channel:

        def set_skip_special_tokens(self, flag):
            pass

        def wait_pop(self, timeout_ms=0):
            return None if state["cancelled"] else _Chunk()

        def is_finished(self):
            return False

        def is_cancelled(self):
            return state["cancelled"]

        def cancel(self):
            state["cancelled"] = True

    class _RT:

        class StreamChannel:

            @staticmethod
            def create():
                return _Channel()

    llm._rt = _RT()
    llm._handle_request = lambda request: None

    class _Req:
        stream_channels = None

    gen = llm.generate_stream([],
                              eng.SamplingParams(max_tokens=4),
                              prebuilt_request=_Req())
    first = next(gen)
    assert first.text == "t"
    gen.close()
    assert state["cancelled"]


def test_cli_parser_builds():
    # Guards against duplicate argparse registrations (a rebase once left two
    # --multimodal-engine-dir definitions and the server could not start;
    # the Jedha CI --help smoke failed on exactly this).
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "-m", "experimental.server.api_server", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={
            **os.environ, "PYTHONPATH": REPO_ROOT
        })
    assert r.returncode == 0, r.stderr[-500:]
    assert "--multimodal-engine-dir" in r.stdout


def test_video_frame_limits_from_engine_config(tmp_path):
    # builder_config token bounds + preprocessor geometry feed the sampler's
    # profile clamping.
    eng = _engine()
    (tmp_path / "visual").mkdir()
    (tmp_path / "visual" / "config.json").write_text(
        '{"model_type": "qwen2_5_vl", "builder_config": '
        '{"min_image_tokens": 128, "max_image_tokens": 4096, '
        '"max_image_tokens_per_image": 4096}}')
    (tmp_path / "visual" / "preprocessor_config.json").write_text(
        '{"patch_size": 14, "merge_size": 2, "temporal_patch_size": 2}')
    llm = eng.LLM.__new__(eng.LLM)
    llm._multimodal_engine_dir = str(tmp_path)
    limits = llm._video_frame_limits()
    assert limits["max_image_tokens"] == 4096
    assert limits["patch_size"] == 14
    assert limits["model_type"] == "qwen2_5_vl"


def test_video_frame_limits_carry_recorded_cu_capacity(tmp_path):
    # config/server wiring: a cu_seqlens capacity recorded by the builder in
    # builder_config must reach the limits dict the sampler and the engine
    # budget consume, taking precedence over the fallback formula.
    eng = _engine()
    (tmp_path / "visual").mkdir()
    (tmp_path / "visual" / "config.json").write_text(
        '{"model_type": "qwen3_vl", "builder_config": '
        '{"min_image_tokens": 4096, "max_image_tokens": 8192, '
        '"max_cu_seqlen_entries": 512}}')
    (tmp_path / "visual" / "preprocessor_config.json").write_text(
        '{"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2}')
    llm = eng.LLM.__new__(eng.LLM)
    llm._multimodal_engine_dir = str(tmp_path)
    limits = llm._video_frame_limits()
    assert limits["max_cu_seqlen_entries"] == 512


def test_chat_streaming_invalid_video_returns_400(chain_client):
    # The streaming path must reject invalid video BEFORE the SSE response
    # starts; without prevalidation the same input is a 400 non-streaming but
    # a 200 SSE with an in-band error.
    client, _ = chain_client
    resp = client.post("/v1/chat/completions",
                       json={
                           "messages": [{
                               "role":
                               "user",
                               "content": [{
                                   "type": "video_url",
                                   "video_url": {
                                       "url": "file:///no/such/clip.mp4"
                                   }
                               }]
                           }],
                           "stream":
                           True,
                           "max_tokens":
                           8
                       })
    assert resp.status_code == 400, resp.text


# ---------------------------------------------------------------------------
# Request-parsing chains: OpenAI JSON in -> parsed generation request out,
# asserting on buffer order, placeholder sequence, and decoded bytes.
# ---------------------------------------------------------------------------


class _CapturingRt:
    """Stub of the pybind module at the C++ boundary; records buffer loads."""

    class Message:

        def __init__(self):
            self.role = ""
            self.contents = []

    class LLMGenerationRequest:
        pass

    class Request:

        def __init__(self, messages=None):
            self.messages = messages

    @staticmethod
    def MessageContent(ctype, data=""):
        c = _Content(ctype, data)
        return c

    @staticmethod
    def load_image_from_path(path):
        return _FakeBuffer("image", path)

    @staticmethod
    def load_video_from_paths(paths, fps):
        return _FakeBuffer("video_paths", list(paths), fps)

    @staticmethod
    def load_video_from_array(frames, fps, timestamps=()):
        return _FakeBuffer("video_array", frames, fps, list(timestamps))

    @staticmethod
    def load_audio_buffer_from_bytes(raw):
        return ("audio_bytes", raw)


@pytest.fixture
def chain_client(tmp_path):
    """TestClient over a real _create_app + a real-code LLM whose only stubs
    are the pybind module and the engine runtime (captures the request)."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    try:
        from fastapi.testclient import TestClient

        from experimental.server.api_server import _create_app
        from experimental.server.engine import LLM
    except ImportError as exc:  # optional deps / pybind runtime chain
        pytest.skip(f"server imports unavailable: {exc}")

    captured = {}

    class _RT:

        @staticmethod
        def has_draft_model():
            return False

        def handle_request(self, request):
            captured["request"] = request

            class _Resp:
                output_texts = ["ok"]
                output_ids = [[1]]
                finish_reasons = []

            return _Resp()

    llm = LLM.__new__(LLM)  # bypass __init__: no engine on a unit host
    llm._rt = _CapturingRt()
    llm._runtime = _RT()
    llm._model_dir = str(tmp_path)
    llm._model_id = "chain-test"
    visual_dir = tmp_path / "mm" / "visual"
    visual_dir.mkdir(parents=True)
    (visual_dir / "config.json").write_text('{"model_type": "qwen3_vl"}')
    llm._multimodal_engine_dir = str(tmp_path / "mm")
    llm._tool_template_formatter = None
    return TestClient(_create_app(llm)), captured


def test_chat_text_request_parses(chain_client):
    client, captured = chain_client
    resp = client.post("/v1/chat/completions",
                       json={
                           "messages": [{
                               "role": "user",
                               "content": "Hello there"
                           }],
                           "max_tokens": 8,
                       })
    assert resp.status_code == 200, resp.text
    req = captured["request"].requests[0]
    assert [c.type for c in req.messages[0].contents] == ["text"]
    assert req.image_buffers == [] and req.audio_buffers == []


def test_chat_video_frames_request_parses_to_ordered_buffers(
        chain_client, tmp_path):
    client, captured = chain_client
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    f0, f1 = tmp_path / "f0.jpg", tmp_path / "f1.jpg"
    f0.write_bytes(b"x")
    f1.write_bytes(b"x")
    resp = client.post("/v1/chat/completions",
                       json={
                           "messages": [{
                               "role":
                               "user",
                               "content": [
                                   {
                                       "type": "image",
                                       "image": str(img)
                                   },
                                   {
                                       "type": "video",
                                       "frames": [str(f0), str(f1)],
                                       "fps": 1.0
                                   },
                                   {
                                       "type": "text",
                                       "text": "describe"
                                   },
                               ],
                           }],
                           "max_tokens":
                           8,
                       })
    assert resp.status_code == 200, resp.text
    req = captured["request"].requests[0]
    # Buffers arrive in message order (the C++ runner pairs them positionally
    # with the visual placeholders).
    assert req.image_buffers == [("image", str(img)),
                                 ("video_paths", [str(f0), str(f1)], 1.0)]
    # The chat-template message carries the placeholder sequence.
    types = [c.type for c in req.messages[0].contents]
    assert types == ["image", "video", "text"]


def test_chat_video_request_end_to_end(chain_client, monkeypatch):
    # One streaming request covers the whole chain: the generation request is
    # built before the SSE response and reused in the generator (exactly one
    # decode); sampler args and buffer + timestamps arrive untouched.
    client, captured = chain_client
    from experimental.server import video_sampling
    sentinel = object()
    calls = {"n": 0}
    seen = {}

    def fake_sample_video(source, **kw):
        calls["n"] += 1
        seen["source"] = source
        seen["kw"] = kw
        return sentinel, 2.0, [0.0, 0.5], 0, 0

    monkeypatch.setattr(video_sampling, "sample_video", fake_sample_video)
    resp = client.post("/v1/chat/completions",
                       json={
                           "messages": [{
                               "role":
                               "user",
                               "content": [{
                                   "type": "video",
                                   "video": "/clips/x.mp4",
                                   "nframes": 8
                               }, {
                                   "type": "text",
                                   "text": "describe"
                               }]
                           }],
                           "stream":
                           True,
                           "max_tokens":
                           8
                       })
    assert resp.status_code == 200, resp.text
    assert calls["n"] == 1
    assert seen["source"] == "/clips/x.mp4" and seen["kw"]["nframes"] == 8
    assert seen["kw"]["family"] == "qwen"
    # The streaming stub short-circuits before the C++ request; a non-stream
    # round-trip checks the buffer + timestamps reach it untouched.
    resp = client.post("/v1/chat/completions",
                       json={
                           "messages": [{
                               "role":
                               "user",
                               "content": [{
                                   "type": "video",
                                   "video": "/clips/x.mp4",
                                   "nframes": 8
                               }],
                           }],
                           "max_tokens":
                           8,
                       })
    assert resp.status_code == 200, resp.text
    req = captured["request"].requests[0]
    assert req.image_buffers == [("video_array", sentinel, 2.0, [0.0, 0.5])]


@pytest.mark.parametrize("content", [
    {
        "type": "video_url",
        "video_url": {
            "url": "https://evil/clip.mp4"
        }
    },
    {
        "type": "video_url",
        "video_url": {
            "url": "file:///no/such/clip.mp4"
        }
    },
    {
        "type": "input_audio",
        "input_audio": {}
    },
])
def test_chat_rejects_bad_media_via_api(chain_client, content):
    # The security/validation boundary holds through the full request chain:
    # remote video URLs and malformed audio come back as 400, not 500.
    client, _ = chain_client
    resp = client.post("/v1/chat/completions",
                       json={
                           "messages": [{
                               "role": "user",
                               "content": [content]
                           }],
                           "max_tokens": 8
                       })
    assert resp.status_code == 400, resp.text


def test_chat_audio_request_parses_to_bytes(chain_client):
    import base64
    client, captured = chain_client
    raw = b"RIFF0000WAVEfmt "
    resp = client.post("/v1/chat/completions",
                       json={
                           "messages": [{
                               "role":
                               "user",
                               "content": [{
                                   "type": "input_audio",
                                   "input_audio": {
                                       "data": base64.b64encode(raw).decode(),
                                       "format": "wav"
                                   },
                               }],
                           }],
                           "max_tokens":
                           8,
                       })
    assert resp.status_code == 200, resp.text
    req = captured["request"].requests[0]
    assert req.audio_buffers == [("audio_bytes", raw)]
    assert [c.type for c in req.messages[0].contents] == ["audio"]
