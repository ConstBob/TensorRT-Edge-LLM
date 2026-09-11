#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenPI websocket front-end for the Edge-LLM Cosmos3 policy.

RoboLab Isaac workers pass ``POLICY_ENDPOINT_URL=wss://…`` into
``policies/cosmos3/client.py`` (OpenPI msgpack). This process is the B-arm
counterpart of cosmos-framework ``action_policy_server_robolab``: same port
8000 / ``/healthz``, same first-frame metadata + infer loop.

Requires: ``websockets``, ``msgpack``, ``numpy``, ``Pillow`` (image decode).
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http
import logging
import traceback

import msgpack
import numpy as np
import websockets.asyncio.server as ws_server
import websockets.frames

from .openpi_adapter import EdgeLLMOpenPIPolicy
from .policy_server import Cosmos3PolicyBackend

logger = logging.getLogger("cosmos3.robolab.openpi")


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"],
                          dtype=np.dtype(obj[b"dtype"]),
                          shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_Packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


class OpenPIPolicyServer:
    """Same framing as ``openpi.serving.websocket_policy_server.WebsocketPolicyServer``."""

    def __init__(self, policy: EdgeLLMOpenPIPolicy, host: str, port: int) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = policy.metadata()

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with ws_server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
        ) as server:
            logger.info("OpenPI Cosmos3 policy on ws://%s:%d", self._host,
                        self._port)
            await server.serve_forever()

    async def _handler(self, websocket):
        logger.info("connection from %s", websocket.remote_address)
        packer = _Packer()
        await websocket.send(packer.pack(self._metadata))
        while True:
            try:
                obs = _unpackb(await websocket.recv())
                action = self._policy.infer(obs)
                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                logger.info("connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        description="Edge-LLM Cosmos3 RoboLab policy server (OpenPI websocket)")
    ap.add_argument("--binary",
                    required=True,
                    help="path to cosmos3_policy_inference")
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--domain", default="droid_lerobot")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--viewpoint", default="concat_view")
    ap.add_argument("--action-chunk-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    backend = Cosmos3PolicyBackend(args.binary, args.engine_dir, args.domain,
                                   args.steps, args.guidance, args.viewpoint,
                                   args.action_chunk_size, args.seed)
    OpenPIPolicyServer(EdgeLLMOpenPIPolicy(backend), args.host,
                       args.port).serve_forever()


if __name__ == "__main__":
    main()
