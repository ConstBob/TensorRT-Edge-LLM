# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OpenPI <-> Edge-LLM request mapping (no GPU, no Isaac)."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from experimental_models.cosmos3.robolab.openpi_adapter import (
    ACTION_CHUNK_SIZE,
    RAW_ACTION_DIM,
    EdgeLLMOpenPIPolicy,
    backend_result_to_openpi,
    openpi_obs_to_request,
)
from experimental_models.cosmos3.robolab.policy_server import Cosmos3PolicyBackend


class _FakeBackend:
    domain = "droid_lerobot"
    steps = 4

    def infer(self, request):
        assert request["instruction"] == "pick the mug"
        action = np.zeros((1, ACTION_CHUNK_SIZE, RAW_ACTION_DIM), np.float32)
        action[..., -1] = 0.2
        return {"action": action.tolist(), "meta": {"server_latency_s": 0.01}}


class OpenPIAdapterTest(unittest.TestCase):

    def test_obs_to_request(self):
        img = np.zeros((540, 640, 3), np.uint8)
        req = openpi_obs_to_request({
            "observation/image": img,
            "prompt": "pick the mug",
            "observation/joint_position": np.zeros(7, np.float32),
            "observation/gripper_position": np.array([0.3], np.float32),
        })
        self.assertEqual(req["instruction"], "pick the mug")
        self.assertEqual(req["domain"], "droid_lerobot")
        self.assertIs(req["image"], img)
        np.testing.assert_allclose(req["state"], [0, 0, 0, 0, 0, 0, 0, 0.7])

    def test_obs_requires_prompt(self):
        with self.assertRaises(ValueError):
            openpi_obs_to_request(
                {"observation/image": np.zeros((8, 8, 3), np.uint8)})

    def test_result_reshape(self):
        raw = {"action": np.arange(ACTION_CHUNK_SIZE * RAW_ACTION_DIM,
                                   dtype=np.float32)}
        out = backend_result_to_openpi(raw)
        self.assertEqual(out["action"].shape,
                         (ACTION_CHUNK_SIZE, RAW_ACTION_DIM))

    def test_policy_infer(self):
        policy = EdgeLLMOpenPIPolicy(_FakeBackend())  # type: ignore[arg-type]
        img = np.zeros((540, 640, 3), np.uint8)
        out = policy.infer({
            "observation/image": img,
            "prompt": "pick the mug",
            "observation/joint_position": np.zeros(7, np.float32),
            "observation/gripper_position": np.array([0.3], np.float32),
        })
        self.assertEqual(out["action"].shape,
                         (ACTION_CHUNK_SIZE, RAW_ACTION_DIM))
        self.assertIn("infer_ms", out["server_timing"])

    def test_backend_forwards_matched_recipe(self):
        with TemporaryDirectory() as root:
            binary = Path(root) / "cosmos3_policy_inference"
            binary.write_text("")
            gen = Path(root) / "engines" / "gen"
            gen.mkdir(parents=True)
            (gen / "config.json").write_text(
                '{"action_chunk_size": 32, "fps": 15.0, '
                '"raw_action_dim": 8, "state_rows": 1, "use_state": true, '
                '"action_start_frame_offset": 0}')
            backend = Cosmos3PolicyBackend(str(binary),
                                           str(Path(root) / "engines"))
            output = np.zeros((32, 8), np.float32).tolist()

            def fake_run(cmd, **_kwargs):
                out_path = Path(cmd[cmd.index("--output") + 1])
                out_path.write_text(
                    '{"action": ' + repr(output) +
                    ', "meta": {}, "shape": [1, 32, 8]}')
                return mock.Mock(returncode=0, stderr="")

            image = np.zeros((8, 8, 3), np.uint8)
            with mock.patch("subprocess.run", side_effect=fake_run) as run:
                backend.infer({
                    "image": image,
                    "instruction": "pick the mug",
                    "state": [0.0] * 8,
                })
            cmd = run.call_args.args[0]
            self.assertEqual(cmd[cmd.index("--guidance") + 1], "3.0")
            self.assertEqual(cmd[cmd.index("--viewPoint") + 1], "concat_view")
            self.assertEqual(cmd[cmd.index("--action-chunk-size") + 1], "32")
            self.assertEqual(cmd[cmd.index("--state") + 1],
                             "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0")
            self.assertIn("--seed", cmd)

    def test_backend_rejects_chunk_16_engine(self):
        with TemporaryDirectory() as root:
            binary = Path(root) / "cosmos3_policy_inference"
            binary.write_text("")
            gen = Path(root) / "engines" / "gen"
            gen.mkdir(parents=True)
            (gen / "config.json").write_text(
                '{"action_chunk_size": 16, "fps": 5.0, '
                '"raw_action_dim": 10, "state_rows": 0, "use_state": false, '
                '"action_start_frame_offset": 1}')
            with self.assertRaisesRegex(ValueError, "action_chunk_size=16"):
                Cosmos3PolicyBackend(str(binary), str(Path(root) / "engines"))


if __name__ == "__main__":
    unittest.main()
