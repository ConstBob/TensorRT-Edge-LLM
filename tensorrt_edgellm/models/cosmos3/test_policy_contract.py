# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the released Policy-DROID export contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .export import _policy_defaults
from .modeling_gen import Cosmos3GenConfig, make_gen_config


class PolicyContractTest(unittest.TestCase):

    def test_policy_metadata_fills_droid_representation(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "checkpoint.json").write_text(
                json.dumps({
                    "policy": {
                        "action_chunk_size": 32,
                        "conditioning_fps": 15.0,
                        "domain_name": "droid_lerobot",
                    }
                }))
            policy = _policy_defaults(root)
        self.assertEqual(policy["raw_action_dim"], 8)
        self.assertTrue(policy["use_state"])

    def test_gen_contract_has_clean_state_row(self):
        cfg = Cosmos3GenConfig(action_chunk_size=33, latent_t=9)
        transformer = {
            "rope_theta": 100000000.0,
            "rope_scaling": {
                "mrope_section": [24, 20, 20]
            },
        }
        contract = make_gen_config(cfg,
                                   transformer,
                                   max_und_len=512,
                                   fps=15.0,
                                   future_action_chunk_size=32,
                                   raw_action_dim=8,
                                   use_state=True)
        self.assertEqual(contract["action_chunk_size"], 32)
        self.assertEqual(contract["action_token_count"], 33)
        self.assertEqual(contract["state_rows"], 1)
        self.assertEqual(contract["history_length"], 1)
        self.assertTrue(contract["use_state"])
        self.assertEqual(contract["raw_action_dim"], 8)
        self.assertEqual(contract["action_start_frame_offset"], 0)
        self.assertEqual(
            contract["optimization_profile"]["action_latent"]["opt"],
            [1, 33, 64])


if __name__ == "__main__":
    unittest.main()
