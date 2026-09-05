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
"""Generate the Marlin-vs-Blackwell NVFP4-A16 MoE plugin benchmark inputs.

One synthetic ModelOpt-style checkpoint (random E2M1 codes, E4M3 block scales
in [0.25, 2), fp32 weight_scale_2) is repacked twice from the same bytes:
Marlin (``Nvfp4A16MoePlugin``) and ``BLACKWELL_MOE_N128_K64_V1``
(``Nvfp4A16BlackwellMoePlugin``). Router logits, hidden states and the
correction bias are written for every token count in ``--tokens`` and for two
routing distributions: ``uniform`` (N(0,1) logits) and ``skewed`` (per-expert
offsets N(0, 1.5) so a subset of experts is hot, closer to real models).

Run on any host with torch (CPU is fine):

    python genMoeBenchData.py --out /path/to/bench_data

then on Thor:

    EDGELLM_MOE_BENCH_DIR=/path/to/bench_data ./unittests/unitTestPlugins \\
        --gtest_filter='Nvfp4A16MoePluginBench*'
"""

import argparse
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                 ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tensorrt_edgellm.checkpoint.repacking import (  # noqa: E402
    repack_nvfp4_a16_blackwell_moe_experts,
    repack_nvfp4_a16_marlin_moe_experts)


def _random_expert(n, k, generator):
    packed = torch.randint(0,
                           256, (n, k // 2),
                           generator=generator,
                           dtype=torch.int64).to(torch.uint8)
    scales = torch.randint(0x28,
                           0x40, (n, k // 16),
                           generator=generator,
                           dtype=torch.int64).to(torch.int8)
    ws2 = (torch.rand(
        (1, ), generator=generator) * 0.008 + 0.004).to(torch.float32)
    return packed, scales, ws2


def _write(path, tensor):
    tensor.contiguous().numpy().tofile(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=2688)
    ap.add_argument("--inter", type=int, default=1856)
    ap.add_argument("--tokens",
                    default="1,2,4,8,16,32,64,128,256,512,1024,2048")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    e, h, i = args.experts, args.hidden, args.inter
    i_pad = (i + 127) // 128 * 128
    tokens = [int(t) for t in args.tokens.split(",")]
    g = torch.Generator().manual_seed(args.seed)
    os.makedirs(os.path.join(args.out, "marlin"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "blackwell"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "inputs"), exist_ok=True)

    fc1 = [_random_expert(i, h, g) for _ in range(e)]
    fc2 = [_random_expert(h, i, g) for _ in range(e)]
    lists = ([t[0] for t in fc1], [t[1] for t in fc1], [t[2] for t in fc1],
             [t[0] for t in fc2], [t[1] for t in fc2], [t[2] for t in fc2])

    print("repacking Marlin ...", flush=True)
    m = repack_nvfp4_a16_marlin_moe_experts(*lists, moe_inter_padded=i_pad)
    print("repacking Blackwell ...", flush=True)
    b = repack_nvfp4_a16_blackwell_moe_experts(*lists)
    names = ("fc1_qweights", "fc1_block_scales", "fc1_global_scales",
             "fc2_qweights", "fc2_block_scales", "fc2_global_scales")
    for name, tm, tb in zip(names, m, b):
        _write(os.path.join(args.out, "marlin", name + ".bin"), tm)
        _write(os.path.join(args.out, "blackwell", name + ".bin"), tb)
        print(f"  {name}: marlin {tuple(tm.shape)} {tm.dtype} | blackwell "
              f"{tuple(tb.shape)} {tb.dtype}")

    bias = (torch.randn((e, ), generator=g) * 0.05).to(torch.float32)
    _write(os.path.join(args.out, "inputs", "bias.bin"), bias)
    skew = (torch.randn((e, ), generator=g) * 1.5).to(torch.float32)
    for t in tokens:
        hidden = torch.randn((t, h), generator=g).to(torch.float16)
        logits = torch.randn((t, e), generator=g).to(torch.float32)
        _write(os.path.join(args.out, "inputs", f"uniform_T{t}_hidden.bin"),
               hidden)
        _write(os.path.join(args.out, "inputs", f"uniform_T{t}_logits.bin"),
               logits)
        _write(os.path.join(args.out, "inputs", f"skewed_T{t}_hidden.bin"),
               hidden)
        _write(os.path.join(args.out, "inputs", f"skewed_T{t}_logits.bin"),
               (logits + skew[None, :]).contiguous())

    with open(os.path.join(args.out, "manifest.txt"), "w") as f:
        f.write(f"num_experts={e}\ntop_k={args.top_k}\nhidden_size={h}\n"
                f"moe_inter_size={i}\nmoe_inter_size_padded={i_pad}\n"
                f"n_group=1\ntopk_group=1\nnorm_topk_prob=1\n"
                f"routed_scaling_factor=2.5\nseed={args.seed}\n"
                f"tokens={','.join(str(t) for t in tokens)}\n"
                f"sets=uniform,skewed\n")
    print("done:", args.out)


if __name__ == "__main__":
    main()
