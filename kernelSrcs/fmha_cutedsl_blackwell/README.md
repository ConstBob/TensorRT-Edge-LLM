# CuTe DSL FMHA Kernels (Blackwell SM10X/SM110)

Fused multi-head attention kernels compiled ahead-of-time from CuTe DSL Python
source. Kernel artifacts (static library + headers) are generated locally by
`kernelSrcs/build_cutedsl.py`. CMake simply links those local artifacts — no
Python, CUTLASS DSL, CuPy, or Blackwell GPU is needed at CMake build time.

> **Dependencies, `build_cutedsl.py` options, CMake integration, and
> cross-compiling/runtime deployment are shared across all CuTe DSL groups and
> documented in [`kernelSrcs/README.md`](../README.md).** This document covers
> only FMHA-specific details.

## Supported Hardware

| GPU | SM | Status |
|---|---|---|
| Blackwell datacenter (B200, GB200) | SM100 / SM103 | Primary target |
| NVIDIA Thor | SM110 | Cross-compile target (aarch64) |

## Kernel Variants

The build produces AOT-compiled kernel objects (`.o` + `.h` pairs):

| Variant | Head Dim | SWA | Mode | Causal |
|---|---|---|---|---|
| `fmha_d64` | 64 | No | LLM | Yes |
| `fmha_d128` | 128 | No | LLM | Yes |
| `fmha_d256` | 256 | No | LLM | Yes |
| `fmha_d64_sw` | 64 | Yes | LLM | Yes |
| `fmha_d128_sw` | 128 | Yes | LLM | Yes |
| `fmha_d256_sw` | 256 | Yes | LLM | Yes |
| `fmha_d64_fp8` | 64 | No | LLM (FP8) | Yes |
| `fmha_d128_fp8` | 128 | No | LLM (FP8) | Yes |
| `fmha_d256_fp8` | 256 | No | LLM (FP8) | Yes |
| `fmha_d64_sw_fp8` | 64 | Yes | LLM (FP8) | Yes |
| `fmha_d128_sw_fp8` | 128 | Yes | LLM (FP8) | Yes |
| `fmha_d256_sw_fp8` | 256 | Yes | LLM (FP8) | Yes |
| `fmha_d64_paged` | 64 | No | LLM (paged) | Yes |
| `fmha_d128_paged` | 128 | No | LLM (paged) | Yes |
| `fmha_d256_paged` | 256 | No | LLM (paged) | Yes |
| `fmha_d64_sw_paged` | 64 | Yes | LLM (paged) | Yes |
| `fmha_d128_sw_paged` | 128 | Yes | LLM (paged) | Yes |
| `fmha_d256_sw_paged` | 256 | Yes | LLM (paged) | Yes |
| `fmha_d64_paged_fp8` | 64 | No | LLM (paged+FP8) | Yes |
| `fmha_d128_paged_fp8` | 128 | No | LLM (paged+FP8) | Yes |
| `fmha_d256_paged_fp8` | 256 | No | LLM (paged+FP8) | Yes |
| `fmha_d64_sw_paged_fp8` | 64 | Yes | LLM (paged+FP8) | Yes |
| `fmha_d128_sw_paged_fp8` | 128 | Yes | LLM (paged+FP8) | Yes |
| `fmha_d256_sw_paged_fp8` | 256 | Yes | LLM (paged+FP8) | Yes |
| `vit_fmha_d64` | 64 | No | ViT | No |
| `vit_fmha_d72` | 72 | No | ViT | No |
| `vit_fmha_d80` | 80 | No | ViT | No |
| `vit_fmha_d128` | 128 | No | ViT | No |

The D256 variants use a dedicated TMEM and pipeline layout selected before
`cute.compile`; D256 ViT is not supported.

**LLM variants** use a fused KV cache layout `[B, 2, H_kv, S_k, D]` with causal
masking and bottom-right alignment (`WINDOW_MASK_INFERENCE`).

**ViT variants** use packed variable-length separate Q/K/V tensors
`[total_S, H, D]` with `cu_seqlens` for ragged batching, bidirectional attention.

## Artifact Development

If you modify this kernel or its registry entries, manually regenerate the
`fmha` group before running CMake. Otherwise, CMake uses the matching prebuilt
tarball by default. Follow the shared
[CuTe DSL kernel development workflow](../README.md#cute-dsl-kernel-development-workflow)
for the supported Docker and local-venv commands, dependency versions,
cross-compilation, artifact layout, and CMake configuration.

CMake defines `CUTE_DSL_FMHA_ENABLED` when the generated metadata contains this
group.

## Standalone Test / Export

```bash
cd kernelSrcs/fmha_cutedsl_blackwell

# LLM d128, no sliding window
python3 fmha.py \
  --q_shape 1,1024,14,128 --k_shape 1,1024,1,128 \
  --is_causal --is_persistent --bottom_right_align \
  --export_only --output_dir ./out --file_name fmha_d128 --function_prefix fmha_d128

# LLM d64, with sliding window
python3 fmha.py \
  --q_shape 1,1024,14,64 --k_shape 1,1024,1,64 \
  --is_causal --is_persistent --bottom_right_align \
  --window_size 4096,-1 \
  --export_only --output_dir ./out --file_name fmha_d64_sw --function_prefix fmha_d64_sw

# ViT d64
python3 fmha.py \
  --q_shape 1,1024,14,64 --k_shape 1,1024,14,64 \
  --is_persistent --vit_mode \
  --export_only --output_dir ./out --file_name vit_fmha_d64 --function_prefix vit_fmha_d64
```

Each invocation produces `<file_name>.h` and `<file_name>.o` in `--output_dir`.

To run reference accuracy checks (without `--export_only`):

```bash
# LLM accuracy reference
python3 fmha.py \
  --q_shape 1,8,8,128 --k_shape 1,64,8,128 \
  --is_causal --is_persistent --bottom_right_align

# ViT accuracy reference
python3 fmha.py \
  --q_shape 1,8,8,72 --k_shape 1,8,8,72 \
  --is_persistent --vit_mode
```

For an AArch64 (Thor) host-target export, `export_fmha_aarch64.sh` wraps the
above through `kernelSrcs/cutedsl_utils/cutedsl_compile_wrapper.py` (see the shared
[cross-compile section](../README.md#cross-compiling-for-aarch64-thor-and-runtime-deployment)).

## C++ Integration

`CuteDslFMHARunner` (`cpp/kernels/contextAttentionKernels/cuteDslFMHARunner.{h,cpp}`)
provides the C++ interface:

- **Module loading**: `loadLLMKernelModule()` / `loadViTKernelModule()` — loads
  the AOT-compiled CUDA libraries. Thread-safe (static, guarded by mutex).
- **Dispatch**: `canImplement(headSize, smVersion)` — returns `true` for
  SM >= 100 and head dim 64 or 128.
- **LLM run**: `run(qPtr, kvPtr, oPtr, cuKVSeqLens, stream, slidingWindowSize)`
  — dispatches to the appropriate d64/d128 + SWA/non-SWA variant.
- **ViT run**: `run(qPtr, kPtr, vPtr, oPtr, cuSeqLens, totalSeqLen, maxSeqLen, batchSize, stream)`
  — dispatches to the appropriate d64/d72/d80/d128 variant.

Plugin (`cpp/plugins/attentionPlugin/attentionPlugin.cpp`): uses CuTe DSL FMHA
as the primary path on Blackwell, with automatic fallback to FMHA_v2.

### Sliding Window Attention

- Plugin attribute `sliding_window_size`: `-1` means disabled (default).
- At the C++ runtime boundary, `-1` is converted to `INT_MAX`.
- Runner dispatches to `_sw` variants when `slidingWindowSize < INT_MAX`.
- `window_size_right` is always `0` (causal-only), baked as a compile-time
  constant.
- `bottom_right_align` is always enabled, producing correct masking for both
  normal prefill and chunked prefill.

## Origin

`fmha.py` is derived from the CUTLASS example at
`examples/python/CuTeDSL/blackwell/fmha.py`, and `fmha_helpers.py` from
`examples/python/CuTeDSL/helpers/fmha_helpers.py`, both taken from CUTLASS commit
[`b9847690c5838ac3d909ebc163ed16c388802485`](https://github.com/NVIDIA/cutlass/commit/b9847690c5838ac3d909ebc163ed16c388802485).

Key adaptations from upstream:
- Replaced PyTorch with CuPy/NumPy
- Fused KV cache layout `(B, 2, H_kv, S_k, D)` instead of separate K/V
- Dynamic batch/seq_len/nheads as runtime arguments
- Sliding window attention with compile-time dispatch
- ViT mode with packed varlen bidirectional attention
- AOT export via `export_to_c()`

## Skip-Softmax (BLASST) Threshold Calibration

The kernel implements BLASST skip-softmax ([arXiv:2512.12087](https://arxiv.org/abs/2512.12087)):
with `skip_softmax_threshold` (lambda) set at construction, a KV tile whose local
row max falls below the running max by more than `ln(lambda)` is skipped whole
(exp / row-sum / P*V elided). `None` (default) compiles the feature out — the
kernel is bit-identical to the dense build. Restricted to plain causal attention
(constructor assert; no sliding window, no ViT/bidirectional) and used by the
prefill/context path only.

`calibrate_skip_softmax.py` covers the full lambda lifecycle with two
subcommands and staged, verbose output:

```
calibrate (default) ── ModelOpt official calibration ──▶ a, b, deploy lambda
      │                                                        │
      │                                    bake lambda into build_cutedsl.py,
      │                                    rebuild artifact + relink (manual)
      ▼                                                        ▼
evaluate ── RULER accuracy of the deployed engine ──▶ PASS/FAIL + recommendation
```

### `calibrate` — lambda via ModelOpt (official)

A fixed lambda yields wildly different sparsity across context lengths, so the
threshold follows `lambda = scale_factor / L` with a model-specific scale
factor. The subcommand wraps the official calibration in
`modelopt.torch.sparsity.attention_sparsity` (the same machinery behind
TensorRT-LLM's `threshold_scale_factor`): ModelOpt auto-generates a RULER
calibration set (default 24 samples across power-of-2 length bins), runs one
forward pass evaluating 20 built-in threshold trials at once, and fits
`scale_factor = a * exp(b * sparsity)` with scipy. Requires `torch`,
`transformers`, `nvidia-modelopt`, `scipy`, `wonderwords`; the model loads
with `attn_implementation="eager"`.

```bash
python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py calibrate \
    --model-dir /path/to/Qwen3-1.7B --max-seqlen 4096 \
    --target-sparsity 0.3 0.5 --max-context 4096 \
    --cache-dir /path/with/room/modelopt-cache   # RULER gen cache; ModelOpt
                                                 # defaults to ~/.cache (quota!)
# [calibrate 1/3] load model ... [calibrate 2/3] ModelOpt calibration
#   (library output is dim and '│'-indented, this tool's lines are plain)
# [calibrate 3/3] fitted parameters and deployment thresholds
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ a = 115.037   b = 4.6992   R^2 = 0.733   (278 points)         ┃
# ┃ observed sparsity range: [10.3%, 74.7%]  (beyond = extrapolated)
# ┃ target  30%  max_ctx 4096    lambda = 0.115008  (log2 -3.12)  ┃
# ┃ target  50%  max_ctx 4096    lambda = 0.294372  (log2 -1.76)  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

ModelOpt's sparsity is a simulated, all-layers-pooled metric — the deployed
kernel's per-layer skip ratio at the same lambda can differ substantially.
Treat the calibrated lambda as the ecosystem-consistent starting point and let
`evaluate` arbitrate which target actually deploys.

### Deploying a candidate lambda

`lambda` is baked at AOT-compile time (there is no runtime knob): add
`--skip_softmax_threshold <lambda>` to the `fmha_d64`/`fmha_d128` variant args
in `kernelSrcs/build_cutedsl.py`, rebuild the artifact
(`build_cutedsl.py --kernels fmha`), and relink with `ENABLE_CUTE_DSL=fmha`.

### `evaluate` — RULER accuracy verdict for the deployed engine

The paper's accuracy instrument is RULER (its ~50%-sparsity safe-zone
conclusions come from it; retrieval-style tasks degrade first). The subcommand
samples real RULER items (HF `simonjegou/ruler`, tokenizer-filtered to the
engine's max input length), runs the deployed engine greedily, scores by
exact-answer matching per task, and — given a baseline — prints a PASS/FAIL
verdict plus a deployment recommendation (exit code follows, so it can gate
CI). Pair with `llm_bench --mode prefill` for TTFT.

```bash
# 1) dense baseline: save its scores
python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py evaluate \
    --model-dir /path/to/Qwen3-1.7B \
    --engine-dir engines/qwen3-1.7b --llm-inference build/examples/llm/llm_inference \
    --max-context 4096 --save-results ruler_dense.json

# 2) each skip build: compare, get the verdict
python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py evaluate \
    --model-dir /path/to/Qwen3-1.7B \
    --engine-dir engines/qwen3-1.7b --llm-inference build/examples/llm/llm_inference \
    --max-context 4096 --baseline ruler_dense.json --label "lambda=0.115"
# ...per-task score table with baseline/delta columns...
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ 
# ┃ VERDICT: PASS [lambda=0.115]                          ┃
# ┃ overall  0.7685 -> 0.7653   drop +0.0032  (gate 0.03) ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# 
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ VERDICT: FAIL [lambda=0.294]                          ┃
# ┃ overall  0.7685 -> 0.7147   drop +0.0537  (gate 0.03) ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# RECOMMENDATION: this build [lambda=0.115] is validated for deployment. ...
```

### Reference result (Qwen3-1.7B NVFP4, max_context 4096, B200)

Calibrated `a = 115.0, b = 4.70` (R² 0.73). RULER 200 samples x 10 tasks,
gate 0.03; TTFT from `llm_bench --mode prefill --inputLen 4096`:

| build | RULER overall | verdict | TTFT S=4096 |
|---|---|---|---|
| dense | 0.7685 | baseline | 12.18 ms |
| lambda=0.115 (target 30%) | 0.7653 (-0.003) | **PASS** | 11.70 ms (1.04x) |
| lambda=0.294 (target 50%) | 0.7147 (-0.054, qa/multiquery collapse) | **FAIL** | 11.68 ms (1.04x) |

Kernel time is threshold-insensitive at these shapes, so deploy the SMALLEST
lambda that passes the gate — a larger lambda buys no speed and only spends
accuracy margin.

## File Map

| File | Description |
|---|---|
| `kernelSrcs/fmha_cutedsl_blackwell/fmha.py` | CuTe DSL kernel source (LLM + ViT variants) |
| `kernelSrcs/fmha_cutedsl_blackwell/fmha_helpers.py` | Helper utilities from CUTLASS |
| `kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py` | Skip-softmax threshold scale-factor calibration tool |
| `kernelSrcs/fmha_cutedsl_blackwell/fmha.patch` | Diff against upstream CUTLASS example |
| `kernelSrcs/fmha_cutedsl_blackwell/fp8_prescale.patch` | FP8 pre-scaling patch (future) |
| `kernelSrcs/fmha_cutedsl_blackwell/export_fmha_aarch64.sh` | AArch64 host-target standalone export using the CuTe DSL compile wrapper |
| `kernelSrcs/build_cutedsl.py` | Unified pre-build script: compiles all CuTe DSL kernel groups |
| `cmake/CuteDsl.cmake` | Unified CMake module: validates and links prebuilt artifacts |
| `cpp/kernels/cuteDSLArtifact/{arch}/{artifact_tag}/` | Local artifacts generated by `build_cutedsl.py` |
| `cpp/kernels/contextAttentionKernels/cuteDslFMHARunner.h` | C++ runner header |
| `cpp/kernels/contextAttentionKernels/cuteDslFMHARunner.cpp` | C++ runner implementation |
| `cpp/plugins/attentionPlugin/attentionPlugin.cpp` | TRT plugin integration |
| `cpp/kernels/posEncoding/applyRopeWriteKV.cu` | RoPE kernel for CuTe DSL KV layout |
