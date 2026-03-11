# CuTe DSL FMHA Kernels (Blackwell SM100+)

Fused multi-head attention kernels compiled ahead-of-time from CuTe DSL Python
source. Unlike FMHA_v2 (pre-compiled cubins), these kernels are compiled during
the CMake build on a machine with a Blackwell GPU.

## Origin

`fmha.py` is derived from the CUTLASS example at
`examples/python/CuTeDSL/blackwell/fmha.py`, and `fmha_helpers.py` from
`examples/python/CuTeDSL/helpers/fmha_helpers.py`, both taken from CUTLASS commit
[`b9847690c5838ac3d909ebc163ed16c388802485`](https://github.com/NVIDIA/cutlass/commit/b9847690c5838ac3d909ebc163ed16c388802485).

Local modifications are captured in `fmha.patch`.

## Key Improvements

- **Runtime Parameter Flexibility** — converted batch size, sequence length, and
  number of heads from compile-time constants to runtime arguments, with
  negligible performance overhead.
- **Sliding Window Attention** — added sliding window attention support per
  Xiaomi's requirements.
- **Prefix Cache Optimization** — eliminated temporary KV cache allocation and
  layout conversion before FMHA, reducing memory footprint.
- **Dependency Removal** — removed PyTorch dependency for standalone C++
  execution.

## 1. Kernel Variants

The build produces eight AOT-compiled kernel objects (`.o` + `.h` pairs):

| Variant | Head Dim | SWA | Mode | Causal |
|---|---|---|---|---|
| `fmha_d64` | 64 | No | LLM | Yes |
| `fmha_d128` | 128 | No | LLM | Yes |
| `fmha_d64_sw` | 64 | Yes | LLM | Yes |
| `fmha_d128_sw` | 128 | Yes | LLM | Yes |
| `vit_fmha_d64` | 64 | No | ViT | No |
| `vit_fmha_d72` | 72 | No | ViT | No |
| `vit_fmha_d80` | 80 | No | ViT | No |
| `vit_fmha_d128` | 128 | No | ViT | No |

**LLM variants** use a fused KV cache layout `[B, 2, H_kv, S_k, D]` with causal
masking and bottom-right alignment (`WINDOW_MASK_INFERENCE`).

**ViT variants** use packed variable-length separate Q/K/V tensors
`[total_S, H, D]` with `cu_seqlens` for ragged batching, bidirectional attention.

## 2. Building

Kernel compilation happens automatically during the CMake build when
`ENABLE_CUTE_DSL_FMHA=ON`. A Blackwell GPU must be present on the build machine.

### 2.1. Prerequisites

| Dependency | Version |
|---|---|
| `nvidia-cutlass-dsl` | 4.4.1 |
| `cupy-cuda12x` | 12.3.0 (for CUDA 12.x) |
| `cupy-cuda13x` | 13.6.0 (for CUDA 13.x) |

These are installed automatically by CMake if missing or mismatched (see
`cmake/CuteDslFMHA.cmake`).

### 2.2. CMake Configuration

```bash
cmake -DENABLE_CUTE_DSL_FMHA=ON \
      -DTRT_PACKAGE_DIR=/path/to/TensorRT \
      ..
```

The `CuteDslFMHA.cmake` module:

1. Verifies / installs `nvidia-cutlass-dsl` and `cupy` pip packages.
2. Invokes `fmha.py` eight times (one per variant) with `--export_only` to produce
   `.o` and `.h` artifacts under `cpp/kernels/contextAttentionKernels/cuteDSLArtifact/`.
3. Links the `.o` files and `cute_dsl_runtime` library into the plugin shared
   library.
4. Defines `CUTE_DSL_FMHA_ENABLED` for conditional compilation.

### 2.3. Standalone Kernel Compilation

To compile a single variant outside CMake (e.g. for testing):

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

To run reference accuracy checks without exporting artifacts (`--export_only` not
set):

```bash
# LLM accuracy reference: multi-round prefill check (plugin-aligned behavior)
python3 fmha.py \
  --q_shape 1,8,8,128 --k_shape 1,64,8,128 \
  --is_causal --is_persistent --bottom_right_align

# ViT accuracy reference: single-shot packed-input check
python3 fmha.py \
  --q_shape 1,8,8,72 --k_shape 1,8,8,72 \
  --is_persistent --vit_mode
```

## 3. Runtime Integration

### 3.1. C++ Runner

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

### 3.2. Plugin Integration

The attention plugin (`cpp/plugins/attentionPlugin/attentionPlugin.cpp`) uses
CuTe DSL FMHA as the primary path on Blackwell, with automatic fallback to
FMHA_v2:

1. At construction, checks `CUTE_DSL_FMHA_ENABLED` compile flag and
   `canImplement()`.
2. Attempts to load kernel modules; falls back to FMHA_v2 on failure.
3. At runtime, CuTe DSL path uses a dedicated RoPE kernel
   (`launchApplyRopeWriteKVSplitQKV`) that writes directly into the fused KV
   cache layout `[B, 2, H_kv, S, D]`.

### 3.3. Runtime Switch Back to FMHA_v2

To force runtime fallback to FMHA_v2 (even when CuTe DSL FMHA is compiled and
available), set:

```bash
export DISABLE_CUTE_DSL_FMHA=1
```

Then run inference as usual (plugin will take the FMHA_v2 path).

To re-enable CuTe DSL FMHA (default behavior), unset it:

```bash
unset DISABLE_CUTE_DSL_FMHA
```

Notes:

- Any non-empty `DISABLE_CUTE_DSL_FMHA` value disables CuTe DSL FMHA.
- The env var is read when the plugin instance is created, so set/unset it
  before launching the process.

### 3.4. Sliding Window Attention

- Plugin attribute `sliding_window_size`: `-1` means disabled (default).
- At the C++ runtime boundary, `-1` is converted to `INT_MAX`.
- Runner dispatches to `_sw` variants when `slidingWindowSize < INT_MAX`.
- `window_size_right` is always `0` (causal-only), baked as a compile-time
  constant.
- `bottom_right_align` is always enabled, producing correct masking for both
  normal prefill and chunked prefill.

## 4. Patch Details

The patch adapts the upstream FMHA example for ahead-of-time (AOT) compilation
and integration into TensorRT Edge-LLM:

- **Replace PyTorch with CuPy/NumPy** — removes the `torch` dependency entirely;
  GPU tensor operations use CuPy and CPU reference computations use NumPy.
- **Fused KV cache layout** — instead of separate K and V tensors `(B, S, H, D)`,
  uses a single interleaved KV cache `(B, 2, H_kv, S_k, D)`, eliminating
  temporary allocation and layout conversion at runtime.
- **Tensor-based kernel API** — `__call__` now accepts `cute.Tensor` objects
  (`q_tensor`, `kv_cache`, `o_tensor`) directly, extracting problem dimensions
  from tensor shapes rather than a separate `problem_size` tuple.
- **Dynamic tensor marking** — marks batch size, sequence length, and number of
  heads as dynamic dimensions (`mark_bshd_dynamic`, `mark_kv_cache_dynamic`),
  allowing these to be runtime arguments instead of compile-time constants.
- **Compile-time sliding window dispatch** — adds a `use_sliding_window` flag;
  when `False`, `window_size_left` is passed as `None` at compile time to
  eliminate left-side window masking code for better performance.
- **AOT export support** — adds `--output_dir`, `--export_only`, `--file_name`,
  and `--function_prefix` CLI arguments. `export_to_c()` is called only when
  `--export_only` is set, producing `.h` and `.o` artifacts in `--output_dir`.
  `--export_only` also skips the reference check and benchmarking.
- **Causal-only window_size_right** — `window_size_right` is always 0 (causal)
  and set as a compile-time constant.
- **Remove variable-length sequence support** — `cum_seqlen_q`/`cum_seqlen_k`
  (nested tensor) paths are removed.
- **ViT mode** — `--vit_mode` compiles a separate variant with packed varlen
  separate Q/K/V and bidirectional (non-causal) attention for vision
  transformer workloads.

## 5. File Map

| File | Description |
|---|---|
| `kernelSrcs/fmha_cutedsl_blackwell/fmha.py` | CuTe DSL kernel source (LLM + ViT variants) |
| `kernelSrcs/fmha_cutedsl_blackwell/fmha_helpers.py` | Helper utilities from CUTLASS |
| `kernelSrcs/fmha_cutedsl_blackwell/fmha.patch` | Diff against upstream CUTLASS example |
| `kernelSrcs/fmha_cutedsl_blackwell/fp8_prescale.patch` | FP8 pre-scaling patch (future) |
| `cmake/CuteDslFMHA.cmake` | CMake module for build-time kernel compilation |
| `cpp/kernels/contextAttentionKernels/cuteDslFMHARunner.h` | C++ runner header |
| `cpp/kernels/contextAttentionKernels/cuteDslFMHARunner.cpp` | C++ runner implementation |
| `cpp/plugins/attentionPlugin/attentionPlugin.cpp` | TRT plugin integration |
| `cpp/kernels/posEncoding/applyRopeWriteKV.cu` | RoPE kernel for CuTe DSL KV layout |
