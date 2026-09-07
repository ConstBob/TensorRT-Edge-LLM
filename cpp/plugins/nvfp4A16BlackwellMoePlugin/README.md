# Nvfp4A16BlackwellMoePlugin

Thor (SM110) plugin for NVFP4-weight / FP16-activation routed MoE (issue #944).
It replaces `Nvfp4A16MoePlugin` (Marlin) for Thor exports only; every other
platform keeps the Marlin plugin unchanged. The two plugins have different
weight layouts and distinct ONNX identities: an engine never carries both.

## Supported contract

| Field | Value |
|---|---|
| SM | 110 only (rejected at `configurePlugin` elsewhere) |
| hidden_states / output | FP16 `[B, S, H]`, `H % 128 == 0` |
| router_logits / expert_score_bias | FP32 `[B*S, E]` / `[E]` |
| `activation_type` | 4 (ReLU2) |
| `routing_mode` | 1 (sigmoid group top-k); `n_group`, `topk_group`, `norm_topk_prob`, `routed_scaling_factor` as in `Nvfp4A16MoePlugin` |
| `num_experts` / `top_k` | {128, 256, 512} / 1..32 |
| `moe_inter_size` | logical I, `I % 64 == 0`; FC1 N is padded to 128 inside the layout |
| weights | `fc{1,2}_qweights` int8 `[E, N/128, K/64, 128, 32]`, `fc{1,2}_block_scales` int8 `[E, N/128, K/64, 128, 4]`, `fc{1,2}_global_scales` fp32 `[E]` (`BLACKWELL_MOE_N128_K64_V1`; each 32-byte code row carries the TMA 32B swizzle image: rows 4-7 of every 8 swap their 16-byte halves) |
| `layout` | 1 (`BLACKWELL_MOE_N128_K64_V1`) |
| `backend` | 0 auto, 1 force decode kernels, 2 force grouped tcgen05 GEMM |
| `max_routed_rows` | padded permuted-row capacity, 0 = resolve from the profile (`T*top_k + E*127`, rounded to 128) |

## Execution

* `T = 1` (decode): warp-per-token sigmoid top-k routing (contracts with
  `n_group > 1` or `E > 512` use the shared `moeSigmoidGroupTopk` instead) ->
  `decodeFc1Kernel` (dequant GEMV over the routed expert, split-K 2) + reduce
  (alpha, ReLU2) -> `decodeFc2Kernel` (per-slot alpha * router weight, fp32
  accumulation, split-K 8) + reduce: 5 GPU ops, deterministic. FC1 is
  launch-bounded to two CTAs per SM and FC2 to three; routing is a separate
  kernel because fusing it into every FC1 CTA cost 131 registers (one CTA per
  SM) and ~40% of FC1's streaming bandwidth inside the engine. The decode
  workspace is sized for the largest FC1 split-K the benchmark override
  (`EDGELLM_MOE_DECODE_FC1_SPLITK`) can select, so the size TensorRT records at
  build time never depends on the environment.
* **Programmatic Dependent Launch**: every kernel above issues `griddepcontrol.wait`
  before its first read of data a previous kernel (or the preceding TensorRT
  layer) produced and `griddepcontrol.launch_dependents` once its outputs are
  written; the runner launches them with
  `cudaLaunchAttributeProgrammaticStreamSerialization` (CUDA kernels through
  `cudaLaunchKernelEx`, the grouped GEMMs through the AOT wrapper's `enable_pdl`
  argument) so each kernel's prologue overlaps the previous kernel's tail. On by
  default; `EDGELLM_ENABLE_PDL=0` (read once, before the first enqueue) disables
  it for an A/B, the same knob as `Nvfp4MoePlugin`. The shared grouped-routing
  kernels used for `n_group > 1` carry no wait and are launched without the
  attribute, so that contract simply serializes. Measured on Thor (CUDA-graph
  decode step, Nemotron 3.5 Lightning): consecutive plugin kernels now start
  0.5-5.6 us before their predecessor ends (nsys), which is worth about 1% of
  the prefill step at ISL 2048 and is within noise at decode, because the
  dependent kernels' pre-wait prologue is short; a pre-wait weight prefetch in
  the decode kernels is the follow-up that would turn the overlap into
  bandwidth.
* `T >= 2` (prefill and batched decode): warp-per-token sigmoid top-k routing -> single-CTA
  expert-contiguous tile layout (`permuted_idx`, `tile_group_idx`,
  `num_valid_tiles`) -> permuted-row gather (routed rows only; pad rows are
  never read into anything that survives) + output zeroing -> FC1 grouped GEMM
  (ReLU2 fused) -> FC2 grouped GEMM (router weight + scatter-add fused): 5 GPU
  ops. Token tile (per-expert padding granularity) tn8 up to 16 tokens, tn16 up
  to 32, tn32 up to 256, tn64 up to 2048, tn128 above. Contracts
  with `n_group > 1` fall back to the shared `moeSigmoidGroupTopk` +
  `buildLayoutGpu` pair for the first two ops.
* Modules are loaded in `onShapeChange` (`Nvfp4A16BlackwellMoeRunner::prepare`);
  `enqueue` never loads modules or queries the device, so CUDA-graph capture is
  safe after one uncaptured warmup.

## Files

* `cpp/plugins/nvfp4A16BlackwellMoePlugin/nvfp4A16BlackwellMoePlugin.{h,cpp}`
* `cpp/kernels/moe/nvfp4A16BlackwellMoe/` — runner, dispatch policy, decode
  kernels, routing / layout / gather support kernels, device routing math
* `kernelSrcs/nvfp4_a16_blackwell_moe/` — CuTe DSL grouped GEMM (AOT group
  `nvfp4_a16_blackwell_moe`) and the on-board oracle
* `tensorrt_edgellm/checkpoint/repacking.py` — `repack_nvfp4_a16_blackwell_moe_experts`

## Validation

* `tests/python-unittests/test_nvfp4_a16_blackwell_moe_layout.py` — layout pin (CPU).
* `tests/python-unittests/test_nvfp4_a16_blackwell_moe_plugin.py` — engine build,
  serialization round-trip, decode/prefill numerics (SM110) and rejections.
* `unittests/cpp/kernels/moe/nvfp4A16BlackwellMoeRunnerTests.cu` — runner vs
  double-precision reference, CUDA-graph replay, dispatch policy.
* `unittests/cpp/plugins/nvfp4A16BlackwellMoePlugin/` — creator contract and the
  Marlin-vs-Blackwell plugin benchmark (`EDGELLM_MOE_BENCH_DIR`).

## Thor sign-off checklist

1. `python kernelSrcs/build_cutedsl.py --kernels ALL --gpu_arch sm_110 --arch aarch64`
2. `cmake .. -DTRT_PACKAGE_DIR=/usr -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake -DEMBEDDED_TARGET=jetson-thor -DCUDA_CTK_VERSION=13.2 -DENABLE_CUTE_DSL=ALL -DBUILD_UNIT_TESTS=ON`
3. `./unittests/unitTestKernelsMoe --gtest_filter='Nvfp4A16BlackwellMoe*'`
4. `./unittests/unitTestPlugins --gtest_filter='Nvfp4A16BlackwellMoePlugin*'`
5. `pytest tests/python-unittests/test_nvfp4_a16_blackwell_moe_plugin.py`
