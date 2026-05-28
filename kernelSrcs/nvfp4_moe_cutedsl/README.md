# SM110 NVFP4 MoE CuTeDSL Kernels

This directory contains the Thor SM110 split FC1/FC2 CuTeDSL backend used by
the unified `Nvfp4MoePlugin`. It keeps the plugin tensor ABI and exports the
`nvfp4_moe` artifact group from `kernelSrcs/build_cutedsl.py` (today the only
variants in that group target SM110/Thor). The SM120 / SM121 fused decode +
prefill kernels live in
[`kernelSrcs/nvfp4_fused_moe_cutedsl/`](../nvfp4_fused_moe_cutedsl/) under the
same plugin.

The legacy `nvfp4_moe` prefill/decode export sources were removed from this
directory.

## Tensor Contract

- FC1 weights: `[E, N1, H / 2]`
- FC1 scales: `[E, ceil(N1 / 128), ceil((H / 16) / 4), 32, 4, 4]`
- FC2 weights: `[E, H, I / 2]`
- FC2 scales: `[E, ceil(H / 128), ceil((I / 16) / 4), 32, 4, 4]`
- Grouped MoE metadata: tile-to-expert, tile limits, and permuted-to-expanded row mapping

The AOT group is specialized for the current Thor Qwen-style plugin contract:
`num_experts=128` and `top_k=8`. Hidden size and intermediate size remain
runtime dimensions. The SM110 runner requires `H % 128 == 0`, `I % 64 == 0`,
and `FC1_N % 128 == 0` (`FC1_N = 2 * I` for SwiGLU, `I` for ReLU2).

## Build

Set up the AOT Python environment on Thor:

```bash
python3 -m pip install 'nvidia-cutlass-dsl[cu13]==4.5.1' cupy-cuda13x==13.6.0 cuda-python
python3 kernelSrcs/nvfp4_moe_cutedsl/patch_cutlass_dsl_sm110a.py --check
```

Apply the patch first if the check fails:

```bash
python3 kernelSrcs/nvfp4_moe_cutedsl/patch_cutlass_dsl_sm110a.py
python3 kernelSrcs/nvfp4_moe_cutedsl/patch_cutlass_dsl_sm110a.py --check
```

Generate the split FC1/FC2 artifact pack:

```bash
python3 kernelSrcs/build_cutedsl.py \
  --kernels nvfp4_moe \
  --gpu_arch sm_110 \
  --arch aarch64 \
  --clean
```

Build the plugin with:

```bash
cmake -S . -B build \
  -DENABLE_CUTE_DSL=nvfp4_moe \
  -DCMAKE_CUDA_ARCHITECTURES=110a \
  ...
```

## Exported Variants

- `nvfp4_moe_sm110_fc1_relu2_n{128,256}`
- `nvfp4_moe_sm110_fc1_swiglu_n{128,256}`
- `nvfp4_moe_sm110_fc2_n{128,256}_fp16`

## Probe

No standalone reference probe is shipped. The SM110 NVFP4 MoE contract is
validated end-to-end through
[`tests/python-unittests/test_nvfp4_moe_sm110_plugin_accuracy.py`](../../tests/python-unittests/test_nvfp4_moe_sm110_plugin_accuracy.py)
(opt-in via `EDGELLM_RUN_SM110_PLUGIN_ACCURACY=1`).

## File Map

| File | Description |
|---|---|
| `blockscaled_contiguous_gather_grouped_gemm_act_fusion.py` | FC1 gather grouped GEMM + activation + FP4 requant kernel |
| `blockscaled_contiguous_grouped_gemm_finalize_fusion.py` | FC2 grouped GEMM + router-scale finalize/scatter kernel |
| `export_fc1_kernel.py` | FC1 AOT export script |
| `export_fc2_kernel.py` | FC2 AOT export script |
| `export_common.py` | Shared AOT export helpers |
| `custom_pipeline.py` | SM110 CuTeDSL pipeline helper |
| `cute_utils.py` | CuTeDSL utility helpers |
| `moe_compat.py` | Compatibility helpers for the split SM110 path |
| `patch_cutlass_dsl_sm110a.py` | Idempotent CuTeDSL 4.5.1 SM110a patch/preflight script |
