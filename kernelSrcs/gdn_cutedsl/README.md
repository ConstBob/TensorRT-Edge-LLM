# CuTe DSL GDN Kernels (Ampere SM80+)

Prefill and decode kernels for Gated Delta Net (GDN), AOT-compiled from CuTe
DSL Python source. Prebuilt artifacts (static library + headers) are checked
into the repo; CMake links them directly — no Python or GPU needed at build time.

Adapted from [SGLang CuTe DSL GDN kernel](https://github.com/sgl-project/sglang/blob/v0.5.9/python/sglang/jit_kernel/cutedsl_gdn.py)
(Apache-2.0). Local changes: remove PyTorch dependency, add prefill support,
AOT export and C++ plugin integration.

## Kernel Variants

| Variant | seq_len | Notes |
|---|---|---|
| `gdn_decode` | 1 | small/large-batch dispatch at runtime (threshold n=32) |
| `gdn_prefill` | > 1 | per-row context masking via `context_lengths` |

Both target SM80+; tile sizes K=128, V=128 are compile-time constants, all
other shapes are dynamic.

## Building Prebuilt Artifacts

Run on a machine with SM80+ GPU and CUDA 12.x/13.x:

```bash
pip install nvidia-cutlass-dsl==4.4.1
pip install cupy-cuda12x==12.3.0   # or cupy-cuda13x==13.6.0 for CUDA 13

cd tensorrt-edge-llm

# With explicit SM (x86 or cross-compile):
python kernelSrcs/build_cutedsl.py --kernels gdn --gpu_arch sm_87 [--clean]

# On-device (device-native SM, e.g. Thor SM110):
python kernelSrcs/build_cutedsl.py --kernels gdn [--clean]
```

Output under `cpp/kernels/cuteDSLArtifact/{arch}/`:

```
libcutedsl_{arch}.a    — decode + prefill .o + libcuda_dialect_runtime_static.a
metadata.json          — build provenance (groups, variants, CUDA ver, DSL ver)
include/
    cutedsl_all.h      — umbrella header
    gdn_decode.h
    gdn_prefill.h
```

Commit the `cuteDSLArtifact/{arch}/` directory so downstream builds need no
Python or GPU.

Key script flags: `--kernels gdn`, `--gpu_arch` (e.g. `sm_87` for Orin,
omit for device-native SM on Thor), `--arch` (default: auto), `--verbose`,
`--clean`.

> **Note on Thor (SM110):** due to a cutlass-dsl 4.4.1 PIC relocation limitation,
> `--gpu_arch` cannot be specified when building on Thor. Omit it to compile for
> the device's native SM.

## CMake

```bash
cmake -DENABLE_CUTE_DSL=gdn ...
```

`cmake/CuteDsl.cmake` reads `metadata.json`, validates the prebuilt artifacts,
and links `libcutedsl_{arch}.a`; defines `CUTE_DSL_GDN_ENABLED`. Fails with a
clear error if artifacts are missing.

To enable both GDN and FMHA:

```bash
cmake -DENABLE_CUTE_DSL=ALL ...
```

## Standalone Test / Export

```bash
cd kernelSrcs/gdn_cutedsl

# accuracy check
python3 gdn_decode.py --n 4 --h 8 --hv 8 --k 128 --v 128
python3 gdn_prefill.py --n 8 --h 8 --hv 8 --k 128 --v 128 --seq_len 16

# AOT export (single variant)
python3 gdn_decode.py --export_only --output_dir ./out --file_name gdn_decode --function_prefix gdn_decode
python3 gdn_prefill.py --export_only --output_dir ./out --file_name gdn_prefill --function_prefix gdn_prefill
```

`context_lengths_preset` options — decode: `all_ones`, `first_half_active`;
prefill: `full`, `half`, `staggered`.

## Tensor Shapes

| Tensor | Decode | Prefill | Dtype |
|---|---|---|---|
| `q`, `k` | `(N,1,H,K)` | `(N,T,H,K)` | FP16 |
| `v`, `o` | `(N,1,HV,V)` | `(N,T,HV,V)` | FP16 |
| `a`, `b` | `(N,1,HV)` | `(N,T,HV)` | FP16 |
| `A_log`, `dt_bias` | `(HV,)` | same | FP16 |
| `h0_source` | `(N,HV,K,V)` batch-dense | same | FP32 |
| `context_lengths` | `(N,)` device | same | INT32 |

`h0_source` base pointer ≥ 16-byte aligned; vectorized-dim offset multiple of
`4×elem_size`. `context_lengths[i]==0` in decode skips row i (zeros output,
leaves h0 unchanged).

## C++ Integration

`CuteDslGDNRunner` (`cpp/kernels/gdnKernels/`): call `loadKernelModules()` once,
then `run(GDNParams, stream)` — dispatches decode/prefill based on `seq_len`.
`canImplement(kDim, vDim, smVersion)` guards SM80+, K=V=128.

Plugin (`cpp/plugins/gatedDeltaNet/`): 9 inputs — `q, k, v, a, b, A_log,
dt_bias, h0_source, context_lengths`.
