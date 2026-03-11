# CuTe DSL FMHA Kernels (Blackwell)

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

## Patch Details

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
  and `--function_prefix` CLI arguments. After compilation, `export_to_c()` is
  called to produce `.h` and `.o` artifacts. `--export_only` skips the reference
  check and benchmarking.
- **Causal-only window_size_right** — `window_size_right` is always 0 (causal)
  and set as a compile-time constant.
- **Remove variable-length sequence support** — `cum_seqlen_q`/`cum_seqlen_k`
  (nested tensor) paths are removed.
- **Improved logging** — uses a `[file_name]` tag prefix for identifiable output
  during parallel compilations.
