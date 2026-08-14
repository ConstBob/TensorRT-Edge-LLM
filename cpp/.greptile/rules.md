# C++ Review Rules

- C++/CUDA changes should preserve dtype and accumulation semantics, synchronization behavior, Tensor ownership, plugin serialization, and platform build coverage.
- Prefer Tensor-based dispatch and project runtime abstractions over ad-hoc raw-buffer contracts.
- Keep ownership and lifetimes explicit. Avoid hidden allocations, hidden copies, and stream synchronization in request/runtime paths.
- New builders, runtimes, plugins, examples, and kernels must be registered through the smallest owning target and validated by the nearest behavior test.
- Flag `cudaMemcpyAsync` calls using pageable host buffers (`std::vector`, stack arrays) — CUDA falls back to synchronous copy internally. Async H2D/D2H transfers require pinned buffers (`cudaMallocHost`/`cudaHostAlloc`).
- Avoid synchronous CUDA APIs (those without an `Async` suffix, `cudaDeviceSynchronize`, `cudaThreadSynchronize`) in library and hot-path code — they block the CPU and may serialize the entire device. Prefer async variants with an explicit stream argument.
- For cross-stream data dependencies, require explicit `cudaStreamWaitEvent` synchronization; never assume ordering between different streams.
- Never use the default/NULL stream in library code — it carries implicit device-wide synchronization semantics (all streams wait for it and vice versa). Always require an explicit `cudaStream_t` argument.
- Flag `cudaMallocManaged` — unified memory is non-deterministic on Tegra/edge SoCs. Use pinned memory (`cudaMallocHost`) for transfers and device memory (`cudaMalloc`) otherwise.
