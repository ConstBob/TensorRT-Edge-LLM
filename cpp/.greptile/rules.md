# C++ Review Rules

- C++/CUDA changes should preserve dtype and accumulation semantics, synchronization behavior, Tensor ownership, plugin serialization, and platform build coverage.
- Prefer Tensor-based dispatch and project runtime abstractions over ad-hoc raw-buffer contracts.
- Keep ownership and lifetimes explicit. Avoid hidden allocations, hidden copies, and stream synchronization in request/runtime paths.
- New builders, runtimes, plugins, examples, and kernels must be registered through the smallest owning target and validated by the nearest behavior test.
