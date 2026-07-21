# Runtime Rules

- Preserve public I/O contracts for inputs, outputs, limits, engine artifacts, decoding behavior, and config parsing.
- Do not allocate in per-token, per-step, prefill, decode, or request hot paths.
- Flag new `rt::Tensor` construction, `cudaMalloc`/`cudaFree`, heap-backed `std::string`, repeated `std::vector` growth, per-item `cudaMemsetAsync` loops, blocking stream synchronization, host dumps, and unnecessary H2D/D2H copies inside runtime paths.
- Allocate tensors and buffers during setup/init. Reuse scratch storage, call `reserve` or pre-size vectors outside hot loops, and prefer `std::array`, `std::span`, `std::string_view`, or fixed stack buffers where sizes are bounded.
- Use zero-copy/runtime tensor bindings when plugin or engine outputs can safely be reused.
- Runtime changes should update runtime docs, relevant examples/input JSON, pipeline tests, test cases, and matching C++ runtime unit tests.
