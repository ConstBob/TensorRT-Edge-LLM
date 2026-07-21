# Kernel Rules

- Kernel changes must update the matching kernel README, generated artifact workflow, CMake/build registration, and closest correctness test.
- Performance-sensitive kernel changes need reproducible benchmark or profiling notes, preferably kernel-only for kernel-only comparisons, with NVTX/nsys methodology.
- Review dtype conversions, accumulation precision, memory coalescing, workspace requirements, architecture guards, generated artifacts, and launch parameter assumptions.
- Do not mix end-to-end model overhead into kernel-only comparisons unless the MR explicitly changes integration behavior.
