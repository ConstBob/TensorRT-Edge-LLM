# Unit Test Rules

- C++ unit tests should use Tensor RAII instead of raw CUDA allocation unless the allocation API itself is under test.
- Plugin and kernel rewrites require targeted numerical correctness tests with clear shape, dtype, and tolerance coverage.
- Avoid tests that only validate local mock behavior while skipping the production producer/consumer contract.
