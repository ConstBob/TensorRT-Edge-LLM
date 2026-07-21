# Test Rules

- Tests should exercise real producer/consumer behavior rather than speculative framework generality.
- Export tests are smoke coverage. Model correctness requires build plus runtime inference pipeline tests and deterministic reference comparison when applicable.
- Test-list changes should be scoped to the feature under review and should not add broad infrastructure usage for experimental coverage.
- CI failures should not be ignored; classify and document them as change regression, unrelated known breakage, or infrastructure issue.
