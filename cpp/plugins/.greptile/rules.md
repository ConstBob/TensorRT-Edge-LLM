# Plugin Rules

- Keep the full plugin contract synchronized: custom-op schema, Python export translation, plugin creator fields, serialization/deserialization versioning, TensorRT registration, builder parsing, runtime input/output shapes and dtypes, C++/Python tests, and plugin documentation.
- Review plugin field additions for backward-compatible serialization and explicit version handling.
- Validate shape, dtype, workspace, tactic, and format assumptions in the closest plugin or kernel correctness test.
- Plugin documentation should describe the exported op, TensorRT plugin inputs/outputs, supported dtype/layout combinations, and known limitations.
