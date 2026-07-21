# Python Exporter and Model Rules

- Keep exporter code checkpoint-driven: inspect `config.json`, safetensors indexes, tensor names/shapes, and quantization metadata.
- Do not rely on HuggingFace FX/tracing as the implementation source of truth.
- Prefer checkpoint-driven Edge-LLM-owned implementations and self-contained HuggingFace-Transformers-style model folders over generic cross-family abstractions.
- Model implementations should be explicit and family-owned. Do not push family-specific config, token, artifact, runner, or limitation handling into generic/default code unless it is a true shared primitive with a single clear owner.
- User-facing behavior belongs in documented CLI flags/config files. Do not add environment variables as the primary API without documentation and tests.
- Export or quantization command changes should update command-generation helpers, export tests, test lists, and the relevant docs in the same change.
