# TODOS

## Runtime Unification Follow-ups

### Temperature/topP in EAGLE speculative decoding path
**Priority:** High
**What:** Apply temperature and topP to base model logits before EAGLE acceptance (rejection sampling) in `runBaseModelVerification()`.
**Why:** Currently EAGLE path is greedy-only. Users requesting temperature sampling with spec-decode get silently wrong results (greedy instead of sampled).
**Context:** Design doc Phase 3 (original spec-decode unification doc). The vanilla decoding path gets sampling in Phase 1. The EAGLE path is deferred because modifying acceptance logic requires algorithm-level changes, not just code movement.
**Depends on:** Phase 1 (sampling params plumbed through context).

### Dynamic EAGLE tree sizing (TreeSizePolicy)
**Priority:** Medium
**What:** Add `TreeSizePolicy` to `EagleDraftingConfig` — a lookup table mapping batch size → (draftTopK, treeSize, draftingStep). CUDA graph capture covers all combinations; graph selected at runtime by (batchSize, treeConfig) key.
**Why:** Enables optimal throughput at batch=1 (large tree) and memory-efficient operation at batch=8 (small tree). Currently tree config is fixed at construction. Aligns with TurboSpec adaptive speculation research. Allocate-max approach keeps tensor allocation simple.
**Depends on:** Phase 2 (strategy pattern must be in place).

### Runtime unit test infrastructure (mock engine runner)
**Priority:** Low
**What:** Create a mock `LLMEngineRunner` that returns pre-computed tensors. Use it to unit-test runtime construction, request validation, tensor allocation, and strategy dispatch without real TensorRT engines.
**Why:** Currently zero unit test coverage for the runtime classes. All testing depends on integration tests via SSH to edge devices. Makes development iteration slow and risky.
**Depends on:** Phase 2 (stable interface to mock against).

## New Frontend Follow-ups (experimental/llm_loader)

The `llm_loader` package is experimental and supports:
- **Standard LLMs** — Llama, Qwen2/3, Mistral, Gemma, Phi, and all other attention-only architectures (via the default `CausalLM`)
- **Hybrid models** — NemotronH (Mamba2 + Attention)
- **VLMs** — Qwen3-VL, Qwen3.5-VL, Qwen2.5-VL, InternVL3, InternVL3.5, Phi-4 Multimodal
- **ASR / Omni** — Qwen3-ASR, Qwen3-Omni (visual + audio + LLM)
- **TTS** — Qwen3-TTS (thinker + talker + code predictor)
- **MoE** — supported through the default CausalLM path (e.g. Qwen-MoE uses standard `int4_groupwise_gemm` plugin for expert GEMMs)

See `design/llm/llm_loader_design.adoc` for architecture notes.

### EAGLE draft model support
**Priority:** High
**What:** Export the EAGLE draft model as a separate ONNX graph sharing the base embedding.  Wire to the existing `LLMInferenceSpecDecodeRuntime` in C++.
**Why:** Speculative decoding with EAGLE is already supported in the C++ runtime; the Python export side is missing.

### Alpamayo model support
**Priority:** Medium
**What:** Add model class and ONNX export spec for the Alpamayo model family.  Exact architecture and required custom ops TBD.
**Why:** Alpamayo is a planned NVIDIA model family targeted at edge inference.

---
