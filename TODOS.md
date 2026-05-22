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

## New Frontend Follow-ups (tensorrt_edgellm)

The `tensorrt_edgellm` package is experimental and supports:
- **Standard LLMs** — Llama, Qwen2/3, Mistral, Gemma, Phi, and all other attention-only architectures (via the default `CausalLM`)
- **Hybrid models** — NemotronH (Mamba2 + Attention)
- **VLMs** — Qwen3-VL, Qwen3.5-VL, Qwen2.5-VL, InternVL3, InternVL3.5, Phi-4 Multimodal
- **ASR / Omni** — Qwen3-ASR, Qwen3-Omni (visual + audio + LLM)
- **TTS** — Qwen3-TTS (thinker + talker + code predictor)
- **MoE** — supported through the default CausalLM path (e.g. Qwen-MoE uses standard `int4_groupwise_gemm` plugin for expert GEMMs)

See `design/llm/tensorrt_edgellm_design.adoc` for architecture notes.

### EAGLE draft model support
**Priority:** High
**What:** Export the EAGLE draft model as a separate ONNX graph sharing the base embedding.  Wire to the existing `LLMInferenceRuntime` in C++.
**Why:** Speculative decoding with EAGLE is already supported in the C++ runtime; the Python export side is missing.

### Alpamayo model support
**Priority:** Medium
**What:** Add model class and ONNX export spec for the Alpamayo model family.  Exact architecture and required custom ops TBD.
**Why:** Alpamayo is a planned NVIDIA model family targeted at edge inference.

---

### Evaluate generateMultimodalIndices D2H performance
**Priority:** Low
**What:** Profile the D2H memcpy + CPU computation + H2D memcpy overhead of `generateMultimodalIndices` for typical audio sequence lengths (1K-8K tokens). If overhead exceeds 1ms, implement a CUDA kernel equivalent.
**Why:** `runBaseModelPrefill` copies full input IDs D2H to compute multimodal indices on CPU, then copies results H2D. This is a synchronization point on the prefill hot path for audio multimodal requests.
**Depends on:** Phase 1 landed + audio multimodal integration test available.

### Unify multimodal embedding creation around `mMultimodalIndices`
**Priority:** Medium
**What:** Fix inconsistent handling of multimodal input by unifying embedding creation around the `mMultimodalIndices` schema as the common path for multimodal embedding lookup.
**Why:** The current runtime uses different embedding creation paths for visual-only, audio+visual, and text-only fallback cases. A unified indexing/model would simplify the logic and reduce the risk of request-mode-specific drift or stale-state bugs.
**Depends on:** Current runtime unification cleanup landing cleanly.

### Add sampling dispatch smoke test
**Priority:** Low
**What:** Unit test in `unittests/` that validates the `useNonGreedySampling` boolean logic with various (temperature, topK, topP) combinations. Test the dispatch decision boundary without requiring a TRT engine: default params → greedy, topK>1 → non-greedy, topP<1.0 → non-greedy, temperature ∈ (0.001, 1.0) → non-greedy, temperature ≤ 0.001 → greedy.
**Why:** The sampling dispatch logic is critical for correctness and currently has no unit test coverage. The logic is pure boolean — testable without engine infrastructure.
**Depends on:** Nothing — can be built immediately.

