# Experimental Python Server

A vLLM-style Python API and OpenAI-compatible HTTP server for TensorRT Edge-LLM.

> **Status:** Experimental. API may change between releases.

> **Prerequisites:** Build the C++ core and pybind extension first. See [Installation Guide](../getting_started/installation.md).

### Reference Specifications

This server follows these upstream API conventions:

| Convention | Reference | Our status |
|---|---|---|
| OpenAI Chat Completions | [API reference](https://platform.openai.com/docs/api-reference/chat/create) | Supported: `messages`, `temperature`, `top_p`, `max_tokens`, `stream`. Not yet: `tools`, `n`, `logprobs`, `response_format`. |
| SSE streaming format | [Streaming guide](https://platform.openai.com/docs/api-reference/streaming) | Same `data: {JSON}\n\n` format with `delta` objects and `data: [DONE]` sentinel. |
| `reasoning` field | [vLLM reasoning streaming](https://docs.vllm.ai/en/latest/features/reasoning.html) | Same: `delta.reasoning` during thinking, `delta.content` for answer. Uses `reasoning` (not the deprecated `reasoning_content`). |
| `finish_reason` values | [OpenAI spec](https://platform.openai.com/docs/api-reference/chat/object) | `stop` and `length` match OpenAI. We add `cancelled` and `error` (not in OpenAI spec). |
| `enable_thinking` | [vLLM thinking support](https://docs.vllm.ai/en/latest/features/reasoning.html) | Same field name as vLLM. Not part of the OpenAI spec. |
| Chat template handling | — | **Different from vLLM:** we always apply chat templates in the C++ runtime (`apply_chat_template=True`), not in Python. The runtime reads `processed_chat_template.json` from the engine directory. |

---

## Quick Start

### From HuggingFace Checkpoint (auto pipeline)

```python
from experimental.server import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-1.7B")
outputs = llm.generate(
    ["What is the capital of France?"],
    SamplingParams(temperature=0.7, max_tokens=256),
)
print(outputs[0].text)
```

### From ONNX Directory (build only)

```python
from experimental.server import LLM, SamplingParams

# LLM only
llm = LLM(onnx_dir="/path/to/llm_onnx")

# VLM: LLM + visual ONNX
llm = LLM(
    onnx_dir="/path/to/llm_onnx",
    visual_onnx_dir="/path/to/visual_onnx",
)
```

### From Pre-built Engine (load only)

```python
from experimental.server import LLM, SamplingParams

# LLM engine only
llm = LLM(engine_dir="/path/to/llm_engine")

# VLM: LLM + visual engine
llm = LLM(
    engine_dir="/path/to/llm_engine",
    visual_engine_dir="/path/to/visual_engine",
)
```

### Streaming

```python
from experimental.server import LLM, SamplingParams

llm = LLM(engine_dir="/path/to/llm_engine")

for delta in llm.generate_stream(
    [{"role": "user", "content": "Tell me a story."}],
    SamplingParams(temperature=0.7, max_tokens=512),
):
    print(delta.text, end="", flush=True)
    if delta.finished:
        print(f"\n[done: {delta.finish_reason}]")
```

### OpenAI-compatible Server

```python
from experimental.server import LLM
llm = LLM(engine_dir="/path/to/llm_engine")
llm.serve(port=8000)
```

Or via CLI:

```bash
python -m experimental.server --model Qwen/Qwen3-1.7B --port 8000
```

Query (non-streaming):

```bash
curl -sN http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 128}'
```

Query (streaming):

```bash
curl -sN http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 128, "stream": true}'
```

---

## Engine Directory Layout

The C++ runtime expects specific files. See `experimental/server/engine_layout.py` for the canonical reference.

### LLM

```
{engine_dir}/
    llm.engine                    # TensorRT engine
    config.json                   # Model config
    tokenizer.json                # Tokenizer vocab
    tokenizer_config.json
    processed_chat_template.json  # Chat template
    embedding.safetensors         # Embedding weights
```

### VLM (LLM + Visual Encoder)

```
{llm_engine_dir}/    → same as LLM above
{visual_engine_dir}/
    visual.engine
    config.json
```

### ONNX (input to engine build)

```
{onnx_dir}/
    model.onnx            # ONNX model
    config.json            # Model config (copied into engine dir)
    tokenizer.json         # Tokenizer (copied into engine dir)
    tokenizer_config.json
    embedding.safetensors
```

### EAGLE

```
{eagle_engine_dir}/
    eagle_base.engine
    eagle_draft.engine
    base_config.json / draft_config.json
    d2t.safetensors
    tokenizer.json / tokenizer_config.json
    embedding.safetensors
```

---

## Thinking / Reasoning

For models with thinking mode (e.g., Qwen3), the server routes `<think>...</think>` tags to the `reasoning` field following the [vLLM convention](https://docs.vllm.ai/en/latest/features/reasoning.html).

Non-streaming response:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "reasoning": "Let me think about this...",
      "content": "The answer is 42."
    }
  }]
}
```

Streaming SSE (same format as [OpenAI streaming](https://platform.openai.com/docs/api-reference/streaming)):

```
data: {"choices": [{"delta": {"role": "assistant"}, "index": 0}]}
data: {"choices": [{"delta": {"reasoning": "Let me think..."}, "index": 0}]}
data: {"choices": [{"delta": {"content": "The answer is 42."}, "index": 0}]}
data: {"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}
data: [DONE]
```

---

## Speculative Decoding (EAGLE)

```python
llm = LLM(
    eagle_engine_dir="/path/to/eagle/engines",
    draft_top_k=10, draft_step=6, verify_tree_size=60,
)
outputs = llm.generate(["Explain quantum computing."], SamplingParams(max_tokens=512))

# Disable per-request
outputs = llm.generate(["Hello!"], SamplingParams(disable_spec_decode=True))
```

---

## API Reference

### `LLM` Class

Exactly one of `model`, `onnx_dir`, or `engine_dir` must be provided.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `""` | HuggingFace model ID or local checkpoint (export → build → load) |
| `onnx_dir` | `str` | `""` | ONNX directory (build → load) |
| `visual_onnx_dir` | `str` | `""` | Visual ONNX directory (VLM, with `onnx_dir`) |
| `engine_dir` | `str` | `""` | Pre-built LLM engine directory (load only) |
| `visual_engine_dir` | `str` | `""` | Pre-built visual engine directory (VLM, with `engine_dir`) |
| `max_input_len` | `int` | 4096 | Max input sequence length (for engine build) |
| `max_batch_size` | `int` | 1 | Max batch size (for engine build) |
| `max_kv_cache_capacity` | `int` | 8192 | Max KV cache capacity (for engine build) |
| `use_trt_native_ops` | `bool` | `False` | Use TRT native attention ops |
| `eagle_engine_dir` | `str` | `""` | Pre-built EAGLE engine directory |
| `draft_top_k` | `int` | 10 | Eagle: tokens per predecessor |
| `draft_step` | `int` | 6 | Eagle: draft tree steps |
| `verify_tree_size` | `int` | 60 | Eagle: verification tree size |

**Methods:**

| Method | Description |
|---|---|
| `generate(prompts, sampling_params)` | Batch generation. Accepts strings, list of strings, or message lists. |
| `generate_stream(messages, sampling_params)` | Streaming generation. Yields `StreamDelta` objects. |
| `chat(messages, sampling_params)` | Single-turn convenience wrapper over `generate`. |
| `serve(host, port)` | Start OpenAI-compatible HTTP server. |

### `SamplingParams`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `temperature` | `float` | 0.7 | Sampling temperature |
| `top_p` | `float` | 0.9 | Nucleus sampling threshold |
| `top_k` | `int` | 50 | Top-K sampling |
| `max_tokens` | `int` | 2048 | Max tokens to generate |
| `enable_thinking` | `bool` | `False` | Enable thinking mode ([vLLM extension](https://docs.vllm.ai/en/latest/features/reasoning.html)) |
| `disable_spec_decode` | `bool` | `False` | Disable speculative decoding |

### `StreamDelta`

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Delta text |
| `token_ids` | `list[int]` | Delta token IDs |
| `finished` | `bool` | Final delta flag |
| `finish_reason` | `str` or `None` | `"stop"` / `"length"` ([OpenAI](https://platform.openai.com/docs/api-reference/chat/object)), or `"cancelled"` / `"error"` (Edge-LLM extension) |

### HTTP Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | List models ([OpenAI spec](https://platform.openai.com/docs/api-reference/models/list)) |
| `POST` | `/v1/chat/completions` | Chat completion ([OpenAI spec](https://platform.openai.com/docs/api-reference/chat/create)) |

---

## Building the Pybind Extension

```bash
# Option 1: Standalone
mkdir -p build && cd build
cmake .. -DTRT_PACKAGE_DIR=$TRT_PACKAGE_DIR && make -j$(nproc) && cd ..
TRT_PACKAGE_DIR=$TRT_PACKAGE_DIR python experimental/server/setup_pybind.py build_ext --inplace

# Option 2: Single CMake build
mkdir -p build && cd build
cmake .. -DTRT_PACKAGE_DIR=$TRT_PACKAGE_DIR -DBUILD_PYTHON_BINDINGS=ON && make -j$(nproc)
```

---

## Architecture

```
LLM(model=...)           LLM(onnx_dir=...)       LLM(engine_dir=...)
 │                        │                        │
 ├─ ONNX export           │ [skip export]          │ [skip export]
 ├─ Engine build ─────────┤─ Engine build           │ [skip build]
 │                        │                        │
 └────────────────────────┴────────────────────────┘
                          │
                   Load LLMRuntime
                     ├─ Vanilla mode
                     ├─ Eagle spec-decode mode
                     └─ generate / generate_stream / chat / serve
```

Streaming uses the C++ `StreamChannel` API: `handleRequest` runs in a background thread while the consumer pops `StreamChunk` deltas via `waitPop()`.

---

## Threading Model

The streaming pipeline uses a two-thread producer-consumer pattern. Understanding the GIL behavior is critical for correct Python integration.

### Thread Roles

| Thread | Role | GIL held? |
|--------|------|-----------|
| **Background** (Python `threading.Thread`) | Calls `runtime.handle_request(request)` — runs TRT inference, pushes `StreamChunk`s to channel | **Released** (`py::call_guard<py::gil_scoped_release>`) |
| **Main** (consumer) | Calls `channel.wait_pop(timeout_ms)` — blocks on condition variable until a chunk arrives | **Released** (`py::call_guard<py::gil_scoped_release>`) |

Both blocking C++ calls release the GIL, so they run concurrently. Non-blocking methods (`try_pop`, `is_finished`, `cancel`) hold the GIL since they return immediately.

### Sequence Diagram

```
Main thread                          Background thread
───────────                          ─────────────────
channel = StreamChannel.create()
thread.start() ───────────────────→  handle_request(req)
                                       [GIL released]
wait_pop(timeout_ms=500)               │
  [GIL released]                       ├─ prefill
  │ blocks on condition_variable       ├─ decode step 1
  │                                    │   push(chunk₁) ──→ cv.notify_one()
  ◄─── wakeup ─────────────────────────┘
  return chunk₁                        ├─ decode step 2
  [GIL re-acquired]                    │   push(chunk₂)
  process chunk₁                       │
wait_pop(timeout_ms=500)               │  ...
  [GIL released]                       ├─ final step
  ◄─── wakeup ─────────────────────────│   push(final_chunk, finished=true)
  return final_chunk                   │   finish(reason)
                                       └─ return
thread.join()
```

### C++ Thread Safety

The C++ `StreamChannel` is designed for cross-thread use:

- **Mutex + condition variable** protects the internal `std::deque<StreamChunk>`. Lock is held only for push/pop operations.
- **Atomic flags** (`mFinished`, `mCancelled`, `mStreamInterval`, `mSkipSpecialTokens`) use `acquire`/`release` ordering — no lock needed.
- **`cancel()`** is fire-and-forget safe from any thread (atomic store + `notify_all`).
- **`finish()`** is idempotent (first-writer-wins).
- **`StreamChannelFinalizer`** (RAII) guarantees `finish()` is called on every exit path from `handleRequest`, including exceptions — consumers never block indefinitely.

### Constraints

- **`LLMRuntime` is NOT thread-safe.** Only one `handle_request` call may be in flight at a time on a given runtime instance. The `LLM` Python class serializes requests through a single runtime.
- **One `StreamChannel` per batch slot.** Each slot gets an independent channel; channels do not share state.
- **`consume()` releases lock during handler callback** to prevent deadlock if the handler interacts with the channel.
