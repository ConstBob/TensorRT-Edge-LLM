# Experimental Rules

- Experimental model support should still be complete for its declared scope: exporter, model-specific builder, runtime, examples, README, and accuracy or parity verification method.
- Keep model families self-contained unless two models share a true primitive with the same contract.
- Experimental code may ignore legacy compatibility, but it should not introduce vague shared abstractions that hide model-specific I/O contracts or limitations.
- Use the common Tensor/runtime utilities when helpful, but keep per-model runtime contracts explicit.
- Experimental OpenAI-compatible server changes must preserve the documented API contract: `/health`,
  `/v1/models`, `/v1/chat/completions`, streaming and non-streaming response shapes, clear 4xx validation,
  usage and finish-reason fields, `stop`, `logprobs`, `top_logprobs`, `logit_bias`, speculative-decoding
  interactions, tool-calling fields, and supported multimodal content forms such as local/base64 audio.
- Server changes should update `docs/source/user_guide/examples/experimental-server.md`, Docker or setup
  instructions when dependencies change, and tests or curl/Python examples that exercise the changed
  OpenAI-compatible behavior.
