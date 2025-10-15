# API Reference


## Status: TBD

This section is planned for future development and will include comprehensive API documentation covering:

### Planned API Documentation

**C++ Runtime API:**
- `LLMInferenceRuntime` class reference
- `LLMInferenceSpecDecodeRuntime` class reference
- `LLMEngineRunner` class reference
- `EagleDraftEngineRunner` class reference
- `MultimodalRunner` class reference
- `Tokenizer` class reference
- Configuration structures and enums

**Python Export API:**
- Model quantization APIs
- ONNX export APIs
- LoRA processing APIs
- Validation utilities

**Callback Interfaces:**
- Token streaming callbacks
- Progress monitoring callbacks
- Error handling callbacks

**Data Structures:**
- `LLMGenerationRequest`
- `LLMGenerationResponse`
- `SamplingParams`
- `ModelConfig`
- `RuntimeConfig`

**Utility Functions:**
- Logging configuration
- Error handling
- Memory management
- Performance profiling

---

## Current Resources

While we develop the full API reference, you can:

**1. Explore Example Code:**
The `examples/` directory contains working examples demonstrating API usage:
- `examples/llm/llm_chat.cpp` - Interactive inference
- `examples/llm/llm_inference.cpp` - Batch processing
- `examples/multimodal/vlm_chat.cpp` - Multimodal inference

**2. Read Header Files:**
API declarations are well-documented in header files:
- `cpp/runtime/llmInferenceRuntime.h`
- `cpp/runtime/llmInferenceSpecDecodeRuntime.h`
- `cpp/runtime/llmEngineRunner.h`
- `cpp/runtime/eagleDraftEngineRunner.h`
- `cpp/multimodal/multimodalRunner.h`
- `cpp/tokenizer/*.h`

**3. Review Design Documents:**
- [Key Components Documentation](03_Key_Components.md)

---

## API Quick Reference

### Basic Inference Example

```cpp
#include "runtime/llmInferenceRuntime.h"
#include "tokenizer/tokenizer.h"

// Initialize runtime
auto runtime = std::make_unique<LLMInferenceRuntime>();
runtime->loadEngine("/path/to/engine");

// Create request
LLMGenerationRequest request;
request.prompt = "Hello, how are you?";
request.maxOutputLen = 100;
request.temperature = 0.7;
request.topK = 50;
request.topP = 0.9;

// Execute inference
LLMGenerationResponse response;
bool success = runtime->handleRequest(request, response, stream);

// Access result
std::cout << "Generated: " << response.text << std::endl;
```

### Streaming Example

```cpp
// Set streaming callback
auto callback = [](const std::string& token) {
    std::cout << token << std::flush;
};
request.streamingCallback = callback;

// Execute with streaming
runtime->handleRequest(request, response, stream);
```

---

## Contributing

If you'd like to contribute to API documentation:

1. See [Contributing Guidelines](../CONTRIBUTING.md)
2. Open an issue on our [GitHub repository](https://github.com/NVIDIA/TensorRT-Edge-LLM)
3. Submit pull requests with documentation improvements

---

## Support

For API questions:
- **GitHub Discussions**: Community Q&A
- **GitHub Issues**: Bug reports and feature requests
- **NVIDIA Developer Forums**: Technical support

---

**Check back for updates or watch our [GitHub repository](https://github.com/NVIDIA/TensorRT-Edge-LLM) for notifications.**




