# TensorRT Edge-LLM

**High-Performance Large Language Model Inference Framework for NVIDIA Edge Platforms**

---

## Overview

TensorRT Edge-LLM is NVIDIA's production-ready C++ inference runtime for Large Language Models (LLMs) and Vision-Language Models (VLMs) on embedded platforms. Designed for NVIDIA Jetson Thor and DRIVE Thor, it delivers state-of-the-art performance with minimal memory footprint through advanced quantization and TensorRT optimization.

### Key Features

- **🚀 High Performance**: Optimized CUDA kernels and TensorRT integration for maximum throughput
- **💾 Memory Efficient**: Advanced quantization support (FP8, INT4, NVFP4) with intelligent KV-cache management
- **🔄 Production Ready**: C++-only runtime with zero Python dependencies for deployment
- **🎯 Edge Optimized**: Purpose-built for NVIDIA Jetson and DRIVE platforms
- **🔧 Flexible**: LoRA adapters, EAGLE speculative decoding, and multimodal models
- **📊 Complete Toolkit**: Integrated Python export pipeline, engine builder, and runtime

### Supported Platforms

| Platform | Architecture | Supported Precisions |
|----------|--------------|---------------------|
| **Jetson Thor** | Blackwell (SM100+) | FP16, FP8, INT4, NVFP4 |
| **DRIVE Thor** | Blackwell (SM100+) | FP16, FP8, INT4, NVFP4 |

### Supported Models

**Language Models:**
- Llama 3/3.1/3.2 (3B-8B)
- Qwen 2/2.5/3 (0.5B-7B)
- DeepSeek-R1 Distilled (1.5B, 7B)

**Vision-Language Models:**
- Qwen2/2.5-VL (2B-7B)
- InternVL3 (1B-2B)

**Quantization Support:** FP16, FP8, INT4 (AWQ/GPTQ), NVFP4

---

## Quick Start

Get started with TensorRT Edge-LLM in ~15 minutes. For complete installation and usage instructions, see the [**Developer Guide**](docs/developer_guide/01_Getting_Started.md).

### 1. Install Python Export Tools (x86 host)

```bash
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
pip install .
```

### 2. Export and Quantize Model (x86 host)

```bash
# Quantize to FP8
tensorrt-edgellm-quantize-llm \
    --model_dir Qwen/Qwen3-0.6B \
    --output_dir ./quantized/qwen3-0.6b \
    --quantization fp8

# Export to ONNX
tensorrt-edgellm-export-llm \
    --model_dir ./quantized/qwen3-0.6b \
    --output_dir ./onnx/qwen3-0.6b
```

### 3. Build C++ Runtime (Thor device)

```bash
mkdir build && cd build
cmake .. \
    -DTRT_PACKAGE_DIR=/path/to/TensorRT \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-thor
make -j$(nproc)
```

### 4. Build TensorRT Engine (Thor device)

```bash
./build/examples/llm/llm_build \
    --onnxDir ./onnx/qwen3-0.6b \
    --engineDir ./engines/qwen3-0.6b \
    --maxBatchSize 1
```

### 5. Run Inference (Thor device)

```bash
./build/examples/llm/llm_inference \
    --engineDir ./engines/qwen3-0.6b \
    --inputFile input.json \
    --outputFile output.json
```

---

## Documentation

### Developer Guide

Complete documentation for installation, usage, and deployment:

- **[Getting Started](docs/developer_guide/01_Getting_Started.md)** - Installation and quick start
- **[Supported Models](docs/developer_guide/02_Supported_Models.md)** - Complete model compatibility matrix
- **[Key Components](docs/developer_guide/03_Key_Components.md)** - System architecture overview
  - [Python Export Pipeline](docs/developer_guide/03.1_Python_Export_Pipeline.md)
  - [Engine Builder](docs/developer_guide/03.2_Engine_Builder.md)
  - [C++ Runtime](docs/developer_guide/03.3_C++_Runtime.md)
- **[Examples](docs/developer_guide/04_Examples.md)** - Working code examples

### Additional Resources

- **[Examples Directory](examples/)** - LLM and VLM inference examples
- **[Tests](tests/)** - Comprehensive test suite for contributors

---

## Use Cases

**🚗 Automotive**
- In-vehicle AI assistants
- Voice-controlled interfaces
- Scene understanding
- Driver assistance systems

**🤖 Robotics**
- Natural language interaction
- Task planning and reasoning
- Visual question answering
- Human-robot collaboration

**🏭 Industrial IoT**
- Equipment monitoring with NLP
- Automated inspection
- Predictive maintenance
- Voice-controlled machinery

**📱 Edge Devices**
- On-device chatbots
- Offline language processing
- Privacy-preserving AI
- Low-latency inference

---

## Tech Blogs

*Coming soon*

Stay tuned for technical deep-dives, optimization guides, and deployment best practices.

---

## Latest News

*Coming soon*

Follow our [GitHub repository](https://github.com/NVIDIA/TensorRT-Edge-LLM) for the latest updates, releases, and announcements.

---

## Support

- **Documentation**: [Developer Guide](docs/developer_guide/01_Getting_Started.md)
- **Issues**: [GitHub Issues](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NVIDIA/TensorRT-Edge-LLM/discussions)
- **Forums**: [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

---

## License

[Apache License 2.0](LICENSE)

---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

---
