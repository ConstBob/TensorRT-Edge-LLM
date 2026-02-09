# TensorRT Edge-LLM

**High-Performance Large Language Model Inference Framework for NVIDIA Edge Platforms**

---

## Overview

TensorRT Edge-LLM is NVIDIA's high-performance C++ inference runtime for Large Language Models (LLMs) and Vision-Language Models (VLMs) on embedded platforms. It enables efficient deployment of state-of-the-art language models on resource-constrained devices such as NVIDIA Jetson and NVIDIA DRIVE platforms. TensorRT Edge-LLM provides convenient Python scripts to convert HuggingFace checkpoints to [ONNX](https://onnx.ai). Engine build and end-to-end inference runs entirely on Edge platforms.

---

## Getting Started

For the supported platforms, models and precisions, see the [**Overview**](docs/source/developer_guide/getting-started/overview.md). Get started with TensorRT Edge-LLM in <15 minutes. For complete installation and usage instructions, see the [**Quick Start Guide**](docs/source/developer_guide/getting-started/quick-start-guide.md).

---

## Documentation

### Introduction

- **[Overview](docs/source/developer_guide/getting-started/overview.md)** - What is TensorRT Edge-LLM and key features
- **[Supported Models](docs/source/developer_guide/getting-started/supported-models.md)** - Complete model compatibility matrix

### User Guide

- **[Installation](docs/source/developer_guide/getting-started/installation.md)** - Set up Python export pipeline and C++ runtime
- **[Quick Start Guide](docs/source/developer_guide/getting-started/quick-start-guide.md)** - Run your first inference in ~15 minutes
- **[Examples](docs/source/developer_guide/getting-started/examples.md)** - End-to-end LLM, VLM, EAGLE, and LoRA workflows
- **[Input Format Guide](docs/source/developer_guide/getting-started/input-format.md)** - Request format and specifications
- **[Chat Template Format](docs/source/developer_guide/getting-started/chat-template-format.md)** - Chat template configuration

### Developer Guide

#### Software Design

- **[Python Export Pipeline](docs/source/developer_guide/software-design/python-export-pipeline.md)** - Model export and quantization
- **[Engine Builder](docs/source/developer_guide/software-design/engine-builder.md)** - Building TensorRT engines
- **[C++ Runtime Overview](docs/source/developer_guide/software-design/cpp-runtime-overview.md)** - Runtime system architecture
  - [LLM Inference Runtime](docs/source/developer_guide/software-design/llm-inference-runtime.md)
  - [LLM SpecDecode Runtime](docs/source/developer_guide/software-design/llm-inference-specdecode-runtime.md)
  - [Advanced Runtime Features](docs/source/developer_guide/features/advanced-runtime-features.md)

#### Advanced Topics

- **[Customization Guide](docs/source/developer_guide/customization/customization-guide.md)** - Customizing TensorRT Edge-LLM for your needs
- **[TensorRT Plugins](docs/source/developer_guide/customization/tensorrt-plugins.md)** - Custom plugin development
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

- **Documentation**: [Developer Guide](docs/source/developer_guide/getting-started/overview.md)
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
