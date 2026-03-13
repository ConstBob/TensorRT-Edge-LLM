# Limitations and Known Issues

This page documents the known limitations and issues for each release version of TensorRT Edge-LLM.

## 0.6.0

- TensorRT 10.15 may cause accuracy degradation with NVFP4 for some models. Use TensorRT 10.13.3.9 shipped with Jetpack 7.1 instead.
- Setting `--maxBatchSize=1` may cause accuracy degradation for some models. Setting `--maxBatchSize=2` can resolve this.
- For Qwen3-30B-A3B-GPTQ-INT4, setting `--maxBatchSize > 1` may cause a hanging issue for some inputs, and therefore `--maxBatchSize=1` is recommended.

