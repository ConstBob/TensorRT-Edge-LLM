# Release Notes
## 0.0.3
- Fixed CUDA Graph capture errors
- Fixed NVFP4 accuracy issue by changing quantization recipe

## 0.0.2
- Added VLM support with Qwen2-VL-2B, Qwen2-VL-7B examples
- Added Qwen2-0.5B and Llama3-1B support by extending `AttentionPlugin`
- Added NVFP4 precision support for all models
- Added CUDA Graph support to improve inference latency for all models
- Improved INT4 performance by `Int4GroupwiseGemmPlugin`
- Improved usage and coding style
- Fixed C++ memory leak

## 0.0.1
- Completed end-to-end Llama and Qwen < 7B inference workflow with FP16, FP8 and INT4 support
    - Completed Python export script to quantize and export the LLM into desired precision and surgeon the graph to required format
    - Completed `AttentionPlugin` to support static shape attention with RoPE
    - Completed `decoder` and `sampler` to support end-to-end LLM inference workflow with
    - Completed Llama and Qwen `tokenizer` support
    - Completed `examples` with `chat`, `benchmark` and `accuracy` to showcase the usage and benchmark accuracy and performance
