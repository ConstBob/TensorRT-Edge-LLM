# Release Notes
## 0.2.0
- Added formal CUDA13.0 support
- Refactored SM120 and SM121 support
- Unified `int32_t` for input_ids and `float` for logits
- Replaced `jsmn` with `nlohmann/json` for better Json read/write support.
- Improved Attention performance by passing `rope_rotary_cos_sin` as model inputs
- Supported longrope
- Refactored Multimodal Runners and added them into `cpp` folder
- Improved runtime parsing from config files and folder structure
- Added runtime `Tensor` class

## 0.1.1
- Added EAGLE support for Qwen2.5-VL
- Added model support for DeepSeek-Distilled Qwen, InternVL3-1B
- Improved Sampler API
- Improved EAGLE pipeline

## 0.1.0
- Initial bring up of EAGLE2 & EAGLE3 with tree attention kernels
- Initial bring up of static and dynamic LoRA
- Added model support for Qwen2.5 3B, Qwen2.5-VL 3B, Qwen2.5-VL 7B
- Added FP8 VIT recipe for VLM
- Add JSON parser implementation
- Improved NVFP4 and FP8 performance with TensorRT10.10
- Improved unit tests and coding style
- Fixed C++ memory leak

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
