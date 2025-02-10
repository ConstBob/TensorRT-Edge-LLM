## 0.0.2
**TensorRT**: **TBA**
**Platform**: **TBA**

## 0.0.1
**TensorRT**: 10.4
**Platform**: Thor-X with clock fixed at 756MHz with FP16 precision.

Model | Precision | Batch Size | Input/Output | First Token Latency(ms) | Generation Tokens/sec | Total Latency
--- | --- | --- | --- | --- | --- | ---
Llama3-8b-instruct | FP16 | 1 | 512+128 | 263.26 | 15.55 | 8492.5
Llama3.1-8B | FP16 | 1 | 512+128 | 263.21 | 15.91 | 8308.82
Llama3.2-3B | FP16 | 1 | 512+128 | 119.32 | 33.87 | 3899.13
Qwen2.5-7B-instruct | FP16 | 1 | 512+128 | 256.90 | 16.29 | 7858.04
Qwen2-7B-instruct | FP16 | 1 | 512+128 | 256.75 | 15.61 | 8455.55