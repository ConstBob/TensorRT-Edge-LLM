# Drive-LLM: TensorRT's Large Language Model Inference Sample in Auto Platform

## Introduction

This project showcases TensorRT's capability and performance in Auto Platform, and therefore is a C++ project to run LLM Inference. It uses ONNX with customized TensorRT Attention Plugin to build the TensorRT engine. It also contains implementation of C++ tokenizers, C++ builder and C++/CUDA generation logics to run e2e LLM inference.


## Prerequisite

You need a standard host to export the model into ONNX format. This host should have `requirements.txt` installed. Once you have the ONNX model ready for inference, the only dependency is TensorRT C++. This demo does not have any external dependency.   

## Supported models and precisions

Currently only LLaMa3-8B-instruct with fp16 and fp8 are supported.

## Getting started

### Build the C++ project

1. Build the project
```
cd drive-llm
mkdir build
cd build
cmake .. -DTRT_PACKAGE_DIR={TRT-Package-Path}
make
```
You will find 2 binaries: `builder`, `runtime` in the `build` folder. Those will be used later.
You will also see libLLaMaPlugin.so in your `build/plugins` folder. This will be used in the builder and runtime.

2. Cross Compilation for aarch64 platform.  
Add toolchain flag to the above CMake command
```
cd drive-llm
mkdir build
cd build
cmake .. -DTRT_PACKAGE_DIR={TRT-Package-Path} -DCMAKE_TOOLCHAIN_FILE={drive-llm-path}/cmake/aarch64_cross_toolchain.cmake
make
```
The build commands will generate the same set of executables and shared library. These can be used on Orin aarch64 board.
The engine generation and inference process on Orin is the same as on x86 machine.

### Export

In standard Linux system, you will first need to export the model from PyTorch to ONNX. We also use `onnx_graphsurgeon` to convert the Attention module into a TensorRT Plugin in the same script. To export fp16 model:

```
python3 export_to_onnx.py --torch_dir llama-v3-8b-instruct-hf/ --output_dir llama_v3_fp16_onnx --dtype fp16
```

To export fp8 model:
```
python3 quantize.py --torch_dir llama-v3-8b-instruct-hf/ --output_dir llama_v3_fp8_onnx
```

### Build the engine

You will use `builder` binary to build the TensorRT engine. For fp8 and fp16, it is the same for now since we are not using fp8 kv cache. Example command:
```
./builder --onnxPath=llama_v3_fp8_onnx/quantized_model.onnx --enginePath=llama.engine --batchSize=1 --maxInputLen=128 --maxSeqLen=256
```

**Notes:** `--maxSeqLen` includes `--maxInputLen`, so it must be greater than `--maxInputLen`. The maximum new token would equal to `maxSeqLen - maxInputLen`.

### Infer the engine

You will use `runtime` binary to infer the built TensorRT engine. Example command:

1. Inference with prompt

```
./runtime --tokenizerPath=llama-v3-8b-instruct-hf/ --enginePath=llama.engine --maxLength=256 --inputString="What is the result of 1+1?"
```

2. Benchmark the engine

```
./runtime --enginePath=llama.engine --maxLength=256 --inputLength=24 --mode=benchmark
```

3. Static multi-batch
```
./builder --onnxPath=llama_v3_onnx/model.onnx --enginePath=llama.bs2.engine --batchSize=2

./runtime --tokenizerPath=llama-v3-8b-instruct-hf/ --enginePath=llama.bs2.engine --maxLength=256 --inputString="What is the result of 1+1?" --inputString="Where is Iceland?"

./runtime --enginePath=llama.bs2.engine --maxLength=256 --inputLength=24 --mode=benchmark
```

4. Evaluate with MMLU

First download and extract the dataset.

```
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xf data.tar
```

Build engine

```
./builder --onnxPath=../export/llama_v3_onnx/model.onnx --enginePath=llama.engine --batchSize=1 --maxInputLen=2048 --maxSeqLen=2050
```

Run engine

```
./runtime --tokenizerPath=/home/scratch.trt_llm_data/llm-models/llama-models-v3/llama-v3-8b-instruct-hf/  --enginePath=llama.engine --mode evaluate --datasetPath ../data --debug
```

Python reference 

```
python mmlu.py
```
