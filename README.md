# Drive-LLM: TensorRT's Large Language Model Inference Sample in Auto Platform

## Introduction

This project showcases TensorRT's performance in Auto Platform, and therefore is a C++ project to run LLM Inference. It uses ONNX with customized TensorRT Attention Plugin to build the TensorRT engine. It also contains implementation of C++ tokenizers, C++ builder and C++/CUDA generation logics to run e2e LLM inference. The main challenge for the sample is that TensorRT Safety does not support dynamic shapes. 


## Prerequisite

You need a standard host to export the model into ONNX format. This host should have `requirements.txt` installed. Once you have the ONNX model ready for inference, the only dependency is TensorRT C++. This demo does not have any external dependency due to safety constraints.   

## Supported models and precisions

Currently only LLaMa3-8B-instruct with fp16 is supported. 

## Getting started

### Build the C++ Sample

1. Build the Attention Plugin
```
cd drive-llm/plugins
mkdir build
cd build
cmake ..
make
```
You will see libLLaMaPlugin.so in your `plugins/build` folder. This will be used in the builder and runtime.

2. Build the sample
```
cd drive-llm
mkdir build
cd build
cmake ..
make
```

You will find 2 binaries: `build` and `runtime` in the `build` folder. Those will be used later. 

### Export

In standard Linux system, you will first need to export the model from PyTorch to ONNX. We also use `onnx_graphsurgeon` to convert the Attention module into a TensorRT Plugin in the same script. An example command is:

```
python3 export_to_onnx.py --torch_dir llama-v3-8b-instruct-hf/ --output_dir llama_v3_onnx --dtype fp16
```

### Build the engine

You will use `builder` binary to build the TensorRT engine. Example command:
```
./builder --onnxPath=llama_v3_onnx/model.onnx --enginePath=llama.engine --batchSize=1 --maxInputLen=128 --maxSeqLen=256
```

**Notes:** `--maxSeqLen` includes `--maxInputLen`, so it must be greater than `--maxInputLen`. The maximum new token would equal to `maxSeqLen - maxInputLen`. 

### Infer the engine

You will use `runtime` binary to infer the built TensorRT engine. Example command:

```
./runtime --tokenizerPath=llama-v3-8b-instruct-hf/ --enginePath=llama.engine --maxLength=256
```

