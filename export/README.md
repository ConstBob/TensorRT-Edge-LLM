# DriveOS LLM SDK ONNX exporter

This export folder contains script to export ONNX model from PyTorch model. The ONNX model follows the required format that can later be converted into TensorRT engine and run LLM inference on Auto platform.

## Prerequisite

1. Since ONNX export is platform agnostic, it is strongly recommended to run the script in Linux x86 platform with Ampere or above GPUs.
1. If you want to run fp8 quantization, it is required to run on SM>=89.
1. To avoid OOM, it is recommended to run the quantization on H100 80GB GPU to avoid OOM.

## Usage

1. Download HF checkpoint from transformers and save it locally
1. `cd export`
1. `pip3 install -r requirements.txt`
1. Call export script
```
python3 onnx_export.py --torch_dir $TORCH_DIR --dtype [fp16|fp8|int4] --output_dir $ONNX_DIR
```

The ONNX with desired data type will be exported in your designated `$ONNX_DIR`.

## Supported models and precisions
The export script can export the following PyTorch models into ONNX.

Model | fp16 | fp8 | int4_awq
--- | --- | --- | ---
[LLaMa3-8b-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/tree/main) | Yes | Yes | Yes
[QWen2.5-7B-instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Yes | Yes | Yes


## Precisions explained
Other than fp16 inference, this sample supports fp8(W8A8) and int4_awq(W4A16) quantization methods:
1. fp8: GEMMs will run in fp8 precision. Both input and weights are quantized. However, KV Cache, Attention and lm_head will run in fp16 precision.
1. int4_awq: only GEMM weights are quantized to int4 using AWQ recipe. All the computation is in fp16.

## Advanced Usage
1. Pass in `--keep_original` if you want the original exported ONNX to be saved. For fp16 and int4, this will be fp16 onnx, while for fp8 this will be fp8 onnx with fp32 weight storage. This ONNX can be reused by passing in `--onnx_path` to save ONNX export time.
1. Pass `--dataset_dir` if you want to skip downloading quantization calibration dataset

## Limitations
1. Even though fp8 is supported for ONNX export, Orin does not support fp8. We are actively preparing for the Thor fp8 release bringup.
1. QWen export requires torch<2.5.0. You might encounter this error if you use torch>=2.5.0:
```
    _C._jit_pass_onnx_graph_shape_type_inference(
RuntimeError: The serialized model is larger than the 2GiB limit imposed by the protobuf library. Therefore the output file must be a file path, so that the ONNX external data can be written to the same directory. Please specify the output file name.
```
