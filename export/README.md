# DriveOS LLM SDK ONNX exporter

This folder contains script to export ONNX model from PyTorch model. The exported ONNX model follows the format required by DriveOS LLM SDK runtime, so it can later be converted into TensorRT engine for E2E LLM inference application on Auto platform.

## Prerequisite

1. Since ONNX export is platform agnostic, it is strongly recommended to run the script in Linux x86 platform with Ampere or above GPUs.
1. To run FP8 quantization, it is required to run on SM>=89.
1. To avoid OOM during quantization and ONNX export, it is recommended to run the quantization on H100 80GB GPU to avoid OOM.

## Usage

1. Download HF checkpoint from transformers and save it locally
1. `cd export`
1. `pip3 install -r requirements.txt`
1. Call export script
```
python3 llm_export.py --torch_dir $TORCH_DIR --dtype [fp16|fp8|int4] --output_dir $ONNX_DIR
```

The ONNX with desired data type will be exported in `$ONNX_DIR`.

**Notes:**
1. Even though FP8 is supported for ONNX export, Orin does not support FP8. Thor FP8 is also not ready for DriveOS 7.0.1 release with TensorRT 10.4, but we are actively working on it.
1. Pass in `--keep_original` to save the original exported ONNX in `$ONNX_DIR_raw` folder. For FP16 and INT4, this is FP16 onnx, while for FP8 this will be FP8 onnx with FP32 weight storage. This ONNX can be reused by passing in `--onnx_path` to save ONNX export time.
1. Pass `--dataset_dir` to skip downloading quantization calibration dataset

## Supported models and precisions

The export script can export the following PyTorch models into ONNX.

Model | FP16 | INT4 | FP8
--- | --- | --- | ---
[Llama3-8b-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | Yes | Yes | Yes
[Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | Yes | Yes | Yes
[Llama3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) | Yes | Yes | Yes
[Qwen2.5-7B-instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Yes | Yes | Yes
[Qwen2-7B-instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | Yes | Yes | Yes

## Limitations

1. Qwen export requires torch<2.5.0. With torch>=2.5.0, you will encounter this issue:
```
    _C._jit_pass_onnx_graph_shape_type_inference(
RuntimeError: The serialized model is larger than the 2GiB limit imposed by the protobuf library. Therefore the output file must be a file path, so that the ONNX external data can be written to the same directory. Please specify the output file name.
```
