# DriveOS LLM SDK ONNX exporter

This folder contains script to export ONNX model from PyTorch model. The exported ONNX model follows the format required by DriveOS LLM SDK runtime, so it can later be converted into TensorRT engine for E2E LLM inference application on Auto platform.

## Prerequisite

1. Since ONNX export is platform agnostic, it is strongly recommended to run the script in Linux x86 platform with Ampere or above GPUs. Even though FP8 deployment only works with Ada and above, and NVFP4 deployment only works with Blackwell and above, the simulated quantization script can be run on any GPU.
1. To avoid OOM during quantization and ONNX export, it is recommended to run the quantization on GPUs with 80GB memory to avoid OOM.

## Usage

1. Download HF checkpoint from transformers and save it locally
1. `cd export`
1. `pip3 install -r requirements.txt`
1. If you are working with NVFP4, you need to unintall `onnx` and `nvidia-modelopt` using `pip3 uninstall onnx` and `pip3 uninstall nvidia-modelopt`, and then install `pip3 install -r requirements_nvfp4.txt`. Please refer to the instruction in [../README.md](../README.md#limitations-and-known-issues) to properly configure the environment.
1. Call export script.
```
# LLM model
python3 llm_export.py --torch_dir $TORCH_DIR --dtype [fp16|fp8|int4|nvfp4|int4_ootb] --output_dir $ONNX_DIR

# VLM model
python3 multimodal_export.py --torch_dir $TORCH_DIR --dtype [fp16|fp8|int4|nvfp4|int4_ootb] --output_dir $ONNX_DIR
```

For models with LoRA weights, you can use the following command:
```
python3 llm_export.py --torch_dir $TORCH_DIR --lora_dir $LORA_DIR --lora_mode merged --dtype [fp16|fp8|int4|nvfp4|int4_ootb] --output_dir $ONNX_DIR
```

The ONNX with desired data type will be exported in `$ONNX_DIR`.

**Notes:**
1. TensorRT Out-of-the-box(OOTB) has a known performance issue with INT4 GEMV. Even though the accuracy is good, the performance is not as desired. Therefore a Int4GroupwiseGemmPlugin is written and the dq+gemms are replaced by the plugin as a temporary solution for now for int4 by default. If you do not want to use this plugin, you can pass in int4_ootb as the datatype for export script.
1. Even though FP8 or NVFP4 is supported for ONNX export, Orin does not support FP8 or NVFP4.
1. Pass in `--keep_original` to save the original exported ONNX in `${ONNX_DIR}_raw` folder. For FP16 and INT4, this is FP16 onnx, while for FP8 or NVFP4 this will be FP8 or NVFP4 onnx with FP32 weight storage. This ONNX can be reused by passing in `--onnx_path` to save ONNX export time.
1. Pass `--dataset_dir` to skip downloading quantization calibration dataset
1. Default `--max_seq_length=4096`, which corresponds to `kv_cache_capacity` field in AttentionPlugin. Please change this field if other sequence length is required. [prepare_mmmu_onnx.py](../../scripts/prepare_mmmu_onnx.py) provides a script to change `kv_cache_capacity` in existing LLM ONNX to avoid exporting again.
1. Qwen2.5-VL 3B VIT has FP16 overflow issue. Apply [upcast_fp32_gemm_war.py](../scripts/upcast_fp32_gemm_war.py) after exporting VIT ONNX as work-around.
1. For LoRA support, two modes are available:
   - `merged`: LoRA weights are merged into the base model before export (recommended for most use cases)
   - `static`: LoRA weights are kept separate and applied during inference using static LoRA patterns

## Supported models and precisions

The `llm_export.py` script can export the following LLM models into ONNX. There is a potential that other LLMs can be supported.

We also provide ONNX files for some of the models so you can download them directly.

Model | FP16 | INT4 | FP8 | NVFP4 | ONNX
--- | --- | --- | --- | --- | ---
[Llama3-8b-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | Yes | Yes | Yes | Yes |
[Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | Yes | Yes | Yes | Yes |
[Llama3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) | Yes | Yes | Yes | Yes |
[Qwen2-0.5B-instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) | Yes | Yes | Yes | Yes | [qwen2_0.5b.tgz]()
[Qwen2-1.5B-instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) | Yes | Yes | Yes | Yes | [qwen2_1.5b.tgz]()
[Qwen2-7B-instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | Yes | Yes | Yes | Yes | [qwen2_7b.tgz]()
[Qwen2.5-0.5B-instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Yes | Yes | Yes | Yes | [qwen2.5_0.5b.tgz]()
[Qwen2.5-1.5B-instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Yes | Yes | Yes | Yes | [qwen2.5_1.5b.tgz]()
[Qwen2.5-3B-instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | Yes | Yes | Yes | Yes | [qwen2.5_3b.tgz]()
[Qwen2.5-7B-instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Yes | Yes | Yes | Yes | [qwen2.5_7b.tgz]()

The `multimodal_export.py` script can export the following multimodal models into ONNX. Currently it only supports Qwen2-VL.

Model | FP16 | INT4 | FP8 | NVFP4 | ONNX
--- | --- | --- | --- | --- | ---
[Qwen2-VL-2B-instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | Yes | Yes | Yes | Yes | [qwen2_vl_2b.tgz](https://nvidia.box.com/shared/static/p1r5fv10qwuq5nvj2ffwpv0ndfbvzgv0)
[Qwen2-VL-7B-instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | Yes | Yes | Yes | Yes | [qwen2_vl_7b.tgz](https://nvidia.box.com/shared/static/zzkstqg4cojfknm1azsb1qfk1in7if51)
[Qwen2.5-VL-3B-instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Yes | Yes | Yes | Yes | [qwen2.5_vl_3b.tgz](https://nvidia.box.com/shared/static/531he8t7k5r59qedzfrch4cj13wl5hfe)
[Qwen2.5-VL-7B-instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Yes | Yes | Yes | Yes | [qwen2.5_vl_2b.tgz](https://nvidia.box.com/shared/static/cgqo6ngxp3dw5ussgpct290ud2kd34kk)