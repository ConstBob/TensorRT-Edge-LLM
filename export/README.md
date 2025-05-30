# DriveOS LLM SDK ONNX exporter

This folder contains script to export ONNX model from PyTorch model. The exported ONNX model follows the format required by DriveOS LLM SDK runtime, so it can later be converted into TensorRT engine for E2E LLM inference application on Auto platform.

## Prerequisite

1. Since ONNX export is platform agnostic, it is strongly recommended to run the script in Linux x86 platform with Ampere or above GPUs. Even though FP8 deployment only works with Ada and above, and NVFP4 deployment only works with Blackwell and above, the simulated quantization script can be run on any GPU.
1. To avoid OOM during quantization and ONNX export, it is recommended to run the quantization on GPUs with 80GB memory to avoid OOM.

## Usage

1. Download HF checkpoint from transformers and save it locally
1. `cd export` at the top-level of LLM SDK repository
1. `pip3 install -r requirements.txt`
   1. If you are working with INT4, you need to downgrade `nvidia-modelopt` to 0.19.0.
   ```
   pip3 uninstall -y nvidia-modelopt
   pip3 install -r requirements_int4.txt
   ```
   1. Please refer to the instruction in [../README.md](../README.md#limitations-and-known-issues) to properly configure the environment.
1. Call export script.
   ```
   # LLM model
   python3 llm_export.py --torch_dir $TORCH_DIR --dtype [fp16|fp8|int4|nvfp4|int4_ootb] --output_dir $ONNX_DIR

   # VLM model
   python3 multimodal_export.py --torch_dir $TORCH_DIR --dtype [fp16|fp8|int4|nvfp4|int4_ootb] --output_dir $ONNX_DIR --visualType [fp16|fp8]
   ```

1. The ONNX with desired data type will be exported in `$ONNX_DIR`.

**Notes:**
1. TensorRT Out-of-the-box(OOTB) has a known performance issue with INT4 GEMV. Even though the accuracy is good, the performance is not as desired. Therefore a Int4GroupwiseGemmPlugin is written and the dq+gemms are replaced by the plugin as a temporary solution for now for int4 by default. If you do not want to use this plugin, you can pass in int4_ootb as the datatype for export script.
1. Even though FP8 or NVFP4 is supported for ONNX export, Orin does not support FP8 or NVFP4.
1. Pass in `--keep_original` to save the original exported ONNX in `${ONNX_DIR}_raw` folder. For FP16 and INT4, this is FP16 onnx, while for FP8 or NVFP4 this will be FP8 or NVFP4 onnx with FP32 weight storage. This ONNX can be reused by passing in `--onnx_path` to save ONNX export time.
1. Pass `--dataset_dir` to skip downloading quantization calibration dataset
1. Default `--max_seq_length=4096`, which corresponds to `kv_cache_capacity` field in AttentionPlugin. Please change this field if other sequence length is required. [prepare_mmmu_onnx.py](../../scripts/prepare_mmmu_onnx.py) provides a script to change `kv_cache_capacity` in existing LLM ONNX to avoid exporting again.

## LoRA
For models with LoRA weights, you can use the following command:
```
python3 llm_export.py --torch_dir $TORCH_DIR --lora_dir $LORA_DIR --lora_mode merged --dtype [fp16|fp8|int4|nvfp4|int4_ootb] --output_dir $ONNX_DIR
```
For LoRA support, two modes are available:
   - `merged`: LoRA weights are merged into the base model before export (recommended for most use cases)
   - `static`: LoRA weights are kept separate and applied during inference using static LoRA patterns

## Eagle Deocding
For Eagle decoding, we only verify LLAMA-based model with FP16 precision now.

### Eagle3
1. For Eagle3, we use the model from HuggingFace [yuhuili/EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B), which is a LLAMA-based model.

   ```
   export TORCH_DIR="../Meta-Llama-3.1-8B-Instruct"
   export EAGLE3_TORCH_DIR="../EAGLE3-LLaMA3.1-Instruct-8B"
   git lfs install
   git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct $TORCH_DIR
   git clone https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B $EAGLE3_TORCH_DIR
   ```

1. Export ONNX for base model:
   ```
   export EAGLE3_ONNX_BASE_DIR="../Meta-Llama-3.1-8B-Instruct-Eagle3-Base"
   python3 llm_export.py --torch_dir $TORCH_DIR --dtype fp16 --output_dir $EAGLE3_ONNX_BASE_DIR --eagle_base True --eagle3 True
   ```
1. Export ONNX for draft model:
   ```
   export EAGLE3_ONNX_DRAFT_DIR="../Meta-Llama-3.1-8B-Instruct-Eagle3-Draft"
   python3 llm_export.py --torch_dir $TORCH_DIR --dtype fp16 --output_dir $EAGLE3_ONNX_DRAFT_DIR --eagle_torch_dir $EAGLE3_TORCH_DIR --eagle_draft True --eagle3 True
   ```
### Eagle2
1. Download model
   ```
   export TORCH_DIR="../Meta-Llama-3.1-8B-Instruct"
   export EAGLE2_TORCH_DIR="../EAGLE-LLaMA3.1-Instruct-8B"
   git lfs install
   git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct $TORCH_DIR
   git clone git clone https://huggingface.co/yuhuili/EAGLE-LLaMA3.1-Instruct-8B $EAGLE2_TORCH_DIR
   ```
1. Export ONNX for base model:
   ```
   export EAGLE2_ONNX_BASE_DIR="../Meta-Llama-3.1-8B-Instruct-Eagle-Base"
   python3 llm_export.py --torch_dir $TORCH_DIR --dtype fp16 --output_dir $EAGLE2_ONNX_BASE_DIR --eagle_base True
   ```
1. Export ONNX for draft model:
   ```
   export EAGLE2_ONNX_DRAFT_DIR="../Meta-Llama-3.1-8B-Instruct-Eagle-Draft"
   python3 llm_export.py --torch_dir $TORCH_DIR --dtype fp16 --output_dir $EAGLE2_ONNX_DRAFT_DIR --eagle_torch_dir $EAGLE2_TORCH_DIR --eagle_draft True
   ```

## Supported models and precisions

The `llm_export.py` script can export the following LLM models into ONNX. There is a potential that other LLMs can be supported.

We also provide ONNX files for some of the models with open-source license so you can download them directly.

Model | FP16 | INT4 | FP8 | NVFP4 | ONNX
--- | --- | --- | --- | --- | ---
[Llama3-8b-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | Yes | Yes | Yes | Yes | [llama3_8b.tgz](https://nvidia.box.com/shared/static/2r5xez6bg3sodpg3xjy7v9kiuw270c8h)
[Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | Yes | Yes | Yes | Yes | [llama3.1_8b.tgz](https://nvidia.box.com/shared/static/2my40zw33m7a3s23hv8a2iowd2oy4ykz)
[Llama3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) | Yes | Yes | Yes | Yes | [llama3.2_3b.tgz](https://nvidia.box.com/shared/static/cz794nwsb5y4vn8sm3b5x9g3xax2m0dj)
[Qwen2-0.5B-instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) | Yes | Yes | Yes | Yes | [qwen2_0.5b.tgz](https://nvidia.box.com/shared/static/9buz5igx2unkerl1o23k4cpbvo2hvigf)
[Qwen2-1.5B-instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) | Yes | Yes | Yes | Yes | [qwen2_1.5b.tgz](https://nvidia.box.com/shared/static/0t4ucre0kc5nuuqqjlgj2ed6tzkkvjw1)
[Qwen2-7B-instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | Yes | Yes | Yes | Yes | [qwen2_7b.tgz](https://nvidia.box.com/shared/static/bfowygk8lj0vt55jxfl1cizenur6pjo4)
[Qwen2.5-0.5B-instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Yes | Yes | Yes | Yes | [qwen2.5_0.5b.tgz](https://nvidia.box.com/shared/static/5rvd43zi8b3ha4x3wm1vjfryb9az1xht)
[Qwen2.5-1.5B-instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Yes | Yes | Yes | Yes | [qwen2.5_1.5b.tgz](https://nvidia.box.com/shared/static/0kg77vm50jw3nheci628mrse5sj1j4yn)
[Qwen2.5-3B-instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | Yes | Yes | Yes | Yes | /
[Qwen2.5-7B-instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Yes | Yes | Yes | Yes | [qwen2.5_7b.tgz](https://nvidia.box.com/shared/static/tjqxajzqz2ko25b3tuft7vsl3ffp56jm)
[Llama3.1-8B-Eagle2-Base](https://huggingface.co/meta-llama/Llama-3.1-8B)  | Yes | No | No | No | [llama3.1_8b_eagle2_base.tgz](https://nvidia.box.com/shared/static/7j4tgbr1hxaj6a2ngv0ynrkyq6rghvq0)
[Llama3.1-8B-Eagle2-Draft](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) | Yes | No | No | No | [llama3.1_8b_eagle2_draft.tgz](https://nvidia.box.com/shared/static/apv5wtpd3twl8y4glz171t6eterht78q)
[Llama3.1-8B-Eagle3-Base](https://huggingface.co/meta-llama/Llama-3.1-8B)  | Yes | No | No | No | [llama3.1_8b_eagle3_base.tgz](https://nvidia.box.com/shared/static/7d5diet4ze770tv93yzev3gf9eemclbi)
[Llama3.1-8B-Eagle3-Draft](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) | Yes | No | No | No | [llama3.1_8b_eagle3_draft.tgz](https://nvidia.box.com/shared/static/wvqyb348j800l6yfks7icmlrog4oiil1)

The `multimodal_export.py` script can export the following multimodal models into ONNX.

Model | FP16 | INT4 | FP8 | NVFP4 | ONNX
--- | --- | --- | --- | --- | ---
[Qwen2-VL-2B-instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | Yes | Yes | Yes | Yes | [qwen2_vl_2b.tgz](https://nvidia.box.com/shared/static/wt0c4ydbkwlqy33hc5u65c7hbqzydas4)
[Qwen2-VL-7B-instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | Yes | Yes | Yes | Yes | [qwen2_vl_7b.tgz](https://nvidia.box.com/shared/static/818242meioy5ms3g0hgpb74uqw9pxhxl)
[Qwen2.5-VL-3B-instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Yes | Yes | Yes | Yes | /
[Qwen2.5-VL-7B-instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Yes | Yes | Yes | Yes | [qwen2.5_vl_7b.tgz](https://nvidia.box.com/shared/static/kwcqornn39km3ujzaor1erjflb9vlajs)