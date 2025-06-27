# DriveOS LLM SDK ONNX exporter

This folder contains scripts to export ONNX models from PyTorch models. The exported ONNX model follows the format required by DriveOS LLM SDK runtime, so it can later be converted into a TensorRT engine for E2E LLM inference applications on the Auto platform.

## Prerequisites

1. Since ONNX export is platform agnostic, it is strongly recommended to run the scripts on a Linux x86 platform with Ampere or above GPUs. Even though FP8 deployment only works with Ada and above, and NVFP4 deployment only works with Blackwell and above, the simulated quantization scripts can be run on any GPU.
2. To avoid OOM during quantization and ONNX export, it is recommended to run the quantization on GPUs with 80GB memory.

## Usage

1. Download HF checkpoint from transformers and save it locally
2. `cd export` at the top-level of LLM SDK repository
3. `pip3 install -r requirements.txt`
   a. If you are working with INT4, you need to downgrade `nvidia-modelopt` to 0.19.0.
      ```
      pip3 uninstall -y nvidia-modelopt
      pip3 install -r requirements_int4.txt
      ```
   b. Please refer to the instruction in [../README.md](../README.md#limitations-and-known-issues) to properly configure the environment.
4. Call export script.
   ```
   # LLM model
   python3 llm_export.py --torch_dir $TORCH_DIR --dtype [fp16|fp8|int4|nvfp4|int4_ootb] --output_dir $ONNX_DIR

   # VLM model
   python3 multimodal_export.py --torch_dir $TORCH_DIR --dtype [fp16|fp8|int4|nvfp4|int4_ootb] --output_dir $ONNX_DIR --visualType [fp16|fp8]
   ```

5. The ONNX with desired data type will be exported in `$ONNX_DIR`.

**Notes:**
1. TensorRT Out-of-the-box(OOTB) has a known performance issue with INT4 GEMV. Even though the accuracy is good, the performance is not as desired. Therefore a Int4GroupwiseGemmPlugin is written and the dq+gemms are replaced by the plugin as a temporary solution for now for int4 by default. If you do not want to use this plugin, you can pass in int4_ootb as the datatype for export script.
2. Even though FP8 or NVFP4 is supported for ONNX export, Orin does not support FP8 or NVFP4.
3. Pass in `--keep_original` to save the original exported ONNX in `${ONNX_DIR}_raw` folder. For FP16 and INT4, this is FP16 onnx, while for FP8 or NVFP4 this will be FP8 or NVFP4 onnx with FP32 weight storage. This ONNX can be reused by passing in `--onnx_path` to save ONNX export time.
4. Pass `--dataset_dir` to skip downloading quantization calibration dataset
5. Default `--max_seq_length=4096`, which corresponds to `kv_cache_capacity` field in AttentionPlugin. Please change this field if other sequence length is required. [prepare_mmmu_onnx.py](../../scripts/prepare_mmmu_onnx.py) provides a script to change `kv_cache_capacity` in existing LLM ONNX to avoid exporting again.

## LoRA
For models with LoRA weights, you can use the following command:
```
python3 llm_export.py --torch_dir $TORCH_DIR --lora_dir $LORA_DIR --lora_mode [merged|static|dynamic] --dtype [fp16|fp8|int4|nvfp4|int4_ootb] --output_dir $ONNX_DIR
```

For LoRA support, three modes are available:
- `merged`: LoRA weights are merged into the base model before quantization and ONNX export. This gives the performance without any loss compared to no LoRA.
- `static`: LoRA weights are kept in separate GEMMs as weights. This requires passing in the LoRA weights during model export. LoRA GEMMs are in FP16 and the mainstream GEMMs are quantized in lower precisions. This causes at most 20% performance loss. 
- `dynamic`: LoRA weights are passed in as model inputs. You do not need actual LoRA weights during model export, but you need to specify `--maxLoraRank` during engine build, and the LoRA weights need to be passed in during runtime. Multiple LoRA weights can be loaded and switched during runtime. When no LoRA weights are loaded, the performance is expected to be close to no LoRA/merged LoRA, and when LoRA weights are loaded, the performance is similar to static LoRA.

To avoid quantizing the model multiple times, we provide a convenient script for adding dynamic LoRA to an exported model with no LoRA. 

```
python3 add_dynamic_lora.py --onnx_path $ONNX_PATH --dtype [fp16|fp8|int4|nvfp4|int4_ootb] --output_dir $OUTPUT_DIR
```

Note: Before using dynamic LoRA at runtime, you need to process your LoRA weights using `process_lora_weights.py`:
```
python3 process_lora_weights.py --input_dir $LORA_WEIGHTS_DIR --output_dir $PROCESSED_LORA_DIR
```

## Eagle Decoding
For Eagle decoding, we have only verified LLAMA-based models with FP16 precision so far.

### Eagle3
1. For Eagle3, we use the model from HuggingFace [yuhuili/EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B), which is a LLAMA-based model.

   ```
   export TORCH_DIR="../Meta-Llama-3.1-8B-Instruct"
   export EAGLE3_TORCH_DIR="../EAGLE3-LLaMA3.1-Instruct-8B"
   git lfs install
   git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct $TORCH_DIR
   git clone https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B $EAGLE3_TORCH_DIR
   ```

2. Export ONNX for the base model:
   ```
   export EAGLE3_ONNX_BASE_DIR="../Meta-Llama-3.1-8B-Instruct-Eagle3-Base"
   python3 llm_export.py --torch_dir $TORCH_DIR --dtype fp16 --output_dir $EAGLE3_ONNX_BASE_DIR --eagle_base True --eagle3 True
   ```

3. Export ONNX for the draft model:
   ```
   export EAGLE3_ONNX_DRAFT_DIR="../Meta-Llama-3.1-8B-Instruct-Eagle3-Draft"
   python3 llm_export.py --torch_dir $TORCH_DIR --dtype fp16 --output_dir $EAGLE3_ONNX_DRAFT_DIR --eagle_torch_dir $EAGLE3_TORCH_DIR --eagle_draft True --eagle3 True
   ```

### Eagle2
1. Download the models:
   ```
   export TORCH_DIR="../Meta-Llama-3.1-8B-Instruct"
   export EAGLE2_TORCH_DIR="../EAGLE-LLaMA3.1-Instruct-8B"
   git lfs install
   git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct $TORCH_DIR
   git clone https://huggingface.co/yuhuili/EAGLE-LLaMA3.1-Instruct-8B $EAGLE2_TORCH_DIR
   ```

2. Export ONNX for the base model:
   ```
   export EAGLE2_ONNX_BASE_DIR="../Meta-Llama-3.1-8B-Instruct-Eagle-Base"
   python3 llm_export.py --torch_dir $TORCH_DIR --dtype fp16 --output_dir $EAGLE2_ONNX_BASE_DIR --eagle_base True
   ```

3. Export ONNX for the draft model:
   ```
   export EAGLE2_ONNX_DRAFT_DIR="../Meta-Llama-3.1-8B-Instruct-Eagle-Draft"
   python3 llm_export.py --torch_dir $TORCH_DIR --dtype fp16 --output_dir $EAGLE2_ONNX_DRAFT_DIR --eagle_torch_dir $EAGLE2_TORCH_DIR --eagle_draft True
   ```

## Supported models and precisions

The `llm_export.py` script can export the following LLM models into ONNX. There is a potential that other LLMs can be supported.

We also provide ONNX files for some of the models with open-source license so you can download them directly.

Model | FP16 | INT4 | FP8 | NVFP4 | ONNX
--- | --- | --- | --- | --- | ---
[Llama3-8b-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | Yes | Yes | Yes | Yes | /
[Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | Yes | Yes | Yes | Yes | /
[Llama3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) | Yes | Yes | Yes | Yes | /
[Qwen2-0.5B-instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) | Yes | Yes | Yes | Yes | [qwen2_0.5b.tgz](https://nvidia.box.com/shared/static/9buz5igx2unkerl1o23k4cpbvo2hvigf)
[Qwen2-1.5B-instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) | Yes | Yes | Yes | Yes | [qwen2_1.5b.tgz](https://nvidia.box.com/shared/static/0t4ucre0kc5nuuqqjlgj2ed6tzkkvjw1)
[Qwen2-7B-instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | Yes | Yes | Yes | Yes | [qwen2_7b.tgz](https://nvidia.box.com/shared/static/bfowygk8lj0vt55jxfl1cizenur6pjo4)
[Qwen2.5-0.5B-instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Yes | Yes | Yes | Yes | [qwen2.5_0.5b.tgz](https://nvidia.box.com/shared/static/5rvd43zi8b3ha4x3wm1vjfryb9az1xht)
[Qwen2.5-1.5B-instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Yes | Yes | Yes | Yes | [qwen2.5_1.5b.tgz](https://nvidia.box.com/shared/static/0kg77vm50jw3nheci628mrse5sj1j4yn)
[Qwen2.5-3B-instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | Yes | Yes | Yes | Yes | /
[Qwen2.5-7B-instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Yes | Yes | Yes | Yes | [qwen2.5_7b.tgz](https://nvidia.box.com/shared/static/tjqxajzqz2ko25b3tuft7vsl3ffp56jm)
[Llama3.1-8B-Eagle2-Base](https://huggingface.co/meta-llama/Llama-3.1-8B)  | Yes | No | No | No | /
[Llama3.1-8B-Eagle2-Draft](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) | Yes | No | No | No | [llama3.1_8b_eagle2_draft.tgz](https://nvidia.box.com/shared/static/apv5wtpd3twl8y4glz171t6eterht78q)
[Llama3.1-8B-Eagle3-Base](https://huggingface.co/meta-llama/Llama-3.1-8B)  | Yes | No | No | No | /
[Llama3.1-8B-Eagle3-Draft](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) | Yes | No | No | No | [llama3.1_8b_eagle3_draft.tgz](https://nvidia.box.com/shared/static/wvqyb348j800l6yfks7icmlrog4oiil1)

The `multimodal_export.py` script can export the following multimodal models into ONNX.

Model | FP16 | INT4 | FP8 | NVFP4 | ONNX
--- | --- | --- | --- | --- | ---
[Qwen2-VL-2B-instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | Yes | Yes | Yes | Yes | [qwen2_vl_2b.tgz](https://nvidia.box.com/shared/static/wt0c4ydbkwlqy33hc5u65c7hbqzydas4)
[Qwen2-VL-7B-instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | Yes | Yes | Yes | Yes | [qwen2_vl_7b.tgz](https://nvidia.box.com/shared/static/818242meioy5ms3g0hgpb74uqw9pxhxl)
[Qwen2.5-VL-3B-instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Yes | Yes | Yes | Yes | /
[Qwen2.5-VL-7B-instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Yes | Yes | Yes | Yes | [qwen2.5_vl_7b.tgz](https://nvidia.box.com/shared/static/kwcqornn39km3ujzaor1erjflb9vlajs)