# DriveOS LLM SDK Example: Multimodal Models

This document shows how to run multimodal pipelines with DrivsOS LLM SDK, e.g. from image+text input modalities to text output.

Multimodal models' LLM part and multimodal part are separated to two TensorRT engines. While LLM part is similar to LLM-only models, multimodal part is model-specific. Multimodal runner combines the two parts together. The multimodal features of shape `[batch_size, num_multimodal_features, multimodal_hidden_dim]` is flattened as `[batch_size * num_multimodal_features, multimodal_hidden_dim]` and passed like a prompt embedding table together with other model specific inputs.

We describes how to run supported models in the below section.

- [Qwen2-VL](#qwen2-vl)


## Qwen2-VL
### Prerequisite
1. Downdload Huggingface weights
    ```bash
    git lfs install
    export MODEL_NAME="Qwen2-VL-7B-Instruct" # or Qwen2-VL-2B-Instruct
    git clone https://huggingface.co/Qwen/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    ```

2. Export to ONNX.

    An ONNX that complies with the DriveOS LLM SDK runtime should be ready following [ONNX export](../../export/README.md). To run inference with real data, a tokenizer file is also required. Visual and LLM part is exported to two separate ONNX files.
    ```
    python3 ../../export/multimodal_export.py \
    --torch_dir tmp/hf_models/${MODEL_NAME} \
    --output_dir tmp/onnx/${MODEL_NAME} \
    --dtype [fp16|fp8|int4|nvfp4] \
    --model_type qwen2_vl
    ```

### Build engine
The `vlm_build` binary is used to build TensorRT engines. Corresponding to ONNX, we build visual engine and LLM engine respectively.
1. Static shape. Specify `--batchSize` and `--imageTokens`.
    ```
    ./build/examples/multimoal/vlm_build \
    --llmOnnxPath=tmp/onnx/${MODEL_NAME}/llm_onnx/model.onnx \
    --llmEnginePath=tmp/trt_engines/${MODEL_NAME}/llm.engine \
    --visualOnnxPath=tmp/onnx/${MODEL_NAME}/visual_enc_onnx/model.onnx \
    --visualEnginePath=tmp/trt_engines/${MODEL_NAME}/visual_enc_fp16.engine \
    --modelType="qwen2_vl" \
    --maxInputLen=1024 --maxSeqLen=4096 \
    --batchSize=1 --imageTokens=512
    ```
2. Dynamic shape. Specify `--maxBatchSize`, `--minImageTokens` and `--minImageTokens`.
    ```
    ./build/examples/multimoal/vlm_build \
    --llmOnnxPath=tmp/onnx/${MODEL_NAME}/llm_onnx/model.onnx \
    --llmEnginePath=tmp/trt_engines/${MODEL_NAME}/llm.engine \
    --visualOnnxPath=tmp/onnx/${MODEL_NAME}/visual_enc_onnx/model.onnx \
    --visualEnginePath=tmp/trt_engines/${MODEL_NAME}/visual_enc_fp16.engine \
    --modelType="qwen2_vl" \
    --maxInputLen=1024 --maxSeqLen=4096 \
    --dynamicShape \
    --maxBatchSize=2 --minImageTokens=4 --maxImageTokens=1024
    ```

### Infer engine

The `vlm_chat` and `vlm_accuracy` binaries are examples to show E2E C++ VLM inference using greedy decoding. Example usages:

#### VLM Chat

```
# BS=2
./build/examples/multimodal/vlm_chat \
--tokenizerPath=tmp/hf_models/${MODEL_NAME} \
--llmEnginePath=tmp/trt_engines/${MODEL_NAME}/llm.engine \
--visualEnginePath=tmp/trt_engines/${MODEL_NAME}/visual_encoder_fp16.engine \
--modelType="qwen2_vl" \
--maxLength=1024 \
--inputString="Describe the picture." \
--imagePaths="examples/multimodal/qwen2vl/pics/demo.jpeg" \
--inputString="Identify the similarities between these images." \
--imagePaths="examples/multimodal/qwen2vl/pics/image1.jpeg,examples/multimodal/qwen2vl/pics/image2.jpeg"
```
**Note**:
1. `--inputString` takes input prompt for one batch. `--imagePaths` takes image paths for one batch. Multiple image paths in one batch should be separated with comma `','`.
1. One `--inputString` and one `--imagePaths` are paired as inputs for one batch. `batchSize` equals to the maximum of number of `--inputString` and number of `--imagePaths`.
1. For any batch that contains `--imagePaths` only, `--inputString` is set to default prompt `Describe this image.`. For any batch that contains `--inputString` only, `--imagePaths` is set to empty, which is equivalent to pure LLM inference.

#### Benchmark Performance

1. Benchmark the E2E pipeline performance on certain input size

    ```
    ./build/examples/multimodal/vlm_benchmark \
    --llmEnginePath=tmp/trt_engines/${MODEL_NAME}/llm.engine \
    --visualEnginePath=tmp/trt_engines/${MODEL_NAME}/visual_encoder_fp16.engine \
    --modelType="qwen2_vl" \
    --textTokenLength=512 --imageTokenLength=512 --outputLength=256
    [--warmUp=2 --numRuns=10]
    ```
2. Benchmark visual encoder and LLM separately

    - Use Qwen2 model as an approximation for Qwen2-VL LLM part performance.
        - Build [Qwen2 engine](../llm/README.md) of the same size and precision.
        - Use `../llm/llm_benchmark` binary to benchmark.
    - Use `trtexec` to benchmark visual encoder engine.
    - E2E latency = visual encoder latency + LLM latency


#### Evaluate accuracy with MMMU

To match MMMU evaluation [config](https://github.com/open-compass/VLMEvalKit/blob/9ca28fd06bac52d0c42845dac8891dd9e6354611/vlmeval/config.py#L253-L264) and MMMU images size, we need to generate ONNX and TensorRT engines with the following config: `--minImageTokens=1280`, `--maxImageTokens=6620`, `--maxInputLen=7168`, `--maxSeqLen=8192`.
1. Export ONNX
    ```
    python3 ../../export/multimodal_export.py \
    --torch_dir tmp/hf_models/${MODEL_NAME} \
    --output_dir tmp/onnx/${MODEL_NAME} \
    --dtype [fp16|fp8|int4] \
    --model_type qwen2_vl \
    --max_seq_length 8192
    ```
2. Build engine
    ```
    ./build/examples/multimoal/vlm_build \
    --llmOnnxPath=tmp/onnx/${MODEL_NAME}/llm_onnx/model.onnx \
    --llmEnginePath=tmp/trt_engines/${MODEL_NAME}/llm.engine \
    --visualOnnxPath=tmp/onnx/${MODEL_NAME}/visual_enc_onnx/model.onnx \
    --visualEnginePath=tmp/trt_engines/${MODEL_NAME}/visual_enc_fp16.engine \
    --modelType="qwen2_vl" \
    --maxInputLen=7168 --maxSeqLen=8192 \
    --dynamicShape \
    --maxBatchSize=1 --minImageTokens=1280 --maxImageTokens=6620
    ```
3. Collect inference results on MMMU-val dataset.

    ```
    wget https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv

    ./build/examples/multimodal/vlm_accuracy \
    --tokenizerPath=tmp/hf_models/${MODEL_NAME} \
    --llmEnginePath=tmp/trt_engines/${MODEL_NAME}/llm.engine \
    --visualEnginePath=tmp/trt_engines/${MODEL_NAME}/visual_encoder_fp16.engine \
    --modelType=qwen2_vl \
    --datasetPath=./MMMU_DEV_VAL.tsv \
    --outputPath=./mmmu-qwen2vl.csv \
    ```
4. Evaluate results with python script.

    ```
    python scripts/mmmu.py --csv_path=./mmmu-qwen2vl.tsv --output_path=./mmmu-qwen2vl-eval.json
    ```