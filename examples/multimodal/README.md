# Guide to Qwen2-VL deployment pipeline

### Download Qwen2-VL
```bash
git lfs install
export MODEL_NAME="Qwen2-VL-7B-Instruct" # or Qwen2-VL-2B-Instruct
git clone https://huggingface.co/Qwen/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
```

### Export to ONNX
In standard Linux system, you will first need to export the model from PyTorch to ONNX. We will export visual encoder part and language model part separately. For language model onnx, We also use `onnx_graphsurgeon` to convert the Attention module into TensorRT Plugin.
```
python3 ../../export/multimodal_export.py \
--torch_dir tmp/hf_models/${MODEL_NAME} --dtype [fp16|fp8|int4] \
--output_dir tmp/onnx/${MODEL_NAME} \
--model_type qwen2_vl
```

### Build visual encoder engine
You will use trtexec to build the engine. Assign desired optimization profile for dynamic input shapes if needed:
- min_hw_dims = 128, max_hw_dims = 5184, max_batch_size = 4
- multi_size_min = min_hw_dims * min_batch_size = 128
- multi_size_max = max_hw_dims * max_batch_size = 20736
- multi_size_opt = max(multi_size_min, int(multi_size_max / 2)) = 10368

```
export MIN=128 OPT=10368 MAX=20736
mkdir -p tmp/trt_engines/${MODEL_NAME}/vision_encoder
trtexec \
--onnx=tmp/onnx/${MODEL_NAME}/visual_enc_onnx/model.onnx \
--saveEngine=tmp/trt_engines/${MODEL_NAME}/vision_encoder/visual_encoder_fp16.engine \
--fp16 \
--skipInference \
--minShapes=input:${MIN}x1176,rotary_pos_emb:${MIN}x40,attention_mask:1x${MIN}x${MIN} \
--optShapes=input:${OPT}x1176,rotary_pos_emb:${OPT}x40,attention_mask:1x${OPT}x${OPT} \
--maxShapes=input:${MAX}x1176,rotary_pos_emb:${MAX}x40,attention_mask:1x${MAX}x${MAX}
```

### Build language model engine
You will use `llm_build` binary to build the TensorRT engine. 
```
cd ../../
./build/examples/llm/llm_build --onnxPath=examples/multimodal/tmp/onnx/${MODEL_NAME}/llm_onnx/model.onnx \
--enginePath=examples/multimodal/tmp/trt_engines/${MODEL_NAME}/llm_model/model.engine \
--maxInputLen=2048 --maxImageTokens=5184 --minImageTokens=128 --maxSeqLen=4096 \
--modelType="qwen2_vl" \
--batchSize=2 \
--imageTokens=1122
```

**Notes:** `--maxSeqLen` includes `--maxInputLen`, so it must be greater than `--maxInputLen`. The maximum new token would equal to `maxSeqLen - maxInputLen`.

### Infer the engine
```
./build/examples/multimodal/multimodal_runner \
--tokenizerPath=examples/multimodal/tmp/hf_models/${MODEL_NAME} \
--lmEnginePath=examples/multimodal/tmp/trt_engines/${MODEL_NAME}/llm_model/model.engine \
--visualEnginePath=examples/multimodal/tmp/trt_engines/${MODEL_NAME}/vision_encoder/visual_encoder_fp16.engine \
--modelType="qwen2_vl" \
--inputString="Describe the picture." \
--imagePaths="examples/multimodal/qwen2vl/pics/demo.jpeg" \
--inputString="Describe the picture." \
--imagePaths="examples/multimodal/qwen2vl/pics/image1.jpeg" \
--maxLength=4096
```