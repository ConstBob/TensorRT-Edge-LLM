# TensorRT Edge-LLM Example: Multimodal Models

## Prerequisites

An ONNX model that complies with the TensorRT Edge-LLM runtime should be ready following [ONNX export](../../export/README.md). To run inference with real data, a tokenizer file is also required.

### ONNX Folder Structure

The ONNX export process creates the following folder structure:

```
onnx_models/${MODEL_NAME}/
├── model.onnx                    # Main LLM model ONNX file
├── config.json                   # LLM model configuration
├── tokenizer_config.json         # Tokenizer configuration
├── tokenizer.json               # Tokenizer vocabulary
└── visual_enc_onnx_${visualType}/
    └── model.onnx               # Visual encoder ONNX file
```

### Image Tokens and Preprocessing

Multimodal models process images into tokens that are fed into the LLM. The number of image tokens depends on the model architecture:

#### Qwen2-VL and Qwen2.5-VL
- An image with height × width = `N×28×28` pixels generates `N` image tokens
- By default resize with `minPixels = 128×28×28, maxPixels = 512×28×28` to keep token count low. User can change this when calling `QwenViTRunner::resizeImage`.
- Example: `demo.jpg` is resized to `1944×1176` and generates `486` image tokens

#### InternVL3
- Images are resized to multiples of `448x448` while maintaining aspect ratio as much as it can
- Each `448x448` generates `256` image tokens.
- By default resize with maximum 6 patches per image to keep token count low. User can change this when calling `internViTRunner::resizeImage`
- By default use thumbnail image, which adds one more patch and `256` additional tokens
- Example: `demo.jpg` is resized to 6 patches with thumbnail generates `1792` image tokens

**Key Points:**
- Total image tokens in a batch must match `--imageTokens` for static engines
- For dynamic engines, tokens must be within `[--minImageTokens, --maxImageTokens]` range
- Image preprocessing is handled by model-specific runners (`Qwen2ViTRunner`, `InternVLViTRunner`)

## Engine Build

The `llm_build` binary builds LLM TensorRT engines, while `visual_build` builds visual encoder engines. Both engines are required for multimodal inference. Image tokens configuration should match for LLM engines and visual encoder engines.

### Standard VLM (Naive Decoding)

The following config can run `vlm_chat` on `demo.jpg` with default code. User should configure `IMAGE_TOKENS` for different input image.

```bash
# For QwenVL models
export IMAGE_TOKENS=486
export MAX_INPUT_LEN=1024

# For InternVL3 models  
export IMAGE_TOKENS=1792
export MAX_INPUT_LEN=2048
```

**Build LLM Engine:**
```bash
./build/examples/llm/llm_build \
--onnxDir=onnx_models/${MODEL_NAME} \
--engineDir=engines/${MODEL_NAME} \
--batchSize=1 \
--maxInputLen=${MAX_INPUT_LEN} \
--maxSeqLen=4096 \
--imageTokens=${IMAGE_TOKENS} \
--usePromptTuning
```

**Build Visual Engine:**
```bash
./build/examples/multimodal/visual_build \
--onnxDir=onnx_models/${MODEL_NAME}/visual_enc_onnx_${visualType} \
--engineDir=visual_engines/${MODEL_NAME} \
--imageTokens=${IMAGE_TOKENS}
```

### Dynamic Shape Support

**LLM Engine:**
```bash
./build/examples/llm/llm_build \
--onnxDir=onnx_models/${MODEL_NAME} \
--engineDir=engines/${MODEL_NAME} \
--maxInputLen=1024 \
--maxSeqLen=4096 \
--dynamicShape \
--maxBatchSize=1 --minImageTokens=128 --maxImageTokens=512 \
--usePromptTuning
```

**Visual Engine:**
```bash
./build/examples/multimodal/visual_build \
--onnxDir=onnx_models/${MODEL_NAME}/visual_enc_onnx_${visualType} \
--engineDir=visual_engines/${MODEL_NAME} \
--dynamicShape \
--minImageTokens=128 --maxImageTokens=512
```

### EAGLE VLM (Speculative Decoding)

For EAGLE VLM, build separate base and draft LLM engines plus visual engine:

**Base LLM Engine:**
```bash
./build/examples/llm/llm_build \
--onnxDir=onnx_models/${MODEL_NAME}_eagle3_base \
--engineDir=engines/${MODEL_NAME}_eagle3_base \
--batchSize=1 \
--maxInputLen=1024 \
--maxSeqLen=4096 \
--dynamicShape \
--maxBatchSize=1 \
--isEagleBase \
--isEagle3 \
--usePromptTuning
```

**Draft LLM Engine:**
```bash
./build/examples/llm/llm_build \
--onnxDir=onnx_models/${MODEL_NAME}_eagle3_draft \
--engineDir=engines/${MODEL_NAME}_eagle3_draft \
--batchSize=1 \
--maxInputLen=1024 \
--maxSeqLen=4096 \
--dynamicShape \
--maxBatchSize=1 \
--isEagleDraft \
--isEagle3 \
--usePromptTuning
```

**Visual Engine (shared):**
```bash
./build/examples/multimodal/visual_build \
--onnxDir=onnx_models/${MODEL_NAME}_eagle3_base/visual_enc_onnx_${visualType} \
--engineDir=visual_engines/${MODEL_NAME} \
--dynamicShape \
--minImageTokens=128 \
--maxImageTokens=512
```

**Notes:**
- `--maxSeqLen` must be greater than `--maxInputLen`
- `maxSeqLen` must match the `kv_cache_capacity` field in the ONNX `AttentionPlugin` node
- For EAGLE2 models, remove the `--isEagle3` flag

## Required Folder Structure

### Standard VLM Engine

```
engines/${MODEL_NAME}/
├── llm.engine                    # LLM TensorRT engine file
├── config.json                   # LLM model configuration
├── tokenizer_config.json         # Tokenizer configuration
└── tokenizer.json               # Tokenizer vocabulary

visual_engines/${MODEL_NAME}/
└── visual_enc_${visualType}.engine  # Visual encoder TensorRT engine
```

### EAGLE VLM Engine Structure

```
engines/${MODEL_NAME}_base/
├── eagle_base.engine            # Base LLM TensorRT engine
├── base_config.json             # Base LLM configuration
├── tokenizer_config.json        # Tokenizer configuration (shared)
└── tokenizer.json              # Tokenizer vocabulary (shared)

engines/${MODEL_NAME}_draft/
├── eagle_draft.engine           # Draft LLM TensorRT engine
├── draft_config.json            # Draft LLM configuration
└── d2t.bin                     # Draft-to-target mapping (Eagle3 only, optional)

visual_engines/${MODEL_NAME}/
└── visual_enc_${visualType}.engine  # Visual encoder TensorRT engine
```

## Running Inference

### Chat Interface

**Standard VLM:**
```bash
./build/examples/multimodal/vlm_chat \
--engineDir=engines/${MODEL_NAME} \
--visualEngineDir=visual_engines/${MODEL_NAME} \
--inputString="Describe the picture." \
--imagePaths="examples/multimodal/pics/demo.jpeg"
```

**EAGLE VLM:**
```bash
./build/examples/multimodal/vlm_chat \
--baseModelDir=engines/${MODEL_NAME}_eagle3_base \
--draftModelDir=engines/${MODEL_NAME}_eagle3_draft \
--visualEngineDir=visual_engines/${MODEL_NAME} \
--inputString="Describe the picture." \
--imagePaths="examples/multimodal/pics/demo.jpeg" \
--isEagle3
```

**Notes:**
- Multiple image paths in one batch should be separated with comma `','`
- For batches with only `--imagePaths`, `--inputString` defaults to "Describe this image."
- For batches with only `--inputString`, `--imagePaths` is set to empty (pure LLM inference)

### Benchmark Performance

**Standard VLM:**
```bash
./build/examples/multimodal/vlm_benchmark \
--engineDir=engines/${MODEL_NAME} \
--visualEngineDir=visual_engines/${MODEL_NAME} \
--textTokenLength=512 \
--imageTokenLength=486 \
--outputLength=256 \
--warmUp=2 \
--numRuns=10
```

**EAGLE VLM:**
```bash
./build/examples/multimodal/vlm_benchmark \
--baseModelDir=engines/${MODEL_NAME}_eagle3_base \
--draftModelDir=engines/${MODEL_NAME}_eagle3_draft \
--visualEngineDir=visual_engines/${MODEL_NAME} \
--textTokenLength=512 \
--imageTokenLength=486 \
--outputLength=256 \
--isEagle3 \
--warmUp=2 \
--numRuns=10
```

### Accuracy Evaluation (MMMU)

**Download Dataset:**
```bash
wget https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv
```

**MMMU Engine Build Requirements:**

To match MMMU evaluation [config](https://github.com/open-compass/VLMEvalKit/blob/9ca28fd06bac52d0c42845dac8891dd9e6354611/vlmeval/config.py#L253-L264) and TensorRT shape requirements, you need specific engine configurations:

**For Qwen-VL Models:**
```bash
# Use prepare_mmmu_onnx.py to set kv_cache_capacity=8192 in LLM ONNX
python3 ./scripts/prepare_mmmu_onnx.py \
--input_path onnx_models/${MODEL_NAME}/model.onnx \
--output_path onnx_models/${MODEL_NAME}/llm_onnx_mmmu/model.onnx

# Build LLM engine
./build/examples/llm/llm_build \
--onnxDir=onnx_models/${MODEL_NAME}/llm_onnx_mmmu \
--engineDir=engines/${MODEL_NAME} \
--maxInputLen=7168 --maxSeqLen=8192 \
--dynamicShape \
--maxBatchSize=1 --minImageTokens=1280 --maxImageTokens=6620 \
--usePromptTuning

# Build visual engine
./build/examples/multimodal/visual_build \
--onnxDir=onnx_models/${MODEL_NAME}/visual_enc_onnx_${visualType} \
--engineDir=visual_engines/${MODEL_NAME} \
--dynamicShape \
--minImageTokens=1280 --maxImageTokens=6620
```

**For InternVL3 Model:**
```bash
# Use prepare_mmmu_onnx.py to set kv_cache_capacity=10240 in LLM ONNX
python3 ./scripts/prepare_mmmu_onnx.py \
--input_path onnx_models/${MODEL_NAME}/model.onnx \
--output_path onnx_models/${MODEL_NAME}/llm_onnx_mmmu/model.onnx \
-kv 10240

# Build LLM engine
./build/examples/llm/llm_build \
--onnxDir=onnx_models/${MODEL_NAME}/llm_onnx_mmmu \
--engineDir=engines/${MODEL_NAME} \
--maxInputLen=9216 --maxSeqLen=10240 \
--dynamicShape \
--maxBatchSize=1 --minImageTokens=512 --maxImageTokens=8960 \
--usePromptTuning

# Build visual engine
./build/examples/multimodal/visual_build \
--onnxDir=onnx_models/${MODEL_NAME}/visual_enc_onnx_${visualType} \
--engineDir=visual_engines/${MODEL_NAME} \
--dynamicShape \
--minImageTokens=512 --maxImageTokens=8960
```

**Standard VLM:**
```bash
./build/examples/multimodal/vlm_accuracy \
--engineDir=engines/${MODEL_NAME} \
--visualEngineDir=visual_engines/${MODEL_NAME} \
--datasetPath=./MMMU_DEV_VAL.tsv \
--outputPath=./mmmu-results.csv
```

**EAGLE VLM:**
```bash
./build/examples/multimodal/vlm_accuracy \
--baseModelDir=engines/${MODEL_NAME}_eagle3_base \
--draftModelDir=engines/${MODEL_NAME}_eagle3_draft \
--visualEngineDir=visual_engines/${MODEL_NAME} \
--datasetPath=./MMMU_DEV_VAL.tsv \
--outputPath=./mmmu-results.csv \
--isEagle3
```

**Evaluate Results:**
```bash
python ./scripts/mmmu.py \
--csv_path=./mmmu-results.csv \
--output_path=./mmmu-results-eval.json
```

**Important Notes for InternVL3:**
- **Accuracy Score Differences**: TensorRT Edge-LLM MMMU scores for InternVL3 are lower than official results because:
  1. **Evaluation Framework**: Official results use VLMEvalKit, while TensorRT Edge-LLM follows MMMU-Benchmark methodology
  2. **Patch Limitations**: We limit images to maximum 6 patches (448×448×3) per image vs 12 patches in official implementation to reduce memory requirements
  3. **Precision**: Our accuracy is achieved with float16 precision vs bf16 precision in official implementation
- **Score Validation**: When VLMEvalKit is run with our constraints (6 max patches, float16 precision), we achieve the same accuracy score as TensorRT Edge-LLM
- **Memory Optimization**: The patch limitation was implemented to reduce memory requirements for edge devices while maintaining reasonable accuracy

**Notes:**
TensorRT Edge-LLM SDK's MMMU_VAL scores are aligned with original HuggingFace models under the same preprocessing setup. Our MMMU implementation follows [MMMU-Benchmark](https://github.com/MMMU-Benchmark/MMMU).

## Runtime LoRA Switching

TensorRT Edge-LLM supports dynamic LoRA (Low-Rank Adaptation) for efficient model adaptation.

### LoRA Weights Processing

Before using LoRA weights, process them using the provided script:

```bash
python export/process_lora_weights.py --input_dir /path/to/lora/adapter --output_dir /path/to/processed/lora --dtype fp16
```

### Build with LoRA Support

Add `--maxLoraRank=<max_rank>` to enable dynamic LoRA support:

```bash
./build/examples/llm/llm_build \
--onnxDir=onnx_models/${MODEL_NAME} \
--engineDir=engines/${MODEL_NAME} \
--batchSize=1 \
--maxInputLen=1024 \
--maxSeqLen=4096 \
--maxLoraRank=16 \
--usePromptTuning
```

### Runtime LoRA Usage

Use `--loraWeights=name:path_to_lora_weights.safetensors` in any inference binary:

```bash
./build/examples/multimodal/vlm_chat \
--engineDir=engines/${MODEL_NAME} \
--visualEngineDir=visual_engines/${MODEL_NAME} \
--inputString="Describe the picture." \
--imagePaths="examples/multimodal/pics/demo.jpeg" \
--loraWeights=my_lora:processed_lora_weights.safetensors
```