# ASR (Automatic Speech Recognition)

Complete workflow for speech recognition with audio understanding capabilities.

**Example model:** Qwen3-ASR-0.6B

> **Prerequisites:** Complete the [Installation Guide](../getting_started/installation.md) before proceeding.

---

## Step 0: Install ASR Dependency (x86 Host)

The export pipeline loads the Qwen3-ASR model via the `qwen-asr` package. Install it before exporting:

> **Warning:** Installing `qwen-asr` may break package versions in your current environment (e.g. `transformers`, `torch`). **Use a dedicated virtual environment for Qwen3-ASR export only** — do not share it with other model workflows. We need this special workflow until Qwen3-ASR gets merged into HuggingFace transformers.

```bash
cd TensorRT-Edge-LLM
python3 -m venv venv-qwen3-asr
source venv-qwen3-asr/bin/activate
pip3 install -r requirements.txt
pip3 install -r experimental/llm_loader/requirements.txt
pip3 install qwen-asr       # install qwen3-asr and its required dependencies.
```

---

## Step 1: Export (x86 Host)

```bash
export EDGE_LLM_PATH=/path/to/TensorRT-Edge-LLM
export PYTHONPATH=$EDGE_LLM_PATH:$EDGE_LLM_PATH/experimental:$PYTHONPATH
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-ASR-0.6B
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Export language model and audio encoder
python -m llm_loader.export_all_cli \
  Qwen/Qwen3-ASR-0.6B \
  $MODEL_NAME/onnx
```

## Step 2: Transfer to Device

```bash
# Transfer ONNX to device
scp -r $MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

## Step 3: Build Engines (Thor Device)

```bash
# Set up workspace directory on device
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-ASR-0.6B
cd ~/TensorRT-Edge-LLM

# Build language model engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/llm \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096

# Build audio encoder engine
./build/examples/multimodal/audio_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/audio \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/audio \
  --minTimeSteps 1000 \
  --maxTimeSteps 3000
```

## Step 4: Preprocess Audio Input (x86 Host or Thor Device)

Audio files must be converted to mel-spectrogram safetensors format before inference.

> **Note:** This step uses the `tensorrt_edgellm.scripts.preprocess_audio` utility, which requires the deprecated `tensorrt_edgellm` package. This is an audio preprocessing utility (not the deprecated LLM exporter) and is expected for this release. Set `EDGE_LLM_PATH` if running on a device where it was not defined in Step 1.

```bash
export EDGE_LLM_PATH=/path/to/TensorRT-Edge-LLM
export PYTHONPATH=$EDGE_LLM_PATH:$EDGE_LLM_PATH/experimental:$PYTHONPATH
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace

pip3 install -e $EDGE_LLM_PATH  # required for tensorrt_edgellm.scripts.preprocess_audio

python -m tensorrt_edgellm.scripts.preprocess_audio \
  --input /path/to/audio.wav \
  --output $WORKSPACE_DIR/audio_input.safetensors
```

**Note:** Supported audio formats include `.wav`, `.mp3`, `.flac`, `.ogg`, and `.m4a`. If preprocessing is done on the x86 host, transfer the output safetensors file to the device before running inference.

## Step 5: Run Inference (Thor Device)

Create an input file `$WORKSPACE_DIR/input_asr.json` (replace `/path/to/audio_input.safetensors` with the actual preprocessed audio file path):

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "max_generate_length": 256,
    "requests": [
        {
            "messages": [
                {
                    "role": "system",
                    "content": ""
                },
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio": "/path/to/audio_input.safetensors"}]
                }
            ]
        }
    ]
}
```

Run inference:

```bash
cd ~/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --multimodalEngineDir $WORKSPACE_DIR/$MODEL_NAME/engines/audio \
  --inputFile $WORKSPACE_DIR/input_asr.json \
  --outputFile $WORKSPACE_DIR/output_asr.json
```

Check `output_asr.json` for the speech recognition transcription.

---

## Quantization (Optional)

For deployments where engine size matters more than peak accuracy, the
new pipeline supports independent precision selection for the LLM
backbone and the audio encoder. Numbers below are measured on
Qwen3-ASR-0.6B; 1.7B is noted where it diverges.

### Supported precision combinations

| LLM backbone | Audio encoder | 0.6B status | 1.7B status |
|---|---|:---:|:---:|
| FP16 | FP16 | ✅ | ✅ |
| FP8 | FP16 | ✅ | ✅ |
| FP8 | FP8 | ✅ | ✅ |
| NVFP4 | FP16 | ✅ | ✅ |
| NVFP4 | FP8 | ❌ Empty output (combined quant noise exceeds the first-token EOS-vs-correct logit margin on 0.6B) | ✅ (1.7B tolerates the combined noise) |

> **Mixed-precision rules** (`experimental/quantization/cli.py`):
> - `--quantization` quantizes the LLM backbone only.
> - `--audio_quantization fp8` opts the audio tower in. **Omit it to
>   keep the audio tower at FP16** regardless of `--quantization` --
>   audio encoders are quantization-sensitive, so the default is
>   conservative (mirrors the `--visual_quantization` behaviour for
>   VLMs).
> - `--lm_head_quantization` is similarly opt-in.

### Three featured recipes

Insert a quantization step before [Step 1: Export](#step-1-export-x86-host),
then point the export at the quantized output instead of the Hugging
Face checkpoint. Joint multimodal calibration streams LibriSpeech
(audio, transcript) pairs through the model -- no `--dataset` flag is
required.

**Recipe A: 0.6B NVFP4 LLM + FP16 audio**

```bash
python -m experimental.quantization.cli llm \
  --model_dir Qwen/Qwen3-ASR-0.6B \
  --output_dir $WORKSPACE_DIR/$MODEL_NAME-nvfp4-lh.nvfp4 \
  --quantization nvfp4 \
  --lm_head_quantization nvfp4

# Then in Step 1, replace ``Qwen/Qwen3-ASR-0.6B`` with the quantized dir:
python -m llm_loader.export_all_cli \
  $WORKSPACE_DIR/$MODEL_NAME-nvfp4-lh.nvfp4 \
  $MODEL_NAME/onnx
```

**Recipe B: 0.6B FP8 LLM + FP8 audio**

```bash
python -m experimental.quantization.cli llm \
  --model_dir Qwen/Qwen3-ASR-0.6B \
  --output_dir $WORKSPACE_DIR/$MODEL_NAME-fp8-a.fp8 \
  --quantization fp8 \
  --audio_quantization fp8

# Then in Step 1, replace ``Qwen/Qwen3-ASR-0.6B`` with the quantized dir:
python -m llm_loader.export_all_cli \
  $WORKSPACE_DIR/$MODEL_NAME-fp8-a.fp8 \
  $MODEL_NAME/onnx
```

**Recipe C: 1.7B NVFP4 LLM + FP8 audio**

```bash
python -m experimental.quantization.cli llm \
  --model_dir Qwen/Qwen3-ASR-1.7B \
  --output_dir $WORKSPACE_DIR/$MODEL_NAME-nvfp4-a.fp8 \
  --quantization nvfp4 \
  --audio_quantization fp8

# Then in Step 1, replace ``Qwen/Qwen3-ASR-1.7B`` with the quantized dir:
python -m llm_loader.export_all_cli \
  $WORKSPACE_DIR/$MODEL_NAME-nvfp4-a.fp8 \
  $MODEL_NAME/onnx
```
