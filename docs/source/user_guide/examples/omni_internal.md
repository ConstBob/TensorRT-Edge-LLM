# Qwen3-Next Omni — Complete Workflow (Internal)

Internal end-to-end reference for **Qwen3-Next Omni**: quantization,
ONNX export, engine build, audio preprocessing, and inference. This file is excluded from the OSS release
tree via `DO_NOT_RELEASE`.

## Supported model

| Model | Backbone | Source | Recipes |
|-------|----------|--------|---------|
| Qwen3-Next Omni | Dense | internal snapshot (closed-source — do NOT reference publicly) | INT4 AWQ Thinker + Talker (ModelOpt) |

Six-engine layout (same as the public Omni recipe):

> **Thinker** (text decoder, generates assistant tokens) +
> **Talker** (text decoder, generates codec tokens) +
> **CodePredictor** (small head over codec tokens) +
> **Audio encoder** (Whisper-style mel) +
> **Visual encoder** +
> **Code2Wav** (codec → 24 kHz PCM).

`config.json` model_type: `qwen3_omni_next`. The quant driver, exporter,
engine build, and runtime all auto-branch on this — the user-facing
commands are identical to the public Omni flow.

## Environment

`Qwen3OmniNextForConditionalGeneration` is NOT in any released
transformers tag. Use the internal transformers build in a dedicated
venv (it may downgrade `torch`).

Shell variables used throughout:

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
mkdir -p $WORKSPACE_DIR

export OMNI_MODEL=Qwen3-Next-Omni-Dense
export HF_ROOT=/path/to/$OMNI_MODEL
export ONNX=$WORKSPACE_DIR/$OMNI_MODEL/onnx
export ENG=$WORKSPACE_DIR/$OMNI_MODEL/engines
```

---

## Part 1: Quantize

The `llm` subcommand auto-detects `qwen3_omni_next` from `config.json`
and dispatches to the joint Thinker+Talker driver
(`tensorrt_edgellm/quantization/qwen3_omni.py`).

Calibration is **joint and multimodal**: a single `mtq.quantize` pass
chains Thinker → hidden projection → Talker per sample, so both
submodels' quantizers observe the same realistic activation
distribution. `--num_samples` (default 512) is split roughly evenly
across three modalities (LibriSpeech audio + MMMU images +
`cnn_dailymail` text).

```bash
export QUANT_ROOT=$WORKSPACE_DIR/$OMNI_MODEL/quant

# INT4 AWQ backbone (W4A16, group=128)
tensorrt-edgellm-quantize llm \
    --model_dir   $HF_ROOT \
    --output_dir  $QUANT_ROOT \
    --quantization int4_awq
```

The following modules stay FP16 in every recipe (encoded in
`hf_quant_config.json` `exclude_modules`):

- visual encoder, audio encoder, code2wav, code_predictor
- `hidden_projection` (Thinker → Talker projection)
- `codec_head` (Talker output projection)
- `codec_embedding`, `lm_head`

**Output** — a self-contained HF root (same layout as `$HF_ROOT`):

```
$QUANT_ROOT/
├── config.json                       # model_type = qwen3_omni_next
├── model-00001-of-000xx.safetensors  # quantized thinker+talker text + sidecars
├── model.safetensors.index.json
├── hf_quant_config.json
├── modelopt_state.pt                 # optional (PyTorch QDQ verification)
├── tokenizer + chat_template + preprocessor files
```

---

## Part 2: Export ONNX

`$QUANT_ROOT` is a complete HF root, so one export invocation covers all
six components:

```bash
mkdir -p $ONNX
tensorrt-edgellm-export $QUANT_ROOT $ONNX
```

Iterate on a single component:

```bash
tensorrt-edgellm-export --components thinker $QUANT_ROOT $ONNX   # only Thinker
tensorrt-edgellm-export --skip-visual $QUANT_ROOT $ONNX          # skip vision
tensorrt-edgellm-export --fp8-embedding $QUANT_ROOT $ONNX        # FP8 E4M3 embedding table
```

Flags: `--components LIST`
(`thinker,talker,code_predictor,visual,audio,code2wav`),
`--fp8-embedding`, `--reduced-vocab-dir DIR`, `--skip-llm`,
`--skip-visual`, `--skip-audio`, `--skip-code2wav`.

**Expected layout**:

```
$ONNX/
├── llm/thinker/            # model.onnx(+.data), config.json, embedding.safetensors,
│                           # processed_chat_template.json, tokenizer files
├── llm/talker/             # model.onnx(+.data), codec embedding, hidden_projection sidecar
├── llm/code_predictor/
├── audio/audio_encoder/
├── audio/code2wav/
└── vision/
```

---

## Part 3: Build Engines

Six engines; `llm_build` covers Thinker / Talker / CodePredictor,
`audio_build` covers audio encoder + Code2Wav, `visual_build` covers the
visual encoder.

```bash
export THINKER_ONNX=$ONNX/llm/thinker
export TALKER_ONNX=$ONNX/llm/talker
export CP_ONNX=$ONNX/llm/code_predictor
export AUDIO_ONNX=$ONNX/audio/audio_encoder
export CODE2WAV_ONNX=$ONNX/audio/code2wav
export VISUAL_ONNX=$ONNX/vision

# 1. Thinker
./build/examples/llm/llm_build \
    --onnxDir $THINKER_ONNX --engineDir $ENG/thinker \
    --maxBatchSize 1 --maxInputLen 8192 --maxKVCacheCapacity 8192

# 2. Talker (sidecars auto-copied from $TALKER_ONNX/*.safetensors)
./build/examples/llm/llm_build \
    --onnxDir $TALKER_ONNX --engineDir $ENG/talker \
    --maxBatchSize 1 --maxInputLen 8192 --maxKVCacheCapacity 8192

# 3. CodePredictor — must sit next to the Talker engine (runtime derives
#    its path from --talkerEngineDir as <parent>/code_predictor)
./build/examples/llm/llm_build \
    --onnxDir $CP_ONNX --engineDir $ENG/code_predictor \
    --maxBatchSize 1 --maxInputLen 512 --maxKVCacheCapacity 2048

# 4. Code2Wav
./build/examples/multimodal/audio_build \
    --onnxDir $CODE2WAV_ONNX --engineDir $ENG/code2wav --maxCodeLen 1000

# 5. Audio encoder
./build/examples/multimodal/audio_build \
    --onnxDir $AUDIO_ONNX --engineDir $ENG/multimodal/audio

# 6. Visual encoder
./build/examples/multimodal/visual_build \
    --onnxDir $VISUAL_ONNX --engineDir $ENG/multimodal/visual
```

`--maxBatchSize` can be raised to the runtime concurrency budget.

---

## Part 4: Preprocess Audio Input

Audio files must be converted to mel-spectrogram safetensors before
inference. Supported: `.wav .mp3 .flac .ogg .m4a`.

```bash
tensorrt-edgellm-preprocess-audio \
    --input  /path/to/audio.wav \
    --output $WORKSPACE_DIR/audio_input.safetensors
```

---

## Part 5: Run Inference

`llm_inference` drives all four scenarios. Audio output is opt-in via
`--enableAudioOutput`; without it only the Thinker generates text. The
runtime detects `model_type` from the engine config.

### Scenario A: Text input → text output

```json
{
    "max_generate_length": 256,
    "requests": [
        {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user",   "content": [{"type": "text", "text": "What is the capital of France?"}]}
            ]
        }
    ]
}
```

```bash
./build/examples/llm/llm_inference \
    --engineDir           $ENG/thinker \
    --multimodalEngineDir $ENG/multimodal \
    --inputFile           input_text.json \
    --outputFile          output_text.json
```

### Scenario B: Audio input → text output

`messages[*].content` carries
`{"type": "audio", "audio": "<preprocessed.safetensors>"}`.

### Scenario C: Image input → text output

`messages[*].content` carries
`{"type": "image", "image": "<path/to/image.jpg>"}`.

### Scenario D: Text/audio/image input → text + speech output

```bash
./build/examples/llm/llm_inference \
    --engineDir              $ENG/thinker \
    --multimodalEngineDir    $ENG/multimodal \
    --talkerEngineDir        $ENG/talker \
    --code2wavEngineDir      $ENG/code2wav \
    --inputFile              input.json \
    --outputFile             output.json \
    --outputAudioDir         ./audio_output \
    --enableAudioOutput \
    --enableThinkerTalkerStreaming
```

`--enableThinkerTalkerStreaming` interleaves Talker prefill/decode with
Thinker token generation, emitting the first audio chunk before the
Thinker text completes (much lower time-to-first-audio).

### Input file extra parameters (audio output)

| Parameter            | Default        | Description |
|----------------------|----------------|-------------|
| `talker_temperature` | 0.9            | Sampling temperature for codec tokens |
| `talker_top_k`       | 10             | Top-K |
| `talker_top_p`       | 1.0            | Top-P |
| `repetition_penalty` | 1.0            | Penalize repeated codec tokens |
| `max_audio_length`   | 4096           | Max codec frames per request |
| `speaker`            | config default | speaker id |

### Output JSON example

```json
{
  "responses": [
    {
      "request_idx": 0,
      "output_text": "The capital of France is Paris.",
      "audio_file": "./audio_output/audio_req0_batch0.wav",
      "audio_samples": 88830,
      "audio_sample_rate": 24000,
      "audio_duration_ms": 3700
    }
  ]
}
```

---

## Notes

- **Calibration quality:** dropping `--num_samples` below ~128 total or
  removing modalities (audio/image) regresses OmniBench and TTS WER.
- **Greedy Talker trap:** `exit_reason=max_length` with a repeating tail
  codec id usually means the input JSON set `talker_temperature: 0`
  (greedy). Check sampling params before suspecting the engine or kernels.
- **Publicity rule:** never mention the model or its checkpoint name in
  public commits, MR descriptions, or public docs. The `qwen3_omni_next`
  model_type string in code is acceptable. Public doc edits stay on
  [`omni.md`](./omni.md); dense-related content lives here.
