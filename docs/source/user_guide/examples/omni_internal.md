# Qwen3-Omni Family — Complete Workflow (Internal)

Internal end-to-end reference for **all Qwen3-Omni family variants**:
quantization, ONNX export, engine build, audio preprocessing, and
inference. The public [`omni.md`](omni.md) is a strict subset of this
document (MoE-only); anything dense- or Next-related and internal-only
lives here. This file is excluded from the OSS release tree via
`DO_NOT_RELEASE`.

## Supported models

| Model | `model_type` | Backbone | Source | Backbone recipes |
|-------|--------------|----------|--------|------------------|
| Qwen3-Omni-30B-A3B-Instruct | `qwen3_omni_moe` | MoE (128 experts) | [HF (public)](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | NVFP4, INT4 AWQ (g=128) |
| Qwen3-Omni-4B-Instruct-multilingual | `qwen3_omni` | Dense | internal snapshot (closed-source — do NOT reference publicly) | NVFP4, INT4 GPTQ (external gptqmodel), FP16 |
| Qwen3.5-Omni dense (Qwen3-Next Omni) | `qwen3_omni_next` | Dense | internal snapshot (closed-source — do NOT reference publicly) | INT4 AWQ (g=128) |

All three share the six-engine layout:

> **Thinker** (text decoder, generates assistant tokens) +
> **Talker** (text decoder, generates codec tokens) +
> **CodePredictor** (small head over codec tokens) +
> **Audio encoder** (Whisper-style mel) +
> **Visual encoder** +
> **Code2Wav** (codec → 24 kHz PCM).

## Unified front-end

One command surface covers the whole family: `tensorrt-edgellm-quantize
llm` → `tensorrt-edgellm-export` → `llm_build` → `llm_inference`. Every
stage auto-branches on `model_type` from `config.json` — the flags are
shared and never model-specific. Internally the quantize stage dispatches
to two backends:

| `model_type` | Quantize backend | Notes |
|--------------|------------------|-------|
| `qwen3_omni` / `qwen3_omni_moe` | `quantize_qwen3_omni` (joint driver) | full flag support incl. sub-encoder FP8 |
| `qwen3_omni_next` | `quantize_and_export_omni` (orchestrator) | adds a transformers in-place patch + Talker amax backfill; unsupported flags fail loudly, never silently |

### Capability matrix

| Capability | MoE (`qwen3_omni_moe`) | Dense (`qwen3_omni`) | Next (`qwen3_omni_next`) |
|------------|------------------------|----------------------|--------------------------|
| `--quantization nvfp4` | ✅ validated | ✅ validated | ⚠️ untested |
| `--quantization int4_awq` | ✅ validated (AWQ→Marlin MoE repack) | ❌ (use external GPTQ) | ✅ validated |
| INT4 GPTQ (external gptqmodel) | — | ✅ validated | — |
| `--visual_quantization fp8` / `--audio_quantization fp8` | ✅ | ✅ | ❌ rejected with explicit error |
| `--cp_quantization fp8` | ✅ | ✅ | ❌ rejected with explicit error |
| `--num_samples` / `--text_dataset` | ✅ | ✅ | ✅ |
| `--lm_head_quantization` / `--kv_cache_quantization` | ✅ | ✅ | ✅ |

## Environment

- **MoE**: released `transformers` (>= 5.x) carries the `Qwen3OmniMoe*`
  classes; the standard installation works.
- **Dense**: `Qwen3OmniForConditionalGeneration` is NOT in any released
  transformers tag. Use the internal transformers build in a dedicated
  venv (it may downgrade `torch`). Known-good venv on scratch:
  `venv-qwen3-omni-export` (transformers 4.57.0.dev0 + pillow +
  torchvision + soundfile + librosa + ninja).
- **Next**: `Qwen3OmniNextForConditionalGeneration` is NOT in any
  released transformers tag either; use the internal transformers build.
  Three latent transformers bugs are patched in-place at quantize time
  (`_patch_qwen3_omni_next_transformers`) — no manual action needed.

Shell variables used throughout:

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
mkdir -p $WORKSPACE_DIR

export OMNI_MODEL=Qwen3-Omni-30B-A3B-Instruct   # or the dense / Next snapshot dir name
export HF_ROOT=Qwen/$OMNI_MODEL                 # or a local snapshot dir
export ONNX=$WORKSPACE_DIR/$OMNI_MODEL/onnx
export ENG=$WORKSPACE_DIR/$OMNI_MODEL/engines
```

---

## Part 1: Quantize

Same CLI for all variants; the `llm` subcommand auto-detects the family
from `config.json` and dispatches per the table above
(`tensorrt_edgellm/quantization/qwen3_omni.py`).

Calibration is **joint and multimodal** for the whole family: a single
`mtq.quantize` pass chains Thinker(audio/image/text) →
`hidden_projection` / `text_projection` → Talker per sample, so both
submodels' quantizers observe the same realistic activation
distribution. `--num_samples` (default 512) is split roughly evenly
across the three modalities (LibriSpeech audio + MMMU images +
`--text_dataset`, default `cnn_dailymail`).

```bash
export QUANT_ROOT=$WORKSPACE_DIR/$OMNI_MODEL/quant

# NVFP4 backbone — MoE / dense
tensorrt-edgellm-quantize llm \
    --model_dir   $HF_ROOT \
    --output_dir  $QUANT_ROOT \
    --quantization nvfp4

# INT4 AWQ backbone (W4A16, group=128) — MoE / Next
tensorrt-edgellm-quantize llm \
    --model_dir   $HF_ROOT \
    --output_dir  $QUANT_ROOT \
    --quantization int4_awq

# Optional sub-encoder FP8 (MoE / dense only), composable with either backbone
tensorrt-edgellm-quantize llm \
    --model_dir   $HF_ROOT \
    --output_dir  $QUANT_ROOT \
    --quantization int4_awq \
    --visual_quantization fp8 \
    --audio_quantization fp8
```

INT4 AWQ is validated on MoE (the AWQ→Marlin repack targets the MoE
expert plugin) and on Next (dense Marlin path from the Qwen3.5 bring-up).
For the `qwen3_omni` dense variant, INT4 goes through the external GPTQ
flow below.

### Optional: CodePredictor FP8 (MoE / dense)

`--cp_quantization fp8` composes with any backbone choice above (CP is
the same dense 5-layer decoder in both). It opts the CP body Linears
(q/k/v/o + gate/up) into FP8; `down_proj`, the 15 lm_heads, and CP
KV-cache BMMs stay FP16 (see `FP8_CP` in `quantization_configs.py` for
the sensitivity rationale). The driver appends a dedicated Thinker →
Talker → `cp.generate` drive (`qwen3_cp_calibration_loop`, 64 text
samples) to the joint forward loop — the multimodal pass alone never
reaches CP, so without the drive every CP input_quantizer would keep an
uninitialised amax.

```bash
tensorrt-edgellm-quantize llm \
    --model_dir   $HF_ROOT \
    --output_dir  $QUANT_ROOT \
    --quantization nvfp4 \
    --cp_quantization fp8
```

A CP-only pass (`--cp_quantization fp8` without `--quantization`) is also
supported for reusing an existing backbone-quantized checkpoint: it skips
the expensive multimodal calibration entirely and runs just the CP drive;
export the `code_predictor` component from its output via
`tensorrt-edgellm-export --components code_predictor`. Note the CP-only
route does NOT apply to the external-GPTQ dense checkpoint (already
quantized ⇒ the ModelOpt pass is skipped); CP FP8 for that flow would
need the CP-only pass on the original BF16 root.

On `qwen3_omni_next` the flag is rejected (`--cp_quantization is not
supported for qwen3_omni_next yet`) — its CodePredictor differs
structurally and the CP calibration drive has not been validated there.

### Dense-only extra: INT4 GPTQ via external gptqmodel

Quantized **outside** trt-edge-llm with
[gptqmodel](https://github.com/modelcloud/gptqmodel) 4.0.0+
(`BaseQwen3_OmniGPTQ`; `bits=4, group_size=128, sym=true, desc_act=true`).
Only Thinker + Talker LLM layers go INT4; visual / audio / code2wav /
projection sidecars stay BF16. Point `QUANT_ROOT` at the gptqmodel output;
the exporter dispatches on `quantization_config.quant_method`
automatically — no extra flag.

### FP16 keep-set (all variants)

The following modules stay FP16 in every recipe (encoded in
`hf_quant_config.json` `exclude_modules`): visual encoder, audio encoder,
code2wav, code_predictor (unless `--cp_quantization` opts it in),
`hidden_projection` / `text_projection` (Thinker → Talker projections),
`codec_head`, `codec_embedding`, `lm_head`, and on MoE additionally the
routers (`mlp.gate`) and `shared_expert_gate` (FP16 for top-k stability).

### Driver internals

- **MoE/dense joint driver** (`quantize_qwen3_omni`, `is_moe` branch):
  INT4 AWQ engages scoped modelopt 0.44.0 patches — fused-experts
  flat-amax slicing fix + uncalibrated-expert per-block fallback
  (`_int4_awq_modelopt_wars`) — and the full-model export patches
  (`_omni_export_wars`: Talker `codec_head` tied-keys strip + resmooth
  skip + `get_expert_linear_names` Qwen3-Omni coverage). AWQ-Lite alpha
  search is default-on for INT4 AWQ (alpha_step 0.25; override with
  `QWEN3_OMNI_AWQ_ALPHA_STEP`), env-var opt-in for NVFP4
  (`QWEN3_OMNI_USE_AWQ_LITE=1`).
- **Next orchestrator** (`quantize_and_export_omni`): patches the
  internal transformers build in-place, runs the same joint multimodal
  calibration, then backfills any Talker `_amax` buffers the calibration
  did not populate (`_backfill_missing_amax`) before export.
- Standalone submodel model_types: `qwen3_omni_moe_text` /
  `qwen3_omni_moe_talker` (MoE) vs `qwen3_omni_thinker` /
  `qwen3_omni_talker` (dense).

**Output** — for every variant, a single self-contained HF root (same
layout as `$HF_ROOT`):

```
$QUANT_ROOT/
├── config.json                       # model_type = qwen3_omni_moe | qwen3_omni | qwen3_omni_next
├── model-00001-of-000xx.safetensors  # quantized thinker+talker text + FP16 vision/audio/code2wav/code_predictor/sidecars
├── model.safetensors.index.json
├── hf_quant_config.json
├── modelopt_state.pt                 # optional (PyTorch QDQ verification via mto.restore)
├── tokenizer + chat_template + preprocessor files
```

---

## Part 2: Export ONNX

`$QUANT_ROOT` is a complete HF root, so one export invocation covers all
six components for every variant. The exporter reads `config.json`,
dispatches per-component, and lays out `$ONNX` per the Omni layout
override.

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

Omni-relevant flags: `--components LIST`
(`thinker,talker,code_predictor,visual,audio,code2wav`),
`--fp8-embedding`, `--reduced-vocab-dir DIR`, `--skip-llm`,
`--skip-visual`, `--skip-audio`, `--skip-code2wav`.

**Expected layout** (identical tree for all variants):

```
$ONNX/
├── llm/thinker/            # model.onnx(+.data), config.json, embedding.safetensors,
│                           # processed_chat_template.json, tokenizer files
├── llm/talker/             # model.onnx(+.data), codec embedding + projection sidecars
├── llm/code_predictor/
├── audio/audio_encoder/
├── audio/code2wav/
└── vision/
```

Per-variant differences:

- **Talker sidecars**: MoE/dense emit `hidden_projection.safetensors` +
  `text_projection.safetensors`; Next emits `hidden_projection` only
  (no `text_projection` in that architecture).
- **Sub-model model_types in the exported configs**: dense reports
  `qwen3_omni_thinker` / `qwen3_omni_talker`; MoE reports
  `qwen3_omni_moe_text` / `qwen3_omni_moe_talker`; Next reports
  `qwen3_omni_next`-prefixed types.

---

## Part 3: Build Engines

Six engines for every variant; `llm_build` covers Thinker / Talker /
CodePredictor, `audio_build` covers audio encoder + Code2Wav,
`visual_build` covers the visual encoder.

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
    --maxBatchSize 1 --maxInputLen 1024 --maxKVCacheCapacity 2048

# 2. Talker (sidecars live next to the engine; symlink or copy)
./build/examples/llm/llm_build \
    --onnxDir $TALKER_ONNX --engineDir $ENG/talker \
    --maxBatchSize 1 --maxInputLen 1024 --maxKVCacheCapacity 2048
ln -sf $TALKER_ONNX/hidden_projection.safetensors $ENG/talker/
ln -sf $TALKER_ONNX/text_projection.safetensors   $ENG/talker/

# 3. CodePredictor — must sit next to the Talker engine (runtime derives
#    its path from --talkerEngineDir as <parent>/code_predictor)
./build/examples/llm/llm_build \
    --onnxDir $CP_ONNX --engineDir $ENG/code_predictor \
    --maxBatchSize 1 --maxInputLen 256 --maxKVCacheCapacity 256

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

Per-variant differences:

| Setting | MoE / dense | Next |
|---------|-------------|------|
| Thinker/Talker `--maxInputLen` / `--maxKVCacheCapacity` | 1024 / 2048 | 8192 / 8192 (bring-up sizing) |
| CodePredictor `--maxInputLen` / `--maxKVCacheCapacity` | 256 / 256 | 512 / 2048 |
| Talker sidecars | symlink/copy `hidden_projection` + `text_projection` next to the engine | auto-copied from `$TALKER_ONNX/*.safetensors` (`hidden_projection` only) |

---

## Part 4: Preprocess Audio Input

Audio files must be converted to mel-spectrogram safetensors before
inference (same as ASR). Supported: `.wav .mp3 .flac .ogg .m4a`.

```bash
tensorrt-edgellm-preprocess-audio \
    --input  /path/to/audio.wav \
    --output $WORKSPACE_DIR/audio_input.safetensors
```

---

## Part 5: Run Inference

`llm_inference` drives all four scenarios for every variant. Audio output
is opt-in via `--enableAudioOutput`; without it only the Thinker
generates text. The runtime detects `model_type` from the engine config.

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

Per-request configuration via the input JSON's top-level `streaming`
block (preferred; the CLI flag takes precedence for `enable`):

```json
"streaming": {
  "enable": true,
  "codec_chunk_frames": 10,
  "talker_prefill_threshold": 4
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `enable` | false | Enable Thinker-Talker streaming for this request |
| `codec_chunk_frames` | 10 | Vocode every N Talker frames (0 = disable inline vocoding); trades time-to-first-audio vs Code2Wav call overhead. Chunks are vocoded without left-context overlap — chunk-boundary PCM is not byte-identical across chunk sizes, perceptual quality unaffected at ≥10 |
| `talker_prefill_threshold` | 4 | Start Talker prefill after this many Thinker assistant tokens |

Contrast with standalone TTS (`qwen3_tts_inference`): that path has no
Thinker interleaving — only chunked vocoding, configured via CLI
(`--streaming --chunkFrames=<N>`, see `tts.md`).

### Input file extra parameters (audio output)

| Parameter            | Default        | Description |
|----------------------|----------------|-------------|
| `talker_temperature` | 0.9            | Sampling temperature for codec tokens |
| `talker_top_k`       | 10             | Top-K (10 empirically best for English TTS) |
| `talker_top_p`       | 1.0            | Top-P (kept at 1.0 with low top_k) |
| `repetition_penalty` | 1.0            | Penalize repeated codec tokens |
| `max_audio_length`   | 4096           | Max codec frames per request |
| `speaker`            | config default | MoE/dense: `chelsie`, `ethan`, or `aiden`; Next: per-checkpoint speaker map |

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

## Validation status (internal)

| Variant | Backbone | Result |
|---------|----------|--------|
| MoE | NVFP4 | full E2E on B100 (SM100, TRT 10.16): text sanity pass, 20-sample TTS WER 1.45%, 0/20 catastrophic |
| MoE | INT4 AWQ g128 | full E2E on B100: TTS WER 6.72%, 0/20 catastrophic (vs 7.80% A100 baseline); quant ~3.3 h with AWQ-Lite alpha_step 0.25 |
| Dense | NVFP4 | full E2E on B100: TTS WER 3.08%, 0/20 catastrophic |
| Dense | INT4 GPTQ | external gptqmodel flow; full E2E on B100: TTS WER 1.67% (requires the talker `quantization_config` inheritance fix in this branch); also verified on DRIVE Orin |
| Next | INT4 AWQ g128 | Qwen3.5 bring-up E2E (text + TTS + audio understanding) |
| Next | NVFP4 | untested |

## Notes

- **Calibration quality:** dropping `--num_samples` below ~128 total or
  removing modalities (audio/image) regresses OmniBench and TTS WER.
- **Talker is the critical path:** Talker MoE is heavier per step than
  Thinker MoE (128 experts × top-6 + shared expert vs × top-8); in
  streaming mode it drives audio-out latency.
- **Talker `text_projection` MLP:** build C++ with
  `-DENABLE_CUTE_DSL=gemm` — without it this MLP silently returns zeros
  on Ampere/Blackwell, producing garbled/empty Talker audio.
- **Greedy Talker trap (internal):** `exit_reason=max_length` with a
  repeating tail codec id usually means the input JSON set
  `talker_temperature: 0` (greedy). Check sampling params before
  suspecting the engine or kernels.
- **INT4 MoE group size:** the Marlin `int4MoePlugin` and the decode
  GEMV (`int4WoQGemvCuda.cu`) support group_size=128 only. The INT4
  CuTe DSL GEMM (SM80+) supports g≥16 at kernel level but has no
  runner/plugin consumer yet.
- **Uncalibrated experts (INT4 AWQ, MoE):** with default calib sizes,
  ~35-40% of experts never fire; the export fallback derives per-block
  amax from weights (warning per expert is expected, not an error).
- **Publicity rule:** never mention the dense or Next models, their
  sizes, or their checkpoint names in public commits, MR descriptions,
  or public docs. The `qwen3_omni` / `qwen3_omni_next` model_type
  strings in code are acceptable. Public doc edits must stay MoE-only;
  put dense/Next-related content here.
