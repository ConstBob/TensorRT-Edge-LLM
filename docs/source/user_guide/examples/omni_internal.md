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
| Qwen3-Omni Next dense (3B) | `qwen3_omni_next` | Dense (GDN-hybrid) | internal snapshot (closed-source — do NOT reference publicly) | INT4 AWQ (g=128) |
| Qwen3-Omni Next MoE (23A2.6B) | `qwen3_omni_next` | MoE (GDN-hybrid, fp16 experts) | internal snapshot (closed-source — do NOT reference publicly) | fp16 MoE plugin (+ MTP draft export) |

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
| `--quantization nvfp4` | ✅ validated | ✅ validated | ⚠️ see note below |

**Next MoE NVFP4 production split** (from the pre-quantized `0422_nvfp4`
checkpoint, ground-truth from its weight dtypes):

- **Thinker** MoE experts → **NVFP4** (layer-0's 256 experts kept FP16;
  layers 1–27 NVFP4). Thinker attention, `shared_expert`, `lm_head`,
  visual/audio towers stay FP16.
- **Talker** MoE experts → **FP16** (fp16 MoE plugin — audio generation
  is quantization-sensitive). `code_predictor`, `talker_projection`,
  `codec_head`, GDN `linear_attn` also FP16.

So the split is Thinker-NVFP4 / Talker-FP16 — the **opposite** of the
dense/MoE Qwen3-Omni intuition. A naive full-model `--quantization
nvfp4` on the BF16 root (0315) NVFP4s *both* Thinker and Talker experts
(the default keep-set only excludes routers/gates/shared_expert/GDN),
which does **not** match production. To reproduce the production layout,
exclude Talker experts (`*talker*mlp.experts.*`) and layer-0 Thinker
experts from the quant config.
| `--quantization int4_awq` | ✅ validated (AWQ→Marlin MoE repack) | ❌ (use external GPTQ) | ✅ validated |
| INT4 GPTQ (external gptqmodel) | — | ✅ validated | — |
| `--visual_quantization fp8` / `--audio_quantization fp8` | ✅ | ✅ | ✅ validated (rides the joint calib pass; see FP8 encoder note below) |
| `--cp_quantization {fp8,nvfp4}` | ✅ CP-only pass | ✅ CP-only pass | ✅ CP-only pass (see Next specifics below) |
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
expert plugin) and on Next (dense Marlin path from the Qwen3-Omni Next bring-up).
For the `qwen3_omni` dense variant, INT4 goes through the external GPTQ
flow below.

### Optional: CodePredictor FP8 / NVFP4 (MoE / dense)

`--cp_quantization` opts the CP body Linears (q/k/v/o + gate/up) into
`fp8` (per-channel weight + per-tensor static input) or `nvfp4`
(per-16-element block weight and input, FP8 E4M3 block scales). Both keep
`down_proj`, the 15 lm_heads, `talker_projection`, the codec embedding
tables, and the CP KV-cache BMMs at FP16 — see `_cp_entries` in
`quantization_configs.py` for the per-exclusion rationale. The driver
appends a dedicated Thinker → Talker → `cp.generate` drive
(`qwen3_cp_calibration_loop`, 64 text samples) to the forward loop — the
multimodal pass alone never reaches CP, so without the drive every CP
input_quantizer would keep an uninitialised amax.

The flag is a **CP-only pass**: it runs without `--quantization` (the
joint combination is rejected on every Qwen3-Omni variant). It skips the expensive multimodal calibration
and runs just the CP drive; export the `code_predictor` component from
its output and take every other component from the backbone-quantized
checkpoint.

```bash
tensorrt-edgellm-quantize llm \
    --model_dir  $HF_ROOT \
    --output_dir $QUANT_ROOT/cp_quant \
    --cp_quantization nvfp4 \
    --text_dataset cnn_dailymail

tensorrt-edgellm-export \
    --components code_predictor \
    $QUANT_ROOT/cp_quant $ONNX
```

`nvfp4` is a bandwidth trade rather than a free win — it pays off on
bandwidth-limited targets and can cost time elsewhere, with TTS WER
unchanged either way. Benchmark both against FP16 on the target part.

Note the CP-only route does NOT apply to the external-GPTQ dense
checkpoint (already quantized ⇒ the ModelOpt pass is skipped); CP
quantization for that flow needs the CP-only pass on the original BF16
root.

On `qwen3_omni_next` only the **CP-only pass** is supported
(`--cp_quantization fp8` without `--quantization`; the joint combination
is rejected with an explicit error). It runs a dedicated Next drive
(`qwen3_next_cp_calibration_loop`: Thinker prefill → Talker codec loop →
`code_predictor` residual steps) and exports a **Talker-root** checkpoint
whose `model_type` is patched to `qwen3_omni_next_talker`. That value is
deliberately NOT the full-root `qwen3_omni_next`: AutoConfig would map it
to the full config class and default-fill the missing thinker/code2wav
sections, while an unregistered type falls back to the raw `config.json`
everywhere. `tensorrt-edgellm-export --components code_predictor` accepts
the Talker-root directly (layout `llm/code_predictor`). The 23A2.6B-only
`talker_projection` bridge (1280 → 1024) is excluded from quantization —
the C++ runtime consumes it as an fp16 sidecar GEMM
(`small_to_mtp_projection.safetensors`), so FP8 there would only add
error. See "Qwen3-Omni Next specifics" below for the environment.

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

### CodePredictor speculative decoding

The CodePredictor loop dominates audio decode time. It speculates with no draft
model: the checkpoint already carries one `lm_head` per RVQ depth over a shared
residual stream, so `lm_heads[s+1]` applied to the hidden state at depth `s` is a
one-step-ahead proposal costing a single GEMV. Verification is standard
speculative sampling (`min(1, p/q)` accept, `norm(p-q)+` residual resample), so the
sampled distribution is unchanged.

Off by default; enable per run with `--cpSpecVerifySize N` (one committed position
plus `N-1` drafted depths, valid range 2-8). Acceptance saturates past four depths.
Applies to every variant — dense, MoE and Next — and to both `llm_inference
--enableAudioOutput` and `qwen3_tts_inference`.

No extra build step: the builder recognises a CodePredictor from the `model` field
in its engine config and widens only that engine's generation profile.

Degrades to the autoregressive loop, with a warning, when the codebook exceeds the
small-vocabulary kernels, when the engine predates the wider generation profile, or
when the model has fewer than two RVQ depths — it never silently mis-decodes.

Under greedy decoding the output differs from the autoregressive path: verification
needs a different `lm_head` per position, which the engine's single `lm_head_idx`
gather cannot express, so the runtime applies the heads itself and the two
computations round differently. Under sampling the distribution is unchanged.

---

## Qwen3-Omni Next specifics (internal)

Everything in this section is closed-source material — never reference
the model names, sizes, or checkpoint paths publicly.

### Checkpoints and environment

| Variant | Snapshot | transformers |
|---------|----------|--------------|
| dense 3B | `qwen3_5_omni_3b_final_multilingual_0324` | `transformers-internal-bk-0316` fork |
| MoE 23A2.6B | `qwen3_5_omni_23a2.6b_final_multilingual_0315` | `transformers-internal-bk-0316` fork |

The 0315/0324 checkpoints require the matching internal fork on
`PYTHONPATH` (known-good venv on scratch: `venv-cp-fp8` — torch 2.9
cu128 + ModelOpt 0.44). The fork already ships gated attention and the
GDN-hybrid Talker, so the in-place transformers patches
(`_patch_qwen3_omni_next_transformers`) detect per-class whether each
fix is still needed and no-op otherwise. Load the model in the
checkpoint-declared dtype (bf16): the fork sizes its GDN cache states
from the config dtype, and an fp16-loaded model fails with mixed-dtype
matmuls inside the Talker generate chain.

That fork also predates Gemma4 and has no `Gemma4AudioConfig`. The lightweight
`import tensorrt_edgellm` package surface does not load exporter models, but
accessing the export API or running the exporter loads
`models/gemma4/__init__.py` → `modeling_gemma4_audio`, whose top-level `from
transformers import Gemma4AudioConfig` raises ImportError. We keep that import
unconditional (no in-code fallback), so run export/quantize in an env whose
transformers ships Gemma4 — either add Gemma4 to the fork, or use a venv that
has both the Next patches and an upstream transformers.

### MoE Talker: fp16 experts

The 23A2.6B Talker/Thinker MoE keeps routed experts in FP16/BF16 and
runs them through the C++ `Fp16MoePlugin` (E ∈ {128, 256}); there is no
weight repack. Export side: `fp16_moe_plugin` op in `models/ops.py` +
the `_use_fp16_moe` path in `modeling_qwen3_moe.py`; the Next Thinker /
Talker MoE classes live in
`models/qwen3_omni_next/modeling_qwen3_omni_next_moe_{text,talker}.py`
(registered as `qwen3_omni_next_text_moe` — the HF config does not
distinguish dense vs MoE, so `model.py` re-dispatches on
`num_experts > 0`). **Do NOT run the 23A2.6B Talker through the fp16
MoE plugin**: this bf16-trained checkpoint has SwiGLU activation
outliers past the fp16 max (65504) — one ±inf row poisons every later
token through attention (all-NaN logits, token 4999+/no-EOS). The same
overflow reproduces in pure PyTorch fp16 (`torch.multinomial` device
assert), i.e. the model is not fp16-able; a saturation clamp in
`activateFc1Kernel` was tried and rejected in review (it silently
swallows NaNs and alters shared-kernel semantics). The Talker ships
quantized (NVFP4 experts) instead; the fp16 MoE plugin remains for the
MTP draft (256 experts, Thinker-side ranges, no overflow observed).

### MTP speculative decoding (E2E validated)

`Qwen3OmniNextMoeMtpDraftModel`
(`modeling_qwen3_omni_next_mtp.py`) reuses the `qwen3_5` MTP draft
infrastructure with the 256-expert sparse-MoE FFN; the exporter selects
it when the checkpoint declares `mtp_num_hidden_layers` under
`talker_config`. The C++ runtime needs **zero changes** — the stock MTP
decoder path (`MTPDecoder` / `gdn_decode_mtp` / `scatterMtpStates`)
drives it as-is. Usage: export `--components thinker,mtp_draft --mtp`
(the draft lands in `<onnx>/mtp_draft/`, NOT `<onnx>/llm/mtp_draft/`);
build base with `--specBase --maxVerifyTreeSize 7` and draft with
`--specDraft` into the same engine dir; run with `--specDecode
--specDraftTopK 1 --specDraftStep 3 --specVerifySize 4`.

**Base export requires the `spec_verify_phase_marker` input.** On a
GDN-hybrid backbone the speculative *verify* step re-runs the whole
drafted window through the GDN mixers, but a GDN layer carries a
recurrent state that mutates per token, so a verify that ends up
rejecting tokens must be able to roll that state back. The marker is a
**shape-only** input (`[0]` = ordinary prefill/decode, `[1]` = spec
verify; the payload is ignored) that flips `gated_delta_net` /
`causal_conv1d` into their spec-verify kernels, which additionally emit
`intermediate_states` — a per-step checkpoint of the recurrent state the
runtime commits by accepted token id. This is **not** new to this MR or
to any rebase: the marker and its `--specBase` validation shipped with
the Qwen3.5 hybrid DFlash/DDTree work (`0b5091a21`, work item #421) and
the `qwen3_5` MTP base has always wired it. The Next MoE Thinker
(`modeling_qwen3_omni_next_moe_text.py`) is a new subclass that inherits
that MTP infrastructure but initially omitted the pass-through, so it
must thread the marker through its backbone / `forward` / flat-wrapper /
OnnxSpec the same way `qwen3_5_text.py` does, or `llm_build --specBase`
fails with *"missing input `spec_verify_phase_marker`"*.

Pitfall worth keeping: the MTP base export must feed the draft the
**final post-norm** hidden states (the backbone's first return value).
Returning `emitted_hidden_states` lets the Talker-inherited
`accept_hidden_layer` (e.g. 18) silently swap in a mid-layer pre-norm
tensor — acceptance collapses to ~1.44 and E2E runs slower than
baseline. Teacher-forced hidden→lm_head agreement (~0.93 when healthy)
is the quick sanity probe.

#### MTP and the Talker on one Thinker engine (`accept_hidden_states`)

The pitfall above is a collision, not just a mistake: the draft's contract
is the post-norm hidden and the Talker's is `hidden_states[accept_hidden_layer]`
(mid-stack, pre-norm), and an MTP base engine has only one `hidden_states`
output to give. Serving both from one forward needs a second output, which
is what `accept_hidden_states` is — the same pair HF exposes as
`outputs.hidden_states[18]` and `[-1]`.

| output | source | consumer |
|---|---|---|
| `hidden_states` | after all 28 layers **and** the final norm | MTP draft |
| `accept_hidden_states` | layer 18's output, **before** the final norm | Talker |

Emitted only when the export is an MTP base **and**
`1 <= accept_hidden_layer <= min(num_hidden_layers, len(layer_types))`; the
`min` matters because the backbone's capture loop is
`zip(layers, layer_types)`, so bounding on `num_hidden_layers` alone would
let a short `layer_types` pass the gate while the loop never captures — the
two outputs would then alias the same tensor. A non-spec export is
unaffected: it emits two outputs and `hidden_states` still carries the
mid-stack tensor.

Runtime binding (`state/pipelineIO.cpp`):

```
isSpecDecodeBase -> "hidden_states"        = baseHiddenStates   (draft reads)
                    "accept_hidden_states" = outputHiddenStates (Talker reads)
otherwise        -> "hidden_states"        = outputHiddenStates (Talker reads)
```

`outputHiddenStates` is always "the Talker's copy"; only the binding name
changes. Wiring the Talker to `hidden_states` on a spec base hands it a
post-final-norm tensor — degraded audio, no error. Allocation follows
`mBaseExecutor->hasIOTensor(kAcceptHiddenStates)`: allocating regardless
would make `outputHiddenStates.isEmpty()` stop meaning "nothing will fill
this", and the Talker would read uninitialised device memory instead of
failing. `llm_bench` must pass the same `hasIOTensor` result, because
`TensorRegistry::bindAll` fails on any engine I/O missing from the map.

Cost of the extra output, measured: 102.37 vs 103.06 tok/s (inside this
board's ~9% run-to-run spread), TRT activation memory byte-identical. The
only real charge is one `batch × max_input_len × hidden × 2 B` buffer
(~16 MB at batch 1 / 4096), and only on a spec-base engine that also has a
Talker.

Measured on B100 (23A2.6B, NVFP4 Thinker). Acceptance rate is strongly
prompt-dependent — it tracks how predictable the continuation is, not
the language: on a 4-prompt mix the per-prompt rate ranged 2.02 (an
open-ended English essay) to 2.88 (structured Chinese Q&A), averaging
~2.5, for **432.8 vs 358.2 tok/s (+21%)**. An earlier run on a
predecessor branch logged 2.91 / +33%, but that used a different draft
implementation (the pre-`moe_text` MTP path) and a different prompt set,
so it is not a same-conditions comparison. Baseline throughput is
unchanged (358 vs 355 tok/s), i.e. the backbone did not regress; the
acceptance delta lives entirely in the prompt spread and the draft
implementation. Report acceptance as a range with the prompt set, not a
single number.

### Trap: `mtp_num_hidden_layers` describes the architecture, not the file

0315 declares `mtp_num_hidden_layers=1` and ships **zero** `thinker.mtp.*`
tensors. Gating MTP quantization on the config alone therefore enters the
MTP path on a checkpoint that has no MTP weights, loads nothing, and
calibrates a randomly initialised 1.35 B-parameter draft — which then gets
quantized and written out as `thinker.mtp.*`. Nothing in the output marks it
as garbage.

Two independent gates are required, and both now apply:

- `_mtp_num_hidden_layers(model.thinker) > 0` — the architecture declares a head
- `_has_mtp_weights(mtp_dir)` — the checkpoint actually ships `mtp.*` tensors

When the first passes and the second fails, the run prints
`Skipping MTP quantization: <dir> declares an MTP head but ships no 'mtp.*'
weights.` and continues with backbone quantization. Independently,
`MtpDraftModel.from_pretrained` now **raises** if any weight outside
`{embed_tokens, rotary_emb, lm_head}` stays unloaded, so an unrecognised
checkpoint layout fails at load instead of at quality-review time. Those
three are legitimately absent: `embed_tokens` is shared from the base,
`rotary_emb` is computed, and Omni ships no `mtp.lm_head` (the draft borrows
the base head at export, or `--lm_head_quantization` copies the real weights
in later).

Observed behaviour, current branch:

```
from_pretrained(0422) -> LOADED: 1353M params, expert0.gate_proj nonzero=True
from_pretrained(0315) -> RuntimeError: MTP draft model has 785 unloaded weights
```

The already-shipped 0315 requant outputs predate this code path and were
checked clean: `thinker.mtp.*` count is 0 in both `q35_0315_gdnq` and
`q35_0315_lmq_nvfp4`, and the loader warning never fired in their logs.

### Quantizing the MTP head and the Talker from 0422

Neither flow has been executed end-to-end yet — the commands below follow
from the code paths and from what the checkpoints are known to contain, and
are **not** backed by a measured run. Treat them as the starting point, not as
a validated recipe.

**Ordering constraint.** The MTP draft calibrates against the Thinker's
final post-norm hidden states, so it must be quantized **before** the Thinker
backbone. `quantize_and_export_omni` already does this internally; it matters
when composing flows by hand.

**MTP.** The head lives in 0422 but calibration needs an *unquantized*
Thinker, which only 0315 has. `--mtp_draft_dir` decouples the two:

```bash
tensorrt-edgellm-quantize llm \
    --model_dir      $CKPT_0315 \
    --mtp_draft_dir  $CKPT_0422 \
    --output_dir     $OUT \
    --quantization nvfp4 --num_samples 512 \
    [--lm_head_quantization nvfp4]
```

`mtp_dir = mtp_draft_dir or model_dir`; dense-vs-MoE dispatch reads
`num_experts` from the *draft* checkpoint's config, so the draft source
decides the architecture even when it differs from the calibration base.

**Talker.** 0422's Thinker is already NVFP4, which rules out the unified
flow (it needs an unquantized backbone to calibrate against). The
swap-in path: load 0315 (all BF16), replace its Talker with 0422's BF16
Talker, append `*thinker.*` disable globs to `quant_cfg`, calibrate
**text-only** with `zh_en_mixed`, `_export_submodel` the Talker, and merge
the result back into 0422. Text-only is a workaround for the B100 venvs
lacking a cu130-matched torchvision, not a quality choice — see
*Talker NVFP4 calibration* above for why mixed-language coverage is
mandatory either way.

### Next MoE recommended pipeline — Thinker read-through, Talker quantized

The Thinker comes straight from the pre-quantized NVFP4 release
checkpoint (its `config.json` embeds the `quantization_config` +
per-layer ignore list) — no quantize step, export reads it directly:

```bash
# Thinker: direct export from the pre-quantized NVFP4 root
tensorrt-edgellm-export --components thinker $PREQUANT_ROOT $ONNX
```

The Talker is quantized from the BF16 training root and only the
`talker` component is extracted (the run also NVFP4s the Thinker, but
that Thinker is neither exportable — GDN gets quantized — nor needed):

```bash
# Talker: NVFP4 with mixed-language calibration, extract talker only
tensorrt-edgellm-quantize llm \
    --model_dir   $BF16_ROOT \
    --output_dir  $QUANT_ROOT \
    --quantization nvfp4 --text_dataset zh_en_mixed --num_samples 128

tensorrt-edgellm-export --components talker $QUANT_ROOT $ONNX
```

CP FP8 (`--cp_quantization fp8` CP-only pass) and encoder FP8
(`--visual_quantization/--audio_quantization fp8`) compose per their
sections; `code2wav`/`visual`/`audio` components export from either
root.

**Thinker quantization policy**: the Thinker is deliberately NOT
quantized in-tree — the pre-quantized release checkpoint is the source
of truth (its ignore list keeps GDN mixers, attention, layer-0 experts
and the shared expert FP16; only routed experts of layers 1..N-1 are
NVFP4). Should a future release require in-house Thinker quantization,
the unified `--quantization` flow applies as-is — mainline already
supports quantized GDN projections for the Qwen3.5 hybrid family
(NVFP4 resmooth equalises the shared-input GDN projections so they
fuse into a single GEMM) — so no preemptive keep-set or code changes
are made here.

### Talker NVFP4 calibration — mixed-language is mandatory

English-only calibration (the `cnn_dailymail` default) leaves Chinese
activations outside the calibrated scale envelope and measurably
destabilizes sampling in BOTH languages: on tts_bench_20, N=3 median
WER was EN 37% / ZH 8.3% with English-only calibration vs
**EN 4.8% / ZH 7.5%** with `zh_en_mixed` (zh-wikipedia / CNN-DailyMail
interleaved; HF bf16 anchor: 4.3/5.8). Also required for correct
pacing: the runtime emits 4 frames per forced chunk call
(`framesPerCall=4`); the HF reference discards the (m+1)-th lookahead
sample and emitting it (the old value 5) rendered one text-starved
orphan frame per chunk.

Runtime mitigation: an explicit `talker_language` id (instead of the
default "auto" / codec_nothink) independently recovers most of the ZH
loss from missing calibration coverage — pooled 3-run ZH WER on
tts_bench_20: English-only-calibrated Talker 18.3% → 10.3%,
mixed-calibrated 9.8% → 8.1%; EN unchanged within sampling noise. Set
it whenever the input language is known, but it is a mitigation, not a
substitute for mixed-language calibration.

### Talker sampling semantics (HF-aligned)

The runtime mirrors the HF reference generate semantics:

- **Repetition-penalty window resets at every chunk re-prefill** — HF
  reissues a fresh `generate()` per chunk call, so only the tokens
  sampled within the current call (<= m+1) are penalized; the post-text
  final stretch accumulates from zero. A cumulative whole-utterance
  window over-penalizes common codec tokens and audibly degrades long
  utterances.
- **The penalty tracks the freshly-sampled token** (host set and GPU
  buffer stay coherent; an earlier off-by-one wrote the new token gated
  on the previous one).
- **CP (sub-talker) sampling defaults are per-arch**: dense/MoE keep the
  HF-hardcoded `temperature=1.0, top_k=50, top_p=0.8`; Next follows the
  HF `subtalker_*` defaults `temperature=0.9, top_k=50, top_p=1.0`.
- `assistant_instruct` is ignored when `prompt_speaker_codes` is given
  (HF applies it to built-in speakers only).

**FP8-CP deployment note**: with an FP8 CodePredictor engine, keep the
old CP sampling via per-request overrides `"subtalker_temperature":
1.0, "subtalker_top_p": 0.8` (subtalker == CodePredictor). HF's bf16 CP tolerates the full distribution tail
(top_p=1.0), but on FP8 logits the opened tail admits noisy codes: on
tts_bench_20 the HF defaults cost SIM (0.48 vs 0.55 mean over N=3) and
destabilized ZH WER. With the overrides plus the penalty-window fixes,
ECAPA SIM reached 0.578 (gate 0.577) on the best round vs 0.507 before
the fixes.

### CodePredictor FP8 (CP-only pass)

```bash
tensorrt-edgellm-quantize llm \
    --model_dir  $HF_ROOT \
    --output_dir $QUANT_ROOT \
    --cp_quantization fp8 --num_samples 16

tensorrt-edgellm-export --components code_predictor $QUANT_ROOT $ONNX
```

Details in "Optional: CodePredictor FP8" above. Expected artifacts: 30
FP8 body weights (q/k/v/o + gate/up × 5 layers) with per-channel
`weight_scale` + per-tensor `input_scale`; ONNX carries per-channel
weight `DequantizeLinear` with `axis=0` (restored post-export — plain
`torch.onnx.export` drops it and TRT then fails with `K == scaleSize`);
`down_proj` / lm_heads / KV BMMs stay FP16.

### Standalone TTS voice-design support matrix

All HF ``generate_talker_only_prefill`` voice-design controls are wired
through ``qwen3_tts_inference`` (per-request or top-level JSON fields):

| HF parameter | input JSON field | Validated |
|---|---|---|
| ``speaker`` (internal name) | ``speaker`` (accepts voice_map friendly names, e.g. "Ryan") | ✅ every bench run |
| ``prompt_speaker_codes`` (reference-voice cloning) | ``prompt_speaker_codes`` ``[[16 ints] per frame]`` | ✅ ECAPA triangle (clone-vs-source 0.477 > independent same-speaker 0.229) |
| ``talker_assistant_instruct`` (21 styles/emotions, e.g. "cheerful") | ``assistant_instruct`` | ✅ injection verified (14-variant A/B vs HF); strong styles (whispering/sluggish/gloomy) match HF direction on duration+loudness; weak styles are inside single-sample sampling noise on BOTH sides |
| ``talker_language`` (29 languages, "auto" = codec_nothink) | ``talker_language`` | ✅ verified (zh/en explicit ids: WER unchanged-or-better, mismatched id safe). NOTE: benches default to "auto" — set explicitly for known-language input |
| ``talker_system_instruct_ids`` (free-text system instruction) | ``system_instruct`` (tokenized by runtime) | ✅ injection verified; instruction-following strength is model-dependent (HF equally weak on single samples) |
| ``talker_text_in_chunk_n`` / ``talker_codec_output_chunk_m`` | fixed 4/4 (``kTextInChunkN`` / ``framesPerCall``), matches HF defaults | ✅ pacing audited frame-level |
| ``fake_user_mm_hidden_path`` | not exposed (HF training/test hook) | N/A |

### Standalone TTS runtime conditioning

The Next Talker runtime accepts optional conditioning (all resolved from
the Talker config token tables): `voice_map.json` friendly speaker
aliases (copied into the engine dir at build), reference-voice codec
codes (HF `prompt_speaker_codes` semantics — replaces built-in speaker
rows), a style/emotion instruction name
(`talker_assistant_prompt_id_mapping`, e.g. "cheerful"), a target
language (`talker_language_id`; "auto" = codec_nothink path), and a
free-text system instruction. Standalone TTS builds the OmniNext chunked
text-feeding prefill (`buildQwen3OmniNextTalkerPrefill`) — the legacy
`prepareTalkerInput` path produces unconditioned prompts and degenerate
speech on this arch.

## Validation status (internal)

| Variant | Backbone | Result |
|---------|----------|--------|
| MoE | NVFP4 | full E2E on B100 (SM100, TRT 10.16): text sanity pass, 20-sample TTS WER 1.45%, 0/20 catastrophic |
| MoE | INT4 AWQ g128 | full E2E on B100: TTS WER 6.72%, 0/20 catastrophic (vs 7.80% A100 baseline); quant ~3.3 h with AWQ-Lite alpha_step 0.25 |
| Dense | NVFP4 | full E2E on B100: TTS WER 3.08%, 0/20 catastrophic |
| Dense | INT4 GPTQ | external gptqmodel flow; full E2E on B100: TTS WER 1.67% (requires the talker `quantization_config` inheritance fix in this branch); also verified on DRIVE Orin |
| Next | INT4 AWQ g128 | Qwen3-Omni Next bring-up E2E (text + TTS + audio understanding) |
| Next | NVFP4 (Talker) | B100 E2E on tts_bench_20: N=3 median TTS WER EN 4.8% / ZH 7.5% (HF bf16 anchor 4.3/5.8) with mixed-language calibration; ECAPA SIM 0.507 vs 0.577 gate — bf16 Talker runtime remains the mainline for production voice quality |
| Next dense 3B | CP FP8 | PTQ (16 samples) + Talker-root ckpt + ONNX QDQ validated on B100; engine E2E pending |
| Next MoE 23A2.6B | CP FP8 | PTQ (~21 min B100) + ONNX QDQ + fp16 projection sidecar validated; engine E2E validated — WER parity with fp16 CP within run noise, SIM 0.510 vs 0.507 |
| Next MoE 23A2.6B | ViT/AuT FP8 | B100 E2E validated. Visual (111 FP8 layers) + audio (199 FP8 layers) towers quantized in the joint calib pass, ONNX carries FP8 QDQ, engines build clean. Audio: LibriSpeech-20 WER 3.29% (FP8) vs 3.05% (FP16), Δ within noise, matches 3.30% FP16 baseline. Visual: shape/color description byte-identical to FP16. Validated against a paged-KV FP16 Thinker rebuilt from the BF16 root on the current branch. |
| All | CP speculative decoding | A100 A/B against the autoregressive loop on the standalone TTS path; kernel-level tests cover sampling/verify equivalence with the DSpark reference |

## Notes

- **Thinking mode:** the runtime supports it (`enable_thinking` in the
  request → `<think>`/`</think>` handling, secondary-EOS suppression until
  the think block closes), but the shipped `0422` checkpoint's
  `chat_template.jinja` is non-thinking: its generation prompt hard-codes an
  empty `<think>\n\n</think>` block and has no `enable_thinking` branch, so
  the flag is a no-op on this checkpoint (`true`/`false` produce byte-identical
  prompts). The model still reasons step-by-step in-line; it just doesn't emit
  an explicit think block. A thinking-enabled checkpoint (template with an
  `enable_thinking` branch) is needed to exercise the runtime's think path.
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
