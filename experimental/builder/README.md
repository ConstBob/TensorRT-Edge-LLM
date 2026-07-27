# Experimental ONNX-less TensorRT Builder

`tensorrt-edgellm-build` builds TensorRT engines directly from an Edge-LLM
checkpoint. It does not export, parse, or cache an ONNX graph. This is an
experimental alternative to the supported
`tensorrt-edgellm-export` plus component-builder workflow.

Use the
[ONNX-less engine build user guide](../../docs/source/user_guide/getting_started/direct-engine-builder.md)
for complete build and runtime commands. The
[developer design](../../docs/source/developer_guide/software-design/onnxless-builder.md)
traces model calls to TensorRT APIs and documents how to add an operation or
model family.

The input must be a Hugging Face-style checkpoint supported by the registry in
[`models/registry.py`](models/registry.py). Plain FP16/BF16 and supported
pre-quantized safetensors checkpoints use the same command.

## Install

Install the repository package and the TensorRT Python wheel that matches the
TensorRT libraries used to compile Edge-LLM:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install .
.venv/bin/python -m pip install \
  /path/to/TensorRT/python/tensorrt-<version>-cp312-none-linux_x86_64.whl
```

Build Edge-LLM before invoking the frontend. The default plugin path is
`build/libNvInfer_edgellm_plugin.so`.

The plugin must include the kernels required by the selected model. In
particular, Qwen3.5 hybrid models require Gated DeltaNet, and Gemma4 can require
FFPA for its larger attention head dimensions. Configure both when those
families are needed:

```bash
cmake -S . -B build \
  -DTRT_PACKAGE_DIR=/path/to/TensorRT \
  -DENABLE_CUTE_DSL=ALL
cmake --build build -j16
```

## Build Every Component

The default component selection is `all`. One command discovers the checkpoint
family and builds each component in runtime order:

```bash
.venv/bin/tensorrt-edgellm-build \
  --model-dir /path/to/checkpoint \
  --engine-dir /path/to/engines \
  --plugin-path /path/to/build/libNvInfer_edgellm_plugin.so \
  --max-input-len 2048 \
  --max-kv-cache-capacity 4096 \
  --max-batch-size 1
```

`--components` is only needed for an intentional partial rebuild:

```bash
.venv/bin/tensorrt-edgellm-build \
  --model-dir /path/to/checkpoint \
  --engine-dir /path/to/engines \
  --components visual,audio \
  --plugin-path /path/to/build/libNvInfer_edgellm_plugin.so
```

The available components are `llm`, `visual`, `audio`, `talker`,
`code-predictor`, `code2wav`, and `action`. A model can expose any subset:

| Model kind | Components built by `all` |
|---|---|
| Text LLM | LLM |
| VLM | LLM, visual |
| ASR | LLM, audio |
| Omni | LLM, visual, audio, and model-owned speech components |
| TTS | Talker, code-predictor, Code2Wav |
| Alpamayo | LLM, visual, action |

Each component is a separate TensorRT engine because components have different
shape profiles and runtime I/O contracts. The command still builds all of them
in one invocation and writes the layout consumed by the existing C++ runtime:

| Component | Engine path |
|---|---|
| LLM | `llm.engine` |
| Visual | `visual/visual.engine` |
| Audio | `audio/audio_encoder.engine` |
| Talker | `talker/llm.engine` |
| Code predictor | `code_predictor/llm.engine` |
| Code2Wav | `code2wav/code2wav.engine` |
| Action | `action/action.engine` |

Runtime configs, tokenizers, chat templates, embeddings, and other model-owned
artifacts are emitted beside those engines.

## Speculative Decoding

One invocation builds both `spec_base.engine` and `spec_draft.engine`, plus any
non-LLM components declared by the target checkpoint.

EAGLE3 uses a paired draft checkpoint:

```bash
.venv/bin/tensorrt-edgellm-build \
  --model-dir /path/to/target \
  --draft-model-dir /path/to/eagle3-draft \
  --spec-type eagle3 \
  --engine-dir /path/to/engines \
  --plugin-path /path/to/build/libNvInfer_edgellm_plugin.so
```

Qwen3.5 native MTP reads the draft layers from the target checkpoint:

```bash
.venv/bin/tensorrt-edgellm-build \
  --model-dir /path/to/qwen3.5-checkpoint \
  --spec-type mtp \
  --engine-dir /path/to/engines \
  --plugin-path /path/to/build/libNvInfer_edgellm_plugin.so
```

DFlash uses its paired draft checkpoint and model-owned DFlash cache contract:

```bash
.venv/bin/tensorrt-edgellm-build \
  --model-dir /path/to/target \
  --draft-model-dir /path/to/dflash-draft \
  --spec-type dflash \
  --engine-dir /path/to/engines \
  --plugin-path /path/to/build/libNvInfer_edgellm_plugin.so
```

Gemma4 MTP uses the matched assistant checkpoint:

```bash
.venv/bin/tensorrt-edgellm-build \
  --model-dir /path/to/gemma4-target \
  --draft-model-dir /path/to/gemma4-assistant \
  --spec-type gemma4_mtp \
  --engine-dir /path/to/engines \
  --plugin-path /path/to/build/libNvInfer_edgellm_plugin.so
```

Use the normal `llm_inference --specDecode` runtime options with the resulting
directory.

## Loader And Modeling Contract

The builder intentionally owns its model definitions instead of importing the
ONNX exporter:

1. `core/config.py` reads checkpoint metadata and determines the exact model
   variant and component set.
2. `core/weights.py` and model-owned `weights.py` files load and convert
   safetensors on CPU.
3. `models/<family>/modeling_*.py` constructs a model-specific
   `NetworkModule` for each component.
4. `ops.functional` is the single PyTorch-like operation surface. TensorRT
   native layers and Edge-LLM extension layers are both called as `F.<op>`;
   model code does not select or expose the lowering kind.
5. `core/builder.py` creates one TensorRT `INetwork` and optimization profile
   per component, serializes the engine, and releases checkpoint storage.
6. Model-owned artifact writers emit the runtime contract consumed by the C++
   executables.

`Module` represents a nested layer and never owns network I/O.
`NetworkModule` represents exactly one TensorRT `INetwork` and is the only
module type allowed to declare component inputs and outputs. Model families
share common operations, but they do not reuse another family's model
definition.

Functional operations are grouped by semantic domain under
`ops/functional/`: core tensor operations, attention, distributed, MoE,
recurrent, and speculative. They all accept symbolic tensors and ordinary
Python attributes. TensorRT creator lookup and attribute encoding are private
lowering details in `ops/backend.py`.

For example, an FP16 `Linear` call reaches
`F.linear_from_weights -> Net.linear_from_weights -> Net.linear`, which emits
`add_constant`, `add_matrix_multiply`, and an optional `add_elementwise`.
RMSNorm is decomposed into native cast/reduce/elementwise/unary layers.
Attention and other Edge-LLM operations pass through the same functional API
and are lowered by `Net.operation` to `add_plugin_v3`. Model code never handles
these TensorRT details.

INT4 GEMM qweights/scales, INT4 MoE qweights, and all NVFP4 MoE plugin weight
buffers are static TensorRT network inputs by default. They are stored in
component-local safetensors files and listed in the runtime config. They remain
module parameters: model `forward()` signatures do not accept weight tensors.
INT4 MoE scales remain embedded to match the production plugin contract.

## Supported Families

The explicit registry currently includes Llama, Mistral, Qwen2, Qwen3,
Qwen3-MoE, Qwen2/2.5/3-VL, Qwen3.5 dense and MoE, Qwen3-ASR, Qwen3-Omni,
Qwen3-TTS, InternVL3/3.5, Phi-4 Multimodal, Nemotron-H/Omni, Gemma4, and
Alpamayo. Unsupported `model_type` values fail before a TensorRT network is
created and report the registered choices.

The user guide contains an explicit
[implementation, CI, and known-gap matrix](../../docs/source/user_guide/getting_started/direct-engine-builder.md#support-and-validation-status).
This includes features such as FP8 embedding, reduced vocabulary, LoRA,
quantized formats, speculative modes, and current differences from the
supported ToT workflow.

## CI

The standard Edge-LLM L0 jobs remain enabled. Direct-builder coverage adds:

- A30 FP16 dense and EAGLE3 target/draft end-to-end cases
- RTX 5090 FP8 and NVFP4 dense end-to-end cases

Each case builds directly from a checkpoint, executes the matching C++ runtime,
checks the runtime artifact layout, and applies the existing accuracy check.
The RTX 5090 cases are part of the existing 50-series job, so they do not add a
new runner allocation.

See the
[ONNX-less builder user guide](../../docs/source/user_guide/getting_started/direct-engine-builder.md#build-time-comparison)
for the reproducible original-versus-direct build timing comparison.
