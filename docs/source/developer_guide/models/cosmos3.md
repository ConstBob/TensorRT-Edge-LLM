# Cosmos3-Edge

Cosmos3-Edge is an omni model for Physical AI. One `nvidia/Cosmos3-Edge`
checkpoint supports two tasks, each on its own Edge-LLM path:

| Use case | Input → Output | Runtime | Export |
|---|---|---|---|
| [Policy generation](#policy-generation) | image + instruction → robot action chunk | experimental component runtime (`cosmos3_policy_*`) | `--task policy` |
| [Multimodal reasoning](#multimodal-reasoning) | image + prompt → reasoning text | standard VLM runtime (`llm_inference`) | `--task reasoning` |

> **Prerequisites:** Complete the [Installation Guide](../../user_guide/getting_started/installation.md).
> Policy generation requires adding `-DBUILD_EXPERIMENTAL_MODELS=ON` to the CMake
> configure step; the `cosmos3_policy_build` / `cosmos3_policy_inference` binaries
> are then part of the regular build. Multimodal reasoning uses only the standard
> binaries. On Thor/SM110 and GB10/SM120, build with `-DENABLE_CUTE_DSL=ALL` so
> the CuTe DSL FMHA kernels the ViT and policy attention use are linked.

Common paths used by both use cases (the runtime environment is set up by the
[Installation Guide](../../user_guide/getting_started/installation.md)):

```bash
export HF_CHECKPOINT=/path/to/Cosmos3-Edge
export ONNX_DIR=$HOME/cosmos3_onnx
export ENGINE_DIR=$HOME/cosmos3_engines
```

## Policy Generation

### 1. Export (x86 host, CPU-only)

```bash
tensorrt-edgellm-export "$HF_CHECKPOINT" "$ONNX_DIR" --task policy --dtype float16
```

This writes the three policy components and the tokenizer artifacts:

```text
$ONNX_DIR/
  und_prefill/ model.onnx config.json embed_tokens.safetensors
  gen/         model.onnx config.json
  vae_encoder/ model.onnx config.json
  text_tokenizer/ tokenizer.json processed_chat_template.json ...
```

Each `config.json` is the component contract (optimization profile, tensor
shapes, and runtime constants) consumed by the builder and the C++ runtime.
`--components und_prefill|gen|vae_encoder` re-runs a single component.

### 2. Build engines

```bash
for comp in und_prefill gen vae_encoder; do
  ./build/experimental_models/cosmos3/examples/cosmos3_policy_build \
      --onnxDir "$ONNX_DIR/$comp" --engineDir "$ENGINE_DIR" --component "$comp"
done
cp -r "$ONNX_DIR/text_tokenizer" "$ENGINE_DIR/"
cp "$ONNX_DIR/und_prefill/embed_tokens.safetensors" "$ENGINE_DIR/"
```

`--maxBatchSize N` widens each engine's batch axis so one engine serves any
request batch in `[1, N]` (default 1).

### 3. Run

```bash
./build/experimental_models/cosmos3/examples/cosmos3_policy_inference \
    --engineDir "$ENGINE_DIR" \
    --image observation.png \
    --prompt "Pick up the banana and place it in the bowl." \
    --output action.json
```

The CLI preprocesses the image, formats the instruction with the model chat
template, and runs VAE encode → UND prefill → GEN denoise loop in process.
Flags:

- `--image`: PNG/JPG conditioning observation (resized to `736x544`, shared across the batch).
- `--prompt`: text instruction; repeat for a batched request (batch = prompt count; requires engines built with `--maxBatchSize` ≥ count and equal-length tokenized prompts — the UND graph carries no attention mask).
- `--steps`: diffusion denoise steps (default 4).
- `--seed`: initial-noise seed (default 0; generation is deterministic per seed).
- `--output`: JSON response by default; a `.safetensors` suffix writes the raw action tensor.

JSON response (`shape` is `[B, chunk, dim]`; for `B == 1` the `action` field is
`[chunk][dim]`, for `B > 1` one such block per prompt):

```json
{
  "action": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
  "shape": [1, 32, 8],
  "domain": "droid_lerobot",
  "num_inference_steps": 4,
  "finite": true
}
```

> **Accuracy note.** Cosmos3-Edge sets `use_und_k_norm_for_gen=True`: the UND K
> that the GEN tower cross-attends to is RMSNorm-normalized (per head, before
> RoPE). The UND-prefill export applies this to the exported `und_k` only; the
> text tower's own self-attention keeps the raw K (`qk_norm_for_text=False`).

## Multimodal Reasoning

The reasoner is a standard Edge-LLM VLM — a SigLIP2 vision encoder +
PatchMerger and an autoregressive text decoder. The only model-specific pieces
are on the export side: the decoder swaps SwiGLU for the Nemotron-H
squared-ReLU MLP (`Cosmos3ReasonerCausalLM`), and the exporter maps the
checkpoint's native flat schema onto the shared decoder names.

### 1. Export (x86 host, CPU-only)

```bash
tensorrt-edgellm-export "$HF_CHECKPOINT" "$ONNX_DIR" --task reasoning --dtype float16
```

```text
$ONNX_DIR/
  llm/    model.onnx config.json embedding.safetensors tokenizer.json ...
  visual/ model.onnx config.json preprocessor_config.json
```

### 2. Build engines

```bash
./build/examples/llm/llm_build \
    --onnxDir "$ONNX_DIR/llm" --engineDir "$ENGINE_DIR/reasoner" \
    --maxInputLen 2048 --maxKVCacheCapacity 4096 --maxBatchSize 1

./build/examples/multimodal/visual_build \
    --onnxDir "$ONNX_DIR/visual" --engineDir "$ENGINE_DIR/reasoner"
```

### 3. Run

```bash
./build/examples/llm/llm_inference \
    --engineDir "$ENGINE_DIR/reasoner" \
    --multimodalEngineDir "$ENGINE_DIR/reasoner" \
    --inputFile input.json --outputFile output.json
```

`input.json` is the standard request format; image messages use
`{"type": "image", "image": "/path/to.jpg"}` (see
[Input Format](../../user_guide/format/input-format.md)). Text-only requests
work with the same engines by omitting `--multimodalEngineDir`.

## RoboLab Integration

`experimental_models/cosmos3/robolab/policy_server.py` exposes the policy JSON
action contract over HTTP by wrapping `cosmos3_policy_inference`;
`cosmos3_client.py` turns simulator observations into action chunks. This
integration is optional and lives outside the core Python package.

```bash
python -m experimental_models.cosmos3.robolab.policy_server \
    --binary ./build/experimental_models/cosmos3/examples/cosmos3_policy_inference \
    --engine-dir "$ENGINE_DIR" \
    --host 0.0.0.0 --port 8080
```
