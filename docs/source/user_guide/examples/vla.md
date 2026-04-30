# VLA (Vision-Language-Action) Models

Complete workflow for Vision-Language-Action (VLA) models — models that combine a VLM backbone with an action expert to produce low-level robot actions from image and text inputs.

**Currently supported model:** [Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)

> **Prerequisites:** Complete the [Installation Guide](../getting_started/installation.md) for both x86 host and edge device before proceeding. The Alpamayo 1 model requires HuggingFace login before downloading.

---

## Architecture Overview

Action models run as a two-stage chained pipeline:

1. **VLM backbone** — processes image, text, and proprioceptive inputs and generates language output
2. **Action expert** — takes VLM output and produces low-level robot actions

Both stages are run with a single `action_inference` invocation. The engines are built separately and loaded together at runtime.

---

## Step 1: Export (x86 Host)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Alpamayo-R1-10B
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Download Alpamayo-R1-10B from HuggingFace
hf download nvidia/Alpamayo-R1-10B --local-dir=Alpamayo-R1-10B

# Export language model backbone
tensorrt-edgellm-export-llm \
  --model_dir Alpamayo-R1-10B \
  --output_dir $MODEL_NAME/onnx/llm

# Export visual encoder
tensorrt-edgellm-export-visual \
  --model_dir Alpamayo-R1-10B \
  --output_dir $MODEL_NAME/onnx/visual

# Export action expert (FP16 only)
tensorrt-edgellm-export-action \
  --model_dir Alpamayo-R1-10B \
  --output_dir $MODEL_NAME/onnx/action \
  --max_kv_cache_capacity 4096
```

> **Note:** Only FP16 is supported for the action expert. The `--max_kv_cache_capacity` value must match the `--maxKVCacheCapacity` used when building the LLM engine in Step 3.

## Step 2: Transfer to Device

```bash
scp -r $MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

## Step 3: Build Engines (Thor Device)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Alpamayo-R1-10B
cd ~/TensorRT-Edge-LLM

# Build language model engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/llm \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --maxInputLen 3424 \
  --maxKVCacheCapacity 4096 \
  --maxBatchSize 6

# Build visual encoder engine
./build/examples/multimodal/visual_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/visual \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --minImageTokens 160 \
  --maxImageTokens 18432 \
  --maxImageTokensPerImage 192

# Build action expert engine
./build/examples/multimodal/action_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/action \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --maxBatchSize 6
```

Build time: < 5 minutes

## Step 4: Run Inference (Thor Device)

Create an input file `$WORKSPACE_DIR/input_action.json`. The `trajectory` content item provides the robot's past trajectory history as a sequence of `[x, y, z]` positions (replace image paths with actual files):

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "max_generate_length": 128,
    "requests": [
        {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a driving assistant that generates safe and accurate actions."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "/path/to/frame_00.png"},
                        {"type": "image", "image": "/path/to/frame_01.png"},
                        {"type": "image", "image": "/path/to/frame_02.png"},
                        {"type": "image", "image": "/path/to/frame_03.png"},
                        {"type": "image", "image": "/path/to/frame_04.png"},
                        {"type": "image", "image": "/path/to/frame_05.png"},
                        {"type": "image", "image": "/path/to/frame_06.png"},
                        {"type": "image", "image": "/path/to/frame_07.png"},
                        {"type": "image", "image": "/path/to/frame_08.png"},
                        {"type": "image", "image": "/path/to/frame_09.png"},
                        {"type": "image", "image": "/path/to/frame_10.png"},
                        {"type": "image", "image": "/path/to/frame_11.png"},
                        {"type": "image", "image": "/path/to/frame_12.png"},
                        {"type": "image", "image": "/path/to/frame_13.png"},
                        {"type": "image", "image": "/path/to/frame_14.png"},
                        {"type": "image", "image": "/path/to/frame_15.png"},
                        {
                            "type": "trajectory",
                            "trajectory": [
                                [-13.570470809936523, 0.06137955188751221, -0.021795138716697693],
                                [-12.617349624633789, 0.05502257123589516, -0.021485785022377968],
                                [-11.672354698181152, 0.04983367770910263, -0.019761288538575172],
                                [-10.733928680419922, 0.04469458386301994, -0.01891801320016384],
                                [-9.802403450012207, 0.03998061269521713, -0.019592544063925743],
                                [-8.87903118133545, 0.034502264112234116, -0.016824787482619286],
                                [-7.963467121124268, 0.03007129393517971, -0.011910118162631989],
                                [-7.054953098297119, 0.02519317716360092, -0.008388019166886806],
                                [-6.15440559387207, 0.020703254267573357, -0.003812047652900219],
                                [-5.261260032653809, 0.016889382153749466, 0.0007052596774883568],
                                [-4.374241828918457, 0.012481293641030788, 0.004680476617068052],
                                [-3.4911155700683594, 0.00914590060710907, 0.0029235705733299255],
                                [-2.611931085586548, 0.005678099580109119, 7.79956899350509e-05],
                                [-1.736931562423706, 0.003552841255441308, -0.0018224255181849003],
                                [-0.8664319515228271, 0.001968142343685031, -0.001738768769428134],
                                [0.0, 0.0, 0.0]
                            ]
                        },
                        {
                            "type": "text",
                            "text": "output the chain-of-thought reasoning of the driving process, then output the future trajectory."
                        }
                    ]
                }
            ]
        }
    ]
}
```

Run inference:

```bash
cd ~/TensorRT-Edge-LLM

./build/examples/multimodal/action_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --multimodalEngineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --inputFile $WORKSPACE_DIR/input_action.json \
  --outputFile $WORKSPACE_DIR/output_action.json
```

Check `output_action.json` for the response. Each entry includes `output_text` (VLM language output) and `output_trajectory` (predicted (accel, kappa) acceleration and curvature pairs).

