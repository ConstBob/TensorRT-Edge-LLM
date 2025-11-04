# Input JSON File Format

This document describes the required format for the input JSON file used with the LLM inference tool.

## Overview

The input JSON file contains configuration parameters and a list of messages to be processed by the LLM. The tool supports both text-only and multimodal (text + images) inputs.

## File Structure

The JSON file must contain the following top-level structure:

```json
{
    "batch_size": <integer>,
    "temperature": <float>,
    "top_p": <float>,
    "top_k": <integer>,
    "max_generate_length": <integer>,
    "default_system_prompt": "<string>",
    "available_lora_weights": {  // optional. Only needed for LoRA engines.
        "<name1>": "<path1>",
        "<name2>": "<path2>",
        ...
    },
    "messages": [
        {
            "user": "<string>",
            "system": "<string>",  // optional
            "images": ["<path1>", "<path2>", ...],  // optional
            "lora_weights": "<name>" // optional. Reference to a name in available_lora_weights.
        }
    ]
}
```

## LoRA (Low-Rank Adaptation) Support

LoRA enables fine-tuned model inference using adapter weights. Requirements:
- TensorRT engine built with LoRA support
- LoRA weights in `.safetensors` format
- Different LoRA weights within the same batch is not supported. LoRA weights within the same batch should be the same.
- LoRA weights must be registered in `available_lora_weights` and referenced by name in messages

## Global Parameters

### Required Parameters

- **`messages`** (array): A list of message objects to be processed. Each message represents a conversation turn.

### Optional Parameters

- **`batch_size`** (integer, default: 1): Number of messages to process in a single batch
- **`temperature`** (float, default: 1.0): Controls randomness in generation (0.0 = deterministic, higher = more random)
- **`top_p`** (float, default: 0.8): Nucleus sampling parameter (0.0-1.0)
- **`top_k`** (integer, default: 50): Top-k sampling parameter
- **`max_generate_length`** (integer, default: 256): Maximum number of tokens to generate
- **`default_system_prompt`** (string, default: ""): Default system prompt applied to all messages unless overridden
- **`available_lora_weights`** (object): Dictionary mapping LoRA weight names to file paths. Only needed for LoRA-enabled engines

## Message Objects

Each message in the `messages` array can contain:

### Required Fields

- **`user`** (string): The user's input prompt or question

### Optional Fields

- **`system`** (string): System prompt specific to this message. If not provided, uses `default_system_prompt`
- **`images`** (array of strings): List of image file paths for multimodal inputs
- **`lora_weights`** (string): Name reference to a LoRA weight defined in `available_lora_weights`. Only used with LoRA-enabled engines

## Examples

### Text-Only Input

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "max_generate_length": 256,
    "default_system_prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
    "messages": [
        {
            "user": "<|im_start|>user\nIntroduce NVIDIA and introduce the CEO of this company.<|im_end|>\n<|im_start|>assistant\n"
        },
        {
            "user": "<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n<|im_start|>assistant\n"
        }
    ]
}
```

### Multimodal Input

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "max_generate_length": 256,
    "default_system_prompt": "You are a helpful assistant.",
    "messages": [
        {
            "user": "Describe this image.",
            "images": [
                "image.jpeg"
            ]
        },
        {
            "user": "Identify the similarities between these images.",
            "images": [
                "image1.jpeg",
                "image2.jpeg"
            ]
        },
        {
            "user": "Give me a short introduction to large language model."
        }
    ]
}
```

### LoRA Input

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "max_generate_length": 256,
    "default_system_prompt": "You are a helpful assistant.",
    "available_lora_weights": {
        "adapter1": "/path/to/lora_weights.safetensors",
        "adapter2": "/path/to/another_adapter.safetensors"
    },
    "messages": [
        {
            "user": "Your prompt here",
            "lora_weights": "adapter1"
        },
        {
            "user": "Another prompt with images",
            "images": [
                "image1.jpg", 
                "image2.jpg"
            ],
            "lora_weights": "adapter2" // This will error out if batch_size = 2.
        }
    ]
}
```

## Processing Behavior

1. **Batching**: Messages are processed in batches according to the `batch_size` parameter
2. **System Prompts**: Each message can have its own system prompt, or it will use the default
3. **Image Loading**: Images are loaded from the specified file paths during processing
4. **LoRA Weights**: When specified, LoRA adapter weights are loaded and applied per batch for fine-tuned inference. LoRA weights must first be defined in `available_lora_weights` and then referenced by name in messages
5. **Error Handling**: The tool will throw errors if:
   - The JSON file cannot be parsed
   - A message is missing the required `user` field
   - The `messages` field is not an array
   - LoRA weights are not the same for different prompts inside the same expected batch
   - A referenced LoRA weight name is not defined in `available_lora_weights`

## Notes

- Image paths should be relative to the working directory or absolute paths
- For LLM models, system prompts and user prompts should follow the model's expected format (e.g., chat templates). For multimodal models, chat template for system prompts and user prompts will be handled in preprocessing.