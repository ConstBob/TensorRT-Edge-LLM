# Input JSON Format

This guide describes the input JSON format for the LLM inference tool. The format supports text-only and multimodal inputs, multi-turn conversations, LoRA adapters, and advanced features.

## Basic Structure

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "max_generate_length": 256,
    "apply_chat_template": true,
    "enable_thinking": false,
    "available_lora_weights": {},
    "requests": [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Your message here"
                }
            ],
            "lora_name": "optional_lora_name",
            "save_system_prompt_kv_cache": false
        }
    ]
}
```


## Parameters

### Required
- **`requests`**: Array of conversation requests

### Optional Global Parameters
- **`batch_size`** (default: 1): Number of requests per batch
- **`temperature`** (default: 1.0): Sampling temperature (0.0 = deterministic)
- **`top_p`** (default: 0.8): Nucleus sampling threshold
- **`top_k`** (default: 50): Top-k sampling parameter
- **`max_generate_length`** (default: 256): Maximum tokens to generate
- **`apply_chat_template`** (default: true): Apply chat template formatting
- **`enable_thinking`** (default: false): Enable thinking mode (Qwen3+)
- **`available_lora_weights`** (default: {}): Map of LoRA adapter names to file paths

### Request Fields
- **`messages`** (required): Array of conversation messages
- **`lora_name`** (optional): LoRA adapter name from `available_lora_weights`
- **`save_system_prompt_kv_cache`** (optional): Cache system prompt KV for reuse

### Message Fields
- **`role`**: `"system"`, `"user"`, or `"assistant"`
- **`content`**: String (text-only) or Array (multimodal)

**Content Array Format:**
- Text: `{"type": "text", "text": "..."}`
- Image: `{"type": "image", "image": "/path/to/image.jpg"}`
- Video: `{"type": "video", "video": "/path/to/video.mp4"}`

## Examples

### Basic Text Input

```json
{
    "batch_size": 1,
    "max_generate_length": 256,
    "requests": [
        {
            "messages": [
                {"role": "user", "content": "What is machine learning?"}
            ]
        }
    ]
}
```

### Multi-Turn Conversation

```json
{
    "requests": [
        {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
                {"role": "user", "content": "What is the population?"}
            ]
        }
    ]
}
```

### Multimodal Input (Vision-Language Models)

```json
{
    "requests": [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "/path/to/image.jpg"},
                        {"type": "text", "text": "Describe this image."}
                    ]
                }
            ]
        }
    ]
}
```

### LoRA Adapters

```json
{
    "available_lora_weights": {
        "french": "/path/to/french_adapter.safetensors"
    },
    "requests": [
        {
            "messages": [
                {"role": "user", "content": "Translate to French."}
            ],
            "lora_name": "french"
        }
    ]
}
```

**Note:** All requests in the same batch must use the same LoRA adapter.

### System Prompt KV Cache

```json
{
    "requests": [
        {
            "messages": [
                {"role": "system", "content": "Long system prompt..."},
                {"role": "user", "content": "Question?"}
            ],
            "save_system_prompt_kv_cache": true
        }
    ]
}
```

### Raw Format (No Chat Template)

```json
{
    "apply_chat_template": false,
    "requests": [
        {
            "messages": [
                {"role": "user", "content": "Raw text without special tokens"}
            ]
        }
    ]
}
```

## Notes

- System prompt: Uses provided system message, or model default from chat template
- LoRA: All requests in same batch must use same adapter
- Paths: Use absolute or relative paths for images/videos
- Format: Follows OpenAI chat completion API structure
