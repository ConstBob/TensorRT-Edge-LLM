# Tokenizer

The tokenizer module provides a C++ implementation of the BPE (Byte Pair Encoding) tokenizer used in Large Language Models. It supports loading HuggingFace tokenizer configurations and provides efficient tokenization and detokenization capabilities.

## Features

- **BPE Tokenization**: Implements Byte Pair Encoding algorithm for efficient text tokenization
- **HuggingFace Support**: Loads tokenizer configurations from HuggingFace model directories
- **Special Token Handling**: Supports special tokens like BOS, EOS, PAD, and UNK
- **Unicode Support**: Handles Unicode text and special characters
- **Memory Efficient**: Optimized for edge platforms with minimal memory footprint

## Usage Example

```cpp
#include <tokenizer.h>
#include <iostream>

int main()
{
    auto tokenizer = std::make_unique<drivellm::tokenizer::Tokenizer>();

    // Load tokenizer from HuggingFace model directory
    tokenizer->loadFromHF("./Meta-Llama-3-8B-Instruct");

    // Tokenize text
    std::string input = "<|begin_of_text|>hello¢ nvidia，你好！👍<|end_of_text|>";
    auto tokens = tokenizer->encode(input, false);

    // Detokenize back to text
    auto output = tokenizer->decode(tokens);
    std::cout << output << std::endl;
    assert(output == input);

    return 0;
}
```

## API Reference

### Tokenizer Class

The main `Tokenizer` class provides the following key methods:

- `loadFromHF(path)`: Load tokenizer configuration from HuggingFace model directory
- `encode(text, addBos, addEos)`: Tokenize text into token IDs
- `decode(tokens, skipSpecialTokens)`: Convert token IDs back to text
- `getNumVocab()`: Get vocabulary size
- `getBosId()`, `getEosId()`, `getPadId()`, `getUnkId()`: Get special token IDs

### BPE Class

The underlying `BPE` class handles the core tokenization algorithm:

- `tokenize(text, output)`: Perform BPE tokenization on text
- `detokenize(tokens, output, skipSpecialTokens)`: Convert tokens back to text
- `specialTokenPartition(text, partitions)`: Handle special token detection and partitioning

## Supported Tokenizer Formats

The tokenizer supports loading from HuggingFace model directories containing:
- `tokenizer.json`: Main tokenizer configuration with vocabulary and merge rules
- `tokenizer_config.json`: Special token definitions and configuration

## Integration

The tokenizer is designed to work seamlessly with the TensorRT Edge-LLM framework and can be used with any supported LLM model that uses BPE tokenization.
