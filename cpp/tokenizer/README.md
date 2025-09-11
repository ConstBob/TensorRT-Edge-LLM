# Tokenizer

The tokenizer module provides a comprehensive C++ implementation of tokenization algorithms used in Large Language Models. It supports loading HuggingFace tokenizer configurations and provides efficient tokenization and detokenization capabilities with robust Unicode support.

## Features

- **BPE Tokenization**: Complete Byte Pair Encoding algorithm implementation with merge rule processing
- **HuggingFace Compatibility**: Full support for HuggingFace tokenizer.json and tokenizer_config.json formats
- **Special Token Handling**: Comprehensive support for special tokens (BOS, EOS, PAD, UNK) with automatic detection and partitioning
- **Advanced Pre-tokenization**: Configurable regex-based pre-tokenization with sequence support
- **Unicode Support**: Full Unicode text processing with category-aware regex handling and UTF-8 encoding/decoding
- **Modular Architecture**: Clean separation between pre-tokenization, encoding, and special token handling
- **Memory Efficient**: Optimized for edge platforms with smart memory management and minimal allocations
- **Robust Error Handling**: Comprehensive error checking and graceful failure handling

## Usage Example

```cpp
#include "tokenizer.h"
#include <iostream>

int main()
{
    // Create tokenizer instance
    drivellm::tokenizer::Tokenizer tokenizer;

    // Load tokenizer from HuggingFace model directory
    if (!tokenizer.loadFromHF("./Qwen2.5-7B-Instruct"))
    {
        std::cerr << "Failed to load tokenizer" << std::endl;
        return 1;
    }

    // Tokenize text with special tokens
    std::string input = "Hello, world! This is a test with Unicode: 你好世界 🌍";
    auto tokens = tokenizer.encode(input, true, true);  // addBos=true, addEos=true

    std::cout << "Input: " << input << std::endl;
    std::cout << "Tokens (" << tokens.size() << "): ";
    for (auto token : tokens)
    {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    // Detokenize back to text
    auto output = tokenizer.decode(tokens, false);  // skipSpecialTokens=false
    std::cout << "Output: " << output << std::endl;

    // Check vocabulary info
    std::cout << "Vocabulary size: " << tokenizer.getNumVocab() << std::endl;
    std::cout << "BOS ID: " << tokenizer.getBosId() << std::endl;
    std::cout << "EOS ID: " << tokenizer.getEosId() << std::endl;

    return 0;
}
```

## Architecture

The tokenizer consists of several key components:

### Core Components

1. **Tokenizer**: Main orchestrator class that coordinates all tokenization steps
2. **PreTokenizer**: Handles text preprocessing and regex-based splitting
3. **TokenEncoder**: Implements the core BPE encoding algorithm
4. **TokenizerUtils**: Provides Unicode utilities and helper functions

### Pre-tokenization

The pre-tokenization stage supports:
- **RegexSplit**: Regex pattern-based text splitting with Unicode category support
- **Sequence**: Chaining multiple pre-tokenization steps
- **Unicode Handling**: Full Unicode category support (\\p{L}, \\p{N}, \\p{P}) with proper collapsing

## API Reference

### Tokenizer Class

The main `Tokenizer` class provides the following methods:

#### Core Methods
- `bool loadFromHF(const std::filesystem::path& modelDir)`: Load tokenizer from HuggingFace model directory
- `std::vector<Rank> encode(const std::string& text, bool addBos = false, bool addEos = false)`: Tokenize text into token IDs
- `std::string decode(const std::vector<Rank>& tokens, bool skipSpecialTokens = false)`: Convert token IDs back to text
- `bool isInitialized()`: Check if tokenizer is properly initialized

#### Accessors
- `int getNumVocab()`: Get vocabulary size
- `Rank getBosId()`, `getEosId()`, `getPadId()`, `getUnkId()`: Get special token IDs

### TokenEncoder Class

The `TokenEncoder` class handles the core encoding algorithm:

- `bool initialize(const TokenToRanks& vocab, const TokenToRanks& specialTokens)`: Initialize with vocabulary
- `bool encode(const std::string& piece, std::vector<Rank>& output)`: Encode text piece using BPE
- `bool decode(const std::vector<Rank>& tokens, std::string& output, bool skipSpecialTokens)`: Decode tokens to text
- `bool hasToken(const std::string& token)`: Check if token exists in vocabulary
- `Rank getTokenRank(const std::string& token)`: Get rank for a token
- `std::string getRankToken(Rank rank)`: Get token for a rank

### PreTokenizer Classes

#### RegexSplit
- `RegexSplit(const std::string& pattern)`: Create regex splitter with pattern
- `std::vector<std::string> process(const std::string& text)`: Process text using regex

#### Sequence
- `void addStep(std::unique_ptr<PreTokenizer> step)`: Add processing step
- `std::vector<std::string> process(const std::string& text)`: Process through all steps

## Supported Tokenizer Formats

The tokenizer supports loading from HuggingFace model directories containing:

### Required Files
- **`tokenizer.json`**: Main tokenizer configuration containing:
  - Vocabulary mappings (token → ID)
  - Pre-tokenizer configuration (regex patterns, sequences)
  - Model type (BPE, Unigram, WordPiece)
  - Added tokens and special tokens

### Optional Files
- **`tokenizer_config.json`**: Special token definitions including:
  - BOS (Beginning of Sequence) token
  - EOS (End of Sequence) token  
  - PAD (Padding) token
  - UNK (Unknown) token

### Supported Model Types
- **BPE**: Fully implemented with merge rule processing
- **Unigram/SentencePiece**: Declared but not yet implemented
- **WordPiece**: Declared but not yet implemented

### Supported Pre-tokenizers
- **Regex-based splitting** with Unicode category support
- **Sequence of pre-tokenizers** for complex preprocessing
- **Case-insensitive matching** with (?i:) pattern support

## File Structure

```
tokenizer/
├── tokenizer.h/cpp          # Main Tokenizer class
├── tokenEncoder.h/cpp       # BPE encoding implementation  
├── preTokenizer.h/cpp       # Pre-tokenization classes
├── tokenizerUtils.h/cpp     # Unicode utilities and helpers
├── unicodeData.h/cpp        # Unicode category data tables
└── README.md                # This documentation
```

## Performance Characteristics

- **Memory Usage**: Optimized for edge deployment with smart memory allocation
- **Size Limits**: 
  - Maximum input text per piece: 1MB
  - Warning threshold: 64KB per piece
- **Unicode Support**: Full Unicode plane support (0x000000 - 0x10FFFF)
- **Thread Safety**: Not thread-safe; use separate instances per thread

## Error Handling

The tokenizer provides robust error handling:
- **File I/O errors**: Graceful handling of missing or corrupted tokenizer files
- **Malformed JSON**: Comprehensive JSON parsing error reporting
- **Unicode errors**: Proper handling of invalid UTF-8 sequences
- **Memory constraints**: Size limit enforcement with clear error messages

## Integration

The tokenizer is designed to work seamlessly with the TensorRT Edge-LLM framework and can be used with any supported LLM model that uses BPE tokenization.
