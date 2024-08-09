## Build
```
cd tokenizer
mkdir build && cd build

cmake ..
# options: -DCUDA_VERSION=12.4 -DTRT_LIB_DIR=/usr/local/tensorrt/lib/ -DTRT_INCLUDE_DIR=/usr/local/tensorrt/include/

make -j$(nproc)
```

## LlamaV3 tokenizer example
```
#include <tokenizer.h>
#include <iostream>

int main()
{
    Tokenizer* tokenizer = new LlamaV3Tokenizer;

    // load tokenizer from tiktoken format
    tokenizer->loadFromTiktoken("/home/builder/depot/llm-models/llama-models-v3/8B-instruct/tokenizer.model");

    // // load tokenizer from hf format is relatively slow. Recommend loading from tiktoken.
    // tokenizer->loadFromHF("/home/builder/depot/llm-models/llama-models-v3/llama-v3-8b-instruct-hf");

    // tokenize
    std::string input = "<|begin_of_text|>hello¢ nvidia，你好！👍<|end_of_text|>";
    auto token = tokenizer->encode(input, false);

    // detokenize
    auto output = tokenizer->decode(token);
    std::cout << output << std::endl;
    assert(output == input);

    return 0;
}
```

## Tests
### LlamaV3 tokenizer unit test
1. Prepare LlamaV3 tokenizer path. Supports both tiktoken and HF format.
    ```
    # tiktoken format
    export MODEL_PATH="${HOME}/llm-models/llama-models-v3/8B-instruct/tokenizer.model"

    # hf format
    export MODEL_PATH="${HOME}/llm-models/llama-models-v3/llama-v3-8b-instruct-hf"
    ```
2. (Optional) Install python dependencies to enable correctness check. 
    - Python script uses official llama 3 tokenizer implementation to generate golden tokens. If not enabled, it's up to the user to determine if the output tokens are correct.
    - Requires `MODEL_PATH` to be of tiktoken format. Otherwise, python script won't be invoked.
    ```
    pip3 install -r ../tests/requirements.txt
    ```
3. Run test
    ```
    cd tokenizer/build

    # default input
    ./llama3Test ${MODEL_PATH}

    # custom input
    ./llama3Test ${MODEL_PATH} "This is my input text.👍"

    # custom input with addSpecial = True
    ./llama3Test ${MODEL_PATH} "This is my input text.👍" 1
    ```
4. Sample output
    ```
    [LOG]: Loaded LlamaV3Tokenizer from /home/builder/depot/llm-models/llama-models-v3/8B-instruct/tokenizer.model
    [LOG]: Input: This is my input text.👍
    [LOG]: Token: 128000, 2028, 374, 856, 1988, 1495, 13, 9468, 239, 235, 128001, 
    [LOG]: Expected: 128000, 2028, 374, 856, 1988, 1495, 13, 9468, 239, 235, 128001, 
    [LOG]: Passed
    ```

### Performance test
1. Download [wikitext-103 dataset](https://dax-cdn.cdn.appdomain.cloud/dax-wikitext-103/1.0.1/wikitext-103.tar.gz)
    ```
    export DATA_ROOT="${HOME}/dataset"
    cd ${DATA_ROOT}
    wget https://dax-cdn.cdn.appdomain.cloud/dax-wikitext-103/1.0.1/wikitext-103.tar.gz
    tar -xzvf wikitext-103.tar.gz
    ```
2. Prepare LlamaV3 tokenizer path. Supports both tiktoken and HF format.
    ```
    # tiktoken format
    export MODEL_PATH="${HOME}/llm-models/llama-models-v3/8B-instruct/tokenizer.model"

    # hf format
    export MODEL_PATH="${HOME}/llm-models/llama-models-v3/llama-v3-8b-instruct-hf"
    ```
3. Run test
    ```
    cd tokenizer/build
    ./perfTest ${MODEL_PATH} ${DATA_ROOT}/wikitext-103/wiki.train.tokens
    ```
4. Performance on A100 (x86_64 AMD EPYC 7313P 16-Core Processor)

    |  | Encode throughput (MB / s) | Decode throughput (MB / s) |
    | ------------------- | ---- | ---- |
    | Drive llm tokenizer | 1.19 | 34.97 |
    | llama.cpp tokenizer | 0.85 | 26.63 |
    | Tiktoken | 4.75 | 62.86 |
