## LlamaV3 tokenizer example
```
#include <tokenizer.h>
#include <iostream>

int main()
{
    auto tokenizer = std::make_unique<Tokenizer>();

    tokenizer->loadFromHF("./Meta-Llama-3-8B-Instruct");

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
### Tokenizer unit test
1. Prepare tokenizer dir.
    ```
    export MODEL_PATH="./Meta-Llama-3-8B-Instruct/"
    ```

2. Build tokenizerTest. Add the following lines to `CMakelists.txt` and build target.
    ```
    add_executable(tokenizerTest tests/tokenizerTest.cpp)
    target_link_libraries(tokenizerTest PUBLIC tokenizer)
    ```
3. Run test
    ```
    # default input
    ./build/tokenizer/tokenizerTest ${MODEL_PATH}

    # custom input
    ./build/tokenizer/tokenizerTest ${MODEL_PATH} "This is my input text.👍"
    ```

### Performance test
1. Download [wikitext-103 dataset](https://dax-cdn.cdn.appdomain.cloud/dax-wikitext-103/1.0.1/wikitext-103.tar.gz)
    ```
    export DATA_ROOT="${HOME}/dataset"
    cd ${DATA_ROOT}
    wget https://dax-cdn.cdn.appdomain.cloud/dax-wikitext-103/1.0.1/wikitext-103.tar.gz
    tar -xzvf wikitext-103.tar.gz
    ```
2. Build performanceTest. Add the following lines to `CMakelists.txt` and build target.
    ```
    add_executable(performanceTest tests/performanceTest.cpp)
    target_link_libraries(performanceTest PUBLIC tokenizer)
    ```
3. Run test
    ```
    ./build/tokenizer/performanceTest ${MODEL_PATH} ${DATA_ROOT}/wikitext-103/wiki.train.tokens
    ```
