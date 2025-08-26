# How to run TensorRT Edge LLM tests

## 1. Unit test (C++)

The C++ unit tests are built with [Google Test](https://github.com/google/googletest) framework and test core functionality of the TensorRT Edge LLM SDK including kernels, utilities, and low-level components.

Unit tests should be small, fast, and test only specific functions or components.

### Local execution

```bash
# In tensorrt-edge-llm source repo root dir
# Build the project first
mkdir -p build
cd build
cmake .. -DTRT_PACKAGE_DIR=/path/to/tensorrt -DBUILD_UNIT_TESTS=ON
make -j$(nproc)

# Run unit tests
./unitTest
```

### Dependencies
- TensorRT package
- CUDA toolkit
- CMake 3.18+
- Google Test (included as submodule)

## 2. Pipeline integration test (Python)

Pipeline integration tests use pytest to test the complete LLM workflow including model export, engine build, inference, benchmarking and accuracy. These tests validate end-to-end functionality.

All pipeline tests are located in `tests/test_llm_pipeline.py`, `tests/test_vlm_pipeline.py` and use configuration files to define test parameters.

### Prepare model files

Pipeline tests require ONNX model files. Set the environment variable `ONNX_MODEL_DIR` to point to your model directory:

```bash
export ONNX_MODEL_DIR=/path/to/onnx/models
export ENGINE_DIR=/path/to/engine/output
export LLM_SDK_DIR=$(pwd)
```

### Local execution

```bash
# Install pytest dependencies
pip install -r tests/requirements.txt

# Run specific test configuration
cd tests/
pytest --priority=l0_pipeline_a30 -v

# Run with HTML report
pytest --priority=l0_pipeline_orin -v --html=report.html --self-contained-html

# Run specific test task
pytest test_common.py::test_unit_tests -v
pytest test_llm_pipeline.py::TestLLMPipeline::test_engine_build[Qwen2.5-0.5B-Instruct-fp16-bs1-mxil2048-mxsl4096-osl128] -v
```

### Test types

1. **Project Build**: `test_build_project` - Auto-detects platform and builds the project
2. **Unit Tests**: `test_unit_tests` - Runs C++ unit tests (requires build)
3. **Engine Build**: `test_engine_build` - Tests TensorRT engine generation
4. **Chat Inference**: `test_inference_chat` - Tests interactive chat inference
5. **Benchmark**: `test_inference_benchmark` - Tests performance benchmarking

### Configuration files

Test configurations are stored in `tests/configs/`:

- `l0_pipeline_a30.yml` - L0 tests for A30 GPU platform
- `l0_pipeline_orin.yml` - L0 tests for Orin platform  
- `l0_pipeline_rtx5080.yml` - L0 tests for RTX5080 platform
- `l0_export.yml` - L0 model export tests

Each configuration defines test cases with model parameters:
```yaml
tests:
  - tests/test_common.py::test_build_project
  - tests/test_common.py::test_unit_tests
  - tests/test_llm_pipeline.py::test_engine_build[Qwen2.5-0.5B-Instruct-fp16-bs1-mxil2048-mxsl4096-osl128]
```

Parameter format: `ModelName-Precision-BatchSize-MaxInputLen-MaxSeqLen-OutputSeqLen`

### Test execution order

Tests should be run in the following order due to dependencies:

1. **`test_build_project`** - Auto-detects platform, CUDA version, TensorRT package and builds the project
2. **`test_unit_tests`** - Requires successful project build
3. **`test_engine_build`** - Requires successful project build  
4. **`test_inference_*`** - Requires successful engine build

The configuration files are ordered to respect these dependencies automatically.

### Common issues

1. **Model files not found**
   ```bash
   FileNotFoundError: ONNX model not found: /path/to/model.onnx
   ```
   Ensure `ONNX_MODEL_DIR` points to correct directory with model structure:
   ```
   models/
   ├── Qwen2.5-0.5B-Instruct-fp16/
   │   ├── model.onnx
   │   ├── onnx_model.data
   |   ├── config.json
   │   ├── tokenizer.json
   │   └── tokenizer_config.json
   └── Qwen2.5-VL-3B-Instruct-int4/
   │   ├── model.onnx
   │   ├── onnx_model.data
   |   ├── config.json
   │   ├── tokenizer.json
   │   ├── tokenizer_config.json
   │   ├── visual_enc_onnx_fp16
   │   │   ├── model.onnx
   │   │   └── config.json
   ```

2. **Build executables not found**
   ```bash
   Unit test executable not found: build/unitTest
   ```
   Build the project with unit tests enabled before running pytest.

3. **TensorRT library not found**
   ```bash
   OSError: libnvinfer.so.x: cannot open shared object file
   ```
   Set `LD_LIBRARY_PATH` to include TensorRT libraries:
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/tensorrt/lib
   ```

## 3. Export test (Python)

Model export tests validate the conversion from PyTorch/Hugging Face models to ONNX format. These tests are in `tests/test_model_export.py`.

```bash
# Install export dependencies
cd export/
pip install -r requirements.txt
pip install -r requirements_int4.txt

# Set model source directory
export TORCH_MODEL_DIR=/path/to/pytorch/models

# Run export tests
cd tests/
pytest test_model_export.py --priority=l0_export -v
pytest test_model_export.py -k "fp16" -v 

# Run specific export test
pytest test_model_export.py::test_llm_model_export[Qwen2.5-0.5B-Instruct-fp16] -v
```

# How to add test to CI

## 1. How does the CI work

The CI pipeline consists of multiple stages running on different platforms:

1. **Pre-commit checks** - Code quality and linting
2. **A30 model export** - Export models to ONNX on A30 GPU
3. **JP6 cross build** - Cross-compile for Jetson Orin
4. **Orin test** - Run tests on physical Orin device
5. **A30 build and test** - Build and test on A30 GPU

Test configurations are managed through YAML files in `tests/configs/`:

- Priority levels: `l0` (smoke tests), `l1` (extended tests)
- Platform specific: `l0_pipeline_a30.yml`, `l0_pipeline_orin.yml`, `l0_pipeline_rtx5080.yml`

## 2. Add a pipeline integration test

Pipeline integration tests run the complete workflow from project build to inference.

1. **Add test case to configuration file**

   Choose appropriate configuration file based on platform and priority:
   ```yaml
   # In tests/configs/l0_pipeline_a30.yml
   tests:
     - tests/test_common.py::test_build_project
     - tests/test_llm_pipeline.py::test_engine_build[MyModel-fp16-bs1-mxil2048-mxsl4096-osl128]
     - tests/test_llm_pipeline.py::test_inference_chat[MyModel-fp16-bs1-mxil2048-mxsl4096-osl128]
   ```

2. **Ensure model files are available**

   Add model preparation logic to export stage if needed.

3. **Test locally first**
   
   Test the build first:
   ```bash
   pytest test_common.py::test_build_project -v
   ```
   
   Then test your new model:
   ```bash
   pytest test_llm_pipeline.py::TestLLMPipeline::test_engine_build[MyModel-fp16-bs1-mxil2048-mxsl4096-osl128] -v
   ```

## 3. Add a unit test

C++ unit tests are automatically run via the `test_unit_tests` pytest case.

1. **Add C++ test file**
   ```cpp
   // In unittests/myNewTest.cpp
   #include <gtest/gtest.h>
   
   TEST(MyNewTest, BasicFunctionality) {
       // Test implementation
       EXPECT_EQ(expected, actual);
   }
   ```

2. **Update CMakeLists.txt**
   ```cmake
   # In unittests/CMakeLists.txt
   target_sources(unitTest PRIVATE
       myNewTest.cpp
   )
   ```

3. **Verify locally**
   ```bash
   # Build and run
   cd build
   make -j$(nproc)
   ./unitTest --gtest_filter="MyNewTest.*"
   ```

## 4. Platform-specific considerations

### A30 GPU
- Full CUDA compute capability
- Suitable for FP16/FP8/INT4 testing
- Larger memory for bigger models

### Jetson Orin
- ARM64 architecture
- Limited memory 
- Optimized for edge deployment
- Cross-compilation required


## 5. Debugging test failures

### Check logs
```bash
# Test logs are saved to logs/ directory
cat logs/test_unit_tests.log
cat logs/test_engine_build_Qwen2_5_0_5B_Instruct_fp16_bs1_mxil2048_mxsl4096_osl128.log
```

### Run with verbose output
```bash
pytest test_llm_pipeline.py -v -s --tb=long
```

### Debug specific test
```bash
pytest test_llm_pipeline.py::TestLLMPipeline::test_engine_build[Qwen2.5-0.5B-Instruct-fp16-bs1-mxil2048-mxsl4096-osl128] -v -s
```

### Check environment
```bash
# Verify required paths
echo $LLM_SDK_DIR
echo $ONNX_MODEL_DIR
echo $ENGINE_DIR
echo $LD_LIBRARY_PATH

# Check executables
ls -la build/unitTest
ls -la build/examples/llm/llm_build
```
