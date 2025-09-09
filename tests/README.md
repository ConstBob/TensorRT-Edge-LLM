# TensorRT Edge LLM Testing

## Quick Start

### 1. Environment Setup
```bash
export ONNX_DIR=/path/to/onnx/models
export ENGINE_DIR=/path/to/engine/outputs
export LLM_SDK_DIR=$(pwd)
export TORCH_DIR=/path/to/pytorch/models  # For export tests
```

### 2. Install Dependencies
```bash
pip install -r tests/requirements.txt
```

### 3. Build Project
```bash
# Install Python package
pip install build
python -m build --wheel --outdir dist .
pip install dist/*.whl

# Build C++ components
mkdir -p build
cd build
cmake .. -DTRT_PACKAGE_DIR=/path/to/tensorrt -DBUILD_UNIT_TESTS=ON
make -j$(nproc)
cd ..
```

### 4. Run Tests
```bash
# Under tensorrt-edgellm root.
pytest --priority=l0_pipeline_a30 -v
```

## Test Types

### Unit Tests (C++)
```bash
cd build && ./unitTest
```

### Export Tests (Python)
```bash
pytest tests/test_model_export.py --priority=l0_export -v
```

### Package Tests (Python)
```bash
pytest tests/test_llm_pipeline.py --priority=l0_pipeline_rtx5080 -v
```

### Pipeline Tests


## Configuration

### Test Configs (`tests/configs/`)
- `l0_pipeline_a30.yml` - A30 GPU tests
- `l0_pipeline_orin.yml` - Jetson Orin tests  
- `l0_pipeline_rtx5080.yml` - RTX5080 tests
- `l0_export.yml` - Export tests

### Parameter Format
```
ModelName-Precision-MaxBatchSize-MaxInputLen-MaxSeqLen-OutputSeqLen
```

**Examples:**
- `Qwen2.5-0.5B-Instruct-fp16-mxbs1-mxil2048-mxsl4096-osl128`
- `InternVL3-1B-hf-int4_awq-mxbs1-mxil2048-mxsl4096-mxbs1-mnit128-mxit1024`

**Parameters:**
- `mxbs1` = max batch size 1
- `mxil2048` = max input length 2048
- `mxsl4096` = max sequence length 4096  
- `osl128` = output sequence length 128
- `mnit128` = min image tokens 128 (VLM)
- `mxit1024` = max image tokens 1024 (VLM)

## Directory Structure

### ONNX Models (`ONNX_DIR/`)
```
Qwen2.5-0.5B-Instruct-fp16-4096/
├── model.onnx
├── onnx_model.data
├── config.json
├── tokenizer.json
└── tokenizer_config.json

InternVL3-1B-hf-int4_awq-4096/
├── model.onnx
├── onnx_model.data
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── visual-fp16/
    ├── model.onnx
    └── config.json
```

### Engine Output (`ENGINE_DIR/`)
```
llm_engines/
└── Qwen2.5-0.5B-Instruct-fp16-4096/

visual_engines/
└── InternVL3-1B-hf-int4_awq-4096/
```

## CI Process

The CI pipeline follows this workflow:

### Stage 1: Pre-commit Checks
- Code quality and linting checks
- Runs on CPU-only runners

### Stage 2: Build
- **`a30_model_export`**: Export PyTorch models to ONNX on A30 GPU
- **`jp6_cross_build`**: Cross-compile for Jetson Orin (ARM64)

### Stage 3: Test
- **`orin_test`**: Run tests on physical Jetson Orin device (remote execution)
- **`a30_build_and_test`**: Build and test on A30 GPU
- **`rtx5080_build_and_test`**: Build and test on RTX5080 GPU

### Local Test Execution Order
1. `test_build_project` - Build project
2. `test_unit_tests` - Run C++ unit tests  
3. `test_engine_build` - Build TensorRT engines
4. `test_inference_*` - Run inference tests

## Troubleshooting

### Model Files Not Found
```bash
FileNotFoundError: ONNX model not found
```
**Fix**: Verify `ONNX_DIR` path and model structure.

### Build Executables Not Found
```bash
Unit test executable not found: build/unitTest
```
**Fix**: 
1. Install package: `python -m build --wheel --outdir dist . && pip install dist/*.whl`
2. Build project with `-DBUILD_UNIT_TESTS=ON`

### TensorRT Library Not Found
```bash
OSError: libnvinfer.so.x: cannot open shared object file
```
**Fix**: Set `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/tensorrt/lib
```

### Debug Commands
```bash
# Check environment
echo $LLM_SDK_DIR $ONNX_DIR $ENGINE_DIR $LD_LIBRARY_PATH

# Check executables
ls -la build/unitTest build/examples/llm/llm_build

# Verbose test output
pytest test_llm_pipeline.py -v -s --tb=long

# Check logs
cat logs/test_unit_tests.log
```

## Adding Tests

### Pipeline Tests
1. Add to config file (`tests/configs/l0_pipeline_a30.yml`):
   ```yaml
   tests:
     - tests/test_llm_pipeline.py::test_engine_build[MyModel-fp16-bs1-mxil2048-mxsl4096-osl128]
   ```
2. Ensure model files in `ONNX_DIR`
3. Test locally first

### Unit Tests
1. Add C++ test file in `unittests/`
2. Update `CMakeLists.txt`
3. Build and verify:
   ```bash
   cd build && make -j$(nproc) && ./unitTest --gtest_filter="MyNewTest.*"
   ```

