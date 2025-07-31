#!/bin/bash
set -e
echo "Running testing on device"

board_password={BOARDPASSWORD}
export LLM_SDK_DIR=$HOME/tensorrt-edge-llm
cd $LLM_SDK_DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LLM_SDK_DIR/TensorRT-jp6/lib
rm -rf build
mv build-jp6 build
cd build
./unitTest
cd $LLM_SDK_DIR
export ONNX_MODEL_DIR=$LLM_SDK_DIR/models
export MODEL_NAME=Qwen2.5-0.5B-Instruct
export ENGINE_DIR=$LLM_SDK_DIR/engines
mkdir -p $ENGINE_DIR
./build/examples/llm/llm_build --onnxPath=$ONNX_MODEL_DIR/$MODEL_NAME-fp16/model.onnx --enginePath=$ENGINE_DIR/$MODEL_NAME-fp16/model.engine --maxSeqLen 4096 --maxInputLen 128 --maxBatchSize 1 --dynamicShape
./build/examples/llm/llm_build --onnxPath=$ONNX_MODEL_DIR/$MODEL_NAME-int4/model.onnx --enginePath=$ENGINE_DIR/$MODEL_NAME-int4/model.engine --maxSeqLen 4096 --maxInputLen 128 --maxBatchSize 1 --dynamicShape
echo "Done testing engine build"
echo "Testing FP16 engine inference"
./build/examples/llm/llm_chat --enginePath=$ENGINE_DIR/$MODEL_NAME-fp16/model.engine --maxLength=64 --inputString="What is NVIDIA?" --tokenizerPath=$ONNX_MODEL_DIR/$MODEL_NAME-fp16/
./build/examples/llm/llm_benchmark --enginePath=$ENGINE_DIR/$MODEL_NAME-fp16/model.engine --inputLength=128 --maxLength=256 
echo "Testing INT4 engine inference"
./build/examples/llm/llm_chat --enginePath=$ENGINE_DIR/$MODEL_NAME-int4/model.engine --maxLength=64 --inputString="What is NVIDIA?" --tokenizerPath=$ONNX_MODEL_DIR/$MODEL_NAME-int4/
./build/examples/llm/llm_benchmark --enginePath=$ENGINE_DIR/$MODEL_NAME-int4/model.engine --inputLength=128 --maxLength=256 
echo "Done testing engine inference"
rm -rf $ENGINE_DIR
echo "Done running testing"
