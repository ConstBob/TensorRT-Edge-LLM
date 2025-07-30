#!/bin/bash
set -e
echo "Running testing on device"

board_password={BOARDPASSWORD}
cd /home/nvidia/tensorrt-edge-llm
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nvidia/tensorrt-edge-llm/TensorRT-jp6/lib
rm -rf build
mv build-jp6 build
cd build
./unitTest

echo "Done running testing"
