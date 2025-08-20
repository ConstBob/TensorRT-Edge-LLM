# Generating FMHA v2 cubin

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout 636c622bb8685b9db7422b3fa064a173cf1ff2a8
git apply gen_fmha_cubin.patch
cd cpp/kernels/fmha_v2
```

## Generate cubin for SM 80/86/87/89/101 with CUDA 12.8

```bash
# 1) Generate the arch–specific .cu sources & headers
export GENERATE_EDGE_LLM=1 GENERATE_CUBIN=1 
python3 setup.py

# 2) Build the cubins (old BERT parameter layout)
make cubin_demobert -j$(nproc)

# 3) Avoid overwrite
mv generated generated_cuda128
```
---

## Generate cubin for SM 120 / 121 with CUDA 12.9

```bash
export GENERATE_EDGE_LLM=1 GENERATE_CUBIN=1 ENABLE_SM12X=1 

# 1) Generate Blackwell-only .cu sources & headers
python3 setup.py

# 2) Build the cubins (old BERT parameter layout)
make cubin_demobert -j$(nproc)

# 3) Avoid overwrite
mv generated generated_cuda129
```

## Kernel Unit Test
```bash
ln -s generated_cuda128 generated

make bin/fmha.exe -j$(nproc)

bin/fmha.exe -v 1 -runs 5 -warm-up-runs 2  -s 1024 -d 128  -b 1 -causal-mask -grouped-query-attention 2 -h 14 -fix-s
bin/fmha.exe -v 1 -runs 5 -warm-up-runs 2  -s 128 -d 64  -b 1 -causal-mask -grouped-query-attention 2 -h 14 -fix-s -force-non-tiled
```