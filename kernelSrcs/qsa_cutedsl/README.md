# QSA sparse-GQA prefill kernels (CuTe DSL)

Qwen Sparse Attention (QSA, Qwen3.8-Flash-Next): the QSA indexer selects, per
query token, the top-512 KV blocks of 4 tokens, expands them to token ids and
appends the current partial block's tail — an int32 list of width
`indexer_budget + compress_ratio - 1 = 2051`, `-1`-padded, unsorted, distinct,
all `< token + 1`. The kernel here attends ONLY to the listed tokens; there is
no causal or positional mask in-kernel.

## Files

- `qsa_sparse_gqa.py` — `QSASparseGQAPrefill`: structural fork of
  `fmha_v2_cutedsl/fmha.py` (same swizzled SMEM atoms, m16n8k16 tiled MMA,
  ldmatrix double-buffered BMM loops and exp2 online softmax) with a sparse
  per-row cp.async gather replacing the affine KV traversal.

## Kernel shape

- Grid `(S, H_kv, B)` — one CTA per (query token, KV head).
- M tile = the token's GQA head group (`H_q / H_kv` rows, zero-padded to
  `m_block_size = 16`); head-padding rows are never loaded or stored.
- The index list is walked in `n_block_size = 32` chunks; one warp-wave
  cp.async-gathers one 256-element K/V row (32 lanes x 128 bits). Chunk
  indices are staged in a ping-pong SMEM buffer shared by the K gather, the
  V gather and the score mask.
- `-1` entries: zero-filled K/V rows AND `-inf` scores (NaN hardening).
- Padding query rows (`t >= context_lengths[b]`) and all-invalid rows store
  exact zeros (`row_sum == 0` guard).
- The softmax scale multiplies the FP32 scores at exp2 time — it is never
  pre-folded into the fp16/bf16 Q (matches the vLLM QSA numerics ordering).

Baked at compile time: `head_dim` (256) and the `(Br, Bc, threads)` tuning.
Runtime-dynamic: batch, seq, `H_q`, `H_kv` (group size <= 16), `topk`,
strides, softmax scale.

## Variants (`build_cutedsl.py` group `qsa`)

| name | dtype | SMs |
|---|---|---|
| `qsa_sparse_d256` | fp16 | 100, 101, 110 |
| `qsa_sparse_d256_bf16` | bf16 | 100, 101, 110 |

## Standalone test / benchmark (needs a GPU)

```bash
python kernelSrcs/qsa_cutedsl/qsa_sparse_gqa.py \
    --batch_size 2 --seqlen 384 --num_head 24 --kv_group_size 12 \
    --head_dim 256 --topk 2051 --ragged
```

The in-file FP32 CuPy oracle checks the gather-softmax result against
NaN-poisoned padding and a sentinel-poisoned output buffer (padding rows must
come back as exact zeros).

## AOT build

```bash
python kernelSrcs/build_cutedsl.py --kernels qsa --gpu_arch sm_110 --arch aarch64
```

Consumed by `cpp/kernels/qsaAttention/cuteDslQsaSparseRunner.{h,cpp}` under
`CUTE_DSL_QSA_ENABLED` (cmake `-DENABLE_CUTE_DSL="fmha;qsa"`).
