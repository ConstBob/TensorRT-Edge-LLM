# QSA Attention Plugin (prefill v1)

Status: in progress (v1 = prefill-only, padded activations, FP16 surface).

## What QSA is

Qwen Sparse Attention, the sparse attention of Qwen3.8-Flash-Next
(`qwen4_exp`): a weight-free block-compressed **indexer** scores KV blocks of
`compress_ratio = 4` tokens and selects the top `indexer_budget /
compress_ratio = 512` blocks per query; the main **GQA attention** (24 Q
heads / 2 KV heads, head_dim 256) attends only to the expanded token list
(width `indexer_budget + compress_ratio - 1 = 2051`, always including the
current partial block's tail). Reference implementations: HF transformers
`qwen4_exp`, sglang PR #36497, vLLM PR #53896.

The model's QSA layers also carry per-head Gemma-style qk-norm
(`normalize(x) * (1 + w)`), partial RoPE (rotary_dim 64 of 256, theta 1e7)
and an attention output gate (`out * sigmoid(gate)`); the gate and all
projections stay in the ONNX graph.

## Indexer math (exact contract)

Inputs: `index_qk = index_qk_proj(h)`, `[B, S, (4 + 1) * 128]`, no bias.

1. Split into 4 index-q heads of 128 and 1 raw index-k of 128.
2. q: Gemma RMSNorm (`(1 + w)`, fp32 math) then partial neox rope (first 64
   dims) at the row position.
3. Compressed keys per complete group `g` (`4g + 3 < L`): **fp32 mean of the
   4 raw keys -> cast to fp16 -> Gemma RMSNorm -> rope at position `4g`**
   (cast ordering is bit-compat with the reference).
4. Scores: `logits[t, g] = sum_h relu(q[t,h] . kbar[g]) / sqrt(128)`, fp32;
   visible blocks `g < (t + 1) / 4` (block-granular causality — no
   partial-block masking).
5. Per row: top-512 blocks -> expand `4g + {0..3}` packed to front -> append
   the `(t + 1) % 4` tail tokens (ALWAYS) -> `-1`-pad to 2051, int32,
   unsorted, distinct.
6. Invariant: when `(t + 1) / 4 <= 512` the selection equals the full causal
   prefix — short sequences are exactly dense.

## Plugin contract

`trt_edgellm::QsaAttentionPlugin` version "1". **All 11 inputs required,
zero optional slots** — no export.py compaction pass. New v2 inputs are
appended at the end (the `swa_kv_cache_mode` evolution pattern).

| # | name | dtype | shape |
|---|------|-------|-------|
| 0 | `qkv` | FP16 | `[B, S, (Hq + 2*Hkv) * D]` |
| 1 | `index_qk` | FP16 | `[B, S, (indexer_n_heads + 1) * indexer_head_dim]` |
| 2 | `past_key_value` | FP16 | `[2, numPages, 128, Hkv, D]` |
| 3 | `context_lengths` | INT32 | `[B]` |
| 4 | `rope_rotary_cos_sin` | FP32 | `[1 or B, maxPos, 64]` (shared by the main partial rope and the indexer rope) |
| 5 | `kvcache_start_index` | INT32 | `[0]` = prefill sentinel (v1 rejects anything else) |
| 6 | `kv_page_table` | INT32 | `[B, 2, maxPagesPerSeq]` |
| 7 | `q_norm_gamma` | FP16 | `[D]`, Constant weight, pre-folded `1 + w` |
| 8 | `k_norm_gamma` | FP16 | `[D]`, Constant weight, pre-folded `1 + w` |
| 9 | `indexer_q_norm_gamma` | FP16 | `[128]`, Constant weight, RAW `w` (kernel applies `1 + w`) |
| 10 | `indexer_k_norm_gamma` | FP16 | `[128]`, Constant weight, RAW `w` |

Outputs: `attn_output [B, S, Hq, D]` FP16; `present_key_value` (in-place
aliased pool; `getAliasedInput` returns -1 per the Myelin WAR precedent).

Attributes: `num_q_heads`, `num_kv_heads`, `head_size`, `indexer_n_heads`,
`indexer_head_dim`, `indexer_budget`, `indexer_compress_ratio`,
`attention_scale` (0 => `1/sqrt(head_size)`), `rms_norm_eps`.

## enqueue flow (prefill)

1. `kernel::launchApplyRopeFromPackedToSplit` — split packed QKV, fused
   per-head qk-norm (Llama convention `normalize(x) * gamma`; the export
   side folds `1 + w` into the gammas), partial rope, paged KV write, and
   dense scratch mirrors `qScratch` / `kScratchOut` / `vScratchOut`
   `[B, S, H, D]` with `cuQSeqLens` ragged Q zeroing.
2. `kernel::runQsaIndexerPrefill` (`cpp/kernels/qsaIndexer/`) — plain CUDA
   pipeline: q-prep, k-compress, fp32 scores,
   `cub::DeviceSegmentedRadixSort` top-512 (row-chunked, ~128 MiB sort
   working set), expand + tail -> `outIdx [B, S, 2051]` int32 in workspace.
3. `CuteDslQsaSparsePrefillRunner::run` (`cpp/kernels/qsaAttention/`,
   CuTe-DSL AOT group `qsa`, `kernelSrcs/qsa_cutedsl/qsa_sparse_gqa.py`) —
   sparse GQA over the dense mirrors + index lists.

## Numerics rules

- fp32 mean -> cast -> gemma-norm, in that order (indexer compressed keys).
- relu per head -> sum over heads -> `/sqrt(128)`, fp32 (indexer scores).
- Sparse kernel: scale applied to FP32 scores at exp2 time, never pre-folded
  into fp16 Q (vLLM ordering).
- All-invalid index rows and padding rows produce exact zeros.
- Main-path gemma fold: `gamma = 1 + w` through the existing fused rope
  kernel (half-precision gamma multiply — a small fp16 deviation from HF's
  fp32 multiply, covered by test tolerances).

## Deferred to v2

Decode (compressed-key side cache + per-row radix-select top-k + sparse
decode kernel), chunked prefill (`splitPagedKV` read-back), BF16 surface,
FP8/NVFP4 KV, paged sparse kernel, MRoPE-3D (VL), compact
block-ids+tail-count index format, indexer chunk-fusion (removes the
`outIdx` workspace term), 2-kv-head fused CTA, full `qwen4_exp` model class
and experimental-builder parity.
