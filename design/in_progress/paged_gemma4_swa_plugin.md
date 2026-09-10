# Paged Gemma4 Sliding-Window Attention

Status: in progress

## Goal

Gemma4 alternates full-attention and sliding-window-attention (SWA) layers.
Keeping the engine's maximum sequence length for every SWA KV cache wastes
memory because attention never reads tokens older than the window `W`.

This design adds a bounded paged-KV policy to the existing `AttentionPlugin`.
It replaces the dense SWA cache with persistent storage bounded by `W`, without
copying or compacting the complete window each token.

## Support Boundary

Eligible Gemma4 FP16 exports are SWA-storage-capable automatically. There is no
user-facing export or build flag.

| Runtime mode | Attention semantics | Physical KV storage | Context reuse |
| --- | --- | --- | --- |
| Context reuse disabled, bounded pages < full pages | Sliding on SWA layers | Bounded paged SWA KV | Bypassed |
| Context reuse disabled, bounded pages >= full pages | Sliding on SWA layers | Full paged KV | Bypassed |
| Context reuse enabled | Sliding on SWA layers | Full paged KV | Supported |
| FP8 KV or speculative decoding | Sliding on SWA layers | Full paged KV | Existing behavior |

All modes use `AttentionPlugin`. `sliding_window_size=W` always controls
attention semantics. The serialized `supports_bounded_kv_cache` capability
advertises that an SWA layer supports bounded storage; it does not force the
runtime policy. The internal one-dimensional `swa_kv_cache_mode` input selects
the policy by shape: length 1 uses bounded storage and length 0 uses full
storage. The tensor data is never read. An engine supports one bounded window
size.

## Architecture

```text
LLMInferenceRuntime
  `-- LLMRankRuntime
        |-- SharedResources: physical KV pools and page tables
        `-- ManagedKVCacheRequest
              |-- reuse on:  ContextCacheRequest -> ContextCacheCoordinator
              `-- bounded:   BoundedSwaKVPageManager::RequestHandle
                                      |
                                      v
                              dual-mode AttentionPlugin
```

Only one physical policy is allocated for a runtime instance. With context
reuse disabled, bounded storage is selected only when its `2M+1` page budget is
strictly smaller than the full pool. In that mode, full-attention and
bounded-SWA layers have independent page-ID spaces, reservations, and page
tables. Otherwise, all layers use full pools and the engine's
`swa_kv_page_table` binding aliases the ordinary full page table. Shared-KV
consumers bind their donor writer's pool and table, advertise the same bounded
capability, and receive the same runtime policy.
They never write or own pages. During bounded prefill they also consume the
donor's normalized, pre-RoPE current K/V as transient graph values because a
chunk larger than `W` is intentionally not all resident in the sparse pool.

The SWA-capable plugin ABI uses paged attention inputs plus a shape carrier:

```text
swa_kv_cache      [2, numPages, 128, Hkv, D]
swa_kv_page_table [B, 2, maxPagesPerSequence]
swa_kv_cache_mode [0] full storage, [1] bounded storage
```

The cache binding's optimization profile covers both the bounded and full page
counts and optimizes for the smaller count. The plugin reads prepared mappings,
writes K/V, and computes attention. Page allocation, mapping, retirement, and
ownership remain runtime responsibilities.

The responsibilities are intentionally separate:

| Component | Owns |
| --- | --- |
| `KVCacheManager` / `SharedResources` | Physical full and SWA buffers |
| `BoundedSwaKVPageManager` | SWA page IDs, reservations, absolute window state, and deferred retirement |
| `KVPageTable` | Sparse execution mappings uploaded to the kernels |
| `ContextCacheCoordinator` | Reusable full-cache records and their existing lookup/publication lifecycle |
| `AttentionPlugin` | Attention computation and K/V reads/writes through prepared mappings |

`ManagedKVCacheRequest` is the runtime facade over the two current request
backends. They are mutually exclusive in this implementation: context reuse selects full
storage, while bounded storage is selected only when context reuse is off. The
facade keeps prefill, decode, compaction, and finish ordering in one runtime
lifecycle without making context reuse the owner of SWA rotation.

## Request Lifecycle

For page size `P`, one active slot reserves:

```text
M = ceil(W / P) + 1
private pages = 2 * M + 1
```

`M` covers one retained window plus page-boundary allowance. Two bounded
images permit a chunk transition while the previous image remains live until
execution completes. The final page guarantees writable decode advancement
after admission.

1. `BoundedSwaKVPageManager::beginRequest()` atomically reserves all private pages
   needed by every sequence in the admitted batch. Admission depends only on
   batch size, not on a provisional prompt length.
2. After embedding assembly and optional visual-token pruning finalize the
   executed prefill lengths, the runtime manager prepares the SWA page-table
   bindings. `KVPageTable::setEntry()`, `clearEntry()`, and `uploadDirty()`
   transfer only changed K/V mapping pairs.
3. `prepareDecodeStep()` maps at most one new page and queues at most one stale
   page at a boundary. Tokens within a page do not update the table.
4. `completeDecodeStep()` runs after the existing completion point and recycles
   queued stale pages. A page is never recycled while a kernel may still read
   it.
5. Batch eviction compacts full and SWA table rows, compacts slot-addressed
   length/state tensors, waits at the existing eviction synchronization point,
   and then releases dropped sequence reservations.
6. Request completion clears the remaining sparse rows and releases all pages.

`SwaKVCacheState` belongs to a runtime request handle, not to a context-cache
lease or record. Its exact resident endpoint and retained logical page range
remain absolute across rotation and batch compaction. The compact physical
bindings are only an execution view; they do not identify a reusable prefix.

## Plugin Execution

- **Normal prefill:** run attention directly on current Q/K/V and persist only
  the final retained window.
- **Chunked prefill:** gather the previous paged window, append the current
  roped K/V in temporary `[B, W + chunk, Hkv, D]` workspace, run bottom-right
  sliding FMHA, and commit only the final retained pages.
- **Decode:** write through the sparse table and launch paged XQA with
  `slidingWindowSize=W`. There is no full-window D2D copy or compaction.
- **Vision and KV sharing:** preserve Gemma4 vision-block masking and bind
  shared-KV consumers to the donor's pool. Bounded consumers combine the
  resident window with transient current donor K/V and leave the donor pool
  immutable. Full-cache consumers read the donor pool directly.

## Context Reuse Boundary

Bounded SWA requests do not instantiate `ContextCacheCoordinator`. Their SWA
pages are not published in `CacheRecord`, shared across requests, pinned by LRU
records, or restored at a reused prefix. The context-cache resource taxonomy,
leases, records, and eviction planner contain no SWA resource type or window
state.

When context reuse is enabled, or when the bounded page budget is not smaller,
runtime initialization selects full storage, aliases the SWA table binding to
the ordinary full page table, and sets `swa_kv_cache_mode` to shape `[0]`.
`AttentionPlugin` still applies `sliding_window_size`, while the existing
context-reuse implementation treats every layer as ordinary attention. Lookup,
publication, branching, rewind, and eviction behavior is unchanged.

Reduced-memory context reuse is follow-up work. It can compose the existing
context request with `BoundedSwaKVPageManager` through the runtime facade, but it
still requires a cache-record representation for sliding-window state, a
reuse-match policy for branch points, immutable shared-page ownership,
copy-on-write continuation, rewind rules, and atomic publication/eviction
across full and SWA pools. Context reuse may seed or publish settled SWA state;
it must not become the owner of window rotation.

The legacy `save_system_prompt_kv_cache` snapshot API is also unavailable in
bounded mode because it captures a complete contiguous KV history. A generation
request carrying that flag logs a warning and continues without saving the
snapshot. Applications that require prompt reuse should enable generic context
reuse, which selects full KV storage for this engine.

## Invariants

- Active mappings use pages from the bounded SWA pool and contain no duplicate
  physical page IDs.
- Sparse holes are allowed only outside the retained logical range. Slot reuse
  clears stale mappings before assigning the row to another request.
- Admission reserves enough pages that an active request cannot fail because
  another request consumed transition capacity.
- Invalid windows, mappings, pool geometry, or pending transitions fail before
  kernel launch.
- Full mode has at least one physical page for every logical table column;
  bounded mode may map a wider logical range onto its smaller sparse page-ID
  space.
- Existing engines must be re-exported and rebuilt for the bounded paged ABI.

## Complexity

- Persistent SWA KV is `O(batch * (W + P))` and independent of the engine's
  maximum sequence length.
- Decode cache maintenance is `O(1)`: zero updates within a page, or one map
  plus one deferred retirement at a boundary.
- Attention remains `O(W)` and is outside the cache-maintenance bound.
- Chunked-prefill metadata scales with pages crossed and its temporary
  workspace scales with `W + chunk`. KV sharing also carries the donor's
  current chunk transiently; persistent storage remains bounded by `W`.

## Validation

Runtime-manager tests cover atomic admission, bounded reservations, prefill
retention for `W<P`, `W=P`, divisible and non-divisible windows, constant-size
decode patches beyond `2W`, page recycling, continued decode after batch
compaction, reference cleanup, and failed-synchronization quarantine during
abandonment and compaction. Page-table tests cover sparse ranges,
non-identity physical IDs, dirty K/V patches, invalid mappings, and slot reuse.
The existing context-cache tests continue unchanged and do not exercise SWA
ownership.

Plugin tests compare bounded and full runtime policies with PyTorch for normal
and ragged prefill, chunked prefill, decode beyond `2W`, scrambled physical
page IDs, a physical pool smaller than the logical page-table range,
vision-block attention, and KV sharing. Export tests verify that one
eligible FP16 engine carries both runtime policies, all nodes use
`AttentionPlugin`, and FP8/speculative modes retain the full-cache fallback.

Thor acceptance validation must use the current source revision and one Gemma4
FP16 engine with context reuse disabled and enabled. It checks output and
benchmark-score parity, exact KV-pool allocation, runtime-ready CUDA memory,
and five-run median decode latency. Acceptance requires material KV-pool
savings, no functional degradation, no full-window copy per decode token, and
no material latency regression.

## Code Map

- `cpp/runtime/managedKVCacheRequest.*`: common runtime lifecycle facade
- `cpp/runtime/state/boundedSwaKVPageManager.*`: bounded page ownership and rotation
- `cpp/runtime/state/contextCache/`: unchanged full-cache reuse lifecycle
- `cpp/runtime/state/kvPageTable.*`: sparse mappings and dirty uploads
- `cpp/runtime/kvCacheManager.*`: independent physical pools
- `cpp/runtime/state/pipelineIO.cpp`: pool and table binding
- `cpp/plugins/attentionPlugin/`: unified plugin and bounded cache policy
- `cpp/kernels/contextAttentionKernels/`: paged writes and chunk assembly
- `tensorrt_edgellm/models/gemma4/`: export routing and plugin attributes
